"""
Wrapper around PoseSwin (BrownBear_ReID, EPFL Mathis Lab) for cross-video bear
identity assignment.

Provides:
  • PoseSwinIdentifier — loads the model once, computes 512-d embeddings
  • Gallery            — JSON-persisted name → embedding store with cosine match

Usage
-----
    identifier = PoseSwinIdentifier()
    gallery    = Gallery.load("data/identity/bear_gallery.json")

    emb = identifier.embed(head_crop_bgr)          # (512,)
    name, sim = gallery.match(emb, threshold=0.6)
    if name is None:
        name = gallery.add_anonymous(emb)          # auto-name e.g. "Bear A"
    gallery.save()
"""

from __future__ import annotations

import json
import logging
import os
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POSESWIN_REPO = PROJECT_ROOT / "external" / "BrownBear_ReID" / "PoseGuidedReID"
CKPT_ROOT = PROJECT_ROOT / "external" / "BrownBear_ReID" / "Public_release" / "checkpoints"

REID_CKPT = CKPT_ROOT / "reid_ckpts" / "katmai_exps" / "6y_model" / "net_best.pth"
POSE_CKPT = CKPT_ROOT / "preprocessing_ckpts" / "pose" / "hrnet_w48_balanced_n13_refined.pth"

# Allow PoseGuidedReID's internal imports to resolve
sys.path.insert(0, str(POSESWIN_REPO))


def _silent_logger():
    log = logging.getLogger("poseswin_identifier")
    log.setLevel(logging.WARNING)
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic head crop from a YOLO bear bbox
# ─────────────────────────────────────────────────────────────────────────────

def head_crop_from_bear(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Heuristic: take upper 50%, central 60% of the bear bbox.

    PoseSwin was trained on faster-rcnn head crops. We don't ship that detector,
    so this proxy gives the right region for typical Brooks Falls footage where
    bears are upright and viewed from the side.
    """
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    yt = y1
    yb = y1 + int(h * 0.50)
    xl = x1 + int(w * 0.20)
    xr = x2 - int(w * 0.20)

    H, W = frame_bgr.shape[:2]
    yt, yb = max(0, yt), min(H, yb)
    xl, xr = max(0, xl), min(W, xr)
    return frame_bgr[yt:yb, xl:xr]


# ─────────────────────────────────────────────────────────────────────────────
# PoseSwin model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PoseSwinIdentifier:
    """Loads the Katmai-trained PoseSwin model once; computes head embeddings."""

    def __init__(self, device: Optional[str] = None):
        from project.config import cfg
        from project.models.build_swint import build_swin_reid
        from project.utils.tools import load_model

        assert REID_CKPT.exists(), f"Missing PoseSwin checkpoint: {REID_CKPT}"
        assert POSE_CKPT.exists(), f"Missing HRNet pose checkpoint: {POSE_CKPT}"

        cfg.defrost()
        swin_cfg = POSESWIN_REPO / "configs" / "swim_transformer" / "swin" / "swin_base_patch4_window7_224_22k.yaml"
        cfg.merge_from_file(str(swin_cfg))
        cfg.MODEL.TYPE = "swin"
        cfg.MODEL.NAME = "swin_base_patch4_window7_224_22k"
        cfg.MODEL.AGG_POSE_FEATURE = True
        cfg.MODEL.POSE_HRNET = "hrnet_w48"
        cfg.MODEL.NUM_JOINTS = 13
        cfg.MODEL.POSE_WEIGHT = str(POSE_CKPT)
        cfg.MODEL.PRETRAIN_PATH = ""
        cfg.MODEL.DEVICE_ID = (0,)
        cfg.MODEL.SWIN.EMBED_DIM = 128
        cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
        cfg.MODEL.SWIN.NUM_HEADS = [4, 8, 16, 32]
        cfg.DATASETS.NAMES = "bear"
        cfg.INPUT.IMG_SIZE = [224, 224]
        cfg.freeze()

        if device is None:
            from src.config import get_device
            device = get_device()
        self.device = torch.device(device)

        self.model = build_swin_reid(
            num_classes=109,
            logger=_silent_logger(),
            linear_num=512,
            cfg=cfg,
            load_weights=False,
            return_feature=True,
            device=self.device,
        )
        self.model = load_model(
            self.model, str(REID_CKPT),
            logger=_silent_logger(), remove_fc=False, local_rank=0,
        )
        self.model.to(self.device).eval()

        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Compute a 512-d L2-normalized embedding for one head crop."""
        if image_bgr.size == 0:
            raise ValueError("Empty image passed to PoseSwinIdentifier.embed()")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        emb = out[1] if isinstance(out, (list, tuple)) else out  # (1, 512)
        emb = emb.squeeze(0).float().cpu().numpy()
        # L2-normalize so cosine == dot product
        n = np.linalg.norm(emb)
        return emb / max(n, 1e-12)

    def embed_batch(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Batched embedding. Returns (N, 512) L2-normalized."""
        if not images_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        tensors = []
        for img in images_bgr:
            if img.size == 0:
                tensors.append(torch.zeros(3, 224, 224))
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensors.append(self.transform(Image.fromarray(rgb)))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            out = self.model(batch)
        emb = out[1] if isinstance(out, (list, tuple)) else out
        emb = emb.float().cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Gallery — persistent JSON store of name → embedding(s)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GalleryEntry:
    name: str               # human-readable identity ("Bear A", "480 Otis", ...)
    embeddings: list        # list of (512,) float32 arrays — multi-shot mean
    n_observations: int     # how many times we've seen this bear

    def mean_embedding(self) -> np.ndarray:
        m = np.mean(self.embeddings, axis=0)
        return m / max(np.linalg.norm(m), 1e-12)


class Gallery:
    """Persistent collection of named bear embeddings."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list[GalleryEntry] = []
        self._next_anon_idx = 0

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        g = cls(Path(path))
        if g.path.exists():
            with open(g.path) as f:
                data = json.load(f)
            for e in data.get("entries", []):
                g.entries.append(GalleryEntry(
                    name=e["name"],
                    embeddings=[np.array(v, dtype=np.float32) for v in e["embeddings"]],
                    n_observations=e["n_observations"],
                ))
            g._next_anon_idx = data.get("next_anon_idx", len(g.entries))
        return g

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_anon_idx": self._next_anon_idx,
            "entries": [
                {
                    "name": e.name,
                    "embeddings": [v.tolist() for v in e.embeddings],
                    "n_observations": e.n_observations,
                }
                for e in self.entries
            ],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def match(self, embedding: np.ndarray, threshold: float = 0.6) -> tuple[Optional[str], float]:
        """Return (best-matching name, cosine similarity) or (None, max_sim) if below threshold."""
        if not self.entries:
            return None, 0.0
        gallery_means = np.stack([e.mean_embedding() for e in self.entries])  # (K, 512)
        sims = gallery_means @ embedding  # (K,) cosine since both L2-normalized
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])
        if best_sim >= threshold:
            return self.entries[idx].name, best_sim
        return None, best_sim

    def _next_anon_name(self) -> str:
        # Bear A, B, ..., Z, AA, AB, ...
        idx = self._next_anon_idx
        chars = string.ascii_uppercase
        if idx < 26:
            name = chars[idx]
        else:
            name = chars[idx // 26 - 1] + chars[idx % 26]
        self._next_anon_idx += 1
        return f"Bear {name}"

    def add_anonymous(self, embedding: np.ndarray) -> str:
        name = self._next_anon_name()
        self.entries.append(GalleryEntry(
            name=name,
            embeddings=[embedding.astype(np.float32)],
            n_observations=1,
        ))
        return name

    def reinforce(self, name: str, embedding: np.ndarray, max_shots: int = 5):
        """Add another exemplar to a known bear, capped at max_shots."""
        for e in self.entries:
            if e.name == name:
                e.embeddings.append(embedding.astype(np.float32))
                if len(e.embeddings) > max_shots:
                    e.embeddings = e.embeddings[-max_shots:]
                e.n_observations += 1
                return
