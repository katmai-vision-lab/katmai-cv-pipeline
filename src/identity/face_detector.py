"""
Bear face / head detector.

Wraps the Faster-RCNN (ResNet-50 + FPN) trained by the BrownBear_ReID team
and reloads it into a torchvision model so we don't need mmdetection / mmcv
installed (the original is mmdet 2.22 + mmcv 1.3.17, incompatible with
PyTorch 2.6).

The architecture is identical:
  • ResNet-50 backbone, frozen stages=1, BN
  • FPN (256 channels, levels P2–P5; we drop the P6 conv weights)
  • RPN with 3 anchors per location (ratios [0.5, 1, 2])
  • Class-agnostic bbox regression (replicated to torchvision's class-specific slots)
  • 1 foreground class: "bear_head"

Usage
-----
    detector = BearFaceDetector()
    boxes = detector(bear_crop_bgr)          # list of (x1, y1, x2, y2, score)
    head_crop = detector.best_head_crop(bear_crop_bgr)  # convenience
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FACE_CKPT = (
    PROJECT_ROOT
    / "external" / "BrownBear_ReID" / "Public_release" / "checkpoints"
    / "preprocessing_ckpts" / "detectors" / "face_detector" / "latest.pth"
)


# ─────────────────────────────────────────────────────────────────────────────
# State-dict converter: mmdet 2.x Faster-RCNN  →  torchvision Faster-RCNN
# ─────────────────────────────────────────────────────────────────────────────

def _convert_mmdet_to_torchvision(mm_sd: dict) -> dict:
    """Translate mmdetection 2.x Faster-RCNN keys to torchvision format."""
    out = {}

    for k, v in mm_sd.items():
        # Backbone: backbone.* → backbone.body.*
        if k.startswith("backbone."):
            out["backbone.body." + k[len("backbone."):]] = v
            continue

        # FPN lateral convs: neck.lateral_convs.{i}.conv.* → backbone.fpn.inner_blocks.{i}.0.*
        if k.startswith("neck.lateral_convs."):
            parts = k.split(".")
            i = parts[2]
            tail = ".".join(parts[4:])  # skip "conv"
            out[f"backbone.fpn.inner_blocks.{i}.0.{tail}"] = v
            continue

        # FPN layer convs P2–P5: neck.fpn_convs.{0..3}.conv.* → backbone.fpn.layer_blocks.{i}.0.*
        # (mmdet's fpn_convs.4 = P6 produced by an extra 3×3 stride-2 conv;
        #  torchvision uses LastLevelMaxPool which has no learned weights, so we drop it)
        if k.startswith("neck.fpn_convs."):
            parts = k.split(".")
            i = int(parts[2])
            if i >= 4:
                continue  # skip P6 conv — torchvision uses parameterless MaxPool
            tail = ".".join(parts[4:])  # skip "conv"
            out[f"backbone.fpn.layer_blocks.{i}.0.{tail}"] = v
            continue

        # RPN: rpn_head.rpn_conv.* → rpn.head.conv.0.0.*
        if k == "rpn_head.rpn_conv.weight":
            out["rpn.head.conv.0.0.weight"] = v;  continue
        if k == "rpn_head.rpn_conv.bias":
            out["rpn.head.conv.0.0.bias"] = v;    continue
        if k == "rpn_head.rpn_cls.weight":
            out["rpn.head.cls_logits.weight"] = v;  continue
        if k == "rpn_head.rpn_cls.bias":
            out["rpn.head.cls_logits.bias"] = v;    continue
        if k == "rpn_head.rpn_reg.weight":
            out["rpn.head.bbox_pred.weight"] = v;  continue
        if k == "rpn_head.rpn_reg.bias":
            out["rpn.head.bbox_pred.bias"] = v;    continue

        # ROI: shared FCs become fc6 / fc7
        if k == "roi_head.bbox_head.shared_fcs.0.weight":
            out["roi_heads.box_head.fc6.weight"] = v;  continue
        if k == "roi_head.bbox_head.shared_fcs.0.bias":
            out["roi_heads.box_head.fc6.bias"] = v;    continue
        if k == "roi_head.bbox_head.shared_fcs.1.weight":
            out["roi_heads.box_head.fc7.weight"] = v;  continue
        if k == "roi_head.bbox_head.shared_fcs.1.bias":
            out["roi_heads.box_head.fc7.bias"] = v;    continue

        # Classification head — class index convention DIFFERS:
        #   mmdet 2.x:    row 0 = bear_head (foreground), row 1 = background
        #   torchvision:  row 0 = background,             row 1 = bear_head
        # → swap rows 0 and 1
        if k == "roi_head.bbox_head.fc_cls.weight":
            assert v.shape == (2, 1024)
            out["roi_heads.box_predictor.cls_score.weight"] = v[[1, 0]].clone()
            continue
        if k == "roi_head.bbox_head.fc_cls.bias":
            assert v.shape == (2,)
            out["roi_heads.box_predictor.cls_score.bias"] = v[[1, 0]].clone()
            continue

        # Regression head:
        #   mmdet (class-agnostic, our config): shape (4, 1024) — single bbox-delta head
        #   torchvision: shape (num_classes * 4, 1024) = (8, 1024)
        #     slots 0-3 → background (never used at inference)
        #     slots 4-7 → bear_head (the one that matters)
        if k == "roi_head.bbox_head.fc_reg.weight":
            assert v.shape == (4, 1024), f"unexpected fc_reg shape {v.shape}"
            new_w = torch.zeros(8, 1024, dtype=v.dtype)
            new_w[4:8] = v
            out["roi_heads.box_predictor.bbox_pred.weight"] = new_w
            continue
        if k == "roi_head.bbox_head.fc_reg.bias":
            assert v.shape == (4,), f"unexpected fc_reg bias shape {v.shape}"
            new_b = torch.zeros(8, dtype=v.dtype)
            new_b[4:8] = v
            out["roi_heads.box_predictor.bbox_pred.bias"] = new_b
            continue

        # everything else (anchor centers, etc.) — drop silently

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inference wrapper
# ─────────────────────────────────────────────────────────────────────────────

class BearFaceDetector:
    """Faster-RCNN bear-head detector with the BrownBear_ReID weights."""

    def __init__(self, device: Optional[str] = None,
                 score_threshold: float = 0.5):
        if device is None:
            from src.config import get_device
            device = get_device()
        self.device = torch.device(device)
        self.score_threshold = score_threshold

        assert FACE_CKPT.exists(), f"Missing face detector checkpoint: {FACE_CKPT}"

        # Build empty torchvision Faster-RCNN with 2 classes (bg + bear_head)
        # Disable score thresholding inside the model — we do it outside.
        self.model = fasterrcnn_resnet50_fpn(
            weights=None, num_classes=2,
            box_score_thresh=0.05,  # keep many candidates; we filter ourselves
        )

        # Convert + load weights
        ckpt = torch.load(str(FACE_CKPT), map_location="cpu", weights_only=False)
        mm_sd = ckpt["state_dict"]
        tv_sd = _convert_mmdet_to_torchvision(mm_sd)
        missing, unexpected = self.model.load_state_dict(tv_sd, strict=False)
        if missing:
            # Allowed missing: anchor utilities the saved weights don't contain
            ignorable = ("anchor_generator", "transform.")
            real_missing = [k for k in missing if not any(p in k for p in ignorable)]
            if real_missing:
                print(f"[BearFaceDetector] Missing keys ({len(real_missing)}):")
                for k in real_missing[:8]:
                    print(f"  {k}")
                if len(real_missing) > 8:
                    print(f"  ... and {len(real_missing)-8} more")
        if unexpected:
            print(f"[BearFaceDetector] Unexpected keys: {len(unexpected)}")

        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Detect bear heads in a BGR image. Returns list of (x1, y1, x2, y2, score)."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = T.ToTensor()(Image.fromarray(rgb)).to(self.device)

        out = self.model([tensor])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        results = []
        for box, score in zip(boxes, scores):
            if score < self.score_threshold:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            results.append((x1, y1, x2, y2, float(score)))
        return results

    def best_head_crop(self, image_bgr: np.ndarray,
                       padding_pct: float = 0.10) -> Optional[np.ndarray]:
        """Return the highest-scoring head crop (with optional padding), or None."""
        dets = self(image_bgr)
        if not dets:
            return None
        x1, y1, x2, y2, _ = max(dets, key=lambda d: d[4])
        h, w = image_bgr.shape[:2]
        pad_x = int((x2 - x1) * padding_pct)
        pad_y = int((y2 - y1) * padding_pct)
        x1 = max(0, x1 - pad_x);  y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x);  y2 = min(h, y2 + pad_y)
        return image_bgr[y1:y2, x1:x2]
