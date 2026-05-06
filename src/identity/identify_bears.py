"""
Augment a feeding-analysis run with cross-video bear identities (PoseSwin).

Takes an existing `analysis.json` produced by `analyze_feeding.py` plus the
source video. For each unique bear (display ID), picks the highest-confidence
frame, crops the head region heuristically, embeds with PoseSwin, and matches
against a persistent gallery on disk.

Outputs `id_mapping.json` next to the analysis file:

    {
      "video": "...",
      "gallery_path": "data/identity/bear_gallery.json",
      "mapping": {
        "1": {"name": "Bear A", "similarity": 0.83, "is_new": false, ...},
        "2": {"name": "Bear B", "similarity": 0.00, "is_new": true,  ...}
      }
    }

The viewer reads this mapping and renders names instead of raw display IDs.

Usage
-----
    venv/bin/python3 -m src.identity.identify_bears \
        --video feed/data_video/<clip>.mp4 \
        --analysis predictions/<...>/analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.identity.poseswin_identifier import (
    PoseSwinIdentifier, Gallery, head_crop_from_bear,
)
from src.identity.face_detector import BearFaceDetector

DEFAULT_GALLERY = PROJECT_ROOT / "data" / "identity" / "bear_gallery.json"


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, a_area + b_area - inter)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> float:
    """Return fraction of `inner` that is inside `outer`."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter = max(0, min(ix2, ox2) - max(ix1, ox1)) * max(0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter / inner_area


def head_crop_from_face_detector(
    frame_bgr,
    bear_bbox: tuple[int, int, int, int],
    face_detector: "BearFaceDetector | None",
    padding_pct: float = 0.10,
):
    """Try the face detector on the FULL frame, pick the head most contained
    in this bear's bbox; return (crop, source) where source is 'face_detector'
    or 'heuristic' (fallback)."""
    if face_detector is not None:
        face_dets = face_detector(frame_bgr)
        # Keep only detections whose center is mostly inside the bear bbox.
        candidates = []
        for x1, y1, x2, y2, score in face_dets:
            inside_frac = _bbox_contains(bear_bbox, (x1, y1, x2, y2))
            if inside_frac >= 0.7:
                candidates.append(((x1, y1, x2, y2), score, inside_frac))
        if candidates:
            (x1, y1, x2, y2), score, frac = max(candidates, key=lambda c: c[1])
            H, W = frame_bgr.shape[:2]
            pad_x = int((x2 - x1) * padding_pct)
            pad_y = int((y2 - y1) * padding_pct)
            x1 = max(0, x1 - pad_x);  y1 = max(0, y1 - pad_y)
            x2 = min(W, x2 + pad_x);  y2 = min(H, y2 + pad_y)
            return frame_bgr[y1:y2, x1:x2], "face_detector", score
    # Fallback to heuristic
    return head_crop_from_bear(frame_bgr, bear_bbox), "heuristic", None


def best_frames_per_bear(entries: list[dict], top_k: int = 3) -> dict[int, list[dict]]:
    """For each display ID, return up to top_k highest-confidence appearances."""
    by_bear: dict[int, list[dict]] = {}
    for entry in entries:
        for bid_str, bear in entry.get("bears", {}).items():
            bid = int(bid_str)
            row = {
                "frame_idx": entry["frame_idx"],
                "timestamp_sec": entry.get("timestamp_sec"),
                "conf": float(bear.get("conf", 0.0)),
                "bbox": tuple(int(v) for v in bear["bbox"]),
            }
            by_bear.setdefault(bid, []).append(row)
    out = {}
    for bid, rows in by_bear.items():
        rows.sort(key=lambda r: r["conf"], reverse=True)
        out[bid] = rows[:top_k]
    return out


def main():
    parser = argparse.ArgumentParser(description="Assign cross-video bear identities via PoseSwin")
    parser.add_argument("--video", required=True, help="Source video")
    parser.add_argument("--analysis", required=True, help="Existing analysis.json from analyze_feeding")
    parser.add_argument("--gallery", default=str(DEFAULT_GALLERY),
                        help=f"Persistent gallery JSON (default: {DEFAULT_GALLERY})")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Cosine similarity above which two bears are considered the same individual")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Use top-K highest-confidence frames per bear for embedding "
                             "(default: 10 — bumped from 3 to give the face detector more chances)")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: id_mapping.json next to --analysis)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute mapping but don't update gallery on disk")
    parser.add_argument("--no-face-detector", action="store_true",
                        help="Skip the Faster-RCNN bear-face detector and always use heuristic crop")
    parser.add_argument("--face-score-threshold", type=float, default=0.3,
                        help="Min Faster-RCNN score to accept a face detection (default: 0.3)")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    analysis_path = Path(args.analysis).resolve()
    out_path = Path(args.output) if args.output else analysis_path.parent / "id_mapping.json"

    print(f"Video    : {video_path}")
    print(f"Analysis : {analysis_path}")
    print(f"Gallery  : {args.gallery}")
    print()

    with open(analysis_path) as f:
        analysis = json.load(f)
    entries = analysis["entries"]

    bears = best_frames_per_bear(entries, top_k=args.top_k)
    print(f"Found {len(bears)} unique display IDs in this video: {sorted(bears.keys())}")

    print("\nLoading PoseSwin model...")
    identifier = PoseSwinIdentifier(device=args.device)
    face_detector = None
    if not args.no_face_detector:
        print("Loading Faster-RCNN bear-face detector...")
        face_detector = BearFaceDetector(
            device=args.device,
            score_threshold=args.face_score_threshold,
        )
    gallery = Gallery.load(args.gallery)
    print(f"Gallery has {len(gallery.entries)} known bears at start.\n")

    cap = cv2.VideoCapture(str(video_path))
    mapping = {}

    for bid in sorted(bears.keys()):
        rows = bears[bid]
        crops = []
        crop_sources = []  # 'face_detector' or 'heuristic'
        face_scores = []
        for r in rows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame_idx"])
            ok, bgr = cap.read()
            if not ok:
                continue
            head, source, fscore = head_crop_from_face_detector(
                bgr, r["bbox"], face_detector,
            )
            if head is not None and head.size > 0:
                crops.append(head)
                crop_sources.append(source)
                if fscore is not None:
                    face_scores.append(fscore)
        if not crops:
            print(f"Bear {bid}: no usable head crops, skipping.")
            continue

        embeddings = identifier.embed_batch(crops)
        mean_emb = embeddings.mean(axis=0)
        mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)

        n_face = sum(1 for s in crop_sources if s == "face_detector")
        n_heur = sum(1 for s in crop_sources if s == "heuristic")
        crop_summary = f"{n_face} face-detector, {n_heur} heuristic"
        if face_scores:
            crop_summary += f"; mean face score {sum(face_scores)/len(face_scores):.2f}"

        name, sim = gallery.match(mean_emb, threshold=args.threshold)
        if name is None:
            new_name = gallery.add_anonymous(mean_emb)
            print(f"Bear {bid}: NEW identity → '{new_name}'  (max sim to gallery: {sim:.3f})  [{crop_summary}]")
            mapping[str(bid)] = {
                "name": new_name,
                "similarity": round(sim, 3),
                "is_new": True,
                "n_shots": len(crops),
                "n_face_crops": n_face,
                "n_heuristic_crops": n_heur,
                "max_conf": round(rows[0]["conf"], 3),
            }
        else:
            gallery.reinforce(name, mean_emb)
            print(f"Bear {bid}: matched '{name}'  (cosine sim {sim:.3f})  [{crop_summary}]")
            mapping[str(bid)] = {
                "name": name,
                "similarity": round(sim, 3),
                "is_new": False,
                "n_shots": len(crops),
                "n_face_crops": n_face,
                "n_heuristic_crops": n_heur,
                "max_conf": round(rows[0]["conf"], 3),
            }

    cap.release()

    if not args.dry_run:
        gallery.save()
        print(f"\n✓ Gallery saved with {len(gallery.entries)} entries → {args.gallery}")
    else:
        print("\n(--dry-run: gallery not saved)")

    out_doc = {
        "video": str(video_path),
        "analysis": str(analysis_path),
        "gallery_path": args.gallery,
        "threshold": args.threshold,
        "mapping": mapping,
    }
    with open(out_path, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"✓ ID mapping saved → {out_path}")


if __name__ == "__main__":
    main()
