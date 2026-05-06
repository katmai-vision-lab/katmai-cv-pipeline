"""
Consumer-grade bear-eating detector — no GPU / VLM required.

Combines two cheap signals computed from each YOLO bbox:

  1. Salmon-flesh color ratio (HSV thresholding for pink, red, white-belly)
     in the bear's mouth region (upper/center of bbox)
  2. Posture/stillness heuristic (bbox aspect ratio + frame-to-frame motion)

Designed to run at hundreds of FPS on a laptop CPU.
Reuses bbox data from a previously-generated analyze_feeding analysis.json,
or runs YOLO directly if no analysis is available.

Usage
-----
    # Re-use existing analysis (fast, no GPU at all):
    venv/bin/python3 -m src.behavior.pixel_eating_detector \
        --video feed/data_video/katmai_2026_05_03_8to20s.mp4 \
        --analysis predictions/.../analysis.json \
        --render

    # Or run from scratch (still CPU-only inference if model fits):
    venv/bin/python3 -m src.behavior.pixel_eating_detector \
        --video <path> --render
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_DIR, TRACKERS_CONFIG_DIR, TRAINED_MODELS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Color masks — tuned to typical salmon flesh on Brooks Falls footage
# ─────────────────────────────────────────────────────────────────────────────

# OpenCV HSV: H in [0, 180], S in [0, 255], V in [0, 255]
# Salmon flesh tuned for SATURATED + BRIGHT only — skips dark brown bear fur
HSV_PINK_LOW   = np.array([  0, 100, 150], dtype=np.uint8)
HSV_PINK_HIGH  = np.array([ 15, 220, 240], dtype=np.uint8)

# Red wraps around 180/0 boundary
HSV_RED_LOW_A  = np.array([  0, 150, 130], dtype=np.uint8)
HSV_RED_HIGH_A = np.array([ 10, 255, 220], dtype=np.uint8)
HSV_RED_LOW_B  = np.array([165, 150, 130], dtype=np.uint8)
HSV_RED_HIGH_B = np.array([180, 255, 220], dtype=np.uint8)

# Light/white salmon belly (low sat, high value)
HSV_LIGHT_LOW  = np.array([  0,   0, 200], dtype=np.uint8)
HSV_LIGHT_HIGH = np.array([180,  40, 255], dtype=np.uint8)

# "Bear fur" — what we want to MASK OUT before computing salmon signals.
# Dark warm tones (brown).
HSV_BEAR_LOW   = np.array([  5,  40,  20], dtype=np.uint8)
HSV_BEAR_HIGH  = np.array([ 28, 220, 130], dtype=np.uint8)


@dataclass
class FrameSignals:
    """Per-bear, per-frame signal vector."""
    pink_ratio:        float  # salmon-flesh pink (vs non-bear pixels)
    red_ratio:         float  # salmon-flesh red
    light_ratio:       float  # salmon-belly white/light
    bright_non_brown:  float  # any bright pixel that isn't bear fur (silver fish, etc)
    aspect_ratio:      float  # bbox width / height
    motion:            float  # bbox center displacement vs prev frame, normalized
    color_score:       float  # combined salmon-color signal in [0, 1]
    posture_score:     float  # bear-shape posture signal in [0, 1]
    eating_score:      float  # final fused score in [0, 1]
    label:             str    # 'eating' / 'maybe' / 'not_eating'


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring
# ─────────────────────────────────────────────────────────────────────────────

def mouth_region(bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop the upper-center part of the bbox where the bear's mouth usually is.

    Drop 15% from each side, keep the top 60% vertically — this is where
    the head appears in both standing-alert and head-down-eating postures.
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    yt = y1 + int(h * 0.00)
    yb = y1 + int(h * 0.60)
    xl = x1 + int(w * 0.15)
    xr = x2 - int(w * 0.15)
    H, W = bgr.shape[:2]
    yt, yb = max(0, yt), min(H, yb)
    xl, xr = max(0, xl), min(W, xr)
    return bgr[yt:yb, xl:xr]


def color_ratios(crop_bgr: np.ndarray) -> tuple[float, float, float, float]:
    """Return (pink_ratio, red_ratio, light_ratio, bright_non_brown_ratio).

    Ratios are computed against the NON-bear-fur portion of the crop, so a
    bear-dominated bbox doesn't dilute the salmon signal.
    """
    if crop_bgr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    bear_mask = cv2.inRange(hsv, HSV_BEAR_LOW, HSV_BEAR_HIGH)
    non_bear  = cv2.bitwise_not(bear_mask)
    non_bear_count = max(1, int((non_bear > 0).sum()))

    pink_mask  = cv2.inRange(hsv, HSV_PINK_LOW,  HSV_PINK_HIGH)
    red_mask_a = cv2.inRange(hsv, HSV_RED_LOW_A, HSV_RED_HIGH_A)
    red_mask_b = cv2.inRange(hsv, HSV_RED_LOW_B, HSV_RED_HIGH_B)
    light_mask = cv2.inRange(hsv, HSV_LIGHT_LOW, HSV_LIGHT_HIGH)

    pink_in_nonbear  = cv2.bitwise_and(pink_mask, non_bear)
    red_in_nonbear   = cv2.bitwise_and(cv2.bitwise_or(red_mask_a, red_mask_b), non_bear)
    light_in_nonbear = cv2.bitwise_and(light_mask, non_bear)

    # "Bright non-brown" — anything inside the bbox that isn't bear fur and isn't
    # too dark. Catches silver salmon body and pink flesh alike.
    bright = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 255, 255]))
    bright_non_brown = cv2.bitwise_and(bright, non_bear)

    return (
        float(pink_in_nonbear.sum() / 255 / non_bear_count),
        float(red_in_nonbear.sum() / 255 / non_bear_count),
        float(light_in_nonbear.sum() / 255 / non_bear_count),
        float(bright_non_brown.sum() / 255 / non_bear_count),
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def score_frame(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    prev_center: tuple[float, float] | None,
) -> FrameSignals:
    """Compute the full per-bear signal vector for one frame."""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = w / h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    motion = 0.0
    if prev_center is not None:
        dx, dy = cx - prev_center[0], cy - prev_center[1]
        motion = float(np.hypot(dx, dy) / max(w, h))

    crop = mouth_region(frame_bgr, bbox)
    pink, red, light, bright_nb = color_ratios(crop)

    # ---- Color score -------------------------------------------------------
    # We score saturated salmon-flesh colors first; bright-non-brown is a
    # secondary signal that catches silver/light salmon bodies.
    color_raw = pink * 25.0 + red * 30.0 + light * 4.0 + bright_nb * 1.5
    color_score = float(np.clip(color_raw, 0.0, 1.0))

    # ---- Posture score -----------------------------------------------------
    # Compact / wide bbox → eating posture; tall bbox → standing alert.
    # Low motion (stillness) increases confidence the bear is feeding, not
    # actively pursuing.
    aspect_term    = _sigmoid((aspect - 0.85) * 4.0)
    stillness_term = float(np.clip(1.0 - motion * 6.0, 0.0, 1.0))
    posture_score  = float(aspect_term * stillness_term)

    # ---- Fused eating score ------------------------------------------------
    eating_score = 0.55 * color_score + 0.45 * posture_score

    if eating_score > 0.60:
        label = "eating"
    elif eating_score > 0.40:
        label = "maybe"
    else:
        label = "not_eating"

    return FrameSignals(
        pink_ratio=round(pink, 4),
        red_ratio=round(red, 4),
        light_ratio=round(light, 4),
        bright_non_brown=round(bright_nb, 4),
        aspect_ratio=round(aspect, 3),
        motion=round(motion, 3),
        color_score=round(color_score, 3),
        posture_score=round(posture_score, 3),
        eating_score=round(eating_score, 3),
        label=label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Two input paths: existing analysis.json OR live YOLO
# ─────────────────────────────────────────────────────────────────────────────

def iter_bbox_per_frame_from_analysis(
    analysis_path: Path,
    cap: cv2.VideoCapture,
):
    """Yield (frame_idx, frame_bgr, {bear_id: bbox}) for each ANALYSIS sample."""
    with open(analysis_path) as f:
        data = json.load(f)

    for entry in data["entries"]:
        idx = entry["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        bears = {
            int(bid): tuple(bear["bbox"]) for bid, bear in entry["bears"].items()
        }
        yield idx, entry.get("timestamp_sec", idx / max(data.get("fps", 30), 1)), bgr, bears


def iter_bbox_per_frame_from_yolo(video_path: Path, model_path: Path, frame_skip: int):
    """Yield (frame_idx, timestamp, frame_bgr, {bear_id: bbox}) using YOLO+ByteTrack."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    tracker_cfg = str(TRACKERS_CONFIG_DIR / "bytetrack.yaml")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    stream = model.track(
        source=str(video_path),
        classes=[0],
        tracker=tracker_cfg,
        save=False,
        stream=True,
        verbose=False,
        persist=True,
        vid_stride=frame_skip,
    )
    for i, result in enumerate(stream):
        idx = i * frame_skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        bears = {}
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
            for tid, box in zip(ids, xyxy):
                bears[int(tid)] = tuple(int(v) for v in box)
        yield idx, idx / fps, bgr, bears
    cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def smooth_scores(per_frame: list[dict], window: int = 5) -> list[dict]:
    """Apply a centered moving-average to per-bear eating scores."""
    by_bear: dict[int, list[tuple[int, float]]] = {}
    for entry in per_frame:
        for bid, sig in entry["bears"].items():
            by_bear.setdefault(int(bid), []).append((entry["frame_idx"], sig["eating_score"]))

    smoothed: dict[int, dict[int, float]] = {}
    for bid, series in by_bear.items():
        scores = np.array([s for _, s in series], dtype=float)
        if len(scores) < 2:
            smoothed[bid] = {fr: s for fr, s in series}
            continue
        kernel = np.ones(window) / window
        smoothed_scores = np.convolve(scores, kernel, mode="same")
        smoothed[bid] = {fr: float(s) for (fr, _), s in zip(series, smoothed_scores)}

    out = []
    for entry in per_frame:
        new_entry = dict(entry)
        new_entry["bears"] = {
            bid: {**sig, "eating_score_smoothed": round(smoothed[int(bid)][entry["frame_idx"]], 3)}
            for bid, sig in entry["bears"].items()
        }
        out.append(new_entry)
    return out


def render_overlay(video_path: Path, results: list[dict], out_path: Path):
    """Draw bbox + eating-score badge onto each frame, write to mp4."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    avi = out_path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(avi), fourcc, fps, (W, H))

    # Index results by frame for quick lookup
    by_frame = {r["frame_idx"]: r for r in results}
    last_seen: dict[int, dict] = {}
    interval = results[1]["frame_idx"] - results[0]["frame_idx"] if len(results) > 1 else 1

    from tqdm import tqdm
    for f in tqdm(range(n_frames), desc="Rendering overlay"):
        ok, bgr = cap.read()
        if not ok:
            break
        if f in by_frame:
            last_seen = by_frame[f]["bears"]
        for bid, sig in last_seen.items():
            bbox = sig.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            score = sig.get("eating_score_smoothed", sig["eating_score"])
            label = sig["label"]
            color = (
                (0, 200, 0)   if score > 0.60 else
                (0, 200, 200) if score > 0.40 else
                (180, 180, 180)
            )
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 3)
            txt = f"Bear {bid}  {label.upper()}  {score:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(bgr, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(bgr, txt, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(bgr)
    writer.release()
    cap.release()

    # Re-encode to mp4 for shareability
    import subprocess
    mp4 = out_path.with_suffix(".mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(avi), "-c:v", "libx264", str(mp4)],
            check=True, capture_output=True,
        )
        avi.unlink()
        return mp4
    except (subprocess.CalledProcessError, FileNotFoundError):
        return avi


def main():
    parser = argparse.ArgumentParser(description="Consumer-grade pixel-based bear-eating detector")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--analysis", default=None,
                        help="Existing analyze_feeding analysis.json (re-uses bboxes, no YOLO needed)")
    parser.add_argument("--model",
                        default=str(TRAINED_MODELS_DIR / "bear_detector3" / "weights" / "best.pt"),
                        help="YOLO model path (used only if --analysis is not provided)")
    parser.add_argument("--frame-skip", type=int, default=15,
                        help="When running YOLO from scratch, sample every N frames (default: 15)")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Moving-average window over eating scores (default: 5 samples)")
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    parser.add_argument("--render", action="store_true",
                        help="Also render an annotated demo video")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    assert video_path.exists(), f"Video not found: {video_path}"

    out_dir = Path(PREDICTIONS_DIR) / f"{video_path.stem}_pixel_eating"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output_json) if args.output_json else out_dir / "pixel_eating.json"

    print(f"Video    : {video_path}")
    print(f"Output   : {out_json}")
    print(f"Mode     : {'analysis-json (no GPU)' if args.analysis else 'live YOLO+ByteTrack'}")
    print()

    cap = cv2.VideoCapture(str(video_path))

    if args.analysis:
        iterator = iter_bbox_per_frame_from_analysis(Path(args.analysis), cap)
    else:
        iterator = iter_bbox_per_frame_from_yolo(video_path, Path(args.model), args.frame_skip)

    prev_center: dict[int, tuple[float, float]] = {}
    per_frame_results: list[dict] = []

    for frame_idx, t_sec, bgr, bears in iterator:
        bears_out = {}
        for bid, bbox in bears.items():
            sig = score_frame(bgr, bbox, prev_center.get(bid))
            bears_out[bid] = {**asdict(sig), "bbox": list(bbox)}
            x1, y1, x2, y2 = bbox
            prev_center[bid] = ((x1 + x2) / 2, (y1 + y2) / 2)
        per_frame_results.append({
            "frame_idx": frame_idx,
            "timestamp_sec": round(t_sec, 3),
            "bears": bears_out,
        })

    cap.release()

    per_frame_results = smooth_scores(per_frame_results, window=args.smooth_window)

    summary = compute_summary(per_frame_results)
    output = {
        "video": str(video_path),
        "n_samples": len(per_frame_results),
        "summary": summary,
        "entries": per_frame_results,
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved {len(per_frame_results)} samples to {out_json}")

    print("\n=== Summary ===")
    for bid, s in summary.items():
        print(f"Bear {bid}: eating frames={s['eating_frames']}/{s['total_frames']} "
              f"({100*s['eating_fraction']:.1f}%)  "
              f"max_score={s['max_score']:.2f}  "
              f"avg_pink_ratio={s['avg_pink_ratio']:.3f}")

    if args.render:
        print("\nRendering annotated demo video...")
        out_video = render_overlay(video_path, per_frame_results, out_dir / f"{video_path.stem}_pixel_eating_demo")
        print(f"✓ Demo video: {out_video}")


def compute_summary(per_frame_results: list[dict]) -> dict:
    by_bear: dict[int, dict] = {}
    for entry in per_frame_results:
        for bid_raw, sig in entry["bears"].items():
            bid = int(bid_raw)
            b = by_bear.setdefault(bid, {
                "total_frames": 0, "eating_frames": 0,
                "max_score": 0.0, "scores": [], "pinks": [],
            })
            b["total_frames"] += 1
            score = sig.get("eating_score_smoothed", sig["eating_score"])
            b["scores"].append(score)
            b["pinks"].append(sig["pink_ratio"])
            if score > 0.60:
                b["eating_frames"] += 1
            b["max_score"] = max(b["max_score"], score)

    out = {}
    for bid, b in by_bear.items():
        out[bid] = {
            "total_frames": b["total_frames"],
            "eating_frames": b["eating_frames"],
            "eating_fraction": round(b["eating_frames"] / max(1, b["total_frames"]), 3),
            "max_score": round(b["max_score"], 3),
            "avg_pink_ratio": round(sum(b["pinks"]) / max(1, len(b["pinks"])), 4),
            "avg_score": round(sum(b["scores"]) / max(1, len(b["scores"])), 3),
        }
    return out


if __name__ == "__main__":
    main()
