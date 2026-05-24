"""
Consumer-grade bear-eating detector — no GPU / VLM required.

Combines three cheap signals computed from each YOLO bbox:

  1. Salmon-flesh color ratio (HSV thresholding for pink, red, white-belly)
     in the bear's head ROI (face detector if --chew-detect, else upper-60% heuristic)
  2. Posture/stillness heuristic (bbox aspect ratio + frame-to-frame motion)
  3. (--chew-detect only) Chewing-rhythm energy: FFT of frame-to-frame
     pixel-difference signal inside the head ROI, fraction of spectral power
     in the chewing band (default 0.5-3 Hz). High when the bear is opening
     and closing its mouth periodically; low when the head is still.

Designed to run at hundreds of FPS on a laptop CPU when --chew-detect is off,
~5-10 fps with --chew-detect (head ROI per sample + 1-sec dense read for FFT).

Usage
-----
    # Re-use existing analysis, no GPU, color + posture only:
    venv/bin/python3 -m src.behavior.pixel_eating_detector \
        --video feed/data_video/katmai_2026_05_03_8to20s.mp4 \
        --analysis predictions/.../analysis.json \
        --render

    # Run from scratch with chewing-rhythm + face-detected head ROI:
    venv/bin/python3 -m src.behavior.pixel_eating_detector \
        --video <path> --chew-detect --render
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
    pink_ratio:        float  # salmon-flesh pink (vs non-bear pixels) inside head ROI
    red_ratio:         float  # salmon-flesh red
    light_ratio:       float  # salmon-belly white/light
    bright_non_brown:  float  # any bright pixel that isn't bear fur (silver fish, etc)
    aspect_ratio:      float  # bbox width / height
    motion:            float  # bbox center displacement vs prev frame, normalized
    color_score:       float  # combined salmon-color signal in [0, 1]
    posture_score:     float  # bear-shape posture signal in [0, 1]
    chew_band_fraction: float  # fraction of spectral power in chew band (0..1)
    chew_peak_freq:    float  # dominant frequency (Hz) within chew band, 0 if disabled
    chew_motion_std:   float  # std of frame-diff signal in head ROI
    head_box:          list   # [x1,y1,x2,y2] of head ROI used (full-frame coords)
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


def head_box_from_face_detector(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    face_detector,
    pad_pct: float = 0.15,
) -> tuple[int, int, int, int] | None:
    """Run the bear face/head detector on the bbox crop and return the best
    head box in FULL-FRAME coordinates, padded slightly so we capture the
    mouth region. Returns None if no face was detected above threshold.
    """
    x1, y1, x2, y2 = bbox
    H, W = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    dets = face_detector(crop)
    if not dets:
        return None
    # Pick highest-scoring head
    fx1, fy1, fx2, fy2, _ = max(dets, key=lambda d: d[4])
    # Pad to capture jawline + mouth a bit better
    fw = fx2 - fx1; fh = fy2 - fy1
    pad_x = int(fw * pad_pct); pad_y = int(fh * pad_pct)
    fx1 -= pad_x; fy1 -= pad_y; fx2 += pad_x; fy2 += pad_y
    # Convert to full-frame coords and clip
    gx1 = max(0, x1 + fx1); gy1 = max(0, y1 + fy1)
    gx2 = min(W, x1 + fx2); gy2 = min(H, y1 + fy2)
    if gx2 - gx1 <= 0 or gy2 - gy1 <= 0:
        return None
    return (gx1, gy1, gx2, gy2)


def chewing_band_energy(
    cap_dense: cv2.VideoCapture,
    center_frame_idx: int,
    head_box: tuple[int, int, int, int],
    src_fps: float,
    window_sec: float = 2.0,
    band: tuple[float, float] = (0.5, 3.0),
) -> tuple[float, float, float]:
    """Read a `window_sec` window of consecutive frames around `center_frame_idx`,
    crop the same head_box from each frame, and compute a per-frame
    salmon-flesh / mouth color ratio time series:

        signal[t] = (pink_mask + light_mask) ∩ non_bear  /  non_bear

    This signal goes UP when the bear opens its mouth (salmon flesh + teeth /
    tongue exposed) and DOWN when it closes (only bear face visible). FFT of
    this time series picks up actual chewing oscillation, unlike grayscale
    pixel-diff which is dominated by 1/f motion noise.

    Returns:
      band_fraction:  spectral power in [band_lo, band_hi] / total power
      peak_freq:      dominant frequency (Hz) inside that band (0 if all-zero)
      signal_std:     std of the raw color-ratio signal (proxy for chew amplitude)

    All return values are 0 if the window can't be read or the head_box is
    degenerate.
    """
    x1, y1, x2, y2 = head_box
    if x2 - x1 <= 4 or y2 - y1 <= 4:
        return 0.0, 0.0, 0.0
    n_frames = max(16, int(window_sec * src_fps))
    half = n_frames // 2
    start = max(0, center_frame_idx - half)
    cap_dense.set(cv2.CAP_PROP_POS_FRAMES, start)

    ratios: list[float] = []
    for _ in range(n_frames):
        ok, frame = cap_dense.read()
        if not ok:
            break
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bear_mask = cv2.inRange(hsv, HSV_BEAR_LOW, HSV_BEAR_HIGH)
        non_bear = cv2.bitwise_not(bear_mask)
        non_bear_count = max(1, int((non_bear > 0).sum()))
        pink_mask  = cv2.inRange(hsv, HSV_PINK_LOW,  HSV_PINK_HIGH)
        light_mask = cv2.inRange(hsv, HSV_LIGHT_LOW, HSV_LIGHT_HIGH)
        # Combined "exposed-mouth-or-flesh" mask: pink salmon flesh OR bright
        # white-tongue / teeth / belly. Both signal "mouth is open over salmon".
        flesh_or_mouth = cv2.bitwise_or(pink_mask, light_mask)
        in_nonbear = cv2.bitwise_and(flesh_or_mouth, non_bear)
        ratios.append(float(in_nonbear.sum() / 255 / non_bear_count))

    if len(ratios) < 16:
        return 0.0, 0.0, 0.0

    sig = np.array(ratios, dtype=np.float64)
    signal_std = float(sig.std())
    if signal_std < 1e-6:
        # Constant signal -> no temporal information at all (e.g., far-field
        # video where the head ROI never has pink or light pixels).
        return 0.0, 0.0, signal_std
    sig = sig - sig.mean()  # detrend (DC removal)
    fft_mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / src_fps)

    total = float(fft_mag.sum())
    if total <= 1e-9:
        return 0.0, 0.0, signal_std
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not band_mask.any():
        return 0.0, 0.0, signal_std
    band_power = float(fft_mag[band_mask].sum())
    band_fraction = band_power / total
    in_band = fft_mag[band_mask]
    peak_freq = float(freqs[band_mask][in_band.argmax()]) if band_power > 0 else 0.0
    return band_fraction, peak_freq, signal_std


def score_frame(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    prev_center: tuple[float, float] | None,
    face_detector=None,
    cap_dense: cv2.VideoCapture | None = None,
    frame_idx: int = 0,
    src_fps: float = 30.0,
    chew_window_sec: float = 2.0,
    chew_band: tuple[float, float] = (0.5, 3.0),
) -> FrameSignals:
    """Compute the full per-bear signal vector for one frame.

    When `face_detector` AND `cap_dense` are both provided, also compute the
    chewing-rhythm signal (FFT band fraction in [chew_band] over a
    `chew_window_sec` window around `frame_idx`), and use it in the fused
    eating score. Otherwise the original (color + posture) score is used.
    """
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = w / h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    motion = 0.0
    if prev_center is not None:
        dx, dy = cx - prev_center[0], cy - prev_center[1]
        motion = float(np.hypot(dx, dy) / max(w, h))

    # ---- Pick the head ROI -------------------------------------------------
    # Prefer the face detector's box (tight on actual head); fall back to the
    # heuristic upper-60% × middle-70% strip of the bbox.
    head_box_global: tuple[int, int, int, int] | None = None
    if face_detector is not None:
        head_box_global = head_box_from_face_detector(frame_bgr, bbox, face_detector)
    if head_box_global is None:
        H, W = frame_bgr.shape[:2]
        hx1 = max(0, x1 + int(w * 0.15))
        hx2 = min(W, x2 - int(w * 0.15))
        hy1 = max(0, y1)
        hy2 = min(H, y1 + int(h * 0.60))
        head_box_global = (hx1, hy1, hx2, hy2)

    hx1, hy1, hx2, hy2 = head_box_global
    crop = frame_bgr[hy1:hy2, hx1:hx2]
    pink, red, light, bright_nb = color_ratios(crop)

    # ---- Color score -------------------------------------------------------
    # Trust pink and red (the only colors that are unambiguously salmon flesh).
    # `light` (white belly / teeth / tongue) kept at low weight as a minor
    # secondary cue. `bright_non_brown` REMOVED — in far/sideview footage it
    # was firing on water reflections, sky, and wet bear fur, inflating
    # color_score even when no salmon was visible (false positives).
    color_raw = pink * 25.0 + red * 30.0 + light * 1.0
    color_score = float(np.clip(color_raw, 0.0, 1.0))

    # ---- Posture score -----------------------------------------------------
    aspect_term    = _sigmoid((aspect - 0.85) * 4.0)
    stillness_term = float(np.clip(1.0 - motion * 6.0, 0.0, 1.0))
    posture_score  = float(aspect_term * stillness_term)

    # ---- Chewing-rhythm score (only if dense reading is enabled) ----------
    chew_band_fraction = 0.0
    chew_peak_freq = 0.0
    chew_motion_std = 0.0
    if cap_dense is not None:
        chew_band_fraction, chew_peak_freq, chew_motion_std = chewing_band_energy(
            cap_dense, frame_idx, head_box_global,
            src_fps=src_fps, window_sec=chew_window_sec, band=chew_band,
        )

    # ---- Fused eating score ------------------------------------------------
    if cap_dense is not None:
        # Joint AMPLITUDE × RHYTHM chew score.
        # band_fraction alone inflates on near-zero signals (FFT of noise in
        # videos with no salmon visible gives band_fraction ≈ 0.5, just like
        # real chewing). Multiply by chew_motion_std (now: std of color-mask
        # ratio time series) so we require BOTH a periodic component AND a
        # meaningful amplitude. Empirically band×std ≈ 0.020 for real chewing
        # vs ~0.0003 for far/no-salmon — scale 50 maps real chewing to ~1.0.
        chew_score = float(np.clip(chew_band_fraction * chew_motion_std * 50.0, 0.0, 1.0))
        eating_score = 0.45 * color_score + 0.45 * chew_score + 0.10 * posture_score
    else:
        # Original color + posture score
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
        chew_band_fraction=round(chew_band_fraction, 4),
        chew_peak_freq=round(chew_peak_freq, 3),
        chew_motion_std=round(chew_motion_std, 3),
        head_box=[int(hx1), int(hy1), int(hx2), int(hy2)],
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
            head_box = sig.get("head_box")
            if head_box and (head_box[2] - head_box[0]) > 0:
                hx1, hy1, hx2, hy2 = head_box
                cv2.rectangle(bgr, (hx1, hy1), (hx2, hy2), (255, 100, 100), 2)
            chew_bf = sig.get("chew_band_fraction", 0.0)
            if chew_bf > 0:
                txt = (f"Bear {bid}  {label.upper()}  s={score:.2f}  "
                       f"pink={sig['pink_ratio']:.2f}  chew={chew_bf:.2f}@{sig.get('chew_peak_freq', 0):.1f}Hz")
            else:
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
    parser.add_argument("--chew-detect", action="store_true",
                        help="Enable face-detector head ROI + chewing-rhythm FFT (slower, more accurate)")
    parser.add_argument("--chew-window-sec", type=float, default=2.0,
                        help="Length of dense window (sec) around each sample for chew FFT (default: 2.0; "
                             "longer = finer frequency resolution)")
    parser.add_argument("--chew-band-low", type=float, default=0.5,
                        help="Chewing frequency band lower bound, Hz (default: 0.5)")
    parser.add_argument("--chew-band-high", type=float, default=3.0,
                        help="Chewing frequency band upper bound, Hz (default: 3.0)")
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
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Optional: face detector for tight head ROI, and a separate dense cap
    # for FFT-window reads. Both lazy-loaded only when --chew-detect is set.
    face_detector = None
    cap_dense: cv2.VideoCapture | None = None
    if args.chew_detect:
        from src.identity.face_detector import BearFaceDetector
        print("Loading bear face detector for head ROI...")
        face_detector = BearFaceDetector(score_threshold=0.5)
        cap_dense = cv2.VideoCapture(str(video_path))
        print(f"Chew detection ON (window={args.chew_window_sec}s, "
              f"band=[{args.chew_band_low}, {args.chew_band_high}] Hz, src_fps={src_fps:.1f})")

    if args.analysis:
        iterator = iter_bbox_per_frame_from_analysis(Path(args.analysis), cap)
    else:
        iterator = iter_bbox_per_frame_from_yolo(video_path, Path(args.model), args.frame_skip)

    prev_center: dict[int, tuple[float, float]] = {}
    per_frame_results: list[dict] = []

    try:
        for frame_idx, t_sec, bgr, bears in iterator:
            bears_out = {}
            for bid, bbox in bears.items():
                sig = score_frame(
                    bgr, bbox, prev_center.get(bid),
                    face_detector=face_detector,
                    cap_dense=cap_dense,
                    frame_idx=frame_idx,
                    src_fps=src_fps,
                    chew_window_sec=args.chew_window_sec,
                    chew_band=(args.chew_band_low, args.chew_band_high),
                )
                bears_out[bid] = {**asdict(sig), "bbox": list(bbox)}
                x1, y1, x2, y2 = bbox
                prev_center[bid] = ((x1 + x2) / 2, (y1 + y2) / 2)
            per_frame_results.append({
                "frame_idx": frame_idx,
                "timestamp_sec": round(t_sec, 3),
                "bears": bears_out,
            })
    finally:
        cap.release()
        if cap_dense is not None:
            cap_dense.release()

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
        chew_str = ""
        if s.get("avg_chew_band_fraction") is not None:
            chew_str = (f"  chew_band={s['avg_chew_band_fraction']:.3f}"
                        f"  chew_peak={s['avg_chew_peak_freq']:.2f}Hz")
        print(f"Bear {bid}: eating frames={s['eating_frames']}/{s['total_frames']} "
              f"({100*s['eating_fraction']:.1f}%)  "
              f"max_score={s['max_score']:.2f}  "
              f"avg_pink_ratio={s['avg_pink_ratio']:.3f}{chew_str}")

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
                "chew_bands": [], "chew_peaks": [],
            })
            b["total_frames"] += 1
            score = sig.get("eating_score_smoothed", sig["eating_score"])
            b["scores"].append(score)
            b["pinks"].append(sig["pink_ratio"])
            b["chew_bands"].append(sig.get("chew_band_fraction", 0.0))
            b["chew_peaks"].append(sig.get("chew_peak_freq", 0.0))
            if score > 0.60:
                b["eating_frames"] += 1
            b["max_score"] = max(b["max_score"], score)

    out = {}
    for bid, b in by_bear.items():
        rec = {
            "total_frames": b["total_frames"],
            "eating_frames": b["eating_frames"],
            "eating_fraction": round(b["eating_frames"] / max(1, b["total_frames"]), 3),
            "max_score": round(b["max_score"], 3),
            "avg_pink_ratio": round(sum(b["pinks"]) / max(1, len(b["pinks"])), 4),
            "avg_score": round(sum(b["scores"]) / max(1, len(b["scores"])), 3),
        }
        if any(v > 0 for v in b["chew_bands"]):
            rec["avg_chew_band_fraction"] = round(sum(b["chew_bands"]) / max(1, len(b["chew_bands"])), 4)
            rec["avg_chew_peak_freq"] = round(sum(b["chew_peaks"]) / max(1, len(b["chew_peaks"])), 3)
        out[bid] = rec
    return out


if __name__ == "__main__":
    main()
