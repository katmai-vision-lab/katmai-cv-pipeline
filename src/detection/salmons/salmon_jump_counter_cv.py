# salmon_jump_counter_cv.py
import cv2
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import json
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_CONFIG_PATH = "config.json"


# ── Config dataclass ──────────────────────────────────────────────────────────
@dataclass
class SalmonConfig:
    salmon_hsv_lower:    list  = field(default_factory=lambda: [0,   0,  40])
    salmon_hsv_upper:    list  = field(default_factory=lambda: [180, 60, 160])
    silver_hsv_lower:    list  = field(default_factory=lambda: [0,   0,  160])
    silver_hsv_upper:    list  = field(default_factory=lambda: [180, 40, 255])
    min_blob_area:       int   = 800
    max_blob_area:       int   = 8000
    roi:                 Optional[list] = None   # [x, y, w, h] or null
    min_jump_gap_sec:    float = 0.5
    sample_rate:         int   = 2
    mog2_history:        int   = 500
    mog2_var_threshold:  int   = 100

    @staticmethod
    def from_file(path: str) -> "SalmonConfig":
        with open(path) as f:
            data = json.load(f)
        cfg = SalmonConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                print(f"[config] Unknown key '{k}' ignored", file=sys.stderr)
        return cfg

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
        print(f"[config] Saved to {path}", file=sys.stderr)

    def summary(self):
        print("[config] Active parameters:", file=sys.stderr)
        for k, v in self.__dict__.items():
            print(f"  {k:25s} = {v}", file=sys.stderr)


# ── Core CV functions (all accept cfg) ───────────────────────────────────────
def extract_frames(video_path: str, sample_rate: int):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {fps:.1f} fps, {total} frames ({total/fps:.1f}s)", file=sys.stderr)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def get_foreground_mask(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)
    return fg_mask


def get_salmon_color_mask(frame, cfg: SalmonConfig):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_salmon = cv2.inRange(hsv,
        np.array(cfg.salmon_hsv_lower), np.array(cfg.salmon_hsv_upper))
    mask_silver = cv2.inRange(hsv,
        np.array(cfg.silver_hsv_lower), np.array(cfg.silver_hsv_upper))
    combined = cv2.bitwise_or(mask_salmon, mask_silver)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return combined


def detect_salmon_blobs(frame, fg_mask, color_mask, cfg: SalmonConfig):
    combined = cv2.bitwise_and(fg_mask, color_mask)

    if cfg.roi:
        x, y, w, h = cfg.roi
        roi_mask = np.zeros_like(combined)
        roi_mask[y:y+h, x:x+w] = combined[y:y+h, x:x+w]
        combined = roi_mask

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (cfg.min_blob_area < area < cfg.max_blob_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1e-5)
        if aspect < 1.5 or aspect > 6.0:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / (hull_area + 1e-5)
        if solidity < 0.5:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            blobs.append((cx, cy, area))
    return blobs


def count_jumps_from_trajectory(blob_signal, fps, cfg: SalmonConfig):
    if not blob_signal:
        return 0, [], np.array([])
    frames, areas = zip(*blob_signal)
    areas = np.array(areas, dtype=float)
    window = max(3, int(fps / cfg.sample_rate))
    smoothed = np.convolve(areas, np.ones(window) / window, mode='same')
    min_gap_frames = int(cfg.min_jump_gap_sec * fps / cfg.sample_rate)
    peaks, _ = find_peaks(
        smoothed,
        height=cfg.min_blob_area * 0.5,
        distance=min_gap_frames,
        prominence=cfg.min_blob_area * 0.3
    )
    timestamps = [frames[p] / fps for p in peaks]
    return len(peaks), timestamps, smoothed


def count_salmon_jumps(video_path: str, cfg: SalmonConfig,
                       debug_output: str = None) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=cfg.mog2_history,
        varThreshold=cfg.mog2_var_threshold,
        detectShadows=False
    )

    blob_signal = []
    for frame_idx, frame in extract_frames(video_path, cfg.sample_rate):
        fg_mask    = get_foreground_mask(frame, bg_subtractor)
        color_mask = get_salmon_color_mask(frame, cfg)
        blobs      = detect_salmon_blobs(frame, fg_mask, color_mask, cfg)
        total_area = sum(a for _, _, a in blobs)
        blob_signal.append((frame_idx, total_area))
        if debug_output and blobs:
            _save_debug_frame(frame, blobs, fg_mask, color_mask,
                              frame_idx, fps, total_frames, cfg, debug_output)

    jump_count, timestamps, _ = count_jumps_from_trajectory(
        blob_signal, fps, cfg)

    return {
        "video":               video_path,
        "fps":                 fps,
        "jump_count":          jump_count,
        "jump_timestamps_sec": [round(t, 2) for t in timestamps],
        "sample_rate":         cfg.sample_rate,
        "config_used":         cfg.__dict__,
    }


def _save_debug_frame(frame, blobs, fg_mask, color_mask,
                      idx, fps, total_frames, cfg: SalmonConfig, out_dir):
    """
    Save a rich annotated debug frame with 4 panels:
      Top-left:     original frame + ROI box + blob circles
      Top-right:    foreground (motion) mask
      Bottom-left:  colour mask
      Bottom-right: combined mask (fg AND colour)
    A status bar at the bottom shows frame index, timestamp, and blob count.
    """
    Path(out_dir).mkdir(exist_ok=True)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    # ── Panel 1: annotated original ───────────────────────────────────────────
    p1 = frame.copy()

    # ROI box
    if cfg.roi:
        rx, ry, rw, rh = cfg.roi
        # dim area outside ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[ry:ry+rh, rx:rx+rw] = 255
        dimmed = (p1 * 0.4).astype(np.uint8)
        p1 = np.where(mask[:, :, None] == 255, p1, dimmed)
        cv2.rectangle(p1, (rx, ry), (rx+rw, ry+rh), (0, 220, 255), 2)
        cv2.putText(p1, "ROI", (rx+6, ry+20), FONT, 0.55, (0, 220, 255), 1)

    # Blob circles with area label and blob index
    for i, (cx, cy, area) in enumerate(blobs):
        r = max(12, int(np.sqrt(area / np.pi)))
        cv2.circle(p1, (cx, cy), r, (0, 255, 80), 2)
        cv2.circle(p1, (cx, cy), 3, (0, 255, 80), -1)
        cv2.putText(p1, f"#{i} {int(area)}px",
                    (cx - 32, cy - r - 6), FONT, 0.42, (0, 255, 80), 1)

    cv2.putText(p1, "annotated", (8, 20), FONT, 0.55, (200, 200, 200), 1)

    # ── Panel 2: foreground mask (motion) ────────────────────────────────────
    p2 = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(p2, "fg mask (motion)", (8, 20), FONT, 0.55, (200, 200, 200), 1)

    # ── Panel 3: colour mask ──────────────────────────────────────────────────
    p3 = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(p3, "colour mask", (8, 20), FONT, 0.55, (200, 200, 200), 1)

    # ── Panel 4: combined mask (fg AND colour) ────────────────────────────────
    combined = cv2.bitwise_and(fg_mask, color_mask)
    if cfg.roi:
        rx, ry, rw, rh = cfg.roi
        roi_only = np.zeros_like(combined)
        roi_only[ry:ry+rh, rx:rx+rw] = combined[ry:ry+rh, rx:rx+rw]
        combined = roi_only
    p4 = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    cv2.putText(p4, "combined (fg & colour & roi)", (8, 20),
                FONT, 0.55, (200, 200, 200), 1)

    # ── Assemble 2x2 grid ─────────────────────────────────────────────────────
    ph = h // 2
    pw = w // 2

    def resize(img):
        return cv2.resize(img, (pw, ph))

    top    = np.hstack([resize(p1), resize(p2)])
    bottom = np.hstack([resize(p3), resize(p4)])
    grid   = np.vstack([top, bottom])

    # ── Status bar ────────────────────────────────────────────────────────────
    bar_h  = 32
    bar    = np.zeros((bar_h, pw * 2, 3), dtype=np.uint8)
    ts_sec = idx / fps
    total_sec = total_frames / fps
    status = (f"frame {idx:5d}  |  {ts_sec:.2f}s / {total_sec:.1f}s  |  "
              f"blobs detected: {len(blobs)}  |  "
              f"min_area={cfg.min_blob_area}  max_area={cfg.max_blob_area}  "
              f"roi={cfg.roi}")
    cv2.putText(bar, status, (8, 22), FONT, 0.42, (180, 180, 180), 1)

    final = np.vstack([grid, bar])
    cv2.imwrite(f"{out_dir}/frame_{idx:05d}.jpg", final)


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description="Count salmon jumps in a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("video", help="Path to input video file")
    p.add_argument("--config", default=None,
                   help="Path to config.json (optional)")
    p.add_argument("--debug", default=None, metavar="DIR",
                   help="Save annotated debug frames to this directory")
    p.add_argument("--save-config", default=None, metavar="PATH",
                   help="Save the final merged config to a JSON file")

    g = p.add_argument_group("detection overrides",
        "Any of these override the value in config.json")
    g.add_argument("--salmon-hsv-lower", nargs=3, type=int, metavar=("H","S","V"),
                   help="e.g. --salmon-hsv-lower 0 0 40")
    g.add_argument("--salmon-hsv-upper", nargs=3, type=int, metavar=("H","S","V"),
                   help="e.g. --salmon-hsv-upper 180 60 160")
    g.add_argument("--silver-hsv-lower", nargs=3, type=int, metavar=("H","S","V"))
    g.add_argument("--silver-hsv-upper", nargs=3, type=int, metavar=("H","S","V"))
    g.add_argument("--min-blob-area",      type=int)
    g.add_argument("--max-blob-area",      type=int)
    g.add_argument("--roi", nargs=4, type=int, metavar=("X","Y","W","H"),
                   help="e.g. --roi 491 731 235 307")
    g.add_argument("--no-roi", action="store_true",
                   help="Disable ROI (use full frame)")
    g.add_argument("--min-jump-gap-sec",   type=float)
    g.add_argument("--sample-rate",        type=int)
    g.add_argument("--mog2-history",       type=int)
    g.add_argument("--mog2-var-threshold", type=int)
    return p


def apply_cli_overrides(cfg: SalmonConfig, args) -> SalmonConfig:
    """Merge CLI args on top of config — CLI always wins."""
    mapping = {
        "salmon_hsv_lower":   args.salmon_hsv_lower,
        "salmon_hsv_upper":   args.salmon_hsv_upper,
        "silver_hsv_lower":   args.silver_hsv_lower,
        "silver_hsv_upper":   args.silver_hsv_upper,
        "min_blob_area":      args.min_blob_area,
        "max_blob_area":      args.max_blob_area,
        "min_jump_gap_sec":   args.min_jump_gap_sec,
        "sample_rate":        args.sample_rate,
        "mog2_history":       args.mog2_history,
        "mog2_var_threshold": args.mog2_var_threshold,
    }
    for attr, val in mapping.items():
        if val is not None:
            setattr(cfg, attr, val)
    if args.no_roi:
        cfg.roi = None
    elif args.roi is not None:
        cfg.roi = args.roi
    return cfg


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # 1. Start with defaults
    cfg = SalmonConfig()

    # 2. Load config file if provided or if default exists
    config_path = args.config or (DEFAULT_CONFIG_PATH
                                  if Path(DEFAULT_CONFIG_PATH).exists() else None)
    if config_path:
        print(f"[config] Loading {config_path}", file=sys.stderr)
        cfg = SalmonConfig.from_file(config_path)

    # 3. CLI overrides on top
    cfg = apply_cli_overrides(cfg, args)
    cfg.summary()

    # 4. Optionally save merged config
    if args.save_config:
        cfg.save(args.save_config)

    result = count_salmon_jumps(args.video, cfg, debug_output=args.debug)
    print(json.dumps(result, indent=2))