"""
Step 2 — Feeding behavior viewer with closed-caption overlay.

Renders bounding boxes + YouTube-style closed captions (bottom of frame)
from a pre-computed analysis.json produced by analyze_feeding.py.

Usage:
    python -m src.behavior.feeding_viewer \
        --video path/to/video.mp4 \
        --analysis predictions/<stem>_feeding_analysis/analysis.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_DIR, RAW_DATA_DIR, TRAINED_MODELS_DIR
from src.detection.detector import BearDetector

PALETTE_BGR = [
    (233, 180, 86), (0, 159, 230), (115, 158, 0),
    (66, 228, 240), (178, 114, 0), (0, 94, 213),
    (167, 121, 204), (255, 255, 0), (255, 0, 255), (0, 255, 0),
]


def bear_bgr(display_id):
    return PALETTE_BGR[(display_id - 1) % len(PALETTE_BGR)]


def resolve_video(path_str):
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    for candidate in [PROJECT_ROOT / p, RAW_DATA_DIR / p]:
        if candidate.exists():
            return candidate
    return p


def wrap_text_cv2(text, max_width_px, font, scale, thickness):
    """Word-wrap text to fit within max_width_px. Returns list of lines."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if current and tw > max_width_px:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [""]


def draw_captions(frame, entry, stale, id_to_name=None):
    """Overlay closed captions on the bottom of frame in-place."""
    if entry is None or not entry.get("bears"):
        return frame

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    SCALE      = 0.62
    THICK      = 1
    WHITE      = (255, 255, 255)
    STALE_DIM  = 0.45

    h, w = frame.shape[:2]
    pad     = 14
    line_h  = 30
    label_gap = 10  # pixels between label and behavior text

    # Build caption lines: list of (label, label_color, behavior_text)
    caption_rows = []
    for bid_str in sorted(entry["bears"], key=lambda x: int(x)):
        bid = int(bid_str)
        behavior = entry["bears"][bid_str].get("behavior", "").strip()
        if not behavior:
            continue
        color = bear_bgr(bid)
        if stale:
            color = tuple(int(c * STALE_DIM) for c in color)
        name = (id_to_name or {}).get(str(bid), f"Bear {bid}")
        label = f"{name}:"

        (lw, _), _ = cv2.getTextSize(label, FONT, SCALE, THICK)
        avail_w = w - 2 * pad - lw - label_gap
        lines = wrap_text_cv2(behavior, avail_w, FONT, SCALE, THICK)
        caption_rows.append((label, color, lines[0]))
        for extra in lines[1:]:
            caption_rows.append(("", color, extra))

    if not caption_rows:
        return frame

    bar_h = len(caption_rows) * line_h + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = h - bar_h + pad + line_h - 4
    for label, color, beh_text in caption_rows:
        x = pad
        if label:
            cv2.putText(frame, label, (x, y), FONT, SCALE, color, THICK, cv2.LINE_AA)
            (lw, _), _ = cv2.getTextSize(label, FONT, SCALE, THICK)
            x += lw + label_gap
        text_color = WHITE if not stale else (160, 160, 160)
        cv2.putText(frame, beh_text, (x, y), FONT, SCALE, text_color, THICK, cv2.LINE_AA)
        y += line_h

    return frame


def draw_summary_overlay(frame, summary_text):
    """Overlay the whole-video summary as CC on a dimmed frame."""
    dimmed = (frame * 0.45).astype(np.uint8)

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.60
    THICK = 1
    h, w  = dimmed.shape[:2]
    pad   = 30
    line_h = 28

    lines = []
    for para in summary_text.strip().split("\n"):
        lines += wrap_text_cv2(para.strip(), w - 2 * pad, FONT, SCALE, THICK)
        lines.append("")

    # Trim trailing blank lines
    while lines and lines[-1] == "":
        lines.pop()

    bar_h = len(lines) * line_h + pad * 2
    bar_top = max(0, h - bar_h - 60)

    overlay = dimmed.copy()
    cv2.rectangle(overlay, (0, bar_top), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, dimmed, 0.35, 0, dimmed)

    y = bar_top + pad + line_h - 4
    for line in lines:
        if y > h - pad:
            break
        cv2.putText(dimmed, line, (pad, y), FONT, SCALE, (230, 230, 230), THICK, cv2.LINE_AA)
        y += line_h

    return dimmed


def find_entry(entries, current_sec, interval_sec):
    """Return (entry, stale) closest to current_sec."""
    if not entries:
        return None, False
    best, best_dist = None, float("inf")
    for e in entries:
        d = abs(e["timestamp_sec"] - current_sec)
        if d < best_dist:
            best_dist = d
            best = e
    stale = best_dist > interval_sec * 0.6
    return best, stale


def render_to_video(video_path, analysis_path, model_path, conf, output_path,
                    id_mapping_path=None):
    with open(analysis_path) as f:
        analysis = json.load(f)
    entries      = analysis["entries"]
    interval_sec = analysis.get("interval_sec", 2.0)
    summary_text = analysis.get("summary", "")

    id_to_name: dict[str, str] = {}
    if id_mapping_path:
        with open(id_mapping_path) as f:
            mapping_doc = json.load(f)
        for did, info in mapping_doc.get("mapping", {}).items():
            id_to_name[str(did)] = info["name"]
        print(f"Loaded identity mapping for {len(id_to_name)} bears: {id_to_name}")

    cap_probe   = cv2.VideoCapture(str(video_path))
    src_fps     = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h     = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w     = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_probe.release()

    canonical_cache = BearDetector.canonical_tracking_cache(video_path)
    if canonical_cache.exists():
        print(f"Loading cached tracks from {canonical_cache} ...")
        with open(canonical_cache) as f:
            cache = json.load(f)
        display_boxes = [
            {int(k): tuple(v) for k, v in frame.items()}
            for frame in cache["display_frame_boxes"]
        ]
        print(f"Done — {len(display_boxes)} frames loaded from cache.\n")
    else:
        print("Pre-computing ByteTrack tracks...")
        detector = BearDetector(model_path=model_path)
        display_boxes = detector.compute_display_boxes(
            video_path, conf=conf, cache_path=canonical_cache,
        )
        print(f"Done — {len(display_boxes)} frames tracked.\n")

    tmp_path = Path(output_path).with_suffix(".avi")
    fourcc   = cv2.VideoWriter_fourcc(*"XVID")
    writer   = cv2.VideoWriter(str(tmp_path), fourcc, src_fps, (frame_w, frame_h))

    # Pre-read last frame for the summary end-card
    last_frame = None
    if summary_text:
        cap_last = cv2.VideoCapture(str(video_path))
        cap_last.set(cv2.CAP_PROP_POS_FRAMES, max(total_frames - 1, 0))
        ret, last_frame = cap_last.read()
        cap_last.release()
    if last_frame is None:
        last_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(str(video_path))
    from tqdm import tqdm
    try:
        for frame_idx in tqdm(range(total_frames), desc="Rendering"):
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx < len(display_boxes):
                for did, (x1, y1, x2, y2, conf_val) in display_boxes[frame_idx].items():
                    color = bear_bgr(did)
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                    name = id_to_name.get(str(did), f"Bear {did}")
                    label = f"{name}  {conf_val:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(bgr, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            current_sec = frame_idx / src_fps
            entry, stale = find_entry(entries, current_sec, interval_sec)
            draw_captions(bgr, entry, stale, id_to_name=id_to_name)
            writer.write(bgr)

        # Summary end-card — written into the same file before closing
        if summary_text:
            summary_frame = draw_summary_overlay(last_frame, summary_text)
            for _ in range(int(src_fps * 4)):
                writer.write(summary_frame)
    finally:
        cap.release()
        writer.release()

    # Transcode temp AVI → MP4 and remove the AVI
    mp4_path = Path(output_path).with_suffix(".mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path)],
            check=True, capture_output=True,
        )
        tmp_path.unlink(missing_ok=True)
        print(f"\n✓ Output video: {mp4_path}")
        return mp4_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"\n✓ Output video: {tmp_path}  (ffmpeg unavailable, kept AVI)")
        return tmp_path


def main():
    parser = argparse.ArgumentParser(description="Bear feeding behavior viewer with CC overlay")
    parser.add_argument("--video",    required=True, help="Input video path")
    parser.add_argument("--analysis", required=True, help="Path to analysis.json")
    parser.add_argument("--model",
                        default=str(TRAINED_MODELS_DIR / "bear_detector3" / "weights" / "best.pt"))
    parser.add_argument("--conf",     type=float, default=0.25)
    parser.add_argument("--output",   default=None,
                        help="Output video path (default: predictions/<stem>_feeding_analysis/)")
    parser.add_argument("--id-mapping", default=None,
                        help="Optional id_mapping.json for bear names")
    args = parser.parse_args()

    video_path = resolve_video(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    if args.output:
        out_path = args.output
    else:
        out_dir = Path(PREDICTIONS_DIR) / f"{video_path.stem}_feeding_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{video_path.stem}_feeding_demo.mp4")

    render_to_video(video_path, args.analysis, args.model, args.conf, out_path,
                    id_mapping_path=args.id_mapping)


if __name__ == "__main__":
    main()
