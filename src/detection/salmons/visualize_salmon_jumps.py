# visualize_jumps.py
# Renders jump detections onto video frames and exports an annotated video.
#
# Usage (standalone):
#   python visualize_jumps.py <video> <result.json> [--config config.json] [--output out.mp4]
#
# Or import and call directly:
#   from visualize_jumps import visualize_jumps
#   visualize_jumps("video.mov", result_dict)

import cv2
import numpy as np
import json
import sys
import argparse
from pathlib import Path

# ── Visual constants (cosmetic only, not user-tunable) ───────────────────────
JUMP_FLASH_SEC    = 1.5
ROI_COLOR         = (0, 220, 255)
BLOB_COLOR        = (0, 255, 80)
JUMP_BANNER_COLOR = (0, 60, 255)
JUMP_TEXT_COLOR   = (255, 255, 255)
TIMELINE_H        = 48
FONT              = cv2.FONT_HERSHEY_SIMPLEX


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_roi_overlay(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    dimmed = (frame * 0.45).astype(np.uint8)
    frame = np.where(mask[:, :, None] == 255, frame, dimmed)
    cv2.rectangle(frame, (x, y), (x+w, y+h), ROI_COLOR, 2)
    cv2.putText(frame, "ROI", (x+6, y+20), FONT, 0.55, ROI_COLOR, 1)
    return frame


def draw_jump_banner(frame, jump_number, total_jumps, alpha):
    if alpha <= 0:
        return frame
    fh, w = frame.shape[:2]
    banner_h = 56
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), JUMP_BANNER_COLOR, -1)
    frame = cv2.addWeighted(overlay, alpha * 0.72, frame, 1 - alpha * 0.72, 0)
    label = f"  JUMP #{jump_number} detected   (total so far: {jump_number}/{total_jumps})"
    cv2.putText(frame, label, (12, 38), FONT, 0.85, JUMP_TEXT_COLOR, 2, cv2.LINE_AA)
    return frame


def draw_timeline(frame, current_sec, total_sec, jump_timestamps):
    fh, w = frame.shape[:2]
    bar_y = fh - TIMELINE_H
    cv2.rectangle(frame, (0, bar_y), (w, fh), (20, 20, 20), -1)
    progress = min(current_sec / max(total_sec, 1), 1.0)
    cv2.rectangle(frame, (0, bar_y + 6), (int(w * progress), fh - 6), (80, 80, 80), -1)
    for ts in jump_timestamps:
        mx = int((ts / total_sec) * w)
        pts = np.array([[mx, bar_y+4], [mx-7, fh-5], [mx+7, fh-5]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 200, 255))
    cv2.putText(frame, f"{current_sec:.1f}s / {total_sec:.1f}s",
                (w - 130, fh - 14), FONT, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, f"Jumps: {len(jump_timestamps)}",
                (10, fh - 14), FONT, 0.45, (180, 180, 180), 1)
    return frame


def draw_frame_info(frame, frame_idx, fps):
    label = f"#{frame_idx}  {frame_idx/fps:.2f}s"
    cv2.putText(frame, label, (frame.shape[1] - 160, 22),
                FONT, 0.45, (200, 200, 200), 1)
    return frame


# ── Main render function ──────────────────────────────────────────────────────
def visualize_jumps(video_path: str,
                    result: dict,
                    output_path: str = None,
                    roi=None,
                    show_roi: bool = True):
    """
    Args:
        video_path:  Input video
        result:      Dict with 'fps', 'jump_timestamps_sec', 'jump_count'.
                     If result contains 'config_used', ROI is read from there
                     automatically unless roi is passed explicitly.
        output_path: Output .mp4 path (default: <stem>_annotated.mp4)
        roi:         [x, y, w, h] list — overrides anything in result['config_used']
        show_roi:    Draw ROI overlay (default True)
    """
    fps          = result["fps"]
    timestamps   = result["jump_timestamps_sec"]
    total_jumps  = result["jump_count"]
    flash_frames = int(JUMP_FLASH_SEC * fps)

    # Auto-read ROI from embedded config if not passed explicitly
    if roi is None and "config_used" in result:
        roi = result["config_used"].get("roi")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_sec = total_frames / fps

    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_annotated.mp4")

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    jump_events      = {int(round(ts * fps)): i+1 for i, ts in enumerate(timestamps)}
    active_flash     = 0
    current_jump_num = 0
    frame_idx        = 0

    print(f"Rendering → {output_path}", file=sys.stderr)
    if roi:
        print(f"ROI: {roi}", file=sys.stderr)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in jump_events:
            current_jump_num = jump_events[frame_idx]
            active_flash = flash_frames
            print(f"  Jump #{current_jump_num} at frame {frame_idx} "
                  f"({frame_idx/fps:.2f}s)", file=sys.stderr)

        if show_roi and roi:
            frame = draw_roi_overlay(frame, roi)

        if active_flash > 0:
            frame = draw_jump_banner(frame, current_jump_num,
                                     total_jumps, active_flash / flash_frames)
            active_flash -= 1

        frame = draw_timeline(frame, frame_idx / fps, total_sec, timestamps)
        frame = draw_frame_info(frame, frame_idx, fps)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done. {frame_idx} frames written.", file=sys.stderr)
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description="Render annotated jump visualization video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("video",  help="Input video file")
    p.add_argument("result", help="JSON result file from salmon_jump_counter_cv.py")
    p.add_argument("--output", default=None,
                   help="Output path (default: <stem>_annotated.mp4)")
    p.add_argument("--config", default=None,
                   help="config.json to read ROI from")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X","Y","W","H"),
                   help="Override ROI, e.g. --roi 491 731 235 307")
    p.add_argument("--no-roi", action="store_true",
                   help="Disable ROI overlay entirely")
    p.add_argument("--no-show-roi", action="store_true",
                   help="Hide ROI rectangle on video")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    with open(args.result) as f:
        res = json.load(f)

    # ROI priority: --roi flag  >  --config file  >  embedded in result JSON
    roi = None
    if args.no_roi:
        roi = None
    elif args.roi:
        roi = args.roi
    elif args.config:
        with open(args.config) as f:
            roi = json.load(f).get("roi")
    # else: visualize_jumps() reads from res["config_used"] automatically

    visualize_jumps(
        video_path  = args.video,
        result      = res,
        output_path = args.output,
        roi         = roi,
        show_roi    = not args.no_show_roi,
    )