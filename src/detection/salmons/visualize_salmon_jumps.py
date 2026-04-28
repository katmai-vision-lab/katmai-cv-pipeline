# visualize_jumps.py
# Renders jump detections onto video frames and exports an annotated video.
#
# Usage:
#   python visualize_jumps.py <video.mp4> <result.json> [output.mp4]
#
# Or call visualize_jumps() directly from salmon_jump_counter_cv.py:
#   from visualize_jumps import visualize_jumps
#   visualize_jumps("video.mov", result_dict, "annotated.mp4")

import cv2
import numpy as np
import json
import sys
from pathlib import Path

# ── Visual config ─────────────────────────────────────────────────────────────
JUMP_FLASH_SEC     = 1.5    # how long the jump banner stays on screen (seconds)
ROI_COLOR          = (0, 220, 255)   # cyan box around ROI
BLOB_COLOR         = (0, 255, 80)    # green circle on detected blob
JUMP_BANNER_COLOR  = (0, 60, 255)    # red banner background
JUMP_TEXT_COLOR    = (255, 255, 255)
TIMELINE_H         = 48             # height of timeline bar at bottom (px)
FONT               = cv2.FONT_HERSHEY_SIMPLEX


def draw_roi_overlay(frame, roi):
    """Draw semi-transparent ROI rectangle."""
    if roi is None:
        return frame
    x, y, w, h = roi
    overlay = frame.copy()
    # Dim everything outside ROI
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    dimmed = (frame * 0.45).astype(np.uint8)
    frame = np.where(mask[:, :, None] == 255, frame, dimmed)
    cv2.rectangle(frame, (x, y), (x+w, y+h), ROI_COLOR, 2)
    cv2.putText(frame, "ROI", (x+6, y+20), FONT, 0.55, ROI_COLOR, 1)
    return frame


def draw_blob_detections(frame, blobs):
    """Draw green circles on each detected salmon blob."""
    for cx, cy, area in blobs:
        r = max(12, int(np.sqrt(area / np.pi)))
        cv2.circle(frame, (cx, cy), r, BLOB_COLOR, 2)
        cv2.circle(frame, (cx, cy), 3, BLOB_COLOR, -1)
        cv2.putText(frame, f"{int(area)}px", (cx - 28, cy - r - 6),
                    FONT, 0.42, BLOB_COLOR, 1)
    return frame


def draw_jump_banner(frame, jump_number, total_jumps, alpha):
    """
    Flash a translucent red banner at the top when a jump is detected.
    alpha: 0.0 (invisible) → 1.0 (fully visible), fades out over time.
    """
    if alpha <= 0:
        return frame
    h, w = frame.shape[:2]
    banner_h = 56
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), JUMP_BANNER_COLOR, -1)
    frame = cv2.addWeighted(overlay, alpha * 0.72, frame, 1 - alpha * 0.72, 0)
    label = f"  JUMP #{jump_number} detected   (total so far: {jump_number}/{total_jumps})"
    cv2.putText(frame, label, (12, 38), FONT, 0.85, JUMP_TEXT_COLOR,
                2, cv2.LINE_AA)
    return frame


def draw_timeline(frame, current_sec, total_sec, jump_timestamps, fps):
    """
    Draw a progress timeline at the bottom with jump markers.
    """
    h, w = frame.shape[:2]
    bar_y = h - TIMELINE_H
    # Background bar
    cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)

    # Progress fill
    progress = min(current_sec / max(total_sec, 1), 1.0)
    fill_w = int(w * progress)
    cv2.rectangle(frame, (0, bar_y + 6), (fill_w, h - 6), (80, 80, 80), -1)

    # Jump markers (yellow triangles)
    for ts in jump_timestamps:
        mx = int((ts / total_sec) * w)
        pts = np.array([[mx, bar_y + 4], [mx - 7, h - 5], [mx + 7, h - 5]],
                       np.int32)
        cv2.fillPoly(frame, [pts], (0, 200, 255))

    # Time label
    elapsed = f"{current_sec:.1f}s / {total_sec:.1f}s"
    cv2.putText(frame, elapsed, (w - 130, h - 14), FONT, 0.45,
                (180, 180, 180), 1)

    # Jump count label
    cv2.putText(frame, f"Jumps: {len(jump_timestamps)}", (10, h - 14),
                FONT, 0.45, (180, 180, 180), 1)

    return frame


def draw_frame_info(frame, frame_idx, fps):
    """Small frame counter top-right."""
    ts = frame_idx / fps
    label = f"#{frame_idx}  {ts:.2f}s"
    cv2.putText(frame, label, (frame.shape[1] - 160, 22),
                FONT, 0.45, (200, 200, 200), 1)
    return frame


def visualize_jumps(video_path: str,
                    result: dict,
                    output_path: str = None,
                    roi: tuple = None,
                    show_roi: bool = True):
    """
    Render an annotated video with:
      - ROI overlay (dimmed outside, cyan border)
      - Jump flash banner at each detected timestamp
      - Timeline bar with jump markers
      - Frame counter

    Args:
        video_path:   Input video path
        result:       Dict from count_salmon_jumps() — needs
                      'fps', 'jump_timestamps_sec', 'jump_count'
        output_path:  Where to save annotated video (default: <input>_annotated.mp4)
        roi:          (x, y, w, h) tuple — pass your ROI constant here
        show_roi:     Whether to draw the ROI overlay
    """
    fps            = result["fps"]
    timestamps     = result["jump_timestamps_sec"]
    total_jumps    = result["jump_count"]
    flash_frames   = int(JUMP_FLASH_SEC * fps)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_sec = total_frames / fps

    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_annotated.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Pre-compute which frames trigger a jump flash
    # jump_frame_idx → jump_number (1-based)
    jump_events = {}
    for i, ts in enumerate(timestamps):
        jump_events[int(round(ts * fps))] = i + 1

    active_flash = 0    # frames remaining for current flash
    current_jump_num = 0

    frame_idx = 0
    print(f"Rendering annotated video → {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this frame triggers a jump
        if frame_idx in jump_events:
            current_jump_num = jump_events[frame_idx]
            active_flash = flash_frames
            print(f"  Jump #{current_jump_num} at frame {frame_idx} "
                  f"({frame_idx/fps:.2f}s)")

        # Draw layers
        if show_roi and roi:
            frame = draw_roi_overlay(frame, roi)

        # Jump banner (fades out)
        if active_flash > 0:
            alpha = active_flash / flash_frames   # 1.0 → 0.0
            frame = draw_jump_banner(frame, current_jump_num,
                                     total_jumps, alpha)
            active_flash -= 1

        frame = draw_timeline(frame, frame_idx / fps, total_sec,
                              timestamps, fps)
        frame = draw_frame_info(frame, frame_idx, fps)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done. {frame_idx} frames written to {output_path}")
    return output_path


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_jumps.py <video> <result.json> [output.mp4]")
        print("")
        print("Example:")
        print("  python visualize_jumps.py salmon_jump_9.mov result.json")
        sys.exit(1)

    video   = sys.argv[1]
    jfile   = sys.argv[2]
    outfile = sys.argv[3] if len(sys.argv) > 3 else None

    with open(jfile) as f:
        res = json.load(f)

    # Import ROI from your counter script if available
    try:
        from salmon_jump_counter_cv import ROI
    except ImportError:
        ROI = None

    visualize_jumps(video, res, outfile, roi=ROI)