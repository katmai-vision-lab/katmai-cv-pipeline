"""
Render a video that overlays each bear's historical trajectory (a fading trail
of past centroids) read from a trajectories.json file produced by the tracker.

Usage:
    python -m src.detection.trajectory_video \
        --trajectories "predictions/<run>/trajectories.json"

    python -m src.detection.trajectory_video \
        --trajectories "predictions/<run>/trajectories.json" \
        --video data/raw/bears/foo.mp4 \
        --output outputs/foo_trails.mp4 \
        --trail-frames 300 --thickness 3

If --video is omitted, the script looks for the video listed in trajectories.json:
  1) next to the JSON file
  2) under data/raw/  (recursively)
"""

import argparse
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_DIR, RAW_DATA_DIR

# Same palette as feeding_viewer.py / detector overlay so colors are consistent.
PALETTE_BGR = [
    (233, 180, 86), (0, 159, 230), (115, 158, 0),
    (66, 228, 240), (178, 114, 0), (0, 94, 213),
    (167, 121, 204), (255, 255, 0), (255, 0, 255), (0, 255, 0),
]


def bear_color(display_id):
    return PALETTE_BGR[(display_id - 1) % len(PALETTE_BGR)]


def resolve_video(traj_json_path, video_arg, video_name_in_json):
    if video_arg:
        p = Path(video_arg)
        if p.is_absolute() and p.exists():
            return p
        for cand in [PROJECT_ROOT / p, RAW_DATA_DIR / p]:
            if cand.exists():
                return cand
        return p

    # 1) sibling of the JSON
    sibling = traj_json_path.parent / video_name_in_json
    if sibling.exists():
        return sibling

    # 2) search data/raw
    matches = list(RAW_DATA_DIR.rglob(video_name_in_json))
    if matches:
        return matches[0]

    return Path(video_name_in_json)


def build_per_frame_index(bears_dict):
    """
    Returns:
        per_frame[frame_idx] = list of dicts {bid, cx, cy, w, h, conf}
        bear_ids = sorted display IDs (1-based)
    """
    per_frame = {}
    bear_ids = []
    # Bear keys look like "bear_1", "bear_2"; preserve numeric order.
    for key in sorted(bears_dict.keys(), key=lambda k: int(k.split("_")[-1])):
        bid = int(key.split("_")[-1])
        bear_ids.append(bid)
        for pt in bears_dict[key]["trajectory"]:
            per_frame.setdefault(pt["frame"], []).append({
                "bid": bid,
                "cx": float(pt["cx"]),
                "cy": float(pt["cy"]),
                "w":  float(pt.get("w", 0)),
                "h":  float(pt.get("h", 0)),
                "conf": float(pt.get("conf", 0)),
            })
    return per_frame, bear_ids


def draw_trail(frame, trail_points, color, thickness):
    """Draw a fading polyline. Older points are dimmer/thinner."""
    if len(trail_points) < 2:
        return
    n = len(trail_points)
    for i in range(1, n):
        t = i / n  # 0 (oldest) -> 1 (newest)
        # Fade: blend toward black for older points.
        c = tuple(int(ch * (0.25 + 0.75 * t)) for ch in color)
        thick = max(1, int(round(thickness * (0.4 + 0.6 * t))))
        p0 = trail_points[i - 1]
        p1 = trail_points[i]
        cv2.line(frame, p0, p1, c, thick, lineType=cv2.LINE_AA)


def draw_legend(frame, bear_ids, font_scale=0.6):
    """Top-left color legend: 'Bear 1', 'Bear 2', ..."""
    pad = 8
    line_h = int(22 * font_scale / 0.6)
    box_w = 130
    box_h = pad * 2 + line_h * len(bear_ids)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, dst=frame)

    for i, bid in enumerate(bear_ids):
        y = 10 + pad + (i + 1) * line_h - 6
        color = bear_color(bid)
        cv2.circle(frame, (24, y - 5), 6, color, -1)
        cv2.putText(frame, f"Bear {bid}", (38, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def render(traj_path, video_path, output_path, trail_frames, thickness,
           draw_box, draw_id, reset_gap_frames=30):
    with open(traj_path) as f:
        traj = json.load(f)

    per_frame, bear_ids = build_per_frame_index(traj["bears"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or float(traj.get("fps") or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(traj.get("total_frames") or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    avi_path = output_path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(avi_path), fourcc, fps, (w, h))

    # One bounded deque of past centroids per bear, capped at trail_frames.
    trails = {bid: deque(maxlen=trail_frames) for bid in bear_ids}
    last_seen = {bid: -10**9 for bid in bear_ids}

    for f_idx in tqdm(range(total), desc="Rendering trails"):
        ret, frame = cap.read()
        if not ret:
            break

        # Push current-frame positions into each bear's trail deque.
        # If this bear has been gone for > reset_gap_frames (camera zoom / cut),
        # clear its old trail — old pixel coords won't match the new view.
        seen_this_frame = set()
        for det in per_frame.get(f_idx, []):
            bid = det["bid"]
            if f_idx - last_seen[bid] > reset_gap_frames:
                trails[bid].clear()
            trails[bid].append((int(round(det["cx"])), int(round(det["cy"]))))
            last_seen[bid] = f_idx
            seen_this_frame.add(bid)

        # Only draw trails for bears visible in the current frame.
        for bid in bear_ids:
            if bid in seen_this_frame:
                draw_trail(frame, list(trails[bid]), bear_color(bid), thickness)

        # Draw current detection markers (dot, optional bbox + ID).
        for det in per_frame.get(f_idx, []):
            bid = det["bid"]
            color = bear_color(bid)
            cx, cy = int(round(det["cx"])), int(round(det["cy"]))
            cv2.circle(frame, (cx, cy), max(thickness + 2, 5), color, -1, cv2.LINE_AA)

            if draw_box and det["w"] > 0 and det["h"] > 0:
                x1 = int(round(cx - det["w"] / 2))
                y1 = int(round(cy - det["h"] / 2))
                x2 = int(round(cx + det["w"] / 2))
                y2 = int(round(cy + det["h"] / 2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if draw_id:
                label = f"Bear {bid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                lx = cx + 8
                ly = max(th + 4, cy - 8)
                cv2.rectangle(frame, (lx - 2, ly - th - 4), (lx + tw + 4, ly + 4), color, -1)
                cv2.putText(frame, label, (lx + 1, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        draw_legend(frame, bear_ids)

        # Frame counter (top-right)
        ts = f"frame {f_idx}/{total - 1}"
        (tw, th), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame, ts, (w - tw - 12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()

    # Try transcoding to MP4 with ffmpeg for portability; fall back to AVI.
    mp4_path = output_path.with_suffix(".mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(avi_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path)],
            check=True, capture_output=True,
        )
        avi_path.unlink()
        print(f"\n✓ Output video: {mp4_path}")
        return mp4_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"\n✓ Output video: {avi_path}  (ffmpeg unavailable, kept AVI)")
        return avi_path


def main():
    parser = argparse.ArgumentParser(
        description="Overlay each bear's historical trajectory on the video"
    )
    parser.add_argument("--trajectories", required=True,
                        help="Path to trajectories.json")
    parser.add_argument("--video", default=None,
                        help="Source video path (auto-detected from JSON if omitted)")
    parser.add_argument("--output", default=None,
                        help="Output video path (default: predictions/<run>/trajectories_overlay.mp4)")
    parser.add_argument("--trail-frames", type=int, default=0,
                        help="Max past frames in each trail (0 = full history; default 0)")
    parser.add_argument("--thickness", type=int, default=3,
                        help="Trail line thickness in pixels (default 3)")
    parser.add_argument("--no-box", action="store_true",
                        help="Do not draw current-frame bounding boxes")
    parser.add_argument("--no-id", action="store_true",
                        help="Do not draw 'Bear N' labels next to each bear")
    parser.add_argument("--reset-gap-frames", type=int, default=30,
                        help="Clear a bear's trail when it has been absent for >N frames "
                             "(handles camera zoom/cut where old pixel coords no longer "
                             "match the current view). Default 30.")
    args = parser.parse_args()

    traj_path = Path(args.trajectories)
    if not traj_path.is_absolute():
        traj_path = (PROJECT_ROOT / traj_path).resolve()
    if not traj_path.exists():
        print(f"Error: trajectories.json not found: {args.trajectories}")
        sys.exit(1)

    with open(traj_path) as f:
        traj_meta = json.load(f)

    video_path = resolve_video(traj_path, args.video, traj_meta.get("video", ""))
    if not video_path.exists():
        print(f"Error: source video not found: {video_path}")
        print(f"  Hint: pass --video <path>  (JSON listed: {traj_meta.get('video')})")
        sys.exit(1)

    if args.output:
        out_path = args.output
    else:
        out_path = str(traj_path.parent / "trajectories_overlay.mp4")

    # trail_frames=0 means "keep entire history" -> use total_frames as deque cap.
    trail_cap = args.trail_frames if args.trail_frames > 0 else \
        max(int(traj_meta.get("total_frames") or 1), 1)

    render(
        traj_path=traj_path,
        video_path=video_path,
        output_path=out_path,
        trail_frames=trail_cap,
        thickness=args.thickness,
        draw_box=not args.no_box,
        draw_id=not args.no_id,
        reset_gap_frames=args.reset_gap_frames,
    )


if __name__ == "__main__":
    main()
