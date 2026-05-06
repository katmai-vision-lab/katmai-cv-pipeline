"""
Step 2 — Side-by-side demo viewer.

Left panel  : ByteTrack video with bounding boxes and bear IDs.
Right panel : Pre-computed behavior descriptions synced to playback timestamp.

Usage:
    python -m src.behavior.feeding_viewer \
        --video feed/data_video/salmon_jump_2.mov \
        --analysis predictions/salmon_jump_2_feeding_analysis/analysis.json

Controls:
    SPACE       pause / resume
    ← / →       seek -10s / +10s
    + / -       speed up / slow down
    Q / Esc     quit
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_DIR, RAW_DATA_DIR, TRACKERS_CONFIG_DIR, TRAINED_MODELS_DIR
from src.detection.detector import BearDetector

# BGR palette — must match detector.py and analyze_feeding.py
PALETTE_BGR = [
    (233, 180, 86), (0, 159, 230), (115, 158, 0),
    (66, 228, 240), (178, 114, 0), (0, 94, 213),
    (167, 121, 204), (255, 255, 0), (255, 0, 255), (0, 255, 0),
]
PALETTE_RGB = [(r, g, b) for (b, g, r) in PALETTE_BGR]

PANEL_BG      = (18, 18, 18)
PANEL_HEADER  = (30, 30, 30)
COLOR_TEXT    = (240, 240, 240)
COLOR_DIM     = (110, 110, 110)
COLOR_HINT    = (75, 75, 75)


def bear_bgr(display_id):
    return PALETTE_BGR[(display_id - 1) % len(PALETTE_BGR)]


def bear_rgb(display_id):
    return PALETTE_RGB[(display_id - 1) % len(PALETTE_RGB)]


def dim(color, factor=0.45):
    return tuple(int(c * factor) for c in color)


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_text(text, max_chars=38):
    """Simple word-wrap to a max character width."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines


def draw_summary_panel(width, height, summary_text, font_title, font_body, font_small):
    """Render a whole-video summary panel."""
    img = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle([(0, 0), (width, 130)], fill=PANEL_HEADER)
    draw.text((30, 38), "VIDEO SUMMARY", font=font_title, fill=COLOR_TEXT)
    draw.line([(30, 162), (width - 30, 162)], fill=(45, 45, 45), width=1)

    y = 200
    body = summary_text.strip() if summary_text else "(no summary available)"
    for paragraph in body.split("\n"):
        for line in wrap_text(paragraph, max_chars=40):
            if y + 56 > height - 80:
                break
            draw.text((30, y), line, font=font_body, fill=COLOR_TEXT)
            y += 56
        y += 18

    _draw_controls(draw, width, height, font_small)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_right_panel(width, height, current_sec, entry, stale,
                     font_title, font_body, font_small,
                     id_to_name: dict | None = None):
    """Render the right panel as a PIL image and return as BGR numpy array."""
    img = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(img)

    # Header bar (taller to accommodate larger title font)
    draw.rectangle([(0, 0), (width, 130)], fill=PANEL_HEADER)
    draw.text((30, 38), "BEAR FEEDING ANALYSIS", font=font_title, fill=COLOR_TEXT)

    # Timestamp + stale indicator
    ts = f"t = {int(current_sec) // 60:02d}:{int(current_sec) % 60:02d}"
    stale_note = "  (last known)" if stale else ""
    draw.text((30, 150), ts + stale_note, font=font_small,
              fill=COLOR_DIM if stale else (170, 170, 170))

    draw.line([(30, 210), (width - 30, 210)], fill=(45, 45, 45), width=1)

    y = 240

    if entry is None or not entry.get("bears"):
        draw.text((30, y), "No bears detected in this segment.", font=font_body, fill=COLOR_DIM)
        _draw_controls(draw, width, height, font_small)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for bid_str in sorted(entry["bears"].keys(), key=lambda x: int(x)):
        if y + 200 > height - 80:
            break

        bid = int(bid_str)
        bear = entry["bears"][bid_str]
        behavior = bear.get("behavior", "").strip()
        color = bear_rgb(bid)
        text_color = dim(color) if stale else COLOR_TEXT
        label_color = dim(color) if stale else color

        # Color dot + bear ID / identity name
        dot_y = y + 18
        draw.ellipse([(30, dot_y), (68, dot_y + 38)], fill=label_color)
        identity_name = None
        if id_to_name:
            identity_name = id_to_name.get(str(bid)) or id_to_name.get(bid)
        label = identity_name if identity_name else f"Bear {bid}"
        draw.text((90, y), label, font=font_body, fill=label_color)
        y += 72

        # Behavior text, word-wrapped
        if behavior:
            for line in wrap_text(behavior, max_chars=34):
                if y + 52 > height - 80:
                    break
                draw.text((90, y), line, font=font_small, fill=text_color)
                y += 52
        else:
            draw.text((90, y), "(no description)", font=font_small, fill=COLOR_DIM)
            y += 52

        y += 36  # gap between bears

    _draw_controls(draw, width, height, font_small)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _draw_controls(draw, width, height, font):
    draw.line([(30, height - 70), (width - 30, height - 70)], fill=(35, 35, 35), width=1)
    draw.text((30, height - 56), "SPACE pause  \u2190\u2192 seek \u00b110s  +/- speed  Q quit",
              font=font, fill=COLOR_HINT)


def resolve_video(path_str):
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    for candidate in [PROJECT_ROOT / p, RAW_DATA_DIR / p]:
        if candidate.exists():
            return candidate
    return p


def precompute_tracks(detector, video_path, conf):
    """Run ByteTrack once over the video and return display-ID-mapped boxes per frame."""
    tracker_cfg = str(TRACKERS_CONFIG_DIR / "bytetrack.yaml")
    results_stream = detector.model.track(
        source=str(video_path),
        conf=conf,
        classes=[0],
        tracker=tracker_cfg,
        save=False,
        stream=True,
        verbose=False,
        persist=True,
    )

    frame_data, raw_boxes = [], []
    for result in results_stream:
        track_ids, positions, boxes = [], {}, {}
        if result.boxes.id is not None:
            tids = result.boxes.id.cpu().numpy().astype(int).tolist()
            xyxy = result.boxes.xyxy.cpu().numpy()
            xywh = result.boxes.xywh.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for i, tid in enumerate(tids):
                positions[tid] = (float(xywh[i][0]), float(xywh[i][1]))
                boxes[tid] = (
                    int(xyxy[i][0]), int(xyxy[i][1]),
                    int(xyxy[i][2]), int(xyxy[i][3]),
                    float(confs[i]),
                )
            track_ids = tids
        frame_data.append({"frame": len(frame_data), "track_ids": track_ids,
                            "track_positions": positions})
        raw_boxes.append(boxes)

    _, id_map = detector._merge_fragmented_tracks(frame_data)
    unique_groups = sorted(set(id_map.values()))
    group_to_display = {g: i + 1 for i, g in enumerate(unique_groups)}

    display_boxes = []
    for rb in raw_boxes:
        disp = {}
        for raw_id, bbox in rb.items():
            gid = id_map.get(raw_id, raw_id)
            did = group_to_display.get(gid, raw_id)
            disp[did] = bbox
        display_boxes.append(disp)

    return display_boxes


def find_entry(entries, current_sec, interval_sec):
    """Return (entry, stale) for the current playback time."""
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


def render_to_video(video_path, analysis_path, model_path, conf, panel_w, output_path,
                    id_mapping_path=None):
    """Render side-by-side video to a file (no GUI required)."""
    with open(analysis_path) as f:
        analysis = json.load(f)
    entries = analysis["entries"]
    interval_sec = analysis.get("interval_sec", 2.0)
    summary_text = analysis.get("summary", "")

    # Optional: load PoseSwin identity mapping (display_id → "Bear A" / "480 Otis")
    id_to_name: dict[str, str] = {}
    if id_mapping_path:
        with open(id_mapping_path) as f:
            mapping_doc = json.load(f)
        for did, info in mapping_doc.get("mapping", {}).items():
            id_to_name[str(did)] = info["name"]
        print(f"Loaded identity mapping for {len(id_to_name)} bears: {id_to_name}")

    font_title = load_font(50)
    font_body  = load_font(44)
    font_small = load_font(36)

    cap_probe = cv2.VideoCapture(str(video_path))
    src_fps     = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h     = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w     = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_probe.release()

    print("Pre-computing ByteTrack tracks...")
    detector = BearDetector(model_path=model_path)
    display_boxes = precompute_tracks(detector, video_path, conf)
    print(f"Done — {len(display_boxes)} frames tracked.\n")

    out_w = frame_w + panel_w
    out_h = frame_h
    avi_path = Path(output_path).with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(str(avi_path), fourcc, src_fps, (out_w, out_h))

    cap = cv2.VideoCapture(str(video_path))
    from tqdm import tqdm
    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        ret, bgr = cap.read()
        if not ret:
            break

        # Draw ByteTrack boxes on frame
        if frame_idx < len(display_boxes):
            for did, (x1, y1, x2, y2, conf_val) in display_boxes[frame_idx].items():
                color = bear_bgr(did)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                # Use identity name if mapping is available
                name_for_box = id_to_name.get(str(did), f"Bear {did}")
                label = f"{name_for_box}  {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(bgr, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        current_sec = frame_idx / src_fps
        entry, stale = find_entry(entries, current_sec, interval_sec)

        right = draw_right_panel(panel_w, frame_h, current_sec, entry, stale,
                                  font_title, font_body, font_small,
                                  id_to_name=id_to_name)
        if right.shape[0] != bgr.shape[0]:
            right = cv2.resize(right, (panel_w, bgr.shape[0]))

        combined = np.hstack([bgr, right])
        writer.write(combined)

    cap.release()

    # ---- Append whole-video summary tail (~4s) ----
    if summary_text:
        last_frame = None
        cap_last = cv2.VideoCapture(str(video_path))
        cap_last.set(cv2.CAP_PROP_POS_FRAMES, max(total_frames - 1, 0))
        ret, last_frame = cap_last.read()
        cap_last.release()
        if last_frame is None:
            last_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Dim the left frame so the summary draws focus.
        dimmed = (last_frame * 0.45).astype(np.uint8)
        summary_panel = draw_summary_panel(panel_w, frame_h, summary_text,
                                            font_title, font_body, font_small)
        summary_combined = np.hstack([dimmed, summary_panel])
        tail_frames = int(src_fps * 4)
        for _ in range(tail_frames):
            writer.write(summary_combined)

    writer.release()

    # Convert to MP4
    mp4_path = Path(output_path).with_suffix(".mp4")
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(avi_path), "-c:v", "libx264", "-c:a", "aac", str(mp4_path)],
            check=True, capture_output=True
        )
        avi_path.unlink()
        print(f"\n✓ Output video: {mp4_path}")
        return mp4_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"\n✓ Output video: {avi_path}")
        return avi_path


def main():
    parser = argparse.ArgumentParser(description="Bear feeding behavior side-by-side viewer")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--analysis", required=True, help="Path to analysis.json")
    parser.add_argument("--model",
                        default=str(TRAINED_MODELS_DIR / "bear_detector3" / "weights" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (default: 1.0)")
    parser.add_argument("--panel-width", type=int, default=900,
                        help="Right panel width in pixels (default: 900)")
    parser.add_argument("--output", default=None,
                        help="Output video path (renders to file instead of GUI window)")
    parser.add_argument("--id-mapping", default=None,
                        help="Optional id_mapping.json from src.identity.identify_bears — "
                             "if provided, the side panel and box labels show bear names "
                             "(e.g. 'Bear A', '480 Otis') instead of raw display IDs")
    args = parser.parse_args()

    video_path = resolve_video(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    # Default: always render to file
    if args.output:
        out_path = args.output
    else:
        out_dir = Path(PREDICTIONS_DIR) / f"{video_path.stem}_feeding_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{video_path.stem}_feeding_demo.mp4")

    render_to_video(video_path, args.analysis, args.model, args.conf,
                    args.panel_width, out_path,
                    id_mapping_path=args.id_mapping)

    # Try GUI playback if display is available
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        has_gui = True
    except cv2.error:
        has_gui = False

    if has_gui:
        print("\nPlaying back... (Q to quit)")
        # ... interactive playback code could go here
        pass


if __name__ == "__main__":
    main()
