"""
Step 1 — Pre-compute bear feeding behavior analysis using Molmo2-8B.

Runs YOLO + ByteTrack on the input video, samples one frame every N seconds,
sends each frame to a local vision model with per-bear position context,
and saves timestamped behavior descriptions to analysis.json.

Usage:
    python -m src.behavior.analyze_feeding --video feed/data_video/salmon_jump_2.mov
    python -m src.behavior.analyze_feeding --video path/to/video.mp4 --interval 2
    python -m src.behavior.analyze_feeding --video path/to/video.mp4 --save-frames
"""

import argparse
import json
import re
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# BGR palette — must match feeding_viewer.py and detector.py
PALETTE_BGR = [
    (233, 180, 86), (0, 159, 230), (115, 158, 0),
    (66, 228, 240), (178, 114, 0), (0, 94, 213),
    (167, 121, 204), (255, 255, 0), (255, 0, 255), (0, 255, 0),
]


def annotate_frame(bgr, bear_boxes):
    """Draw bboxes + Bear IDs on the frame (copy) so the VLM can read them."""
    annotated = bgr.copy()
    for bid in sorted(bear_boxes.keys()):
        x1, y1, x2, y2, _ = bear_boxes[bid]
        color = PALETTE_BGR[(bid - 1) % len(PALETTE_BGR)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"Bear {bid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return annotated

from src.config import PREDICTIONS_DIR, RAW_DATA_DIR, TRACKERS_CONFIG_DIR, TRAINED_MODELS_DIR
from src.detection.detector import BearDetector


def position_hint(x1, y1, x2, y2, frame_w, frame_h):
    """Describe a bounding box location in human-readable terms."""
    cx = (x1 + x2) / 2 / frame_w
    cy = (y1 + y2) / 2 / frame_h
    vert = "top" if cy < 0.35 else ("bottom" if cy > 0.65 else "center")
    horiz = "left" if cx < 0.35 else ("right" if cx > 0.65 else "center")
    if vert == "center" and horiz == "center":
        return "center of frame"
    if vert == "center":
        return f"{horiz} side of frame"
    if horiz == "center":
        return f"{vert} of frame"
    return f"{vert}-{horiz} of frame"


def load_vision_backend(backend_name: str, model_name: str | None = None):
    """Pluggable VLM backend (Molmo2 / OpenAI / Anthropic / Gemini / custom).

    See `src/behavior/backends/` for available backends and how to add new ones.
    Returns an object with `.analyze_frame(pil, prompt)` and
    `.summarize_video(timeline, ref_pil)` methods.
    """
    from src.behavior.backends import get_backend

    kwargs = {}
    if backend_name == "molmo2" and model_name:
        kwargs["model_name"] = model_name
    elif backend_name in ("openai", "anthropic", "gemini") and model_name:
        kwargs["model"] = model_name
    return get_backend(backend_name, **kwargs)


def build_prompt(bear_boxes, frame_w, frame_h):
    bear_lines = []
    for bid in sorted(bear_boxes.keys()):
        x1, y1, x2, y2, _ = bear_boxes[bid]
        hint = position_hint(x1, y1, x2, y2, frame_w, frame_h)
        bear_lines.append(f"  Bear {bid} — {hint}")
    bears_str = "\n".join(bear_lines)

    return (
        "You are a wildlife biologist analyzing a video frame of brown bears at Brooks Falls, Alaska.\n\n"
        "Each bear in the image has been pre-annotated with a colored bounding box and a "
        "label like 'Bear 1', 'Bear 2'. USE THESE ON-IMAGE LABELS to identify each bear. "
        "Do not re-number them based on position.\n\n"
        f"Detected bears in this frame (for reference):\n{bears_str}\n\n"
        "For each bear, classify their EXACT feeding stage as one of:\n"
        "  - WAITING: standing or sitting, scanning water, no fish contact\n"
        "  - LUNGING: diving or snapping at water, mouth open, no fish yet\n"
        "  - CATCHING: mouth clamping down on a salmon, fish visible in jaws or splash\n"
        "  - EATING: holding or chewing a fish, often moving away from water\n"
        "  - MISSED: pulling back from water empty-mouthed after a strike\n\n"
        "Write one sentence per bear: start with the stage in brackets, then describe the specific action.\n"
        "Be precise about whether a fish is visible in the bear's mouth.\n\n"
        "Format:\n"
        "Bear 1: [STAGE] <description>\n"
        "Bear 2: [STAGE] <description>\n\n"
        "Only describe bears listed above."
    )


def parse_response(raw, bear_boxes):
    """Parse Bear N: [STAGE] <text> lines from model response."""
    behaviors = {}
    for line in raw.split("\n"):
        line = line.strip()
        for sep in (":", ":"):
            if line.lower().startswith("bear ") and sep in line:
                parts = line.split(sep, 1)
                try:
                    bid = int(parts[0].strip().split()[-1])
                    if bid in bear_boxes:
                        behaviors[bid] = parts[1].strip()
                except (ValueError, IndexError):
                    pass
                break
    return behaviors


def generate_summary(backend, entries, video_path):
    """Produce a narrative summary of feeding events across the whole video."""
    if not entries:
        return "No bears were detected in the video."

    timeline_lines = []
    last_seen = {}
    for e in entries:
        t = e["timestamp_sec"]
        for bid_str, bear in e["bears"].items():
            beh = (bear.get("behavior") or "").strip()
            if not beh:
                continue
            if last_seen.get(bid_str) != beh:
                timeline_lines.append(f"  t={t:.1f}s  Bear {bid_str}: {beh}")
                last_seen[bid_str] = beh

    timeline_str = "\n".join(timeline_lines) if timeline_lines else "  (no behaviors recorded)"

    # Seek to a middle-of-video frame as visual context — read one frame, then release.
    image_pil = None
    mid_idx = entries[len(entries) // 2]["frame_idx"]
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, ref_frame = cap.read()
    cap.release()
    if ret and ref_frame is not None:
        h, w = ref_frame.shape[:2]
        max_side = 512
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            ref_frame = cv2.resize(ref_frame, (int(w * scale), int(h * scale)))
        image_pil = Image.fromarray(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))

    return backend.summarize_video(timeline_str, reference_image_pil=image_pil)


def resolve_video(path_str):
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    for candidate in [PROJECT_ROOT / p, RAW_DATA_DIR / p]:
        if candidate.exists():
            return candidate
    return p


def run(
    video_path: str,
    output: str | None = None,
    interval: float = 0.5,
    model: str | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    backend: str = "molmo2",
    vision_model: str | None = None,
    dedupe_threshold: float = 0.7,
    save_frames: bool = False,
) -> Path:
    """
    Run feeding behavior analysis on a video and return the output JSON path.

    This is the programmatic entry point used by the TUI and other callers.
    For CLI usage see main() below.
    """
    from types import SimpleNamespace

    model = model or str(TRAINED_MODELS_DIR / "bear_detector3" / "weights" / "best.pt")
    video_path_obj = resolve_video(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = PREDICTIONS_DIR / f"{video_path_obj.stem}_feeding_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(output) if output else out_dir / "analysis.json"
    frames_dir = (out_dir / "sampled_frames") if save_frames else None
    if frames_dir:
        frames_dir.mkdir(exist_ok=True)

    args = SimpleNamespace(
        video=str(video_path_obj),
        output=str(json_path),
        interval=interval,
        model=model,
        conf=conf,
        iou=iou,
        backend=backend,
        vision_model=vision_model,
        dedupe_threshold=dedupe_threshold,
        save_frames=save_frames,
    )
    _run_analysis(args, video_path_obj, json_path, frames_dir)
    return json_path


def _run_analysis(args, video_path, json_path, frames_dir):
    """Core analysis logic, shared by run() and main()."""
    print("=" * 60)
    print("Bear Feeding Behavior Analysis")
    print("=" * 60)
    print(f"Video        : {video_path}")
    print(f"Output       : {json_path}")
    print(f"Interval     : {args.interval}s")
    print(f"YOLO model   : {Path(args.model).name}")
    print(f"Backend      : {args.backend}")
    print(f"Vision model : {args.vision_model or '(backend default)'}")
    print("=" * 60)

    out_dir = json_path.parent
    tracking_cache_path = out_dir / "tracking_cache.json"
    progress_path       = out_dir / "entries_progress.jsonl"

    # ── Steps 1+2: ByteTrack + merge ─────────────────────────────────────────
    if tracking_cache_path.exists():
        print(f"\n[1/3] Loading cached tracking results (delete {tracking_cache_path.name} to re-track)...")
        with open(tracking_cache_path) as f:
            cache = json.load(f)
        src_fps      = cache["src_fps"]
        total_frames = cache["total_frames"]
        frame_w      = cache["frame_w"]
        frame_h      = cache["frame_h"]
        display_frame_boxes = [
            {int(k): tuple(v) for k, v in frame.items()}
            for frame in cache["display_frame_boxes"]
        ]
        print(f"  ✓ {len(display_frame_boxes)} frames loaded from cache")
        print("\n[2/3] Skipped (cached).")
    else:
        print("\n[1/3] Running YOLO + ByteTrack...")
        detector = BearDetector(model_path=args.model)

        cap_probe = cv2.VideoCapture(str(video_path))
        src_fps      = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w      = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h      = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_probe.release()

        tracker_cfg = str(TRACKERS_CONFIG_DIR / "bytetrack.yaml")
        results_stream = detector.model.track(
            source=str(video_path),
            conf=args.conf,
            iou=args.iou,
            classes=[0],
            tracker=tracker_cfg,
            save=False,
            stream=True,
            verbose=False,
            persist=True,
        )

        frame_data  = []
        frame_boxes = []
        for result in tqdm(results_stream, total=total_frames, desc="  Tracking"):
            track_ids, track_positions, box_data = [], {}, {}
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int).tolist()
                xyxy  = result.boxes.xyxy.cpu().numpy()
                xywh  = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for i, tid in enumerate(track_ids):
                    track_positions[tid] = (float(xywh[i][0]), float(xywh[i][1]))
                    box_data[tid] = (
                        int(xyxy[i][0]), int(xyxy[i][1]),
                        int(xyxy[i][2]), int(xyxy[i][3]),
                        float(confs[i]),
                    )
            frame_data.append({
                "frame": len(frame_data),
                "track_ids": track_ids,
                "track_positions": track_positions,
            })
            frame_boxes.append(box_data)

        print("\n[2/3] Merging fragmented tracks...")
        _, id_map = detector._merge_fragmented_tracks(frame_data)
        unique_groups   = sorted(set(id_map.values()))
        group_to_display = {g: i + 1 for i, g in enumerate(unique_groups)}

        display_frame_boxes = []
        for raw_boxes in frame_boxes:
            disp = {}
            for raw_id, bbox in raw_boxes.items():
                gid = id_map.get(raw_id, raw_id)
                did = group_to_display.get(gid, raw_id)
                disp[did] = bbox
            display_frame_boxes.append(disp)

        with open(tracking_cache_path, "w") as f:
            json.dump({
                "src_fps":      src_fps,
                "total_frames": total_frames,
                "frame_w":      frame_w,
                "frame_h":      frame_h,
                "display_frame_boxes": [
                    {str(k): list(v) for k, v in frame.items()}
                    for frame in display_frame_boxes
                ],
            }, f)
        print(f"  ✓ Tracking cache saved → {tracking_cache_path.name}")

    # ── Step 3: VLM analysis ──────────────────────────────────────────────────
    print(f"\n[3/3] Running vision model (every {args.interval}s, backend={args.backend})...")
    backend = load_vision_backend(args.backend, args.vision_model)

    interval_frames = max(1, int(args.interval * src_fps))
    sample_indices  = list(range(0, len(display_frame_boxes), interval_frames))

    # Resume from existing progress if available
    entries = []
    done_indices = set()
    prev_behaviors = {}
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    entries.append(entry)
                    done_indices.add(entry["frame_idx"])
        if entries:
            last = entries[-1]
            prev_behaviors = {
                int(bid): bear["behavior"]
                for bid, bear in last["bears"].items()
                if bear.get("behavior")
            }
            print(f"  ✓ Resuming from {len(entries)} saved entries (last: t={last['timestamp_sec']:.1f}s)")

    stage_re = re.compile(r"\[([A-Z]+)\]")

    def extract_stage(text):
        m = stage_re.search(text or "")
        return m.group(1) if m else None

    def behaviors_changed(new_behaviors, prev_behaviors, threshold):
        if set(new_behaviors.keys()) != set(prev_behaviors.keys()):
            return True
        for bid, new_text in new_behaviors.items():
            old_text = prev_behaviors.get(bid, "")
            if not old_text or not new_text:
                return True
            if extract_stage(new_text) != extract_stage(old_text):
                return True
            ratio = SequenceMatcher(None, old_text.lower(), new_text.lower()).ratio()
            if ratio < threshold:
                return True
        return False

    cap_sample   = cv2.VideoCapture(str(video_path))
    progress_f   = open(progress_path, "a")
    remaining    = [i for i in sample_indices if i not in done_indices]
    try:
        for idx in tqdm(remaining, desc="  Analyzing"):
            cap_sample.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap_sample.read()
            if not ret or frame_bgr is None:
                continue

            bear_boxes    = display_frame_boxes[idx]
            timestamp_sec = idx / src_fps

            if bear_boxes:
                annotated = annotate_frame(frame_bgr, bear_boxes)
                if args.save_frames and frames_dir:
                    cv2.imwrite(str(frames_dir / f"frame_{idx:06d}_t{timestamp_sec:.1f}s.jpg"), annotated)
                prompt       = build_prompt(bear_boxes, frame_w, frame_h)
                image_pil    = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                raw_response = backend.analyze_frame(image_pil, prompt)
                behaviors    = parse_response(raw_response, bear_boxes)
            else:
                raw_response = "(no bears detected)"
                behaviors    = {}

            changed = behaviors_changed(behaviors, prev_behaviors, args.dedupe_threshold)
            if changed:
                prev_behaviors = dict(behaviors)
                status = "NEW"
            else:
                behaviors = dict(prev_behaviors)
                status    = "same"

            entry = {
                "timestamp_sec": round(timestamp_sec, 3),
                "frame_idx":     idx,
                "changed":       changed,
                "bears": {
                    str(bid): {
                        "bbox":          list(bear_boxes[bid][:4]),
                        "conf":          round(float(bear_boxes[bid][4]), 3),
                        "position_hint": position_hint(*bear_boxes[bid][:4], frame_w, frame_h),
                        "behavior":      behaviors.get(bid, ""),
                    }
                    for bid in sorted(bear_boxes.keys())
                },
                "raw_response": raw_response,
            }
            entries.append(entry)
            progress_f.write(json.dumps(entry) + "\n")
            progress_f.flush()
            tqdm.write(f"  t={timestamp_sec:.1f}s  [{status}]  bears={list(bear_boxes.keys())}  behaviors={behaviors}")
    finally:
        cap_sample.release()
        progress_f.close()

    # Sort entries by timestamp in case of out-of-order resume
    entries.sort(key=lambda e: e["frame_idx"])

    print("\n[4/4] Generating whole-video summary...")
    torch.cuda.empty_cache()
    summary = generate_summary(backend, entries, video_path)
    print(f"\nSummary:\n{summary}\n")

    output_data = {
        "video": str(video_path),
        "fps": src_fps,
        "total_frames": total_frames,
        "frame_size": [frame_w, frame_h],
        "interval_sec": args.interval,
        "yolo_model": args.model,
        "backend": args.backend,
        "vision_model": args.vision_model or backend.name,
        "created": datetime.now().isoformat(),
        "summary": summary,
        "entries": entries,
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Done. {len(entries)} entries saved to: {json_path}")
    print(f"\nNext step:")
    print(f"  python -m src.behavior.feeding_viewer --video \"{video_path}\" --analysis \"{json_path}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute bear feeding behavior analysis with Molmo2-8B"
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: predictions/<stem>_feeding_analysis/analysis.json)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Sample one frame every N seconds (default: 0.5)")
    parser.add_argument("--model", default=str(TRAINED_MODELS_DIR / "bear_detector3" / "weights" / "best.pt"),
                        help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU threshold — lower = bboxes less likely to merge "
                             "(default: 0.7; try 0.45 for crowded scenes)")
    parser.add_argument("--backend", default="molmo2",
                        choices=["molmo2", "openai", "anthropic", "gemini"],
                        help="Vision-language backend to use. Defaults to local Molmo2-8B "
                             "(no network required). Cloud APIs need the corresponding API "
                             "key in env (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY). "
                             "See src/behavior/backends/ to add a custom backend.")
    parser.add_argument("--vision-model", default=None,
                        help="Specific model identifier within the chosen backend "
                             "(e.g. 'allenai/Molmo2-8B', 'gpt-4o', 'claude-sonnet-4-6', "
                             "'gemini-1.5-pro'). Backend's default is used if omitted.")
    parser.add_argument("--dedupe-threshold", type=float, default=0.7,
                        help="Similarity threshold (0-1). If new behavior is this similar to previous, "
                             "reuse previous text. Set to 1.0 to disable. (default: 0.7)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save sampled frames to disk for debugging")
    args = parser.parse_args()

    video_path = resolve_video(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    out_dir = PREDICTIONS_DIR / f"{video_path.stem}_feeding_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.output) if args.output else out_dir / "analysis.json"
    frames_dir = (out_dir / "sampled_frames") if args.save_frames else None
    if frames_dir:
        frames_dir.mkdir(exist_ok=True)

    _run_analysis(args, video_path, json_path, frames_dir)


if __name__ == "__main__":
    main()
