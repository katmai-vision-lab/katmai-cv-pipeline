"""
Track bears in a video and save an output video with bounding boxes and track IDs
(dynamic boxes following each bear). Uses ByteTrack by default.

Usage:
    python -m src.detection.track_video --video "2025-09-19 23-30-11_Brooks_Falls_Low_5_bears.mp4"
    python -m src.detection.track_video --video path/to/video.mp4 --frame-skip 5
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import BearDetector
from src.detection.trajectory_video import render as render_trajectory_overlay
from src.config import RAW_DATA_DIR, TRAINED_MODELS_DIR

DEFAULT_MODEL = str(TRAINED_MODELS_DIR / "bear_detector2" / "weights" / "best.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Track bears in video and save output video with boxes + track IDs"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Video path: absolute, or relative to project root / data/raw/ (e.g. bears/xxx.mp4)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to model weights (default: trained bear_detector2)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--classes", type=int, nargs="+", default=[0],
                        help="Class IDs (default: 0 for trained bear)")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth frame (1=every frame). Default 1 for smooth video.")
    parser.add_argument("--tracker", type=str, default="bytetrack",
                        help="Tracker: bytetrack, botsort")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output folder name under predictions/")
    parser.add_argument("--max-gap-frames", type=int, default=3600,
                        help="Merge: max frame gap to still connect same bear (default 3600 = 2min @ 30fps)")
    parser.add_argument("--max-dist-px", type=int, default=150,
                        help="Merge: max pixel distance at transition (default 150)")
    parser.add_argument("--cooccur-tol-frames", type=int, default=60,
                        help="Merge: tolerate up to N co-occurrence frames as detection "
                             "artifact if mean dist < max_dist_px (default 60)")
    parser.add_argument("--cooccur-artifact-iou", type=float, default=0.3,
                        help="Merge: if two co-occurring bboxes overlap with mean IoU >= "
                             "this, treat as same animal regardless of duration (default 0.3)")
    parser.add_argument("--min-duration", type=int, default=150,
                        help="Post-merge filter: drop groups shorter than N frames (default 150 = 5s @ 30fps)")
    parser.add_argument("--min-raw-duration", type=int, default=None,
                        help="Pre-merge filter: drop raw tracks shorter than N frames "
                             "BEFORE merge step. Prevents brief detections from being used "
                             "as merge bridges. Default: same as --min-duration")
    parser.add_argument("--min-mean-conf", type=float, default=0.80,
                        help="Filter: drop groups with mean conf < this (default 0.80)")
    parser.add_argument("--no-trails", action="store_true",
                        help="Skip the trajectory-overlay video (default: also produce one)")
    parser.add_argument("--trail-frames", type=int, default=0,
                        help="Trail length in frames (0 = full history, default 0)")
    parser.add_argument("--trail-thickness", type=int, default=3,
                        help="Trail line thickness in px (default 3)")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        # Try: project root, then data/raw/
        candidates = [
            PROJECT_ROOT / video_path,
            RAW_DATA_DIR / video_path,
            PROJECT_ROOT / "data" / "raw" / video_path,
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        video_path = found if found is not None else (RAW_DATA_DIR / video_path)
    if not video_path.exists():
        print(f"Error: video not found: {args.video}")
        print(f"  Tried: {video_path}")
        if not Path(args.video).is_absolute():
            print(f"  Also try: absolute path, or path relative to project root (e.g. bears/xxx.mp4)")
        sys.exit(1)

    detector = BearDetector(model_path=args.model)
    results, output_dir = detector.track_and_save_video(
        video_path=video_path,
        output_name=args.output_name,
        conf=args.conf,
        frame_skip=args.frame_skip,
        classes=args.classes,
        tracker=args.tracker,
        max_gap_frames=args.max_gap_frames,
        max_dist_px=args.max_dist_px,
        cooccurrence_tolerance_frames=args.cooccur_tol_frames,
        cooccurrence_artifact_iou=args.cooccur_artifact_iou,
        min_duration=args.min_duration,
        min_raw_duration=args.min_raw_duration,
        min_mean_conf=args.min_mean_conf,
    )

    video_files = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.avi"))
    if video_files:
        print(f"\n▶ Play output: {video_files[0]}")

    if not args.no_trails:
        traj_json = output_dir / "trajectories.json"
        if traj_json.exists():
            import json as _json
            with open(traj_json) as _f:
                _meta = _json.load(_f)
            total = max(int(_meta.get("total_frames") or 1), 1)
            cap = args.trail_frames if args.trail_frames > 0 else total
            print("\n→ Rendering trajectory overlay on raw video...")
            render_trajectory_overlay(
                traj_path=traj_json,
                video_path=video_path,
                output_path=str(output_dir / "trajectories_overlay.mp4"),
                trail_frames=cap,
                thickness=args.trail_thickness,
                draw_box=True,
                draw_id=True,
            )
        else:
            print(f"\n(skipped trail overlay — no {traj_json.name})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
