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
    )

    video_files = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.avi"))
    if video_files:
        print(f"\n▶ Play output: {video_files[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
