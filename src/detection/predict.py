"""
src/detection/predict.py
Run bear detection on videos
Usage:
    python -m src.detection.predict --video bears.mkv --conf 0.25
    python -m src.detection.predict --video bears.mkv --model trained_model.pt
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import BearDetector
from src.config import YOLOV8N_PATH, TRAINED_MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description='Run bear detection on video')
    parser.add_argument('--video', type=str, required=True,
                       help='Video filename (in data/raw/) or full path')
    parser.add_argument('--model', type=str, default=str(YOLOV8N_PATH),
                       help='Path to model (default: yolov8n pretrained)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--classes', type=int, nargs='+', default=[21],
                       help='Classes to detect (default: 21 for bear in COCO)')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Custom output directory name')

    args = parser.parse_args()

    # Initialize detector
    detector = BearDetector(model_path=args.model)

    # Run prediction
    results, output_dir = detector.predict_video(
        video_path=args.video,
        output_name=args.output_name,
        conf=args.conf,
        classes=args.classes
    )

    # Summary
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\nSummary:")
    print(f"  Frames: {len(results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections/frame: {total_detections/len(results):.2f}")


if __name__ == '__main__':
    main()