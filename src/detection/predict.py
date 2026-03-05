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
from src.config import TRAINED_BEAR_DETECTOR_PATH


def main():
    parser = argparse.ArgumentParser(description='Run bear detection on video')
    parser.add_argument('--video', type=str, required=True,
                       help='Video filename (in data/raw/) or full path')
    parser.add_argument('--model', type=str, default=str(TRAINED_BEAR_DETECTOR_PATH),
                       help='Path to model (default: fine-tuned bear_detector3)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Classes to detect (default: None = all classes; use 21 for COCO pretrained, omit for fine-tuned model)')
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

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()