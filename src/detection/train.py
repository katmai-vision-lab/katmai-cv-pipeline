"""
src/detection/train.py
Train or fine-tune bear detection model
Usage:
    python -m src.detection.train --data data/bears/bear.yaml --epochs 50
    python -m src.detection.train --data data/bears/bear.yaml --resume
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import BearDetector
from src.config import DATA_DIR, YOLOV8N_PATH

def main():
    parser = argparse.ArgumentParser(description='Train bear detection model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default=str(YOLOV8N_PATH),
                       help='Path to base model (default: yolov8n pretrained)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--name', type=str, default='bear_detector',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')

    args = parser.parse_args()

    # Initialize detector
    detector = BearDetector(model_path=args.model)

    # Train
    results = detector.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        resume=args.resume
    )

    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Final mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Best weights: {detector.model_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
