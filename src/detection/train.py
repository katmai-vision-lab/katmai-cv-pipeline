"""
src/detection/train.py
Train or fine-tune bear detection model
Usage:
    python -m src.detection.train --data data/bears/bear.yaml --epochs 50
    python -m src.detection.train --data data/bears/bear.yaml --config configs/train_config.yaml
    python -m src.detection.train --data data/bears/bear.yaml --resume
"""

import argparse
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import BearDetector
from src.config import DATA_DIR, YOLOV8N_PATH

def main():
    parser = argparse.ArgumentParser(description='Train bear detection model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training config YAML file')
    parser.add_argument('--model', type=str, default=str(YOLOV8N_PATH),
                       help='Path to base model (default: yolov8n pretrained)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Image size (overrides config)')
    parser.add_argument('--name', type=str, default='bear_detector',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')

    args = parser.parse_args()

    # Load config file if provided
    train_params = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                train_params = yaml.safe_load(f)
            print(f"✓ Loaded training config from: {config_path}")
        else:
            print(f"⚠️  Config file not found: {config_path}")
    
    # Command line arguments override config file
    if args.epochs is not None:
        train_params['epochs'] = args.epochs
    if args.batch is not None:
        train_params['batch'] = args.batch
    if args.imgsz is not None:
        train_params['imgsz'] = args.imgsz
    
    # Set defaults if not in config
    train_params.setdefault('epochs', 50)
    train_params.setdefault('batch', 8)
    train_params.setdefault('imgsz', 640)

    # Initialize detector
    detector = BearDetector(model_path=args.model)

    # Train
    results = detector.train(
        data_yaml=args.data,
        name=args.name,
        resume=args.resume,
        **train_params  # Pass all config parameters
    )

    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Final mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Best weights: {detector.model_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
