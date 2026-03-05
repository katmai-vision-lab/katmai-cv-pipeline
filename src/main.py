# Full pipeline orchestrator - train, predict, evaluate

"""
src/main.py
Complete bear detection pipeline
Usage:
    # Full pipeline
    python -m src.main --mode full --video bears.mkv --data bears.yaml
    
    # Just prediction
    python -m src.main --mode predict --video bears.mkv
    
    # Just evaluation
    python -m src.main --mode evaluate --video bears.mkv --ground-truth 5
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import BearDetector
from src.detection.evaluate import evaluate_video, plot_evaluation
from src.config import TRAINED_BEAR_DETECTOR_PATH, DATA_DIR, PREDICTIONS_DIR


def run_full_pipeline(video_path, data_yaml=None, model_path=None, epochs=3, 
                     conf=0.25, ground_truth=None, skip_train=False):
    """Run complete pipeline: train -> predict -> evaluate"""

    print("\n" + "="*60)
    print("BEAR DETECTION PIPELINE - FULL MODE")
    print("="*60)

    # Step 1: Train
    # print("\n[1/3] Prepare model...")
    # detector = BearDetector(model_path=TRAINED_BEAR_DETECTOR_PATH)
    # if not skip_train:
    #     print("\nTrain model...")
    #     detector.train(
    #         data_yaml=data_yaml,
    #         epochs=epochs,
    #         name='pipeline_trained_model'
    #     )
    if skip_train:
        # Use existing model (pretrained or custom)
        if model_path is None:
            model_path = TRAINED_BEAR_DETECTOR_PATH
            print("\n⚠️  Using fine-tuned bear detector (bear_detector3)")
        else:
            print(f"\n⚠️  Using existing model: {model_path}")

        detector = BearDetector(model_path=model_path)
        print("Skipping training step...")
    else:
        # Train new model
        if data_yaml is None:
            raise ValueError("--data is required when training")

        print("\n[1/3] Training model...")
        detector = BearDetector(model_path=model_path or TRAINED_BEAR_DETECTOR_PATH)
        detector.train(
            data_yaml=data_yaml,
            epochs=epochs,
            name='pipeline_trained_model'
        )


    # Step 2: Predict
    print("\n[2/3] Running detection...")
    results, output_dir = detector.predict_video(
        video_path=video_path,
        conf=conf,
        output_name='pipeline_prediction'
    )

    # Step 3: Evaluate
    print("\n[3/3] Evaluating results...")
    df = evaluate_video(detector, video_path, ground_truth, conf)

    # Save evaluation
    eval_dir = PREDICTIONS_DIR / 'evaluations'
    eval_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = eval_dir / f"pipeline_evaluation_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    plot_path = eval_dir / f"pipeline_evaluation_{timestamp}.png"
    plot_evaluation(df, plot_path)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Model: {detector.model_path}")
    print(f"Predictions: {output_dir}")
    print(f"Evaluation: {csv_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Bear Detection Pipeline')
    parser.add_argument('--mode', type=str, choices=['full', 'train', 'predict', 'evaluate'],
                       default='predict', help='Pipeline mode')
    parser.add_argument('--video', type=str, required=False,
                       help='Video file (required for predict/evaluate/full modes)')
    parser.add_argument('--data', type=str, default=None,
                       help='Dataset YAML (required for training)')
    parser.add_argument('--model', type=str, default=str(TRAINED_BEAR_DETECTOR_PATH),
                       help='Model path')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Training epochs')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--ground-truth', type=int, default=None,
                       help='Ground truth bear count')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip training in full mode (use existing model)')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Classes to detect (e.g., 21 for COCO bear, 0 for custom bear)')

    args = parser.parse_args()

    # Validation
    if args.mode == 'train' and not args.data:
        parser.error("--data is required for train mode")

    if args.mode == 'full' and not args.skip_train and not args.data:
        parser.error("--data is required for full mode (or use --skip-train)")

    # Execute based on mode
    if args.mode == 'full':
        run_full_pipeline(
            video_path=args.video,
            data_yaml=args.data,
            model_path=args.model if args.model != str(TRAINED_BEAR_DETECTOR_PATH) else None,
            epochs=args.epochs,
            conf=args.conf,
            ground_truth=args.ground_truth,
            skip_train=args.skip_train
        )

    elif args.mode == 'train':
        detector = BearDetector(model_path=args.model)
        detector.train(data_yaml=args.data, epochs=args.epochs)

    elif args.mode == 'predict':
        detector = BearDetector(model_path=args.model)
        results, output_dir = detector.predict_video(args.video, conf=args.conf)

        # Print summary
        total_detections = sum(len(r.boxes) for r in results)
        print(f"\nSummary:")
        print(f"  Frames: {len(results)}")
        print(f"  Total detections: {total_detections}")
        print(f"  Avg detections/frame: {total_detections/len(results):.2f}")

    elif args.mode == 'evaluate':
        detector = BearDetector(model_path=args.model)
        df = evaluate_video(detector, args.video, args.ground_truth, args.conf, 
                          frame_skip=30)

        # Save results
        eval_dir = PREDICTIONS_DIR / 'evaluations'
        eval_dir.mkdir(parents=True, exist_ok=True)

        csv_path = eval_dir / f"eval_{Path(args.video).stem}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved: {csv_path}")


if __name__ == '__main__':
    main()