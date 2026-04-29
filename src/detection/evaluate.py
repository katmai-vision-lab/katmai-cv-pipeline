"""
src/detection/evaluate.py
Evaluate bear detection model on videos

Usage:
    # Dataset evaluation (uses YOLO's built-in validation)
    python -m src.detection.evaluate --mode dataset --data data/annotation/bears/bear.yaml
    
    # Counting evaluation (simple bear counting)
    python -m src.detection.evaluate --mode counting --video bears.mkv --ground-truth 5
    
    # Legacy simple evaluation
    python -m src.detection.evaluate --mode simple --video bears.mkv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import BearDetector
from src.detection.metrics import VideoEvaluator
from src.config import TRAINED_BEAR_DETECTOR_PATH, PREDICTIONS_DIR, RAW_DATA_DIR, DATA_DIR

def evaluate_video(detector, video_path, ground_truth_count=None, conf=0.25):
    """
    Evaluate model on video and analyze performance
    
    Args:
        detector: BearDetector instance
        video_path: Path to video
        ground_truth_count: Optional ground truth bear count
        conf: Confidence threshold
    
    Returns:
        DataFrame with per-frame statistics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on: {Path(video_path).name}")
    print(f"{'='*60}\n")

    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = RAW_DATA_DIR / video_path
        
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    results = detector.model.predict(
        source=str(video_path),
        conf=conf,
        stream=True,
        verbose=False
    )

    detections_per_frame = []
    
    for frame_id, result in enumerate(results):
        boxes = result.boxes
        num_bears = len(boxes)
        confidences = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
        
        detections_per_frame.append({
            'frame': frame_id,
            'num_bears': num_bears,
            'avg_confidence': confidences.mean() if len(confidences) > 0 else 0,
            'max_confidence': confidences.max() if len(confidences) > 0 else 0,
            'min_confidence': confidences.min() if len(confidences) > 0 else 0
        })
    
    df = pd.DataFrame(detections_per_frame)
    
    # Print statistics
    print(f"Statistics:")
    print(f"  Total frames: {len(df)}")
    print(f"  Frames with bears: {(df['num_bears'] > 0).sum()}")
    print(f"  Avg bears/frame: {df['num_bears'].mean():.2f}")
    print(f"  Max bears in frame: {df['num_bears'].max()}")
    print(f"  Avg confidence: {df['avg_confidence'].mean():.2f}")
    
    if ground_truth_count:
        print(f"\nGround Truth Comparison:")
        print(f"  Ground truth: {ground_truth_count} bears")
        print(f"  Max detected: {df['num_bears'].max()} bears")
        accuracy = "✓ Correct" if df['num_bears'].max() == ground_truth_count else "✗ Incorrect"
        print(f"  Status: {accuracy}")
    
    return df


def plot_evaluation(df, output_path):
    """Plot evaluation metrics"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Bears detected per frame
    axes[0].plot(df['frame'], df['num_bears'], linewidth=0.8)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Number of Bears')
    axes[0].set_title('Bears Detected Per Frame')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence over time
    axes[1].plot(df['frame'], df['avg_confidence'], label='Avg Confidence', linewidth=0.8)
    axes[1].plot(df['frame'], df['max_confidence'], label='Max Confidence', linewidth=0.8, alpha=0.7)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Detection Confidence Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n✓ Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive bear detection evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on validation dataset (recommended - uses YOLO's mAP/precision/recall)
  python -m src.detection.evaluate --mode dataset --data data/annotation/bears/bear.yaml
  
  # Evaluate counting accuracy on video
  python -m src.detection.evaluate --mode counting --video bears.mkv --ground-truth 5
  
  # Simple frame-by-frame analysis
  python -m src.detection.evaluate --mode simple --video bears.mkv
        """
    )
    
    parser.add_argument('--mode', type=str, 
                       choices=['dataset', 'counting', 'simple'],
                       default='simple',
                       help='Evaluation mode: dataset (YOLO val), counting, or simple')
    parser.add_argument('--video', type=str, default=None,
                       help='Video filename or path (required for counting/simple modes)')
    parser.add_argument('--data', type=str, default=None,
                       help='Dataset YAML path (required for dataset mode)')
    parser.add_argument('--model', type=str, default=str(TRAINED_BEAR_DETECTOR_PATH),
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--ground-truth', type=int, default=None,
                       help='Ground truth bear count (for counting mode)')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every Nth frame (for counting mode)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    # Validation
    if args.mode == 'dataset' and not args.data:
        parser.error("--data is required for dataset mode")
    if args.mode in ['counting', 'simple'] and not args.video:
        parser.error("--video is required for counting/simple modes")
    if args.mode == 'counting' and args.ground_truth is None:
        parser.error("--ground-truth is required for counting mode")
    
    # Initialize detector and evaluator
    detector = BearDetector(model_path=args.model)
    evaluator = VideoEvaluator(detector, conf_threshold=args.conf)
    
    # Create output directory
    output_dir = PREDICTIONS_DIR / 'evaluations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute based on mode
    if args.mode == 'dataset':
        # YOLO native validation - comprehensive metrics
        data_yaml = args.data
        if not Path(data_yaml).is_absolute():
            data_yaml = DATA_DIR / 'annotation' / 'bears' / data_yaml
        
        metrics = evaluator.evaluate_dataset_with_yolo(
            data_yaml=str(data_yaml),
            save_dir=output_dir
        )
        
    elif args.mode == 'counting':
        # Counting accuracy evaluation
        df = evaluator.evaluate_counting_accuracy(
            video_path=args.video,
            ground_truth_counts=args.ground_truth,
            frame_skip=args.frame_skip,
            save_dir=output_dir
        )
        
    elif args.mode == 'simple':
        # Simple frame-by-frame analysis (legacy)
        df = evaluate_video(
            detector=detector,
            video_path=args.video,
            ground_truth_count=args.ground_truth,
            conf=args.conf
        )
        
        csv_path = output_dir / f"simple_eval_{Path(args.video).stem}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved: {csv_path}")
        
        if args.plot:
            plot_path = output_dir / f"simple_eval_{Path(args.video).stem}.png"
            plot_evaluation(df, plot_path)


if __name__ == '__main__':
    main()