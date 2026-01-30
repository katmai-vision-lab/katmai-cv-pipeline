"""
src/detection/evaluate.py
Evaluate bear detection model on videos

Usage:
    python -m src.detection.evaluate --video bears.mkv --model model.pt
    python -m src.detection.evaluate --video bears.mkv --ground-truth 5
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import BearDetector
from src.config import YOLOV8N_PATH, PREDICTIONS_DIR, RAW_DATA_DIR

def evaluate_video(detector, video_path, ground_truth_count=None, conf=0.25, classes=None):
    """
    Evaluate model on video and analyze performance
    
    Args:
        detector: BearDetector instance
        video_path: Path to video
        ground_truth_count: Optional ground truth bear count
        conf: Confidence threshold
        classes: Classes to detect (None = all classes)
    
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
        classes=classes,
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
    print(f"\n✓ Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate bear detection model')
    parser.add_argument('--video', type=str, required=True,
                       help='Video filename or path')
    parser.add_argument('--model', type=str, default=str(YOLOV8N_PATH),
                       help='Path to model')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--ground-truth', type=int, default=None,
                       help='Ground truth bear count')
    parser.add_argument('--plot', action='store_true',
                       help='Generate evaluation plots')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Classes to detect (default: None for all classes)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BearDetector(model_path=args.model)
    
    # Evaluate
    df = evaluate_video(
        detector=detector,
        video_path=args.video,
        ground_truth_count=args.ground_truth,
        conf=args.conf,
        classes=args.classes
    )
    
    # Save results
    output_dir = PREDICTIONS_DIR / 'evaluations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"eval_{Path(args.video).stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")
    
    # Generate plots if requested
    if args.plot:
        plot_path = output_dir / f"eval_{Path(args.video).stem}.png"
        plot_evaluation(df, plot_path)


if __name__ == '__main__':
    main()
