import argparse
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import BearDetector
from src.config import YOLOV8N_PATH

def main():
    parser = argparse.ArgumentParser(description='Batch Bear Counter')
    
    # Input options
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument('--videos', nargs='+', 
                            help='List of video files')
    video_group.add_argument('--video-dir', type=str, 
                            help='Directory containing videos')
    
    parser.add_argument('--pattern', type=str, default='*.mkv',
                       help='File pattern for video-dir mode (default: *.mkv)')
    
    # Model options
    parser.add_argument('--model', type=str, default=str(YOLOV8N_PATH),
                       help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--frame-skip', type=int, default=30,
                       help='Process every Nth frame (default: 30, ~1fps)')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Class ID(s) to detect (e.g., 0 for custom bear, 21 for COCO bear)')
    
    # Tracking options
    parser.add_argument('--tracking', action='store_true',
                       help='Enable ByteTrack tracking for accurate bear counting')
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       help='Tracker name: bytetrack, botsort (default: bytetrack)')
    
    # Ground truth
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='JSON file: {"video1.mkv": 5, "video2.mkv": 3}')
    
    # Output
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save results to files')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed processing information')
    
    args = parser.parse_args()
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
        print(f"✓ Loaded ground truth for {len(ground_truth)} videos")
    
    # Initialize detector
    print(f"Initializing detector...")
    detector = BearDetector(model_path=args.model)
    
    # Check if tracking is requested
    if args.tracking:
        results = detector.batch_track_bears(
            video_paths=args.videos,
            video_dir=args.video_dir,
            pattern=args.pattern,
            conf=args.conf,
            frame_skip=args.frame_skip,
            classes=args.classes,
            tracker=args.tracker,
            verbose=args.verbose
        )
        return 0 if results['failed'] == 0 else 1
    
    # Standard batch counting (no tracking)
    results = detector.batch_count_bears(
        video_paths=args.videos,
        video_dir=args.video_dir,
        pattern=args.pattern,
        conf=args.conf,
        frame_skip=args.frame_skip,
        classes=args.classes,
        ground_truth=ground_truth,
        save_results=not args.no_save
    )
    
    # Return success
    return 0 if results['aggregate']['failed_videos'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())