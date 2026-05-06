"""
Video to Annotations Pipeline

End-to-end script to process videos and output labeled data.
Combines frame extraction and auto-annotation in one step.

Usage:
    # Process single video
    python -m src.preprocessing.video_to_annotations --video data/rawdata_oneDrive\ 1_2/2025-09-19\ 00-40-39_Brooks_Falls_multibear_walkabout_1.mkv

    # Process all videos in a directory
    python -m src.preprocessing.video_to_annotations --video-dir data/rawdata_oneDrive\ 1_2/

    # Custom settings (e.g., 1 frame every 10 seconds)
    python -m src.preprocessing.video_to_annotations --video-dir data/rawdata_oneDrive\ 1_2/ --fps 0.1 --conf 0.3

    # Visualize annotations
    python -m src.preprocessing.video_to_annotations --video data/video.mkv --visualize

Requirements:
    pip install ultralytics opencv-python
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.frame_extractor import extract_frames
from src.preprocessing.auto_annotator import auto_annotate
from src.preprocessing.visualize_labels import visualize_directory
from src.config import YOLOV8N_PATH, COCO_BEAR_CLASS, DATA_DIR


def process_video(
    video_path: str,
    output_base_dir: str = None,
    fps: float = 0.2,
    conf: float = 0.25,
    model_path: str = None,
    visualize: bool = False,
    start_time: float = None,
    end_time: float = None,
):
    """
    Process a single video: extract frames and auto-annotate.

    Args:
        video_path: Path to video file
        output_base_dir: Base directory for all outputs (default: data/)
        fps: Frames per second to extract
        conf: Confidence threshold for detection
        model_path: Custom YOLO model path
        visualize: Whether to create visualized annotations
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Dictionary with paths to frames, labels, and visualizations
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return None

    # Set up output directories
    if output_base_dir is None:
        output_base_dir = DATA_DIR
    else:
        output_base_dir = Path(output_base_dir)

    frames_dir = output_base_dir / "frames"
    labels_dir = output_base_dir / "auto_labels"
    viz_dir = output_base_dir / "visualized"

    print("\n" + "="*60)
    print("VIDEO TO ANNOTATIONS PIPELINE")
    print("="*60)
    print(f"Video: {video_path.name}")
    print(f"Output base: {output_base_dir}")
    print("="*60 + "\n")

    # Step 1: Extract frames
    print("[1/3] Extracting frames...")
    num_frames = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        fps=fps,
        start_time=start_time,
        end_time=end_time,
    )

    if num_frames == 0:
        print("Error: No frames extracted")
        return None

    # Step 2: Auto-annotate frames
    print("\n[2/3] Auto-annotating frames...")
    video_frames_dir = frames_dir / video_path.stem

    total_images, images_with_detections, total_detections = auto_annotate(
        input_dir=video_frames_dir,
        output_dir=labels_dir,
        model_path=model_path,
        conf=conf,
        target_class=COCO_BEAR_CLASS,
    )

    # Step 3: Visualize (optional)
    if visualize and images_with_detections > 0:
        print("\n[3/3] Visualizing annotations...")
        visualize_directory(
            images_dir=video_frames_dir,
            labels_dir=labels_dir,
            output_dir=viz_dir,
        )
        print(f"Visualizations saved to: {viz_dir}")
    else:
        print("\n[3/3] Skipping visualization")

    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Video: {video_path.name}")
    print(f"Frames extracted: {num_frames}")
    print(f"Images with detections: {images_with_detections}/{total_images}")
    print(f"Total bear detections: {total_detections}")
    print(f"\nOutputs:")
    print(f"  Frames: {video_frames_dir}")
    print(f"  Labels: {labels_dir}")
    if visualize:
        print(f"  Visualizations: {viz_dir}")
    print("="*60 + "\n")

    return {
        "video": video_path,
        "frames_dir": video_frames_dir,
        "labels_dir": labels_dir,
        "viz_dir": viz_dir if visualize else None,
        "num_frames": num_frames,
        "num_detections": total_detections,
    }


def process_directory(
    video_dir: str,
    output_base_dir: str = None,
    fps: float = 0.2,
    conf: float = 0.25,
    model_path: str = None,
    visualize: bool = False,
):
    """
    Process all videos in a directory.

    Args:
        video_dir: Directory containing video files
        output_base_dir: Base directory for all outputs
        fps: Frames per second to extract
        conf: Confidence threshold for detection
        model_path: Custom YOLO model path
        visualize: Whether to create visualized annotations

    Returns:
        List of result dictionaries
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}")
        return []

    # Find all video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = [f for f in video_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in video_extensions]

    if not video_files:
        print(f"No video files found in {video_dir}")
        return []

    print(f"\nFound {len(video_files)} video(s) to process\n")

    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}")
        print(f"{'='*60}\n")

        result = process_video(
            video_path=video_path,
            output_base_dir=output_base_dir,
            fps=fps,
            conf=conf,
            model_path=model_path,
            visualize=visualize,
        )

        if result:
            results.append(result)

    # Overall summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE!")
    print("="*60)
    print(f"Videos processed: {len(results)}/{len(video_files)}")
    print(f"Total frames: {sum(r['num_frames'] for r in results)}")
    print(f"Total detections: {sum(r['num_detections'] for r in results)}")
    print("="*60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end video to annotations pipeline"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", "-v",
        help="Path to single video file"
    )
    input_group.add_argument(
        "--video-dir", "-d",
        help="Directory containing video files"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output base directory (default: data/)"
    )

    # Processing options
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=0.2,
        help="Frames per second to extract (default: 0.2, i.e., 1 frame every 5 seconds)"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Custom YOLO model path (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of annotations"
    )

    # Video clip options
    parser.add_argument(
        "--start", "-s",
        type=float,
        default=None,
        help="Start time in seconds (only for single video)"
    )
    parser.add_argument(
        "--end", "-e",
        type=float,
        default=None,
        help="End time in seconds (only for single video)"
    )

    args = parser.parse_args()

    if args.video:
        # Process single video
        process_video(
            video_path=args.video,
            output_base_dir=args.output,
            fps=args.fps,
            conf=args.conf,
            model_path=args.model,
            visualize=args.visualize,
            start_time=args.start,
            end_time=args.end,
        )
    else:
        # Process directory
        if args.start or args.end:
            print("Warning: --start and --end are ignored for directory processing")

        process_directory(
            video_dir=args.video_dir,
            output_base_dir=args.output,
            fps=args.fps,
            conf=args.conf,
            model_path=args.model,
            visualize=args.visualize,
        )


if __name__ == "__main__":
    main()
