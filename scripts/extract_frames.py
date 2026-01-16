"""
Frame Extraction Script

Extract frames from video files for annotation and model training.

Usage:
    python scripts/extract_frames.py --input data/raw_videos/video.mp4 --output data/frames/
    python scripts/extract_frames.py --input data/raw_videos/ --output data/frames/ --fps 2

Requirements:
    pip install opencv-python
"""

import argparse
import cv2
import os
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    format: str = "jpg",
    quality: int = 95,
    start_time: float = None,
    end_time: float = None,
):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1.0)
        format: Output image format, 'jpg' or 'png' (default: 'jpg')
        quality: JPEG quality 1-100 (default: 95)
        start_time: Start time in seconds (default: None, from beginning)
        end_time: End time in seconds (default: None, to end)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Create output directory
    video_name = video_path.stem
    frame_output_dir = output_dir / video_name
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Total frames: {total_frames}")

    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1

    # Calculate start and end frames
    start_frame = 0
    end_frame = total_frames
    if start_time is not None:
        start_frame = int(start_time * video_fps)
    if end_time is not None:
        end_frame = int(end_time * video_fps)

    print(f"  Extracting: {fps} fps (every {frame_interval} frames)")
    print(f"  Time range: {start_frame/video_fps:.2f}s - {end_frame/video_fps:.2f}s")
    print(f"  Output: {frame_output_dir}")

    # Set start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Extract frames
    frame_count = start_frame
    saved_count = 0

    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_count - start_frame) % frame_interval == 0:
            # Generate filename with timestamp
            timestamp = frame_count / video_fps
            filename = f"{video_name}_frame{saved_count:05d}_t{timestamp:.2f}s.{format}"
            filepath = frame_output_dir / filename

            # Save frame
            if format == "jpg":
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(filepath), frame)

            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"  Saved: {saved_count} frames\n")
    return saved_count


def process_directory(input_dir: str, output_dir: str, **kwargs):
    """Process all video files in a directory."""
    input_dir = Path(input_dir)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in video_extensions]

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} video(s)\n")

    total_frames = 0
    for video_path in video_files:
        total_frames += extract_frames(video_path, output_dir, **kwargs)

    print(f"Total: {total_frames} frames extracted")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video files")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for frames"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format (default: jpg)"
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)"
    )
    parser.add_argument(
        "--start", "-s",
        type=float,
        default=None,
        help="Start time in seconds"
    )
    parser.add_argument(
        "--end", "-e",
        type=float,
        default=None,
        help="End time in seconds"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        extract_frames(
            args.input,
            args.output,
            fps=args.fps,
            format=args.format,
            quality=args.quality,
            start_time=args.start,
            end_time=args.end,
        )
    elif input_path.is_dir():
        process_directory(
            args.input,
            args.output,
            fps=args.fps,
            format=args.format,
            quality=args.quality,
            start_time=args.start,
            end_time=args.end,
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
