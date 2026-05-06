"""
Auto Annotation Script

Use pretrained YOLO model to automatically generate annotations for extracted frames.
Outputs YOLO format label files (.txt) for each image.

Usage:
    # Annotate frames in a directory
    python -m src.preprocessing.auto_annotator --input data/frames/video_name/ --output data/auto_labels/

    # With custom confidence threshold
    python -m src.preprocessing.auto_annotator --input data/frames/ --output data/auto_labels/ --conf 0.3

    # Use custom model
    python -m src.preprocessing.auto_annotator --input data/frames/ --output data/auto_labels/ --model models/trained/best.pt

Requirements:
    pip install ultralytics opencv-python
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import YOLOV8N_PATH, COCO_BEAR_CLASS


def auto_annotate(
    input_dir: str,
    output_dir: str,
    model_path: str = None,
    conf: float = 0.25,
    target_class: int = COCO_BEAR_CLASS,
    output_class: int = 0,
    limit: int = None,
):
    """
    Auto-annotate images using YOLO model.

    Args:
        input_dir: Directory containing images to annotate
        output_dir: Directory to save YOLO format labels
        model_path: Path to YOLO model (default: yolov8n.pt)
        conf: Confidence threshold (default: 0.25)
        target_class: COCO class ID to detect (default: 21 for bear)
        output_class: Class ID to use in output labels (default: 0)
        limit: Max number of images to process (default: None, process all)

    Returns:
        Tuple of (total_images, images_with_detections, total_detections)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 0, 0, 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if model_path is None:
        model_path = YOLOV8N_PATH
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 0, 0, 0

    print(f"Loading model: {model_path.name}")
    model = YOLO(str(model_path))

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    # Also search subdirectories
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            image_files.extend([
                f for f in subdir.iterdir()
                if f.suffix.lower() in image_extensions
            ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return 0, 0, 0

    # Apply limit if specified
    if limit:
        image_files = image_files[:limit]

    print(f"\nAuto Annotation")
    print(f"{'='*50}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_path.name}")
    print(f"Confidence: {conf}")
    print(f"Target class: {target_class} (COCO bear)")
    print(f"Images found: {len(image_files)}")
    print(f"{'='*50}\n")

    total_images = len(image_files)
    images_with_detections = 0
    total_detections = 0

    for i, image_path in enumerate(image_files):
        # Run inference
        results = model.predict(
            source=str(image_path),
            conf=conf,
            classes=[target_class],
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        # Filter detections and convert to YOLO format
        labels = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == target_class:
                # Get normalized coordinates (YOLO format)
                # boxes.xywhn gives [center_x, center_y, width, height] normalized
                xywhn = box.xywhn[0].tolist()
                center_x, center_y, width, height = xywhn

                # Use output_class (0) instead of COCO class (21)
                label_line = f"{output_class} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                labels.append(label_line)

        # Save label file
        label_filename = image_path.stem + ".txt"
        label_path = output_dir / label_filename

        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        num_detections = len(labels)
        total_detections += num_detections

        if num_detections > 0:
            images_with_detections += 1

        # Progress output
        status = f"{num_detections} bear(s)" if num_detections > 0 else "no detection"
        print(f"[{i+1}/{total_images}] {image_path.name}: {status}")

    print(f"\n{'='*50}")
    print(f"Annotation Complete!")
    print(f"{'='*50}")
    print(f"Total images: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Labels saved to: {output_dir}")

    return total_images, images_with_detections, total_detections


def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate images using YOLO model"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for label files"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to YOLO model (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=COCO_BEAR_CLASS,
        help=f"COCO class ID to detect (default: {COCO_BEAR_CLASS} for bear)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max number of images to process (default: all)"
    )

    args = parser.parse_args()

    auto_annotate(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        conf=args.conf,
        target_class=args.target_class,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
