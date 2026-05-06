"""
Multi-class Auto Annotation using Grounding DINO

Detect multiple object classes (e.g., bear and salmon) in the same image.

Usage:
    python -m src.preprocessing.auto_annotator_gdino_multiclass \
        --input data/frames/video/ \
        --output data/labels/ \
        --classes "bear:0" "salmon:1" "jumping salmon:1" \
        --box-threshold 0.35 \
        --text-threshold 0.30
"""

import argparse
from pathlib import Path
import sys
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.auto_annotator_gdino import GroundingDINOAnnotator


def auto_annotate_multiclass(
    input_dir: str,
    output_dir: str,
    class_prompts: dict,  # {class_id: [prompt1, prompt2, ...]}
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    limit: int = None,
    model_size: str = "base",
):
    """
    Auto-annotate images with multiple classes using Grounding DINO.

    Args:
        input_dir: Directory containing images to annotate
        output_dir: Directory to save YOLO format labels
        class_prompts: Dict mapping class_id to list of text prompts
        box_threshold: Confidence threshold for boxes
        text_threshold: Confidence threshold for text matching
        limit: Max number of images to process
        model_size: "tiny", "base", or "large"

    Returns:
        Tuple of (total_images, images_with_detections, detections_by_class)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 0, 0, {}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model_id = f"IDEA-Research/grounding-dino-{model_size}"
    annotator = GroundingDINOAnnotator(model_id=model_id)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []
    
    for f in input_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in image_extensions:
            image_files.append(f)

    if not image_files:
        print(f"No images found in {input_dir}")
        return 0, 0, {}

    image_files = sorted(image_files, key=lambda x: x.name)

    if limit:
        image_files = image_files[:limit]

    print(f"\nGrounding DINO Multi-Class Auto Annotation")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: grounding-dino-{model_size}")
    print(f"Classes:")
    for class_id, prompts in class_prompts.items():
        print(f"  Class {class_id}: {', '.join(prompts)}")
    print(f"Box threshold: {box_threshold}")
    print(f"Text threshold: {text_threshold}")
    print(f"Images found: {len(image_files)}")
    print(f"{'='*60}\n")

    total_images = len(image_files)
    images_with_detections = 0
    detections_by_class = {class_id: 0 for class_id in class_prompts.keys()}

    for i, image_path in enumerate(image_files):
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        all_labels = []

        # Detect each class
        for class_id, prompts in class_prompts.items():
            # Combine prompts for this class
            combined_prompt = ". ".join(prompts) + "."
            
            # Get detections
            detections = annotator.detect(
                image=image,
                text_prompt=combined_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # Convert to YOLO format
            for det in detections:
                box = det["box"]
                x_min, y_min, x_max, y_max = box

                # Convert to YOLO format (normalized center + width/height)
                center_x = ((x_min + x_max) / 2) / width
                center_y = ((y_min + y_max) / 2) / height
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height

                # Clamp values to [0, 1]
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                box_width = max(0, min(1, box_width))
                box_height = max(0, min(1, box_height))

                label_line = f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
                all_labels.append(label_line)
                detections_by_class[class_id] += 1

        # Save label file
        label_filename = image_path.stem + ".txt"
        label_path = output_dir / label_filename

        with open(label_path, "w") as f:
            f.write("\n".join(all_labels))

        num_detections = len(all_labels)
        if num_detections > 0:
            images_with_detections += 1

        # Progress output with class breakdown
        if num_detections > 0:
            class_counts = {}
            for label in all_labels:
                cls = int(label.split()[0])
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            status_parts = []
            for cls, count in sorted(class_counts.items()):
                class_name = list(class_prompts[cls])[0]
                status_parts.append(f"{count} {class_name}")
            status = ", ".join(status_parts)
        else:
            status = "no detection"
        
        print(f"[{i+1}/{total_images}] {image_path.name}: {status}")

    print(f"\n{'='*60}")
    print(f"Annotation Complete!")
    print(f"{'='*60}")
    print(f"Total images: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Detections by class:")
    for class_id, count in detections_by_class.items():
        class_name = list(class_prompts[class_id])[0]
        print(f"  Class {class_id} ({class_name}): {count}")
    print(f"Detection rate: {images_with_detections/total_images*100:.1f}%")
    print(f"Labels saved to: {output_dir}")

    return total_images, images_with_detections, detections_by_class


def main():
    parser = argparse.ArgumentParser(
        description="Multi-class auto-annotation using Grounding DINO"
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
        "--classes", "-c",
        nargs="+",
        required=True,
        help='Class definitions in format "prompt:class_id". Example: "bear:0" "salmon:1" "jumping salmon:1"'
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Box confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Text matching threshold (default: 0.25)"
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "base", "large"],
        default="base",
        help="Model size (default: base)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to process"
    )

    args = parser.parse_args()

    # Parse class definitions
    class_prompts = {}
    for class_def in args.classes:
        if ":" not in class_def:
            print(f"Error: Invalid class definition '{class_def}'. Format: 'prompt:class_id'")
            return
        
        prompt, class_id_str = class_def.rsplit(":", 1)
        try:
            class_id = int(class_id_str)
        except ValueError:
            print(f"Error: Invalid class ID in '{class_def}'. Must be an integer.")
            return
        
        if class_id not in class_prompts:
            class_prompts[class_id] = []
        class_prompts[class_id].append(prompt)

    auto_annotate_multiclass(
        input_dir=args.input,
        output_dir=args.output,
        class_prompts=class_prompts,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        limit=args.limit,
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
