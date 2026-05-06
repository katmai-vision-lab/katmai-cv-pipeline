"""
Auto Annotation Script using Florence-2

Microsoft's Florence-2 is a multi-task vision model that can perform:
- Object detection
- Dense region captioning
- Grounding (text-to-object detection)
- Image captioning and more

This annotator uses Florence-2's open-vocabulary grounding capability for high-accuracy detection.

Usage:
    # Basic detection
    python -m src.preprocessing.auto_annotator_florence2 --input data/frames/ --output data/auto_labels/

    # Custom text prompt
    python -m src.preprocessing.auto_annotator_florence2 --input data/frames/ --output data/auto_labels/ --prompt "brown bear"

    # Use larger model
    python -m src.preprocessing.auto_annotator_florence2 --input data/frames/ --output data/auto_labels/ --model-size large

Requirements:
    pip install transformers torch torchvision pillow
"""

import argparse
from pathlib import Path
import sys
import torch
from PIL import Image
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Florence2Annotator:
    """Florence-2 based auto-annotator for open-vocabulary detection."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = None,
    ):
        """
        Initialize Florence-2 model.

        Args:
            model_size: Model size - "base", "base-ft", "large", or "large-ft"
                - base: Faster, smaller
                - large: More accurate
                - ft: Fine-tuned versions (generally better)
            device: Device to run on (auto-detect if None)
        """
        from transformers import AutoProcessor, AutoModelForCausalLM

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model_id = f"microsoft/Florence-2-{model_size}"
        print(f"Loading Florence-2 model: {model_id}")
        print(f"Device: {device}")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        print("Model loaded successfully!")

    def detect(
        self,
        image: Image.Image,
        text_prompt: str = None,
        use_grounding: bool = True,
    ) -> list:
        """
        Detect objects in image.

        Args:
            image: PIL Image
            text_prompt: Text describing what to detect (e.g., "brown bear")
                If None, uses general object detection
            use_grounding: If True, uses caption-to-phrase grounding (more accurate)
                If False, uses general object detection

        Returns:
            List of detections, each with keys: box, label, score (if available)
            box is in [x_min, y_min, x_max, y_max] format (absolute pixels)
        """
        if use_grounding and text_prompt:
            # Use caption-to-phrase grounding for specific objects
            task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{text_prompt}"
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
        else:
            # Use general object detection
            task_prompt = "<OD>"
            task = "<OD>"

        # Process inputs
        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        # Post-process to get bounding boxes
        result = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=image.size
        )

        # Parse results
        detections = []
        if task in result:
            task_result = result[task]

            # Florence-2 returns boxes in [x_min, y_min, x_max, y_max] format
            if 'bboxes' in task_result and 'labels' in task_result:
                boxes = task_result['bboxes']
                labels = task_result['labels']

                for box, label in zip(boxes, labels):
                    detections.append({
                        "box": box,  # [x_min, y_min, x_max, y_max]
                        "label": label,
                        "score": 1.0  # Florence-2 doesn't provide confidence scores
                    })

        return detections

    def detect_to_yolo_format(
        self,
        image: Image.Image,
        text_prompt: str = None,
        output_class: int = 0,
    ) -> list:
        """
        Detect and convert to YOLO format.

        Args:
            image: PIL Image
            text_prompt: Text describing objects to detect
            output_class: Class ID to use in YOLO labels

        Returns:
            List of YOLO format strings: "class_id center_x center_y width height"
        """
        detections = self.detect(image, text_prompt, use_grounding=True)

        width, height = image.size
        yolo_labels = []

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

            label_line = f"{output_class} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
            yolo_labels.append(label_line)

        return yolo_labels


def auto_annotate_florence2(
    input_dir: str,
    output_dir: str,
    text_prompt: str = "bear",
    output_class: int = 0,
    limit: int = None,
    model_size: str = "base-ft",
):
    """
    Auto-annotate images using Florence-2.

    Args:
        input_dir: Directory containing images to annotate
        output_dir: Directory to save YOLO format labels
        text_prompt: Text prompt for detection (e.g., "bear", "brown bear")
        output_class: Class ID to use in output labels (default: 0)
        limit: Max number of images to process (default: None, process all)
        model_size: "base", "base-ft", "large", or "large-ft"
                   (ft = fine-tuned, generally better)

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

    # Initialize model
    annotator = Florence2Annotator(model_size=model_size)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    # Also search subdirectories
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            image_files.extend([
                f for f in subdir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return 0, 0, 0

    # Sort by name for consistent ordering
    image_files = sorted(image_files, key=lambda x: x.name)

    # Apply limit if specified
    if limit:
        image_files = image_files[:limit]

    print(f"\nFlorence-2 Auto Annotation")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: Florence-2-{model_size}")
    print(f"Text prompt: \"{text_prompt}\"")
    print(f"Images found: {len(image_files)}")
    print(f"{'='*60}\n")

    total_images = len(image_files)
    images_with_detections = 0
    total_detections = 0

    for i, image_path in enumerate(image_files):
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get YOLO format labels
        labels = annotator.detect_to_yolo_format(
            image=image,
            text_prompt=text_prompt,
            output_class=output_class,
        )

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

    print(f"\n{'='*60}")
    print(f"Annotation Complete!")
    print(f"{'='*60}")
    print(f"Total images: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Detection rate: {images_with_detections/total_images*100:.1f}%")
    print(f"Labels saved to: {output_dir}")

    return total_images, images_with_detections, total_detections


def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate images using Florence-2 (Microsoft multi-task vision model)"
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
        "--prompt", "-p",
        default="bear",
        help="Text prompt for detection (default: 'bear'). Examples: 'brown bear', 'grizzly bear'"
    )
    parser.add_argument(
        "--model-size",
        choices=["base", "base-ft", "large", "large-ft"],
        default="large-ft",
        help="Model size: base, base-ft, large, or large-ft (recommended). ft=fine-tuned (better). Default: large-ft"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max number of images to process (default: all)"
    )

    args = parser.parse_args()

    auto_annotate_florence2(
        input_dir=args.input,
        output_dir=args.output,
        text_prompt=args.prompt,
        model_size=args.model_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
