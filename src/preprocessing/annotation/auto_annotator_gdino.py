"""
Auto Annotation Script using Grounding DINO

Use Grounding DINO (open-vocabulary detection) for higher accuracy auto-annotation.
This model uses text prompts to detect objects, which works much better for wildlife detection.

Usage:
    # Annotate frames with default prompt "bear"
    python -m src.preprocessing.auto_annotator_gdino --input data/frames/video_name/ --output data/auto_labels/

    # With custom text prompt
    python -m src.preprocessing.auto_annotator_gdino --input data/frames/ --output data/auto_labels/ --prompt "brown bear. grizzly bear."

    # Adjust thresholds
    python -m src.preprocessing.auto_annotator_gdino --input data/frames/ --output data/auto_labels/ --box-threshold 0.3 --text-threshold 0.25

Requirements:
    pip install transformers torch torchvision pillow
"""

import argparse
from pathlib import Path
import sys
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GroundingDINOAnnotator:
    """Grounding DINO based auto-annotator for high-accuracy detection."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = None,
    ):
        """
        Initialize Grounding DINO model.

        Args:
            model_id: HuggingFace model ID (tiny, base, or large)
                - "IDEA-Research/grounding-dino-tiny" (smallest, fastest)
                - "IDEA-Research/grounding-dino-base" (balanced)
                - "IDEA-Research/grounding-dino-large" (largest, most accurate)
            device: Device to run on (auto-detect if None)
        """
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading Grounding DINO model: {model_id}")
        print(f"Device: {device}")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

        print("Model loaded successfully!")

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> list:
        """
        Detect objects in image using text prompt.

        Args:
            image: PIL Image
            text_prompt: Text describing objects to detect (e.g., "bear. brown bear.")
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            List of detections, each with keys: box, score, label
            box is in [x_min, y_min, x_max, y_max] format (absolute pixels)
        """
        # Ensure prompt ends with period
        if not text_prompt.strip().endswith('.'):
            text_prompt = text_prompt.strip() + '.'

        # Process inputs
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        width, height = image.size
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)]
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = result["labels"]

            for box, score, label in zip(boxes, scores, labels):
                detections.append({
                    "box": box.tolist(),  # [x_min, y_min, x_max, y_max]
                    "score": float(score),
                    "label": label
                })

        return detections

    def detect_to_yolo_format(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        output_class: int = 0,
    ) -> list:
        """
        Detect and convert to YOLO format.

        Args:
            image: PIL Image
            text_prompt: Text describing objects to detect
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text matching
            output_class: Class ID to use in YOLO labels

        Returns:
            List of YOLO format strings: "class_id center_x center_y width height"
        """
        detections = self.detect(image, text_prompt, box_threshold, text_threshold)

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


def auto_annotate_gdino(
    input_dir: str,
    output_dir: str,
    text_prompt: str = "bear",
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    output_class: int = 0,
    limit: int = None,
    model_size: str = "base",
):
    """
    Auto-annotate images using Grounding DINO.

    Args:
        input_dir: Directory containing images to annotate
        output_dir: Directory to save YOLO format labels
        text_prompt: Text prompt for detection (e.g., "bear", "brown bear. grizzly bear.")
        box_threshold: Confidence threshold for boxes (default: 0.25)
        text_threshold: Confidence threshold for text matching (default: 0.25)
        output_class: Class ID to use in output labels (default: 0)
        limit: Max number of images to process (default: None, process all)
        model_size: "tiny", "base", or "large" (large is most accurate, tiny is fastest)

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
    model_id = f"IDEA-Research/grounding-dino-{model_size}"
    annotator = GroundingDINOAnnotator(model_id=model_id)

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

    print(f"\nGrounding DINO Auto Annotation")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: grounding-dino-{model_size}")
    print(f"Text prompt: \"{text_prompt}\"")
    print(f"Box threshold: {box_threshold}")
    print(f"Text threshold: {text_threshold}")
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
            box_threshold=box_threshold,
            text_threshold=text_threshold,
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
        target = text_prompt.split('.')[0].strip()
        status = f"{num_detections} {target}(s)" if num_detections > 0 else "no detection"
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
        description="Auto-annotate images using Grounding DINO (open-vocabulary detection)"
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
        help="Text prompt for detection (default: 'bear'). Can use multiple: 'brown bear. grizzly bear.'"
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for boxes (default: 0.25)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for text matching (default: 0.25)"
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "base", "large"],
        default="large",
        help="Model size: 'large' (most accurate), 'base' (balanced), or 'tiny' (fastest). Default: large"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max number of images to process (default: all)"
    )

    args = parser.parse_args()

    auto_annotate_gdino(
        input_dir=args.input,
        output_dir=args.output,
        text_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        model_size=args.model_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
