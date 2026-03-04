"""
Auto Annotation Script using OWL-ViT v2

OWL-ViT (Open-vocabulary Learning with Vision Transformers) is a zero-shot
object detection model from Google. It uses CLIP-based architecture and supports
text-based object detection.

Usage:
    # Annotate frames with default prompt "bear"
    python -m src.preprocessing.auto_annotator_owlvit --input data/frames/video_name/ --output data/auto_labels/

    # With custom text prompt
    python -m src.preprocessing.auto_annotator_owlvit --input data/frames/ --output data/auto_labels/ --prompt "brown bear"

    # Adjust threshold
    python -m src.preprocessing.auto_annotator_owlvit --input data/frames/ --output data/auto_labels/ --threshold 0.2

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


class OWLViTAnnotator:
    """OWL-ViT v2 based auto-annotator for zero-shot detection."""

    def __init__(
        self,
        model_id: str = "google/owlv2-large-patch14-ensemble",
        device: str = None,
    ):
        """
        Initialize OWL-ViT model.

        Args:
            model_id: HuggingFace model ID
                - "google/owlv2-large-patch14-ensemble" (best accuracy, ensemble)
                - "google/owlv2-large-patch14" (large, single model)
                - "google/owlv2-base-patch16-ensemble" (faster, ensemble)
                - "google/owlv2-base-patch16" (fastest)
            device: Device to run on (auto-detect if None)
        """
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading OWL-ViT v2 model: {model_id}")
        print(f"Device: {device}")

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

        print("Model loaded successfully!")

    def detect(
        self,
        image: Image.Image,
        text_queries: list,
        threshold: float = 0.1,
    ) -> list:
        """
        Detect objects in image using text queries.

        Args:
            image: PIL Image
            text_queries: List of text queries (e.g., ["bear", "brown bear"])
            threshold: Confidence threshold for detections

        Returns:
            List of detections, each with keys: box, score, label
            box is in [x_min, y_min, x_max, y_max] format (absolute pixels)
        """
        # Ensure text_queries is a list
        if isinstance(text_queries, str):
            text_queries = [text_queries]

        # Process inputs
        inputs = self.processor(
            text=text_queries,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process results manually
        # OWL-ViT v2 returns logits and pred_boxes
        logits = outputs.logits[0]  # (num_queries, num_labels)
        pred_boxes = outputs.pred_boxes[0]  # (num_queries, 4)

        # Get scores and labels
        probs = logits.sigmoid()  # Convert logits to probabilities
        scores, labels = probs.max(dim=-1)

        # Filter by threshold
        keep = scores > threshold
        scores = scores[keep]
        labels = labels[keep]
        pred_boxes = pred_boxes[keep]

        # Convert boxes from normalized [cx, cy, w, h] to [x_min, y_min, x_max, y_max]
        width, height = image.size
        boxes_xyxy = []
        for box in pred_boxes:
            cx, cy, w, h = box.cpu().numpy()
            x_min = (cx - w / 2) * width
            y_min = (cy - h / 2) * height
            x_max = (cx + w / 2) * width
            y_max = (cy + h / 2) * height
            boxes_xyxy.append([x_min, y_min, x_max, y_max])

        detections = []
        for box, score, label_idx in zip(boxes_xyxy, scores.cpu().numpy(), labels.cpu().numpy()):
            # label_idx corresponds to the index in text_queries
            label_text = text_queries[int(label_idx)] if int(label_idx) < len(text_queries) else f"class_{int(label_idx)}"
            detections.append({
                "box": box,  # [x_min, y_min, x_max, y_max]
                "score": float(score),
                "label": label_text,
                "label_idx": int(label_idx)
            })

        return detections

    def detect_to_yolo_format(
        self,
        image: Image.Image,
        text_queries: list,
        threshold: float = 0.1,
        output_class: int = 0,
    ) -> list:
        """
        Detect and convert to YOLO format.

        Args:
            image: PIL Image
            text_queries: List of text queries
            threshold: Confidence threshold
            output_class: Class ID to use in YOLO labels

        Returns:
            List of YOLO format strings: "class_id center_x center_y width height"
        """
        detections = self.detect(image, text_queries, threshold)

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


def auto_annotate_owlvit(
    input_dir: str,
    output_dir: str,
    text_prompt: str = "bear",
    threshold: float = 0.1,
    output_class: int = 0,
    limit: int = None,
    model_size: str = "large",
):
    """
    Auto-annotate images using OWL-ViT v2.

    Args:
        input_dir: Directory containing images to annotate
        output_dir: Directory to save YOLO format labels
        text_prompt: Text prompt for detection (can use comma-separated: "bear, brown bear")
        threshold: Confidence threshold (default: 0.1)
        output_class: Class ID to use in output labels (default: 0)
        limit: Max number of images to process (default: None, process all)
        model_size: "large" or "base" (large is more accurate)

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

    # Parse text queries (split by comma)
    text_queries = [q.strip() for q in text_prompt.split(",")]

    # Initialize model
    if model_size == "large":
        model_id = "google/owlv2-large-patch14-ensemble"
    else:
        model_id = "google/owlv2-base-patch16-ensemble"

    annotator = OWLViTAnnotator(model_id=model_id)

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

    print(f"\nOWL-ViT v2 Auto Annotation")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_id}")
    print(f"Text queries: {text_queries}")
    print(f"Threshold: {threshold}")
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
            text_queries=text_queries,
            threshold=threshold,
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
        status = f"{num_detections} object(s)" if num_detections > 0 else "no detection"
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
        description="Auto-annotate images using OWL-ViT v2 (zero-shot object detection)"
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
        help="Text prompt for detection (default: 'bear'). Use comma for multiple: 'bear, brown bear'"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Confidence threshold (default: 0.1, OWL-ViT typically needs lower threshold)"
    )
    parser.add_argument(
        "--model-size",
        choices=["large", "base"],
        default="large",
        help="Model size: 'large' (more accurate) or 'base' (faster). Default: large"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max number of images to process (default: all)"
    )

    args = parser.parse_args()

    auto_annotate_owlvit(
        input_dir=args.input,
        output_dir=args.output,
        text_prompt=args.prompt,
        threshold=args.threshold,
        model_size=args.model_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
