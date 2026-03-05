"""
Auto Annotation Script using DETR (DEtection TRansformer)

DETR is Meta's end-to-end object detection model using Transformer architecture.
Uses ResNet-101 backbone with Transformer encoder-decoder for detection.

Note: DETR uses COCO pretrained weights. In low-resolution wildlife videos,
bears may be classified as similar animals (e.g., elephant, dog). We detect
all large animal categories and treat them as potential bear detections.

Usage:
    python -m src.preprocessing.auto_annotator_detr --input data/frames/video_name/ --output data/auto_labels/
    python -m src.preprocessing.auto_annotator_detr --input data/frames/ --output data/auto_labels/ --threshold 0.3

Requirements:
    pip install transformers torch torchvision timm
"""

import argparse
from pathlib import Path
import sys
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# COCO animal class IDs that could be bears in low-res wildlife footage
ANIMAL_CLASS_IDS = {
    21,  # bear
    22,  # zebra (unlikely but included)
    23,  # giraffe (unlikely but included)
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
}


class DETRAnnotator:
    """DETR (DEtection TRansformer) based auto-annotator."""

    def __init__(
        self,
        model_id: str = "facebook/detr-resnet-101",
        device: str = None,
        animal_classes: set = None,
    ):
        """
        Initialize DETR model.

        Args:
            model_id: HuggingFace model ID
                - "facebook/detr-resnet-101" (larger, more accurate)
                - "facebook/detr-resnet-50" (faster)
            device: Device to run on (auto-detect if None)
            animal_classes: Set of COCO class IDs to detect as potential bears.
                            If None, uses default animal classes.
        """
        from transformers import DetrForObjectDetection, DetrImageProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.animal_classes = animal_classes or ANIMAL_CLASS_IDS

        print(f"Loading DETR model: {model_id}")
        print(f"Device: {device}")

        self.processor = DetrImageProcessor.from_pretrained(model_id)
        self.model = DetrForObjectDetection.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        print("Model loaded successfully!")

    def detect(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        text_prompt: str = None,
    ) -> list:
        """
        Detect objects in image.

        Args:
            image: PIL Image
            threshold: Confidence threshold for detections
            text_prompt: Ignored (DETR uses fixed COCO classes, not text prompts).
                         Kept for interface compatibility.

        Returns:
            List of detections, each with keys: box, score, label
            box is in [x_min, y_min, x_max, y_max] format (absolute pixels)
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        detections = []
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label_id_int = label_id.item()
            # Only keep animal classes (potential bears)
            if label_id_int in self.animal_classes:
                label_name = self.id2label.get(label_id_int, str(label_id_int))
                detections.append({
                    "box": box.cpu().tolist(),
                    "score": float(score),
                    "label": label_name,
                })

        return detections

    def detect_to_yolo_format(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        output_class: int = 0,
        text_prompt: str = None,
    ) -> list:
        """
        Detect and convert to YOLO format.

        Args:
            image: PIL Image
            threshold: Confidence threshold
            output_class: Class ID to use in YOLO labels
            text_prompt: Ignored, for interface compatibility

        Returns:
            List of YOLO format strings
        """
        detections = self.detect(image, threshold, text_prompt)

        width, height = image.size
        yolo_labels = []

        for det in detections:
            box = det["box"]
            x_min, y_min, x_max, y_max = box

            center_x = ((x_min + x_max) / 2) / width
            center_y = ((y_min + y_max) / 2) / height
            box_width = (x_max - x_min) / width
            box_height = (y_max - y_min) / height

            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            box_width = max(0, min(1, box_width))
            box_height = max(0, min(1, box_height))

            label_line = f"{output_class} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
            yolo_labels.append(label_line)

        return yolo_labels


def auto_annotate_detr(
    input_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    output_class: int = 0,
    limit: int = None,
    model_size: str = "101",
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 0, 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = f"facebook/detr-resnet-{model_size}"
    annotator = DETRAnnotator(model_id=model_id)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions],
        key=lambda x: x.name
    )

    if not image_files:
        print(f"No images found in {input_dir}")
        return 0, 0, 0

    if limit:
        image_files = image_files[:limit]

    print(f"\nDETR Auto Annotation")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: detr-resnet-{model_size}")
    print(f"Threshold: {threshold}")
    print(f"Images found: {len(image_files)}")
    print(f"{'='*60}\n")

    total_images = len(image_files)
    images_with_detections = 0
    total_detections = 0

    for i, image_path in enumerate(image_files):
        image = Image.open(image_path).convert("RGB")
        labels = annotator.detect_to_yolo_format(image, threshold, output_class)

        label_filename = image_path.stem + ".txt"
        label_path = output_dir / label_filename
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        num_detections = len(labels)
        total_detections += num_detections
        if num_detections > 0:
            images_with_detections += 1

        status = f"{num_detections} object(s)" if num_detections > 0 else "no detection"
        print(f"[{i+1}/{total_images}] {image_path.name}: {status}")

    print(f"\n{'='*60}")
    print(f"Annotation Complete!")
    print(f"{'='*60}")
    print(f"Total images: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    if total_images > 0:
        print(f"Detection rate: {images_with_detections/total_images*100:.1f}%")
    print(f"Labels saved to: {output_dir}")

    return total_images, images_with_detections, total_detections


def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate images using DETR (DEtection TRansformer)"
    )
    parser.add_argument("--input", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", required=True, help="Output directory for label files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--model-size", choices=["50", "101"], default="101",
                        help="ResNet backbone: '101' (more accurate) or '50' (faster). Default: 101")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Max number of images to process")

    args = parser.parse_args()
    auto_annotate_detr(
        input_dir=args.input, output_dir=args.output,
        threshold=args.threshold, model_size=args.model_size, limit=args.limit,
    )


if __name__ == "__main__":
    main()
