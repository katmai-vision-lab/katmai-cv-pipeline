"""
Auto Annotation Script using MegaDetector v5

MegaDetector is Microsoft's wildlife detection model, specifically designed for
camera trap and wildlife images. Trained on millions of wildlife images, it is
the gold standard for animal detection in complex natural environments.

Handles occlusion, varying lighting, long distances, and camouflaged animals
far better than general-purpose detectors like OWL-ViT.

MegaDetector Categories:
    0: animal
    1: person
    2: vehicle

Usage:
    python -m src.preprocessing.auto_annotator_megadet \
        --input data/frames/video_name/ \
        --output data/auto_labels/

    # Adjust threshold
    python -m src.preprocessing.auto_annotator_megadet \
        --input data/frames/ --output data/auto_labels/ --threshold 0.3

Requirements:
    pip install PytorchWildlife
"""

import argparse
from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class MegaDetectorAnnotator:
    """MegaDetector v5 based auto-annotator for wildlife detection."""

    CATEGORY_MAP = {0: "animal", 1: "person", 2: "vehicle"}

    def __init__(
        self,
        device: str = None,
        version: str = "v5",
    ):
        """
        Initialize MegaDetector model.

        Args:
            device: Device to run on (auto-detect if None)
            version: Model version, currently only "v5" is supported
        """
        try:
            from PytorchWildlife.models import detection as pw_detection
        except ImportError:
            raise ImportError(
                "PytorchWildlife is required for MegaDetector. "
                "Install it with: pip install PytorchWildlife"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading MegaDetector {version}...")
        print(f"Device: {device}")

        if version == "v5":
            self.model = pw_detection.MegaDetectorV5(
                device=device, pretrained=True
            )
        else:
            raise ValueError(f"Unsupported version: {version}. Use 'v5'.")

        self.target_category = 0  # animal only
        print("Model loaded successfully!")

    def detect(
        self,
        image: Image.Image,
        threshold: float = 0.3,
        text_prompt: str = None,
    ) -> list:
        """
        Detect animals in image.

        Args:
            image: PIL Image
            threshold: Confidence threshold (default: 0.3)
            text_prompt: Ignored. MegaDetector uses fixed categories.
                         Kept for interface compatibility with multi-model system.

        Returns:
            List of detections, each with keys: box, score, label
            box is in [x_min, y_min, x_max, y_max] format (absolute pixels)
        """
        img_np = np.array(image)

        result = self.model.single_image_detection(
            img=img_np,
            det_conf_thres=threshold,
        )

        detections = []
        if result is None:
            return detections

        det_obj = result.get("detections")
        if det_obj is None:
            return detections

        # supervision.Detections object
        boxes = det_obj.xyxy        # (N, 4) [x1, y1, x2, y2]
        scores = det_obj.confidence  # (N,)
        class_ids = det_obj.class_id # (N,)

        if boxes is None or len(boxes) == 0:
            return detections

        for box, score, cls_id in zip(boxes, scores, class_ids):
            if int(cls_id) == self.target_category:
                detections.append({
                    "box": [float(box[0]), float(box[1]),
                            float(box[2]), float(box[3])],
                    "score": float(score),
                    "label": "animal",
                })

        return detections

    def detect_to_yolo_format(
        self,
        image: Image.Image,
        threshold: float = 0.3,
        output_class: int = 0,
    ) -> list:
        """
        Detect and convert to YOLO format.

        Args:
            image: PIL Image
            threshold: Confidence threshold
            output_class: Class ID to use in YOLO labels

        Returns:
            List of YOLO format strings: "class_id center_x center_y width height"
        """
        detections = self.detect(image, threshold)

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


def auto_annotate_megadet(
    input_dir: str,
    output_dir: str,
    threshold: float = 0.3,
    output_class: int = 0,
    limit: int = None,
):
    """
    使用 MegaDetector v5 对图像进行自动标注。

    Args:
        input_dir: 输入图像目录
        output_dir: 输出 YOLO 格式标签目录
        threshold: 置信度阈值 (default: 0.3)
        output_class: 输出标签的类别 ID (default: 0)
        limit: 最大处理图像数量
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return 0, 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    annotator = MegaDetectorAnnotator()

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

    print(f"\nMegaDetector v5 Auto Annotation")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
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

        status = f"{num_detections} animal(s)" if num_detections > 0 else "no detection"
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
        description="Auto-annotate images using MegaDetector v5 (wildlife detection)"
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
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max number of images to process (default: all)"
    )

    args = parser.parse_args()
    auto_annotate_megadet(
        input_dir=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
