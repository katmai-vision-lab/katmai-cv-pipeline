#!/usr/bin/env python3
"""
Visualize YOLO annotations in a nested directory layout.

Handles the case where images live in subfolders but all label files
sit together in a single labels directory.
"""

import argparse
import cv2
from pathlib import Path
import random


def draw_yolo_boxes(image_path: Path, label_path: Path, output_path: Path, class_names: list = None):
    """Draw YOLO-format bounding boxes onto an image."""
    if class_names is None:
        class_names = ["salmon"]

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return False

    h, w = img.shape[:2]

    # Load labels
    if not label_path.exists():
        print(f"⚠️  Label file not found: {label_path.name}")
        return False

    # Skip empty label files
    if label_path.stat().st_size == 0:
        print(f"⚠️  Empty label (no detections): {image_path.name}")
        return False

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return False

    # Draw each detection box
    box_count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        # Convert normalized coords to pixel coords
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Draw bounding box
        color = (0, 255, 0)  # green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Draw label
        label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

        # Background behind the text for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        # Text background
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        # Text
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        box_count += 1

    # Save image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"✓ {image_path.name} ({box_count} detections)")

    return True


def visualize_nested_directory(images_dir: Path, labels_dir: Path, output_dir: Path, limit: int = None):
    """
    Visualize annotations from a nested image layout.

    Args:
        images_dir: image root directory (may contain subfolders)
        labels_dir: labels directory (all .txt files in one place)
        output_dir: where to write visualizations
        limit: max number of images to process (None = all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recursively find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = []

    for ext in image_extensions:
        image_files.extend(images_dir.rglob(f"*{ext}"))

    image_files = sorted(image_files)

    print(f"\nFound {len(image_files)} images")

    # Filter: only keep images that have a non-empty label file
    images_with_labels = []
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists() and label_path.stat().st_size > 0:
            images_with_labels.append(img_path)

    print(f"  {len(images_with_labels)} of them have non-empty labels")

    # Random sample if a limit was given
    if limit and len(images_with_labels) > limit:
        images_with_labels = random.sample(images_with_labels, limit)
        print(f"Random sampling {limit} images for visualization")

    print(f"\nStarting visualization...")

    visualized_count = 0
    for img_path in images_with_labels:
        label_path = labels_dir / (img_path.stem + ".txt")
        output_path = output_dir / img_path.name

        if draw_yolo_boxes(img_path, label_path, output_path):
            visualized_count += 1

    print(f"\n✅ Done! Successfully visualized {visualized_count} images")
    print(f"   Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO annotations in a nested directory layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize the first 50 annotated images
  python visualize_nested.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --output data/visualized/salmon_validation/ \\
      --limit 50

  # Visualize all
  python visualize_nested.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --output data/visualized/salmon_validation/
        """
    )

    parser.add_argument("--images", required=True, help="Image root directory (supports nested subfolders)")
    parser.add_argument("--labels", required=True, help="Labels directory (YOLO-format .txt files)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, help="Max number of images to visualize (random sample)")

    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)

    if not images_dir.exists():
        print(f"❌ Image directory not found: {images_dir}")
        return 1

    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return 1

    print("=" * 70)
    print("🎨 YOLO annotation visualizer")
    print("=" * 70)

    visualize_nested_directory(images_dir, labels_dir, output_dir, args.limit)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    exit(main())
