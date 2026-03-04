"""
Visualize YOLO Labels

Draw bounding boxes on images based on YOLO format label files.

Usage:
    # Visualize single image
    python -m src.preprocessing.visualize_labels --image data/frames/video/frame.jpg --label data/auto_labels/frame.txt

    # Visualize directory (saves to output folder)
    python -m src.preprocessing.visualize_labels --images data/frames/video/ --labels data/auto_labels/ --output data/visualized/
"""

import argparse
import cv2
from pathlib import Path


def draw_yolo_boxes(image_path: str, label_path: str, output_path: str = None, class_names: list = None):
    """
    Draw YOLO format bounding boxes on an image.

    Args:
        image_path: Path to the image
        label_path: Path to the YOLO label file
        output_path: Path to save the output image (optional)
        class_names: List of class names (default: ["salmon"])

    Returns:
        Image with boxes drawn
    """
    if class_names is None:
        class_names = ["salmon"]

    image_path = Path(image_path)
    label_path = Path(label_path)

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    h, w = img.shape[:2]

    # Read labels
    if not label_path.exists():
        print(f"Warning: Label file not found: {label_path}")
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    # Draw each box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        # Convert normalized coordinates to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Draw box
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save or return
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        print(f"Saved: {output_path}")

    return img


def visualize_directory(images_dir: str, labels_dir: str, output_dir: str, limit: int = None):
    """
    Visualize all images in a directory.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        output_dir: Directory to save visualized images
        limit: Max number of images to process (optional)
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if limit:
        image_files = image_files[:limit]

    print(f"Visualizing {len(image_files)} images...")

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        output_path = output_dir / img_path.name

        draw_yolo_boxes(img_path, label_path, output_path)

    print(f"\nDone! Visualized images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on images")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--label", help="Single label path")
    parser.add_argument("--images", help="Directory of images")
    parser.add_argument("--labels", help="Directory of labels")
    parser.add_argument("--output", "-o", help="Output path or directory")
    parser.add_argument("--limit", "-n", type=int, help="Max images to process")

    args = parser.parse_args()

    if args.image and args.label:
        # Single image mode
        output = args.output or args.image.replace(".jpg", "_labeled.jpg")
        draw_yolo_boxes(args.image, args.label, output)
    elif args.images and args.labels and args.output:
        # Directory mode
        visualize_directory(args.images, args.labels, args.output, args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
