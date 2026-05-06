#!/usr/bin/env python3
"""
Filter detection results using CLIP — drop specific classes (e.g. bears).

How it works:
    1. Read the YOLO label file
    2. Crop each detection box
    3. Use CLIP to decide if it's the target class (e.g. salmon) or an excluded class (e.g. bear)
    4. Keep only detections classified as the target class

Usage:
    python filter_detections_with_clip.py \\
        --images data/frames/salmon_validation/ \\
        --labels data/auto_labels/salmon_validation/ \\
        --output data/auto_labels/salmon_filtered/ \\
        --target-class "salmon fish" \\
        --exclude-classes "bear" "rock" "water splash" \\
        --threshold 0.6
"""

import argparse
import cv2
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_clip_model():
    """Load the CLIP model."""
    try:
        from transformers import CLIPProcessor, CLIPModel

        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        return model, processor, device
    except ImportError:
        print("❌ transformers is required: pip install transformers")
        exit(1)


def classify_detection(image_crop, model, processor, device, target_class, exclude_classes):
    """
    Classify a detection crop with CLIP.

    Returns:
        (is_target, confidence, predicted_class)
    """
    # Build the text prompts
    all_classes = [target_class] + exclude_classes
    text_inputs = [f"a photo of {cls}" for cls in all_classes]

    # Convert to PIL
    if isinstance(image_crop, np.ndarray):
        image_crop = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))

    # CLIP inference
    inputs = processor(
        text=text_inputs,
        images=image_crop,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]

    # Decision
    target_prob = probs[0].item()  # index 0 is the target class
    max_exclude_prob = probs[1:].max().item() if len(exclude_classes) > 0 else 0.0

    predicted_idx = probs.argmax().item()
    predicted_class = all_classes[predicted_idx]

    is_target = predicted_idx == 0
    confidence = target_prob

    return is_target, confidence, predicted_class


def filter_labels_file(image_path: Path, label_path: Path, output_path: Path,
                       model, processor, device, target_class, exclude_classes, threshold):
    """
    Filter a single label file.

    Args:
        threshold: confidence threshold for the target class (above = keep)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return 0, 0

    h, w = img.shape[:2]

    # Load labels
    if not label_path.exists() or label_path.stat().st_size == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return 0, 0

    with open(label_path, 'r') as f:
        lines = f.readlines()

    kept_lines = []
    total_detections = len(lines)

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        # Convert to pixel coords
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))

        # Crop the detection box
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # CLIP classification
        is_target, confidence, predicted_class = classify_detection(
            crop, model, processor, device, target_class, exclude_classes
        )

        # Keep or drop
        if is_target and confidence >= threshold:
            kept_lines.append(line)
        # else:
        #     print(f"  Filtered: {predicted_class} (conf={confidence:.2f})")

    # Save the filtered labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(kept_lines)

    return total_detections, len(kept_lines)


def filter_dataset(images_dir: Path, labels_dir: Path, output_dir: Path,
                   target_class: str, exclude_classes: list, threshold: float):
    """Filter the entire dataset."""

    print("\n[1/3] Loading CLIP model...")
    model, processor, device = load_clip_model()
    print(f"  Device: {device}")

    print(f"\n[2/3] Scanning label files...")
    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"  Found {len(label_files)} label file(s)")

    print(f"\n[3/3] Filtering detections...")
    print(f"  Target class    : '{target_class}'")
    print(f"  Exclude classes : {exclude_classes}")
    print(f"  Confidence thr. : {threshold}")
    print()

    total_original = 0
    total_kept = 0
    non_empty_files = 0

    for label_path in tqdm(label_files, desc="Processing"):
        # Find the matching image
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_path = None

        for ext in image_extensions:
            for img_path in images_dir.rglob(f"{label_path.stem}{ext}"):
                image_path = img_path
                break
            if image_path:
                break

        if not image_path or not image_path.exists():
            continue

        output_path = output_dir / label_path.name

        original, kept = filter_labels_file(
            image_path, label_path, output_path,
            model, processor, device, target_class, exclude_classes, threshold
        )

        total_original += original
        total_kept += kept

        if kept > 0:
            non_empty_files += 1

    print(f"\n✅ Done!")
    print(f"  Original detections: {total_original}")
    print(f"  Kept detections    : {total_kept} ({100*total_kept/max(1,total_original):.1f}%)")
    print(f"  Filtered out       : {total_original - total_kept} ({100*(total_original-total_kept)/max(1,total_original):.1f}%)")
    print(f"  Non-empty files    : {non_empty_files}")
    print(f"\n  Output directory   : {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter detection results with CLIP, removing specific classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter out bears, keep only salmon
  python filter_detections_with_clip.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --output data/auto_labels/salmon_filtered/ \\
      --target-class "salmon fish" \\
      --exclude-classes "bear" "rock" "water" \\
      --threshold 0.6

  # Stricter threshold
  python filter_detections_with_clip.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --output data/auto_labels/salmon_filtered/ \\
      --target-class "jumping salmon" \\
      --exclude-classes "bear" "brown bear" "grizzly bear" \\
      --threshold 0.7
        """
    )

    parser.add_argument("--images", required=True, help="Image directory (supports nested subfolders)")
    parser.add_argument("--labels", required=True, help="Labels directory (YOLO format)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--target-class", required=True, help="Target class (e.g. 'salmon fish')")
    parser.add_argument("--exclude-classes", nargs="+", default=["bear"],
                       help="Classes to exclude (default: bear)")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Target-class confidence threshold (default: 0.6)")

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
    print("🔍 CLIP detection filter")
    print("=" * 70)

    filter_dataset(
        images_dir, labels_dir, output_dir,
        args.target_class, args.exclude_classes, args.threshold
    )

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
