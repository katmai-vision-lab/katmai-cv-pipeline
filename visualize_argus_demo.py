"""
ARGUS system demo: visualize the three expert models and the final fused result.

Usage:
    python visualize_argus_demo.py --image data/annotation/salmons/images/525543413_1305151501169403_1183741009494220996_n.jpg --output demo_results/
"""

import argparse
from pathlib import Path
import sys
import torch
from PIL import Image
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.annotation_salmon.auto_annotator_gdino import GroundingDINOAnnotator
from src.preprocessing.annotation_salmon.auto_annotator_owlvit import OWLViTAnnotator
from src.preprocessing.annotation_salmon.auto_annotator_florence2 import Florence2Annotator


def draw_detections(image_np, detections, color, model_name):
    """Draw detection boxes on the image."""
    img = image_np.copy()
    h, w = img.shape[:2]

    for det in detections:
        x_min, y_min, x_max, y_max = det['box']
        score = det['score']
        label = det['label']

        # Draw bounding box
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3)

        # Draw label + confidence
        text = f"{label} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Text background
        cv2.rectangle(img, (int(x_min), int(y_min) - text_size[1] - 10),
                     (int(x_min) + text_size[0], int(y_min)), color, -1)
        # Text
        cv2.putText(img, text, (int(x_min), int(y_min) - 5),
                   font, font_scale, (255, 255, 255), thickness)

    # Add model-name watermark
    watermark = f"{model_name} ({len(detections)} detections)"
    cv2.putText(img, watermark, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    return img


def run_expert_models(image_path, text_prompt="bear. salmon. fish."):
    """Run the three expert models."""
    print(f"\n{'='*60}")
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"Image size: {width}x{height}")

    results = {}

    # 1. Grounding DINO
    print(f"\n{'='*60}")
    print("Running Grounding DINO...")
    gdino = GroundingDINOAnnotator(device="cuda" if torch.cuda.is_available() else "cpu")
    gdino_detections = gdino.detect(image, text_prompt, box_threshold=0.25, text_threshold=0.25)
    print(f"✓ GDINO: {len(gdino_detections)} detections")
    results['gdino'] = gdino_detections
    del gdino
    torch.cuda.empty_cache()

    # 2. OWL-ViT
    print(f"\n{'='*60}")
    print("Running OWL-ViT...")
    owlvit = OWLViTAnnotator(device="cuda" if torch.cuda.is_available() else "cpu")
    owlvit_detections = owlvit.detect(image, [p.strip() for p in text_prompt.split('.') if p.strip()], threshold=0.20)
    print(f"✓ OWL-ViT: {len(owlvit_detections)} detections (threshold=0.20)")
    results['owlvit'] = owlvit_detections
    del owlvit
    torch.cuda.empty_cache()

    # 3. Florence-2
    print(f"\n{'='*60}")
    print("Running Florence-2...")
    florence = Florence2Annotator(device="cuda" if torch.cuda.is_available() else "cpu")
    florence_detections = florence.detect(image, text_prompt, use_grounding=True)
    print(f"✓ Florence-2: {len(florence_detections)} detections")
    results['florence2'] = florence_detections
    del florence
    torch.cuda.empty_cache()

    return results, image


def create_fusion_result(detections_dict, image_size):
    """
    Simplified fusion: use NMS to merge overlapping detections,
    applying per-model weights to produce a weighted consensus.
    """
    width, height = image_size
    all_detections = []

    # Per-model weights (simulating the scene-adaptive weights of a gating net)
    model_weights = {
        'gdino': 0.45,      # Grounding DINO weighted highest
        'owlvit': 0.25,     # OWL-ViT weighted lowest (per user request)
        'florence2': 0.30   # Florence-2 in the middle
    }

    print(f"\n📊 Model Weights:")
    for model, weight in model_weights.items():
        print(f"  {model}: {weight:.2f}")

    # Collect all detections, multiplying each score by its model's weight
    for model_name, dets in detections_dict.items():
        weight = model_weights.get(model_name, 1.0)
        for det in dets:
            all_detections.append({
                'box': det['box'],
                'score': det['score'] * weight,  # apply model weight
                'original_score': det['score'],   # keep raw score for reference
                'label': det['label'],
                'model': model_name,
                'weight': weight
            })

    if not all_detections:
        return []

    # Simple NMS: sort by confidence, drop boxes with high overlap
    all_detections.sort(key=lambda x: x['score'], reverse=True)

    def iou(box1, box2):
        """Compute Intersection-over-Union of two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    fused = []
    iou_threshold = 0.7  # Raised threshold so close-but-distinct boxes survive

    print(f"  NMS IoU threshold: {iou_threshold}")

    for det in all_detections:
        # Skip if this box overlaps an already-selected one
        overlap = False
        for selected in fused:
            if iou(det['box'], selected['box']) > iou_threshold:
                overlap = True
                break

        if not overlap:
            fused.append(det)

    # Final filter: only keep detections with score >= 0.23
    confidence_threshold = 0.23
    fused_filtered = [det for det in fused if det['score'] >= confidence_threshold]

    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Before filtering: {len(fused)} detections")
    print(f"  After filtering: {len(fused_filtered)} detections")

    return fused_filtered


def main():
    parser = argparse.ArgumentParser(description="ARGUS system demo")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="demo_results", help="Output directory")
    parser.add_argument("--prompt", default="bear. salmon. fish.", help="Detection text prompt")

    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    # Run the three expert models
    results, image = run_expert_models(image_path, args.prompt)

    # Convert to numpy array for OpenCV
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Overlay all three models' detections on a single image
    colors = {
        'gdino': (0, 255, 0),      # green
        'owlvit': (255, 0, 0),     # blue
        'florence2': (0, 165, 255) # orange
    }

    model_names = {
        'gdino': 'Grounding DINO',
        'owlvit': 'OWL-ViT',
        'florence2': 'Florence-2'
    }

    print(f"\n{'='*60}")
    print("Visualizing results...")

    # Build a combined visualization with all three models on one image
    img_all_models = image_np.copy()
    h, w = img_all_models.shape[:2]

    # Draw each model's boxes on the same image
    for model_key, detections in results.items():
        for det in detections:
            x_min, y_min, x_max, y_max = det['box']
            score = det['score']
            label = det['label']
            color = colors[model_key]

            # Bounding box
            cv2.rectangle(img_all_models, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

            # Label + confidence
            text = f"{model_names[model_key][:5]}: {label} {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            # Text background
            cv2.rectangle(img_all_models, (int(x_min), int(y_min) - text_size[1] - 8),
                         (int(x_min) + text_size[0], int(y_min)), color, -1)
            # Text
            cv2.putText(img_all_models, text, (int(x_min), int(y_min) - 4),
                       font, font_scale, (255, 255, 255), thickness)

    # Legend
    legend_y = 30
    legend_x = 10
    for model_key, color in colors.items():
        text = f"{model_names[model_key]}: {len(results[model_key])} detections"
        cv2.putText(img_all_models, text, (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        legend_y += 35

    # Title
    title = "Three Expert Models Combined"
    cv2.putText(img_all_models, title, (legend_x, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(img_all_models, title, (legend_x, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    output_path = output_dir / f"{image_path.stem}_three_experts.jpg"
    cv2.imwrite(str(output_path), img_all_models)
    print(f"✓ Saved combined experts: {output_path}")

    # Build the fused result
    print(f"\n{'='*60}")
    print("Creating fusion result...")
    fused_detections = create_fusion_result(results, image.size)
    print(f"✓ Fused: {len(fused_detections)} detections")

    # Visualize the fused result (purple)
    img_fused = image_np.copy()
    h, w = img_fused.shape[:2]

    for det in fused_detections:
        x_min, y_min, x_max, y_max = det['box']
        score = det['score']
        label = det['label']
        color = (255, 0, 255)  # purple

        # Bounding box
        cv2.rectangle(img_fused, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3)

        # Label + confidence
        text = f"{label} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Text background
        cv2.rectangle(img_fused, (int(x_min), int(y_min) - text_size[1] - 10),
                     (int(x_min) + text_size[0], int(y_min)), color, -1)
        # Text
        cv2.putText(img_fused, text, (int(x_min), int(y_min) - 5),
                   font, font_scale, (255, 255, 255), thickness)

    # Watermark
    watermark = f"ARGUS Fusion ({len(fused_detections)} detections)"
    cv2.putText(img_fused, watermark, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(img_fused, watermark, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)

    output_path = output_dir / f"{image_path.stem}_argus_fusion.jpg"
    cv2.imwrite(str(output_path), img_fused)
    print(f"✓ Saved fusion result: {output_path}")

    print(f"\n{'='*60}")
    print("Done! Results saved to:")
    print(f"  {output_dir}/")
    print(f"\n📊 Summary:")
    print(f"  Three Experts Combined: {output_dir / f'{image_path.stem}_three_experts.jpg'}")
    print(f"  ARGUS Fusion: {output_dir / f'{image_path.stem}_argus_fusion.jpg'}")
    print(f"\n  Detection counts:")
    print(f"    GDINO: {len(results['gdino'])} detections")
    print(f"    OWL-ViT: {len(results['owlvit'])} detections")
    print(f"    Florence-2: {len(results['florence2'])} detections")
    print(f"    ARGUS Fusion: {len(fused_detections)} detections")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
