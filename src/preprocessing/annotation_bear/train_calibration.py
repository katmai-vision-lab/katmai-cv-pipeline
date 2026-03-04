"""
Calibration Trainer for Multi-Model Detection System

Collects confidence scores and ground truth matches from a validation set
to train probability calibration curves for each model.

Requires:
- Ground truth annotations in YOLO format
- Same validation images used for model arena evaluation
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.annotation.auto_annotator_gdino import GroundingDINOAnnotator
from src.preprocessing.annotation.auto_annotator_megadet import MegaDetectorAnnotator
from src.preprocessing.annotation.auto_annotator_detr import DETRAnnotator
from src.preprocessing.annotation.probability_calibrator import (
    ProbabilityCalibrator,
    CalibrationData,
)


def load_ground_truth_yolo(label_file: Path, img_width: int, img_height: int) -> List[Tuple[float, float, float, float]]:
    """
    Load ground truth boxes from YOLO format file.
    
    Returns: List of (x_center, y_center, width, height) in normalized [0,1] coordinates
    """
    if not label_file.exists():
        return []
    
    boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # class_id x_center y_center width height
                boxes.append((
                    float(parts[1]),  # x_center
                    float(parts[2]),  # y_center
                    float(parts[3]),  # width
                    float(parts[4]),  # height
                ))
    return boxes


def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate IoU between two boxes in (x_center, y_center, width, height) format.
    All coordinates are normalized [0, 1].
    """
    # Convert to corner coordinates
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def match_detections_to_ground_truth(
    pred_boxes: List[Tuple[float, float, float, float, float]],  # (x, y, w, h, conf)
    gt_boxes: List[Tuple[float, float, float, float]],  # (x, y, w, h)
    iou_threshold: float = 0.5
) -> Tuple[List[bool], List[float]]:
    """
    Match predicted boxes to ground truth using IoU threshold.
    
    Returns:
        correctness: List of boolean (True if matched to GT, False otherwise)
        confidences: List of confidence scores
    """
    if not pred_boxes:
        return [], []
    
    correctness = []
    confidences = []
    matched_gt: Set[int] = set()  # Track which GT boxes are already matched
    
    # Sort predictions by confidence (highest first)
    sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1][4], reverse=True)
    
    for orig_idx, (x, y, w, h, conf) in sorted_preds:
        confidences.append(conf)
        
        # Find best matching GT box
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue  # GT already matched
            
            iou = calculate_iou((x, y, w, h), gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is good enough
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            correctness.append(True)
            matched_gt.add(best_gt_idx)
        else:
            correctness.append(False)  # False positive
    
    return correctness, confidences


def train_calibration(
    images_dir: Path,
    labels_dir: Path,
    output_path: Path,
    prompt: str = "bear",
    iou_threshold: float = 0.5,
    use_gdino: bool = True,
    use_megadet: bool = True,
    use_detr: bool = True,
    device: str = "cuda"
):
    """
    Train calibration curves for each model using validation set.
    
    Args:
        images_dir: Directory containing validation images
        labels_dir: Directory containing ground truth YOLO labels
        output_path: Where to save trained calibrator
        prompt: Detection prompt for Grounding DINO
        iou_threshold: IoU threshold for matching detections to GT
        use_gdino/use_megadet/use_detr: Which models to calibrate
        device: cuda or cpu
    """
    print("=" * 80)
    print("PROBABILITY CALIBRATION TRAINING")
    print("=" * 80)
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Device: {device}")
    print()
    
    # Initialize models
    models: Dict[str, any] = {}
    calibration_data: Dict[str, CalibrationData] = {}
    
    if use_gdino:
        print("Loading Grounding DINO...")
        models['gdino'] = GroundingDINOAnnotator(
            model_size='base',
            box_threshold=0.25,
            text_threshold=0.25,
            device=device
        )
        calibration_data['gdino'] = CalibrationData('gdino')
    
    if use_megadet:
        print("Loading MegaDetector v5...")
        models['megadet'] = MegaDetectorAnnotator(device=device)
        calibration_data['megadet'] = CalibrationData('megadet')
    
    if use_detr:
        print("Loading DETR...")
        models['detr'] = DETRAnnotator(
            model_name='facebook/detr-resnet-101',
            confidence_threshold=0.3,
            device=device
        )
        calibration_data['detr'] = CalibrationData('detr')
    
    print()
    
    # Find all validation images
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"**/*{ext}"))
    
    print(f"Found {len(image_files)} validation images")
    print()
    
    # Process each image
    stats = defaultdict(lambda: {'total': 0, 'tp': 0, 'fp': 0})
    
    for img_path in image_files:
        # Find corresponding label file
        relative_path = img_path.relative_to(images_dir)
        label_path = labels_dir / relative_path.with_suffix('.txt')
        
        if not label_path.exists():
            print(f"Warning: No labels for {img_path.name}, skipping")
            continue
        
        # Load ground truth
        from PIL import Image
        img = Image.open(img_path)
        gt_boxes = load_ground_truth_yolo(label_path, img.width, img.height)
        
        if not gt_boxes:
            continue  # No ground truth objects
        
        # Run each model
        for model_name, model in models.items():
            if model_name == 'gdino':
                detections = model.annotate(img_path, prompt)
            elif model_name == 'megadet':
                detections = model.annotate(img_path)
            elif model_name == 'detr':
                detections = model.annotate(img_path, img.width, img.height)
            else:
                detections = []
            
            # Convert detections to normalized format
            pred_boxes = []
            for det in detections:
                # detections format: (x_min, y_min, x_max, y_max, confidence)
                # Convert to (x_center, y_center, width, height, confidence) normalized
                x_min, y_min, x_max, y_max, conf = det
                x_center = (x_min + x_max) / 2 / img.width
                y_center = (y_min + y_max) / 2 / img.height
                width = (x_max - x_min) / img.width
                height = (y_max - y_min) / img.height
                pred_boxes.append((x_center, y_center, width, height, conf))
            
            # Match to ground truth
            correctness, confidences = match_detections_to_ground_truth(
                pred_boxes, gt_boxes, iou_threshold
            )
            
            # Add to calibration data
            for is_correct, conf in zip(correctness, confidences):
                calibration_data[model_name].add_sample(conf, is_correct)
                stats[model_name]['total'] += 1
                if is_correct:
                    stats[model_name]['tp'] += 1
                else:
                    stats[model_name]['fp'] += 1
        
        # Print progress
        if len(calibration_data[list(calibration_data.keys())[0]].confidences) % 50 == 0:
            print(f"  Processed {len([c for c in calibration_data.values() if c.confidences][0].confidences)} samples...")
    
    # Print collection statistics
    print("\n" + "=" * 80)
    print("CALIBRATION DATA COLLECTION COMPLETE")
    print("=" * 80)
    for model_name, model_stats in stats.items():
        if model_stats['total'] > 0:
            precision = model_stats['tp'] / model_stats['total']
            print(f"{model_name}: {model_stats['total']} detections "
                  f"(TP={model_stats['tp']}, FP={model_stats['fp']}, Precision={precision:.1%})")
    print()
    
    # Train calibrators
    calibrator = ProbabilityCalibrator()
    calibrator.fit(calibration_data)
    
    # Save calibrators
    calibrator.save(output_path)
    print(f"\n✓ Calibrators saved to {output_path}")
    print()
    
    # Print calibration stats
    print("Calibration Statistics:")
    for model_name, stats in calibrator.get_calibration_stats().items():
        print(f"  {model_name}:")
        print(f"    Samples: {stats['n_samples']}")
        print(f"    Confidence range: [{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]")
    
    return calibrator


def main():
    parser = argparse.ArgumentParser(
        description="Train probability calibration for multi-model detection system"
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing validation images"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Directory containing ground truth YOLO labels"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/calibrators.pkl"),
        help="Output path for trained calibrators (default: models/calibrators.pkl)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="bear",
        help="Detection prompt for Grounding DINO (default: bear)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching detections to GT (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run models on (default: cuda)"
    )
    parser.add_argument(
        "--use-gdino",
        type=bool,
        default=True,
        help="Include Grounding DINO (default: True)"
    )
    parser.add_argument(
        "--use-megadet",
        type=bool,
        default=True,
        help="Include MegaDetector v5 (default: True)"
    )
    parser.add_argument(
        "--use-detr",
        type=bool,
        default=True,
        help="Include DETR (default: True)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.images.exists():
        print(f"Error: Images directory not found: {args.images}")
        sys.exit(1)
    
    if not args.labels.exists():
        print(f"Error: Labels directory not found: {args.labels}")
        sys.exit(1)
    
    # Train calibration
    train_calibration(
        images_dir=args.images,
        labels_dir=args.labels,
        output_path=args.output,
        prompt=args.prompt,
        iou_threshold=args.iou_threshold,
        use_gdino=args.use_gdino,
        use_megadet=args.use_megadet,
        use_detr=args.use_detr,
        device=args.device
    )


if __name__ == "__main__":
    main()
