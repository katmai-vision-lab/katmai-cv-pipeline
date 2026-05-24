#!/usr/bin/env python3
"""
Predict using trained Stacking Meta-Learner
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import joblib

# Import annotators
from .auto_annotator_gdino import GroundingDINOAnnotator
from .auto_annotator_owlvit import OWLViTAnnotator
from .auto_annotator_florence2 import Florence2Annotator


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes [x_min, y_min, x_max, y_max]."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    
    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def extract_detection_features(
    detection: Dict,
    model_name: str,
    all_detections: List[Tuple[str, Dict]],
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Extract features for a single detection."""
    features = []
    
    # 1. Model one-hot encoding
    model_map = {'gdino': 0, 'owlvit': 1, 'florence2': 2}
    model_onehot = [0, 0, 0]
    if model_name in model_map:
        model_onehot[model_map[model_name]] = 1
    features.extend(model_onehot)
    
    # 2. Raw confidence
    features.append(detection['score'])
    
    # 3. Box size (normalized)
    box = detection['box']
    box_width = (box[2] - box[0]) / img_width
    box_height = (box[3] - box[1]) / img_height
    features.extend([box_width, box_height])
    
    # 4. Box position (normalized center)
    center_x = ((box[0] + box[2]) / 2) / img_width
    center_y = ((box[1] + box[3]) / 2) / img_height
    features.extend([center_x, center_y])
    
    # 5. IoU features with other models' detections
    max_iou = 0.0
    num_overlaps = 0
    overlap_confidences = []
    
    for other_model, other_det in all_detections:
        if other_model == model_name:
            continue
        
        iou = calculate_iou(detection['box'], other_det['box'])
        max_iou = max(max_iou, iou)
        
        if iou > 0.5:
            num_overlaps += 1
            overlap_confidences.append(other_det['score'])
    
    features.append(max_iou)
    features.append(num_overlaps)
    features.append(np.mean(overlap_confidences) if overlap_confidences else 0.0)
    
    return np.array(features)


def predict_with_stacking(
    images_dir: Path,
    stacker_path: Path,
    output_dir: Path,
    prompt: str = "fish",
    confidence_threshold: float = 0.5,
    device: str = "cuda",
    visualize: bool = True
):
    """
    Run inference using stacking meta-learner.
    
    Args:
        images_dir: Directory with test images
        stacker_path: Path to trained stacker (.pkl)
        output_dir: Output directory for labels and visualizations
        prompt: Detection prompt
        confidence_threshold: Minimum stacking confidence (0-1)
        device: Device for models
        visualize: Whether to create visualizations
    """
    print("="*70)
    print("Stacking Meta-Learner Inference")
    print("="*70)
    print(f"Images: {images_dir}")
    print(f"Stacker: {stacker_path}")
    print(f"Output: {output_dir}")
    print(f"Prompt: {prompt}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("="*70)
    
    # Create output directories
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_dir = output_dir / "visualized"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load stacking model
    print("\n[1/3] Loading stacking model...")
    with open(stacker_path, 'rb') as f:
        stacker_data = pickle.load(f)
    
    stacker = stacker_data['meta_learner']
    print(f"Stacker loaded successfully!")
    print(f"  Trained on prompt: {stacker_data.get('prompt', 'unknown')}")
    if 'metrics' in stacker_data:
        metrics = stacker_data['metrics']
        print(f"  Training metrics: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    # Load base models
    print("\n[2/3] Loading base models...")
    gdino = GroundingDINOAnnotator(device=device)
    owlvit = OWLViTAnnotator(device=device)
    florence = Florence2Annotator(model_size='base', device=device)
    
    # Process images
    print("\n[3/3] Running inference...")
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    stats = {
        'total_images': 0,
        'images_with_detections': 0,
        'total_detections': 0,
        'gdino_detections': 0,
        'owlvit_detections': 0,
        'florence2_detections': 0,
    }
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        stats['total_images'] += 1
        
        # Get detections from all models
        all_detections = []
        
        # Grounding DINO
        dets_g = gdino.detect(img, text_prompt=prompt, box_threshold=0.25)
        for det in dets_g:
            all_detections.append(('gdino', det))
        stats['gdino_detections'] += len(dets_g)
        
        # OWL-ViT
        dets_o = owlvit.detect(img, text_queries=[prompt], threshold=0.1)
        for det in dets_o:
            all_detections.append(('owlvit', det))
        stats['owlvit_detections'] += len(dets_o)
        
        # Florence-2 (with area filtering)
        dets_f = florence.detect(img, text_prompt=prompt, use_grounding=True)
        img_area = img_width * img_height
        for det in dets_f:
            box_area = (det['box'][2] - det['box'][0]) * (det['box'][3] - det['box'][1])
            if box_area < 0.8 * img_area:  # Filter large boxes
                all_detections.append(('florence2', det))
        stats['florence2_detections'] += len([d for d in dets_f if (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]) < 0.8*img_area])
        
        # Extract features and predict
        final_detections = []
        
        for model_name, detection in all_detections:
            features = extract_detection_features(
                detection, model_name, all_detections, img_width, img_height
            )
            
            # Predict probability
            prob = stacker.predict_proba(features.reshape(1, -1))[0][1]
            
            if prob >= confidence_threshold:
                final_detections.append({
                    'model': model_name,
                    'box': detection['box'],
                    'confidence': prob,
                    'raw_score': detection['score']
                })
        
        stats['total_detections'] += len(final_detections)
        if len(final_detections) > 0:
            stats['images_with_detections'] += 1
        
        # Save labels (YOLO format)
        label_path = labels_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            for det in final_detections:
                box = det['box']
                # Convert to YOLO format: class_id center_x center_y width height
                center_x = ((box[0] + box[2]) / 2) / img_width
                center_y = ((box[1] + box[3]) / 2) / img_height
                width = (box[2] - box[0]) / img_width
                height = (box[3] - box[1]) / img_height
                f.write(f"0 {center_x} {center_y} {width} {height}\n")
        
        # Visualize
        if visualize and len(final_detections) > 0:
            vis_img = img.copy()
            draw = ImageDraw.Draw(vis_img)
            
            for det in final_detections:
                box = det['box']
                
                # Draw box
                draw.rectangle(box, outline='lime', width=3)
                
                # Draw label
                label = f"{det['model']}: {det['confidence']:.2f}"
                draw.text((box[0], box[1] - 20), label, fill='lime')
            
            vis_img.save(vis_dir / img_path.name)
    
    # Print statistics
    print("\n" + "="*70)
    print("Inference Complete!")
    print("="*70)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']} ({stats['images_with_detections']/stats['total_images']*100:.1f}%)")
    print(f"\nBase model detections (before stacking):")
    print(f"  Grounding DINO: {stats['gdino_detections']}")
    print(f"  OWL-ViT v2:     {stats['owlvit_detections']}")
    print(f"  Florence-2:     {stats['florence2_detections']}")
    print(f"  Total:          {stats['gdino_detections'] + stats['owlvit_detections'] + stats['florence2_detections']}")
    print(f"\nFinal detections (after stacking filter):")
    print(f"  Kept: {stats['total_detections']}")
    print(f"  Filtered: {stats['gdino_detections'] + stats['owlvit_detections'] + stats['florence2_detections'] - stats['total_detections']}")
    print(f"  Avg per image: {stats['total_detections'] / stats['images_with_detections']:.2f}" if stats['images_with_detections'] > 0 else "  Avg per image: 0.00")
    print(f"\nOutput saved to: {output_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with stacking meta-learner"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory with test images"
    )
    parser.add_argument(
        "--stacker",
        type=str,
        required=True,
        help="Path to trained stacker (.pkl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for labels and visualizations"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="fish",
        help="Detection prompt"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum stacking confidence threshold (0-1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run models on: cuda / mps / cpu (default: auto-detect)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations"
    )
    
    args = parser.parse_args()

    if args.device is None:
        from src.config import get_device
        args.device = get_device()

    predict_with_stacking(
        images_dir=Path(args.images),
        stacker_path=Path(args.stacker),
        output_dir=Path(args.output),
        prompt=args.prompt,
        confidence_threshold=args.confidence,
        device=args.device,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
