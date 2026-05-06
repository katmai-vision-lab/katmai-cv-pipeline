#!/usr/bin/env python3
"""
Train Stacking Meta-Learner for Multi-Model Detection Fusion

Stacking Strategy:
1. Collect predictions from all base models (GDINO, OWL-ViT, Florence-2)
2. Extract features for each detection:
   - Model ID (one-hot encoding)
   - Raw confidence score
   - IoU overlap with other models' detections
   - Box size (normalized)
   - Box position (center coordinates)
   - Number of overlapping detections from other models
3. Train a meta-learner (Random Forest / Gradient Boosting) to predict:
   - Whether this detection is a true positive
   - Final confidence score
4. Use trained stacker for inference

This approach learns optimal fusion weights from validation data.
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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
    
    return inter_area / union_area if union_area > 0 else 0.0


def extract_detection_features(
    detection: Dict,
    model_name: str,
    all_detections: List[Tuple[str, Dict]],
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Extract features for a single detection for stacking.
    
    Features:
    - Model one-hot (gdino, owlvit, florence2): 3 features
    - Raw confidence score: 1 feature
    - Box normalized size (width, height): 2 features
    - Box normalized position (center_x, center_y): 2 features
    - Max IoU with other models: 1 feature
    - Number of overlapping detections (IoU > 0.5): 1 feature
    - Average confidence of overlapping detections: 1 feature
    
    Total: 11 features
    """
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
            continue  # Skip same model
        
        iou = calculate_iou(detection['box'], other_det['box'])
        max_iou = max(max_iou, iou)
        
        if iou > 0.5:
            num_overlaps += 1
            overlap_confidences.append(other_det['score'])
    
    features.append(max_iou)
    features.append(num_overlaps)
    features.append(np.mean(overlap_confidences) if overlap_confidences else 0.0)
    
    return np.array(features)


def load_ground_truth(label_path: Path, img_width: int, img_height: int) -> List[List[float]]:
    """Load ground truth boxes from YOLO format."""
    gt_boxes = []
    
    if not label_path.exists():
        return gt_boxes
    
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # YOLO format: class_id center_x center_y width height (normalized)
            _, cx, cy, w, h = map(float, parts[:5])
            
            # Convert to absolute coordinates [x_min, y_min, x_max, y_max]
            x_min = (cx - w/2) * img_width
            y_min = (cy - h/2) * img_height
            x_max = (cx + w/2) * img_width
            y_max = (cy + h/2) * img_height
            
            gt_boxes.append([x_min, y_min, x_max, y_max])
    
    return gt_boxes


def is_true_positive(detection_box: List[float], gt_boxes: List[List[float]], iou_threshold: float = 0.5) -> bool:
    """Check if detection matches any ground truth box."""
    for gt_box in gt_boxes:
        if calculate_iou(detection_box, gt_box) >= iou_threshold:
            return True
    return False


def train_stacking_meta_learner(
    images_dir: Path,
    labels_dir: Path,
    output_path: Path,
    prompt: str = "jumping salmon",
    iou_threshold: float = 0.5,
    meta_learner: str = "rf",
    device: str = "cuda",
):
    """
    Train stacking meta-learner on validation dataset.
    
    Args:
        images_dir: Directory with validation images
        labels_dir: Directory with ground truth labels (YOLO format)
        output_path: Path to save trained stacker (.pkl)
        prompt: Detection prompt
        iou_threshold: IoU threshold for TP/FP determination
        meta_learner: 'rf' (Random Forest), 'gb' (Gradient Boosting), or 'lr' (Logistic Regression)
        device: Device for models
    """
    print("="*70)
    print("Stacking Meta-Learner Training")
    print("="*70)
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Prompt: {prompt}")
    print(f"Meta-learner: {meta_learner}")
    print("="*70)
    
    # Load models
    print("\n[1/4] Loading base models...")
    gdino = GroundingDINOAnnotator(device=device)
    owlvit = OWLViTAnnotator(device=device)
    florence = Florence2Annotator(model_size='base', device=device)
    
    models = {
        'gdino': gdino,
        'owlvit': owlvit,
        'florence2': florence
    }
    
    # Collect training data
    print("\n[2/4] Extracting features from validation set...")
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    
    X = []  # Features
    y = []  # Labels (1 = TP, 0 = FP)
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Load image
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Load ground truth
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_ground_truth(label_path, img_width, img_height)
        
        if len(gt_boxes) == 0:
            print(f"Warning: No ground truth for {img_path.name}, skipping...")
            continue
        
        # Get detections from all models
        all_detections = []
        
        # GDINO
        dets_g = gdino.detect(img, text_prompt=prompt, box_threshold=0.25, text_threshold=0.25)
        for det in dets_g:
            all_detections.append(('gdino', det))
        
        # OWL-ViT
        dets_o = owlvit.detect(img, text_queries=[prompt], threshold=0.1)
        for det in dets_o:
            all_detections.append(('owlvit', det))
        
        # Florence-2 (with area filtering)
        dets_f = florence.detect(img, text_prompt=prompt, use_grounding=True)
        img_area = img_width * img_height
        for det in dets_f:
            box_area = (det['box'][2] - det['box'][0]) * (det['box'][3] - det['box'][1])
            if box_area < 0.8 * img_area:  # Filter large boxes
                all_detections.append(('florence2', det))
        
        # Extract features for each detection
        for model_name, detection in all_detections:
            features = extract_detection_features(
                detection, model_name, all_detections, img_width, img_height
            )
            label = 1 if is_true_positive(detection['box'], gt_boxes, iou_threshold) else 0
            
            X.append(features)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset collected:")
    print(f"  Total detections: {len(X)}")
    print(f"  True Positives: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  False Positives: {len(y) - y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
    print(f"  Feature dimension: {X.shape[1]}")
    
    # Train meta-learner
    print("\n[3/4] Training meta-learner...")
    
    # Split for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if meta_learner == "rf":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
    elif meta_learner == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    elif meta_learner == "lr":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown meta-learner: {meta_learner}")
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\n[4/4] Evaluating meta-learner...")
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    
    print(f"\nValidation Performance:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    
    # Feature importance (if available)
    if hasattr(clf, 'feature_importances_'):
        feature_names = [
            'model_gdino', 'model_owlvit', 'model_florence2',
            'confidence', 'box_width', 'box_height',
            'center_x', 'center_y',
            'max_iou', 'num_overlaps', 'avg_overlap_conf'
        ]
        importances = clf.feature_importances_
        print("\nFeature Importances:")
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:5]:
            print(f"  {name}: {imp:.3f}")
    
    # Save stacker
    print(f"\nSaving stacking model to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stacker_data = {
        'meta_learner': clf,
        'prompt': prompt,
        'iou_threshold': iou_threshold,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(stacker_data, f)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Train stacking meta-learner for multi-model detection fusion"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory with validation images"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Directory with ground truth labels (YOLO format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/stacker_salmon.pkl",
        help="Output path for trained stacker"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="jumping salmon",
        help="Detection prompt"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to ground truth"
    )
    parser.add_argument(
        "--meta-learner",
        type=str,
        choices=['rf', 'gb', 'lr'],
        default='rf',
        help="Meta-learner type: rf (Random Forest), gb (Gradient Boosting), lr (Logistic Regression)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run models on"
    )
    
    args = parser.parse_args()
    
    train_stacking_meta_learner(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        output_path=Path(args.output),
        prompt=args.prompt,
        iou_threshold=args.iou_threshold,
        meta_learner=args.meta_learner,
        device=args.device
    )


if __name__ == "__main__":
    main()
