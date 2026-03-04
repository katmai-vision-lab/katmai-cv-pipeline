"""
Multi-Model Annotator for Salmon Detection (Adapted from Bear System)

This module implements a multi-model annotation system optimized for salmon:
1. Uses Grounding DINO + DETR (MegaDetector disabled - not trained for fish)
2. Checks for consensus among model predictions
3. Routes disagreements to human review

Usage:
    # 带人工审核队列（默认）
    python -m src.preprocessing.annotation_salmon.multi_model_annotator \
        --input data/frames/salmon_video/ \
        --output data/consensus_labels_salmon/ \
        --review-queue data/review_queue_salmon/ \
        --prompt "salmon"
    
    # 自动批准模式（用于训练数据生成，不需要人工审核）
    python -m src.preprocessing.annotation_salmon.multi_model_annotator \
        --input data/frames/salmon_video/ \
        --output data/auto_labels_salmon/ \
        --review-queue data/review_queue_salmon/ \
        --prompt "salmon" \
        --auto-approve
        
Note: MegaDetector v5 is disabled by default as it was trained for terrestrial animals.
"""

import argparse
from pathlib import Path
import sys
import torch
from PIL import Image
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import gc

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.annotation_salmon.auto_annotator_gdino import GroundingDINOAnnotator
from src.preprocessing.annotation_salmon.auto_annotator_megadet import MegaDetectorAnnotator
from src.preprocessing.annotation_salmon.auto_annotator_detr import DETRAnnotator
from src.preprocessing.annotation_salmon.auto_annotator_owlvit import OWLViTAnnotator
from src.preprocessing.annotation_salmon.auto_annotator_florence2 import Florence2Annotator
from src.preprocessing.annotation_salmon.probability_calibrator import ProbabilityCalibrator


@dataclass
class Detection:
    """Represents a single detection from a model."""
    box: List[float]  # [x_min, y_min, x_max, y_max] in absolute pixels
    score: float
    label: str
    model: str  # Which model produced this detection

    def to_yolo_format(self, img_width: int, img_height: int, class_id: int = 0) -> str:
        """Convert detection to YOLO format string."""
        x_min, y_min, x_max, y_max = self.box
        center_x = ((x_min + x_max) / 2) / img_width
        center_y = ((y_min + y_max) / 2) / img_height
        box_width = (x_max - x_min) / img_width
        box_height = (y_max - y_min) / img_height

        # Clamp values to [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        box_width = max(0, min(1, box_width))
        box_height = max(0, min(1, box_height))

        return f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


@dataclass
class AnnotationResult:
    """Result of multi-model annotation for a single image."""
    image_path: str
    detections_by_model: Dict[str, List[Detection]]
    consensus_detections: List[Detection]  # Detections agreed upon by models
    requires_review: bool
    review_reason: str = ""


class MultiModelAnnotator:
    """Multi-model annotator with consensus checking."""

    def __init__(
        self,
        use_gdino: bool = True,
        use_detr: bool = False,      # Disabled for salmon - low precision (35.4%)
        use_megadet: bool = False,   # Disabled for salmon (trained for terrestrial animals)
        use_owlvit: bool = True,     # Enabled - CLIP architecture, good for "jumping salmon"
        use_florence2: bool = True,  # Enabled - latest VLM, robust to water/splash
        gdino_threshold: float = 0.25,
        detr_threshold: float = 0.5,
        megadet_threshold: float = 0.3,
        owlvit_threshold: float = 0.1,
        florence2_threshold: float = 0.3,
        device: str = None,
        calibrator_path: str = None,
        stacker_path: str = None,
    ):
        """
        Initialize multi-model annotator for jumping salmon detection.
        
        Optimized 3-model system: Grounding DINO + OWL-ViT v2 + Florence-2

        Args:
            use_gdino: Whether to use Grounding DINO (text understanding)
            use_detr: Whether to use DETR (default False - low precision)
            use_megadet: Whether to use MegaDetector v5 (default False - not suitable for fish)
            use_owlvit: Whether to use OWL-ViT v2 (default True - CLIP for action understanding)
            use_florence2: Whether to use Florence-2 (default True - robust to complex scenes)
            gdino_threshold: Confidence threshold for Grounding DINO
            detr_threshold: Confidence threshold for DETR
            megadet_threshold: Confidence threshold for MegaDetector
            owlvit_threshold: Confidence threshold for OWL-ViT
            florence2_threshold: Confidence threshold for Florence-2
            device: Device to run on (auto-detect if None)
            calibrator_path: Path to trained calibrator .pkl file (optional)
            stacker_path: Path to trained stacking model .pkl file (optional)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load probability calibrator if provided
        self.calibrator = None
        if calibrator_path and Path(calibrator_path).exists():
            print(f"\n加载概率校准器: {calibrator_path}")
            self.calibrator = ProbabilityCalibrator.load(Path(calibrator_path))
            print("✓ 校准器加载成功")
        
        # Load stacking meta-learner if provided
        self.stacker = None
        if stacker_path and Path(stacker_path).exists():
            import pickle
            print(f"\n加载Stacking元学习器: {stacker_path}")
            with open(stacker_path, 'rb') as f:
                stacker_data = pickle.load(f)
            self.stacker = stacker_data['meta_learner']
            print(f"✓ Stacking加载成功 (F1={stacker_data['metrics']['f1']:.3f})")
            print("  使用机器学习优化的多模型融合策略")
        elif calibrator_path:
            print(f"\n警告: 校准器文件未找到: {calibrator_path}")
            print("将使用未校准的置信度分数")

        self.models = {}
        self.thresholds = {}
        
        # Model weights for jumping salmon detection (3-model system)
        # Optimized for: "salmon jumping out of water" scenario
        # - Grounding DINO: Best text understanding ("jumping", "leaping")
        # - OWL-ViT v2: CLIP architecture, good for action concepts
        # - Florence-2: Latest VLM, robust to water splash/reflections
        # Weights sum to 1.0 for proper probability combination
        self.model_weights = {
            'gdino': 0.40,        # Strongest text prompt understanding
            'owlvit': 0.35,       # CLIP advantage for "jumping salmon" concept
            'florence2': 0.25,    # Complex scene robustness (water, splash)
            'detr': 0.0,          # Disabled (low precision 35.4% on bears)
            'megadet': 0.0,       # Disabled (terrestrial animals only)
        }

        print("\n" + "="*80)
        print("初始化三文鱼跳跃检测系统 (Grounding DINO + OWL-ViT v2 + Florence-2)")
        print("="*80)
        print("针对场景: 从水面跃出的三文鱼")
        print("模型选择: 文本理解 + CLIP动作识别 + 复杂场景鲁棒性")
        print("="*80)

        # 串行加载模型，避免显存溢出（RTX 2080 8GB）
        # 加载顺序: 最常用 → 最不常用
        model_count = 0
        total_models = sum([use_gdino, use_owlvit, use_florence2, use_detr, use_megadet])
        
        if use_gdino:
            model_count += 1
            print(f"\n[{model_count}/{total_models}] 加载 Grounding DINO Base...")
            self.models['gdino'] = GroundingDINOAnnotator(
                model_id="IDEA-Research/grounding-dino-base",
                device=device
            )
            self.thresholds['gdino'] = gdino_threshold
            print("   显存清理中...")
            torch.cuda.empty_cache()
            gc.collect()

        if use_owlvit:
            model_count += 1
            print(f"\n[{model_count}/{total_models}] 加载 OWL-ViT v2 Base Ensemble...")
            print("   模型: google/owlv2-base-patch16-ensemble")
            print("   特点: CLIP架构, 适合动作理解 (jumping, leaping)")
            self.models['owlvit'] = OWLViTAnnotator(
                model_id="google/owlv2-base-patch16-ensemble",
                device=device
            )
            self.thresholds['owlvit'] = owlvit_threshold
            print("   显存清理中...")
            torch.cuda.empty_cache()
            gc.collect()

        if use_florence2:
            model_count += 1
            print(f"\n[{model_count}/{total_models}] 加载 Florence-2 Base...")
            print("   模型: microsoft/Florence-2-base")
            print("   特点: 最新VLM, 对水花/反光鲁棒")
            self.models['florence2'] = Florence2Annotator(
                model_size="base",
                device=device
            )
            self.thresholds['florence2'] = florence2_threshold
            print("   显存清理中...")
            torch.cuda.empty_cache()
            gc.collect()

        if use_detr:
            model_count += 1
            print(f"\n[{model_count}/{total_models}] 加载 DETR-ResNet-101...")
            self.models['detr'] = DETRAnnotator(
                model_id="facebook/detr-resnet-101",
                device=device
            )
            self.thresholds['detr'] = detr_threshold
            print("   显存清理中...")
            torch.cuda.empty_cache()
            gc.collect()

        if use_megadet:
            model_count += 1
            print(f"\n[{model_count}/{total_models}] 加载 MegaDetector v5...")
            self.models['megadet'] = MegaDetectorAnnotator(
                device=device,
                version="v5",
            )
            self.thresholds['megadet'] = megadet_threshold
            print("   显存清理中...")
            torch.cuda.empty_cache()
            gc.collect()

        print("\n" + "="*80)
        print(f"✓ 成功加载 {len(self.models)} 个模型!")
        print(f"  激活模型: {list(self.models.keys())}")
        print(f"  模型权重: {', '.join([f'{k}={v:.2f}' for k, v in self.model_weights.items() if v > 0])}")
        print("="*80 + "\n")

    def annotate_image(
        self,
        image: Image.Image,
        text_prompt: str,
    ) -> Dict[str, List[Detection]]:
        """
        Run all models on a single image.

        Args:
            image: PIL Image
            text_prompt: Text prompt for detection (e.g., "bear")

        Returns:
            Dict mapping model name to list of detections
        """
        results = {}

        # Run each model serially
        for model_name, model in self.models.items():
            if model_name == 'gdino':
                detections_raw = model.detect(
                    image=image,
                    text_prompt=text_prompt,
                    box_threshold=self.thresholds['gdino'],
                    text_threshold=self.thresholds['gdino'],
                )
            elif model_name == 'owlvit':
                detections_raw = model.detect(
                    image=image,
                    text_queries=[text_prompt],  # OWL-ViT expects list
                    threshold=self.thresholds['owlvit'],
                )
            elif model_name == 'florence2':
                detections_raw = model.detect(
                    image=image,
                    text_prompt=text_prompt,
                )
            elif model_name == 'detr':
                detections_raw = model.detect(
                    image=image,
                    threshold=self.thresholds['detr'],
                )
            elif model_name == 'megadet':
                detections_raw = model.detect(
                    image=image,
                    threshold=self.thresholds['megadet'],
                )
            else:
                continue

            # Convert to Detection objects
            img_width, img_height = image.size
            detections = []
            for d in detections_raw:
                # Calculate box area (for filtering)
                box = d['box']
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                box_area = box_width * box_height
                image_area = img_width * img_height
                area_ratio = box_area / image_area
                
                # Skip detections that cover more than 80% of the image
                # (likely scene labels like "water", not actual objects)
                if area_ratio > 0.8:
                    continue
                
                detections.append(
                    Detection(
                        box=box,
                        score=d.get('score', 1.0),
                        label=d.get('label', text_prompt),
                        model=model_name
                    )
                )
            results[model_name] = detections

            # Clear cache after each model
            torch.cuda.empty_cache()

        return results

    def _extract_stacking_features(
        self,
        detection: Detection,
        all_detections: List[Detection],
        img_width: int,
        img_height: int,
    ) -> 'np.ndarray':
        """
        Extract features for stacking meta-learner.
        
        Features (11 total):
        - Model one-hot (gdino, owlvit, florence2): 3
        - Raw confidence: 1
        - Box size (width, height): 2
        - Box position (center_x, center_y): 2
        - Max IoU with other models: 1
        - Number of overlapping detections: 1
        - Average confidence of overlaps: 1
        """
        import numpy as np
        
        features = []
        
        # 1. Model one-hot
        model_map = {'gdino': 0, 'owlvit': 1, 'florence2': 2}
        model_onehot = [0, 0, 0]
        if detection.model in model_map:
            model_onehot[model_map[detection.model]] = 1
        features.extend(model_onehot)
        
        # 2. Raw confidence
        features.append(detection.score)
        
        # 3. Box size (normalized)
        box = detection.box
        box_width = (box[2] - box[0]) / img_width
        box_height = (box[3] - box[1]) / img_height
        features.extend([box_width, box_height])
        
        # 4. Box position (normalized)
        center_x = ((box[0] + box[2]) / 2) / img_width
        center_y = ((box[1] + box[3]) / 2) / img_height
        features.extend([center_x, center_y])
        
        # 5. IoU features
        max_iou = 0.0
        num_overlaps = 0
        overlap_confidences = []
        
        for other_det in all_detections:
            if other_det.model == detection.model:
                continue
            
            iou = self._calculate_iou(detection.box, other_det.box)
            max_iou = max(max_iou, iou)
            
            if iou > 0.5:
                num_overlaps += 1
                overlap_confidences.append(other_det.score)
        
        features.append(max_iou)
        features.append(num_overlaps)
        features.append(np.mean(overlap_confidences) if overlap_confidences else 0.0)
        
        return np.array(features)

    def check_consensus(
        self,
        detections_by_model: Dict[str, List[Detection]],
        img_width: int,
        img_height: int,
        iou_threshold: float = 0.5,
        min_agreement: int = 2,
    ) -> Tuple[List[Detection], bool, str]:
        """
        Check consensus among model predictions.
        
        If stacker is available, uses ML-based fusion.
        Otherwise, uses traditional voting-based consensus.

        Args:
            detections_by_model: Dict mapping model name to detections
            img_width: Image width (for stacking features)
            img_height: Image height (for stacking features)
            iou_threshold: IoU threshold for matching boxes
            min_agreement: Minimum number of models that must agree (ignored if stacker is used)

        Returns:
            Tuple of (consensus_detections, requires_review, review_reason)
        """
        # Get all detections from all models
        all_detections = []
        for model_name, detections in detections_by_model.items():
            all_detections.extend(detections)

        if len(all_detections) == 0:
            return [], False, ""

        # Use Stacking if available
        if self.stacker is not None:
            import numpy as np
            
            # Extract features for all detections
            X = []
            for det in all_detections:
                features = self._extract_stacking_features(det, all_detections, img_width, img_height)
                X.append(features)
            
            X = np.array(X)
            
            # Predict with stacker
            predictions = self.stacker.predict(X)
            probabilities = self.stacker.predict_proba(X)[:, 1]  # Probability of being TP
            
            # Keep detections predicted as True Positives
            consensus_detections = []
            for det, pred, prob in zip(all_detections, predictions, probabilities):
                if pred == 1:  # Predicted as True Positive
                    # Update detection score with stacker probability
                    det.score = prob
                    consensus_detections.append(det)
            
            # Stacking already filters, no review needed
            return consensus_detections, False, ""

        # Traditional voting-based consensus (fallback)
        # Group detections by IoU overlap
        detection_groups = self._group_detections_by_iou(
            all_detections,
            iou_threshold
        )

        consensus_detections = []
        disagreements = []

        for group in detection_groups:
            # Count unique models in this group
            models_in_group = set(d.model for d in group)
            num_agreeing_models = len(models_in_group)

            if num_agreeing_models >= min_agreement:
                # Consensus reached - use weighted score (model_weight * confidence)
                def weighted_score(d: Detection) -> float:
                    # Apply probability calibration if available
                    score = d.score
                    if self.calibrator:
                        score = self.calibrator.calibrate(d.model, score)
                    return self.model_weights.get(d.model, 0.33) * score
                
                best_detection = max(group, key=weighted_score)
                consensus_detections.append(best_detection)
            else:
                # Not enough models agree
                disagreements.append(group)

        # Only flag for review when there are high-confidence disagreements
        # Low-confidence solo detections are likely false positives, just ignore them
        high_conf_disagreements = []
        for group in disagreements:
            max_score = max(d.score for d in group)
            if max_score >= 0.5:
                high_conf_disagreements.append(group)

        requires_review = len(high_conf_disagreements) > 0
        review_reason = ""

        if requires_review:
            review_reason = (
                f"Found {len(high_conf_disagreements)} high-confidence detection(s) "
                f"with insufficient agreement (< {min_agreement} models)"
            )

        return consensus_detections, requires_review, review_reason

    def _group_detections_by_iou(
        self,
        detections: List[Detection],
        iou_threshold: float,
    ) -> List[List[Detection]]:
        """
        Group detections that overlap significantly (by IoU).

        Args:
            detections: List of detections
            iou_threshold: IoU threshold for grouping

        Returns:
            List of detection groups
        """
        if not detections:
            return []

        # Simple greedy grouping algorithm
        groups = []
        remaining = list(detections)

        while remaining:
            # Start a new group with the first remaining detection
            current = remaining.pop(0)
            group = [current]

            # Find all detections that overlap with this one
            i = 0
            while i < len(remaining):
                if self._calculate_iou(current.box, remaining[i].box) >= iou_threshold:
                    group.append(remaining.pop(i))
                else:
                    i += 1

            groups.append(group)

        return groups

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU (Intersection over Union) between two boxes.

        Args:
            box1, box2: Boxes in [x_min, y_min, x_max, y_max] format

        Returns:
            IoU score (0.0 to 1.0)
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area


def auto_annotate_multi_model(
    input_dir: str,
    output_dir: str,
    review_queue_dir: str,
    text_prompt: str = "bear",
    iou_threshold: float = 0.5,
    min_agreement: int = 2,
    limit: int = None,
    auto_approve: bool = False,
    calibrator_path: str = None,
    stacker_path: str = None,
    gdino_threshold: float = 0.25,
    owlvit_threshold: float = 0.1,
    florence2_threshold: float = 0.3,
):
    """
    使用多模型对图像进行标注，并通过一致性检查提高标注质量。

    Args:
        input_dir: 输入图像目录
        output_dir: 一致性标注结果输出目录
        review_queue_dir: 需要人工审核的样本保存目录（auto_approve=True时不使用）
        text_prompt: 检测目标文本提示
        iou_threshold: IoU阈值，用于判断检测框是否匹配
        min_agreement: 最少需要几个模型同意 (默认: 3个中2个, Stacking时忽略)
        limit: 最大处理图像数量
        auto_approve: 自动批准模式，只保存达成共识的检测，跳过人工审核（用于训练数据生成）
        calibrator_path: 校准器文件路径（可选，使用校准后的概率）
        stacker_path: Stacking元学习器文件路径（可选，使用ML优化的多模型融合）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    review_queue_dir = Path(review_queue_dir)

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 只在非自动批准模式下创建review queue
    if not auto_approve:
        review_queue_dir.mkdir(parents=True, exist_ok=True)
        (review_queue_dir / "images").mkdir(exist_ok=True)
        (review_queue_dir / "detections").mkdir(exist_ok=True)

    # 初始化标注器
    annotator = MultiModelAnnotator(
        calibrator_path=calibrator_path,
        stacker_path=stacker_path,
        gdino_threshold=gdino_threshold,
        owlvit_threshold=owlvit_threshold,
        florence2_threshold=florence2_threshold,
    )

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []

    for f in input_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in image_extensions:
            image_files.append(f)

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    image_files = sorted(image_files, key=lambda x: x.name)

    if limit:
        image_files = image_files[:limit]

    mode = "自动批准模式 (训练数据生成)" if auto_approve else "共识检查模式 (含人工审核)"
    print(f"\nMulti-Model Auto Annotation with Consensus Checking")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    if not auto_approve:
        print(f"Review queue: {review_queue_dir}")
    print(f"Text prompt: \"{text_prompt}\"")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Min agreement: {min_agreement}/{len(annotator.models)} models")
    print(f"Images found: {len(image_files)}")
    print(f"{'='*60}\n")

    stats = {
        "total": len(image_files),
        "consensus": 0,
        "needs_review": 0,
        "total_detections": 0,
    }

    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_path.name}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        # Run all models
        detections_by_model = annotator.annotate_image(image, text_prompt)

        # Check consensus
        consensus_detections, requires_review, review_reason = annotator.check_consensus(
            detections_by_model,
            img_width=img_width,
            img_height=img_height,
            iou_threshold=iou_threshold,
            min_agreement=min_agreement,
        )

        # Print results by model
        for model_name, detections in detections_by_model.items():
            print(f"  {model_name}: {len(detections)} detection(s)")

        if requires_review:
            if auto_approve:
                # 自动批准模式：忽略需要审核的，只保存有共识的
                print(f"  ⚠️ Skipped (no consensus): {review_reason}")
                stats["needs_review"] += 1
            else:
                # 人工审核模式：保存到review queue
                print(f"  ⚠️  NEEDS REVIEW: {review_reason}")
                stats["needs_review"] += 1

                # Save to review queue
                review_data = {
                    "image_path": str(image_path),
                    "text_prompt": text_prompt,
                    "detections_by_model": {
                        model: [asdict(d) for d in dets]
                        for model, dets in detections_by_model.items()
                    },
                    "review_reason": review_reason,
                }

                review_file = review_queue_dir / "detections" / f"{image_path.stem}.json"
                with open(review_file, "w") as f:
                    json.dump(review_data, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))

                # Copy image to review queue
                import shutil
                shutil.copy(image_path, review_queue_dir / "images" / image_path.name)
        else:
            print(f"  ✓ Consensus: {len(consensus_detections)} detection(s)")
            stats["consensus"] += 1

        # Save consensus labels (even if needs review, save what we have)
        label_file = output_dir / f"{image_path.stem}.txt"
        width, height = image.size

        with open(label_file, "w") as f:
            for det in consensus_detections:
                yolo_line = det.to_yolo_format(width, height)
                f.write(yolo_line + "\n")

        stats["total_detections"] += len(consensus_detections)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Multi-Model Annotation Complete!")
    print(f"{'='*60}")
    print(f"Total images: {stats['total']}")
    print(f"Consensus reached: {stats['consensus']} ({stats['consensus']/stats['total']*100:.1f}%)")
    if auto_approve:
        print(f"Skipped (no consensus): {stats['needs_review']} ({stats['needs_review']/stats['total']*100:.1f}%)")
    else:
        print(f"Needs review: {stats['needs_review']} ({stats['needs_review']/stats['total']*100:.1f}%)")
    print(f"Total consensus detections: {stats['total_detections']}")
    print(f"\nLabels saved to: {output_dir}")
    if not auto_approve:
        print(f"Review queue saved to: {review_queue_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model auto-annotation with consensus checking"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for consensus labels"
    )
    parser.add_argument(
        "--review-queue", "-r",
        required=True,
        help="Directory to save images/labels needing review"
    )
    parser.add_argument(
        "--prompt", "-p",
        default="jumping salmon",
        help="Text prompt for detection (default: 'jumping salmon')"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching detections (default: 0.5)"
    )
    parser.add_argument(
        "--min-agreement",
        type=int,
        default=2,
        help="Minimum number of models that must agree (default: 2 for 3-model system)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max number of images to process (default: all)"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="自动批准模式：只保存达成共识的检测，跳过人工审核（用于生成训练数据）"
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        default=None,
        help="Path to trained probability calibrator .pkl file (optional)"
    )
    parser.add_argument(
        "--stacker",
        type=str,
        default=None,
        help="Path to trained stacking meta-learner .pkl file (optional, overrides min-agreement)"
    )
    parser.add_argument("--gdino-threshold", type=float, default=0.25,
                        help="Grounding DINO confidence threshold (default: 0.25)")
    parser.add_argument("--owlvit-threshold", type=float, default=0.1,
                        help="OWL-ViT confidence threshold (default: 0.1)")
    parser.add_argument("--florence2-threshold", type=float, default=0.3,
                        help="Florence-2 confidence threshold (default: 0.3)")
    args = parser.parse_args()

    auto_annotate_multi_model(
        input_dir=args.input,
        output_dir=args.output,
        review_queue_dir=args.review_queue,
        text_prompt=args.prompt,
        iou_threshold=args.iou_threshold,
        min_agreement=args.min_agreement,
        limit=args.limit,
        auto_approve=args.auto_approve,
        calibrator_path=args.calibrator,
        stacker_path=args.stacker,
        gdino_threshold=args.gdino_threshold,
        owlvit_threshold=args.owlvit_threshold,
        florence2_threshold=args.florence2_threshold,
    )


if __name__ == "__main__":
    main()
