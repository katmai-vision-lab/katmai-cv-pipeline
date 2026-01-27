"""Bear detection evaluation metrics"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


class DetectionMetrics:
    """Calculate detection metrics"""
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def match_detections(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """Match predictions to ground truth by IoU"""
        if not pred_boxes:
            return [], [], list(range(len(gt_boxes)))
        if not gt_boxes:
            return [], list(range(len(pred_boxes))), []
        
        iou_matrix = np.array([[self.calculate_iou(p[:4], g) for g in gt_boxes] 
                               for p in pred_boxes])
        
        matches, matched_preds, matched_gts = [], set(), set()
        pred_indices = sorted(range(len(pred_boxes)), 
                            key=lambda i: pred_boxes[i][4], reverse=True)
        
        for pred_idx in pred_indices:
            best_gt = -1
            best_iou = 0
            for gt_idx in range(len(gt_boxes)):
                if gt_idx not in matched_gts and iou_matrix[pred_idx, gt_idx] > best_iou:
                    if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
                        best_iou = iou_matrix[pred_idx, gt_idx]
                        best_gt = gt_idx
            
            if best_gt >= 0:
                matches.append((pred_idx, best_gt, best_iou))
                matched_preds.add(pred_idx)
                matched_gts.add(best_gt)
        
        unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
        unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]
        
        return matches, unmatched_preds, unmatched_gts
    
    def calc_metrics(self, tp, fp, fn):
        """Calculate precision, recall, F1"""
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1
    
    def evaluate_frame(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """Evaluate single frame"""
        matches, fp_idx, fn_idx = self.match_detections(pred_boxes, gt_boxes, iou_threshold)
        tp, fp, fn = len(matches), len(fp_idx), len(fn_idx)
        p, r, f1 = self.calc_metrics(tp, fp, fn)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f1}


class VideoEvaluator:
    """Evaluate detection on videos"""
    
    def __init__(self, detector, conf_threshold=0.25):
        self.detector = detector
        self.conf_threshold = conf_threshold
        self.metrics_calc = DetectionMetrics()
    
    def evaluate_dataset_with_yolo(self, data_yaml, save_dir=None):
        """Use YOLO's built-in validation for mAP, precision, recall"""
        print(f"\n{'='*70}")
        print("YOLO NATIVE VALIDATION")
        print(f"{'='*70}\n")
        
        metrics = self.detector.model.val(data=data_yaml, conf=self.conf_threshold, 
                                         iou=0.6, verbose=True)
        
        results = {
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'precision': float(metrics.box.p),
            'recall': float(metrics.box.r),
            'f1': float(2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r)) 
                  if (metrics.box.p + metrics.box.r) > 0 else 0,
        }
        
        print(f"\n{'='*70}")
        print(f"Precision:     {results['precision']:.4f}")
        print(f"Recall:        {results['recall']:.4f}")
        print(f"F1 Score:      {results['f1']:.4f}")
        print(f"mAP@0.5:       {results['map50']:.4f}")
        print(f"mAP@0.5:0.95:  {results['map50_95']:.4f}")
        print(f"{'='*70}\n")
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = Path(save_dir) / f"yolo_val_{ts}.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Metrics saved: {json_path}\n")
        
        return results
    
    def evaluate_counting_accuracy(self, video_path, ground_truth_counts, 
                                   frame_skip=1, save_dir=None):
        """Evaluate bear counting accuracy"""
        print(f"\n{'='*70}")
        print("COUNTING ACCURACY EVALUATION")
        print(f"{'='*70}\n")
        
        is_constant = isinstance(ground_truth_counts, int)
        if is_constant:
            print(f"Ground Truth: {ground_truth_counts} bears (constant)")
        
        results = self.detector.model.predict(source=str(video_path), 
                                             conf=self.conf_threshold,
                                             stream=True, verbose=False)
        
        frame_data = []
        for frame_id, result in enumerate(results):
            if frame_id % frame_skip != 0:
                continue
            
            num_detected = len(result.boxes)
            gt_count = ground_truth_counts if is_constant else ground_truth_counts.get(frame_id)
            if gt_count is None:
                continue
            
            confs = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else []
            frame_data.append({
                'frame': frame_id,
                'detected_count': num_detected,
                'ground_truth': gt_count,
                'absolute_error': abs(num_detected - gt_count),
                'is_correct': (num_detected == gt_count),
                'avg_confidence': np.mean(confs) if len(confs) > 0 else 0,
            })
        
        df = pd.DataFrame(frame_data)
        accuracy = (df['is_correct'].sum() / len(df)) * 100
        mae = df['absolute_error'].mean()
        rmse = np.sqrt((df['absolute_error'] ** 2).mean())
        
        print(f"Frames: {len(df)}")
        print(f"\nCounting Accuracy:")
        print(f"  Exact Match:  {accuracy:.2f}%")
        print(f"  MAE:          {mae:.3f} bears")
        print(f"  RMSE:         {rmse:.3f} bears")
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = Path(save_dir) / f"counting_{ts}.csv"
            df.to_csv(csv_path, index=False)
            self._plot_counting(df, Path(save_dir), ts)
            print(f"\n✓ Results saved: {csv_path}")
        
        print(f"{'='*70}\n")
        return df
    
    def _plot_counting(self, df, save_dir, timestamp):
        """Plot counting metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Detected vs GT
        axes[0, 0].plot(df['frame'], df['detected_count'], label='Detected', alpha=0.7)
        axes[0, 0].plot(df['frame'], df['ground_truth'], label='Ground Truth', 
                       alpha=0.7, linestyle='--')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Bear Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error
        axes[0, 1].plot(df['frame'], df['absolute_error'], color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution
        axes[1, 0].hist(df['absolute_error'], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Absolute Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Confidence vs accuracy
        correct = df[df['is_correct']]
        incorrect = df[~df['is_correct']]
        if len(correct) > 0:
            axes[1, 1].scatter(correct['avg_confidence'], correct['detected_count'],
                             label='Correct', alpha=0.5, color='green')
        if len(incorrect) > 0:
            axes[1, 1].scatter(incorrect['avg_confidence'], incorrect['detected_count'],
                             label='Incorrect', alpha=0.5, color='red')
        axes[1, 1].set_xlabel('Avg Confidence')
        axes[1, 1].set_ylabel('Detected Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"counting_{timestamp}.png", dpi=150)
        print(f"✓ Plot saved: counting_{timestamp}.png")
        plt.close()
