# Evaluation Metrics Protocol

This document describes the evaluation metrics framework implemented in `src/detection/metrics.py`. It explains what each metric measures, why it was chosen, and how results should be interpreted to guide model development and validation. These metrics map directly to project milestones: M2 (Detection Baseline) uses mAP, precision, and recall, while M4 (Counting MVP) uses MAE, RMSE, and exact match accuracy.

## Detection Metrics

Detection metrics evaluate how well the model localizes and identifies bears in individual frames. The core metrics are precision, recall, F1, and mean average precision (mAP).

**Precision** measures what fraction of the model's detections are correct. It is calculated as TP / (TP + FP), where TP (true positives) are detections that match a ground truth bear with sufficient overlap, and FP (false positives) are detections that don't match any ground truth. High precision means the model rarely produces false alarms, which is critical for avoiding over-counting bears.

**Recall** measures what fraction of actual bears the model successfully detects. It is calculated as TP / (TP + FN), where FN (false negatives) are ground truth bears that the model missed. High recall means the model rarely misses bears, which is critical for accurate population monitoring.

**F1 Score** is the harmonic mean of precision and recall, calculated as 2 × (Precision × Recall) / (Precision + Recall). It provides a single balanced metric that penalizes models with extreme imbalance between precision and recall. A model with 90% precision but 50% recall will have a lower F1 than one with 70% in both.

**mAP@0.5** (mean Average Precision at IoU 0.5) is the standard PASCAL VOC detection metric. A detection is considered correct if it overlaps with ground truth by at least 50% (IoU ≥ 0.5). This threshold is relatively lenient, allowing for some localization error while still requiring substantial overlap. mAP@0.5 summarizes model performance across all confidence thresholds by computing the area under the precision-recall curve.

**mAP@0.5:0.95** is the COCO benchmark standard, which averages mAP across IoU thresholds from 0.5 to 0.95 in steps of 0.05. This metric rewards precise localization more heavily than mAP@0.5 because a model must perform well even when strict overlap is required. In our context, mAP@0.5:0.95 will typically be lower than mAP@0.5 due to challenging conditions like distant bears and motion blur.

## Counting Metrics

Counting metrics evaluate how accurately the model counts bears across video frames, independent of localization quality.

**MAE (Mean Absolute Error)** is the average absolute difference between predicted and ground truth bear counts across all evaluated frames. For example, if the model predicts 4 bears when there are 5, that frame contributes an error of 1. MAE is calculated as (1/N) × Σ|predicted - actual| and is measured in units of bears. An MAE of 0.5 means the model is off by half a bear on average. MAE provides an intuitive measure of counting accuracy without emphasizing outliers.

**RMSE (Root Mean Square Error)** is the square root of the average squared counting errors. It is calculated as sqrt((1/N) × Σ(predicted - actual)²). Unlike MAE, RMSE penalizes large errors more heavily due to the squaring operation. If the model usually counts correctly but occasionally makes large mistakes (e.g., detecting 2 bears when there are 5), RMSE will be disproportionately higher than MAE. Comparing RMSE to MAE helps identify whether errors are consistent or sporadic.

**Exact Match Accuracy** is the percentage of frames where the predicted count exactly equals the ground truth count. This is the strictest counting metric and directly measures how often the model gets the count perfectly right. For use cases that require exact counts (e.g., "there are exactly 5 bears in frame"), this metric provides direct success measurement.

## IoU Calculation and Detection Matching

Understanding how detections are matched to ground truth is essential for interpreting detection metrics. The matching process uses Intersection over Union (IoU) to determine whether a predicted bounding box correctly identifies a ground truth bear.

**IoU (Intersection over Union)** quantifies the overlap between two bounding boxes. Given a predicted box and a ground truth box, IoU is calculated as the area of their intersection divided by the area of their union. If the boxes perfectly overlap, IoU is 1.0. If they don't overlap at all, IoU is 0. An IoU of 0.5 means roughly half of the combined area is shared between the two boxes.

For example, consider a predicted box covering pixels [100, 100, 200, 200] and a ground truth box at [150, 150, 250, 250]. Both boxes are 100×100 pixels (area = 10,000 px² each). Their intersection is the region [150, 150, 200, 200], which is 50×50 pixels (area = 2,500 px²). The union is the total area covered by both boxes: 10,000 + 10,000 - 2,500 = 17,500 px². Thus, IoU = 2,500 / 17,500 = 0.143, indicating poor overlap.

**Detection Matching** uses a greedy algorithm implemented in the `match_detections` method. First, all predictions are sorted by confidence score from highest to lowest. Then, for each prediction in order, the algorithm finds the ground truth box with the highest IoU that hasn't already been matched. If that IoU exceeds the threshold (default 0.5), the prediction and ground truth are matched, and both are marked unavailable for future matches. This greedy approach prioritizes high-confidence detections, which aligns with how YOLO ranks its outputs. After matching completes, matched predictions become true positives, unmatched predictions become false positives, and unmatched ground truths become false negatives.

## Interpretation Guide

Interpreting metric values requires understanding what constitutes good performance in our specific context of bear detection from webcam footage.

For detection metrics, our M2 (Detection Baseline) targets are mAP@0.5 ≥ 0.70, precision ≥ 0.75, and recall ≥ 0.70. Values below 0.60 for precision, recall, or F1 indicate poor performance requiring investigation. Values between 0.60 and 0.75 are acceptable for early iterations but need improvement. Values between 0.75 and 0.85 represent good performance suitable for deployment with monitoring. Values above 0.85 are excellent and indicate a well-tuned model. For mAP@0.5:0.95, expect lower absolute values: below 0.35 is poor, 0.35-0.50 is acceptable, 0.50-0.65 is good, and above 0.65 is excellent.

For counting metrics, our M4 (Counting MVP) targets are MAE ≤ 1.0 and exact match accuracy ≥ 70%. An MAE above 2.0 bears indicates the model is frequently miscounting by significant margins. MAE between 1.0 and 2.0 is acceptable but needs work. MAE between 0.5 and 1.0 is good, and below 0.5 is excellent. For exact match accuracy, below 50% is poor, 50-70% is acceptable, 70-85% is good, and above 85% is excellent. Videos with 1-3 bears should achieve near-perfect counts, while videos with 5+ bears may have higher error due to occlusion.

When diagnosing issues, common trade-off patterns provide guidance. If precision is high but recall is low, the model is being too conservative and missing real bears; lowering the confidence threshold will help. If precision is low but recall is high, the model is detecting too many false positives; raising the confidence threshold will help. If mAP@0.5 is good but mAP@0.5:0.95 is poor, the model detects bears but localizes them imprecisely; focus on tighter bounding box annotation during training. If RMSE is much higher than MAE, the model makes occasional large counting errors that should be investigated individually.

## Failure Mode Categories

Understanding common failure modes helps prioritize model improvements and guides error analysis.

**Distant bears** appear when bears are far from the camera and occupy few pixels in the frame. Symptoms include low recall and increased false negatives, particularly on frames where bears are more than 100 meters from camera. Mitigation strategies include increasing training data with distant bear examples and considering multi-scale detection approaches.

**Occlusion** occurs when bears are partially blocked by terrain, vegetation, or other bears. Symptoms include under-counting and lower precision when partially visible bears are either missed or detected with poor localization. Mitigation includes annotating partially visible bears in the training set and potentially adjusting detection thresholds for partial matches.

**Water splash and spray** can create bear-shaped visual patterns that trigger false positives, particularly near the falls. Symptoms include increased false positive rate and over-counting near water features. Mitigation includes adding hard negative examples of water splash labeled as background and implementing temporal filtering since bears persist across frames while splashes are transient.

**Low contrast and challenging lighting** conditions occur when brown bears blend into brown backgrounds or during dawn, dusk, and overcast conditions. Symptoms include reduced recall and lower confidence scores across the board. Mitigation includes data augmentation with varied brightness and contrast settings, plus ensuring the training set includes samples from different times of day and weather conditions.

**Motion blur** affects frames where bears are moving quickly or camera shake occurs. Symptoms include inconsistent detection with bounding boxes that drift or miss fast-moving subjects. Mitigation includes adding motion-blurred samples to the training set and potentially implementing frame selection strategies that skip heavily blurred frames.

**False positives from similar objects** occur when rocks, logs, or other animals are misclassified as bears. Symptoms include high false positive rate with low-confidence detections on non-bear objects. Mitigation includes adding hard negative examples to the training set and increasing the confidence threshold for deployment.

## Example Evaluation Outputs

Running dataset evaluation with `--mode dataset` against the validation set produces output like the following. The JSON file saved to `predictions/evaluations/` contains the core metrics: `{"map50": 0.856, "map50_95": 0.534, "precision": 0.847, "recall": 0.812, "f1": 0.829}`. This output indicates good detection performance with mAP@0.5 well above the 0.70 target. The lower mAP@0.5:0.95 (0.534) is expected given challenging localization conditions.

Running counting evaluation with `--mode counting` produces per-frame statistics and summary metrics. The terminal output shows something like: Exact Match: 78.44%, MAE: 0.356 bears, RMSE: 0.612 bears. These values exceed our M4 targets (MAE ≤ 1.0, Exact Match ≥ 70%). The CSV file contains columns for frame number, detected count, ground truth count, absolute error, correctness flag, and average detection confidence. This per-frame data enables detailed error analysis to identify which specific conditions cause counting failures.

## Usage

To run dataset evaluation for M2 metrics, use: `python -m src.detection.evaluate --mode dataset --data data/annotation/bears/bear.yaml`. This leverages YOLO's built-in validation to compute mAP, precision, and recall against the annotated validation set.

To run counting evaluation for M4 metrics, use: `python -m src.detection.evaluate --mode counting --video path/to/video.mkv --ground-truth 5 --frame-skip 30`. The ground-truth flag specifies the expected bear count (constant across the video), and frame-skip controls sampling density.

