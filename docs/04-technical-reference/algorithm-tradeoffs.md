# Algorithms Research: Identify and Compare Viable Algorithms

Status: **Draft** | In Review | Approved  
Author: Oorjit Chowdhary, Kaito Yan  
Date: 2026-02-09  
Last Updated: 2026-02-09

## 1\. Context

We aim to develop a computer vision pipeline that automatically detects and tracks Alaskan brown bears, counts salmon, and quantifies feeding behavior in Katmai National Park videos. This decision matters because the choice of algorithms directly affects the accuracy of bear and salmon detection, the ability to track behavior over time, and the feasibility of running analyses on consumer-grade hardware.

## 2\. Options Considered

### Option A: YOLO

A single-stage real-time object detection model that predicts bounding boxes and class probabilities directly from full images in one forward pass.

Pros:

* Pretrained YOLOv8 models available with strong out-of-box accuracy on common objects including animals  
* Ultralytics Python package provides seamless training, fine-tuning, and inference with minimal setup  
* Single-stage architecture enables real-time inference suitable for video processing  
* Extensive community support, documentation, and tutorials

Cons:

* Less accurate on small or densely packed objects compared to two-stage detectors  
* Pretrained weights optimized for COCO dataset—domain shift to Katmai footage requires fine-tuning

Evidence: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/) \- COCO pretrained model includes "bear" class (class 21\)

### Option B: Vision Transformers (ViT-based Detection)

Transformer-based architectures (e.g., DETR, DINO, ViTDet) that apply self-attention mechanisms to image patches for object detection.

Pros:

* Strong performance on complex scenes with global context modeling  
* State-of-the-art results on benchmarks when trained at scale

Cons:

* Pretrained detection models less mature than CNN-based alternatives: fewer plug-and-play options  
* Limited Python tooling compared to YOLO ecosystem; integration requires more custom code  
* Higher computational requirements for both training and inference  
* Available pretrained weights often not optimized for wildlife or animal detection tasks

Evidence: [DETR Paper](https://arxiv.org/abs/2005.12872) | [ViT Paper](https://arxiv.org/abs/2010.11929) \- Initial evaluation showed weaker zero-shot performance on Katmai test frames compared to YOLO

### Option C: Convolutional Neural Network (CNN)

A deep learning model that uses convolution, pooling, and connected layers to learn hierarchical visual features.

Pros:

* Automatically learns spatial feature hierarchies from low to high-level patterns, reducing the need for manual feature engineering  
* Achieves high classification accuracy compared to traditional methods and has become a mainstream approach in computer vision

Cons:

* Training requires large computational resources and significant processing power  
* Models often take a long time to train, even though inference can be fast afterward

Evidence: [https://dl.acm.org/doi/10.1145/3725899.3725936](https://dl.acm.org/doi/10.1145/3725899.3725936) \- Animal classification using CNN

### Option D: RetinaNet

What it is: A single-stage object detector that uses a CNN backbone with a Feature Pyramid Network and focal loss to detect objects efficiently

Pros:

* Faster than two-stage detectors while maintaining high accuracy  
* Focal loss reduces class imbalance, improving detection of harder examples

Cons:

* Still computationally heavy compared to lightweight detectors  
* Requires careful tuning of anchors and focal loss parameters

Evidence: [https://learnopencv.com/finetuning-retinanet/-](https://learnopencv.com/finetuning-retinanet/-) Fine tuned RetinaNet 

## 3\. Decision Criteria

How we evaluated the options. List criteria in priority order.

| Criterion | Weight | Why It Matters |
| ----- | ----- | ----- |
| Accuracy | High | Determines how reliably the model detects or classifies objects, making it the most important metric for evaluating performance. |
| Inference Speed | Medium | Impacts whether the model can be used in real-time applications such as surveillance, robotics, or autonomous systems. |
| Computational Cost | Low | Reflects GPU/CPU and memory requirements, which affect scalability and deployment feasibility. |

## 4\. Evaluation

Score each option against criteria. Use: ✓ (meets), \~ (partial), ✗ (fails)

| Criterion | YOLO | Vision Transformers | CNN (from scratch) | RetinaNet |
| ----- | ----- | ----- | ----- | ----- |
| Accuracy | ✓ | ✓ | \~ | ✓ |
| Inference Speed | ✓ | \~ | ✓ | \~ |
| Computational Cost | ✓ | ✗ | ✗  | \~ |
| Python Ecosystem | ✓ | \~ | ✓ | \~ |

## 5\. Decision

Selected: YOLO (v8)

Rationale:

YOLO meets all criteria with minimal tradeoffs. The pretrained YOLOv8 includes a bear class, providing a strong starting point without training from scratch. The Ultralytics Python package offers the most mature ecosystem for our use case because training, fine-tuning, evaluation, and video inference are all available through a unified API. Given our team's limited prior ML/CV experience and the 2 quarter timeline, reducing integration complexity was a deciding factor. Vision Transformers and RetinaNet offered comparable accuracy but worse tooling and training a CNN from scratch was infeasible given compute and time constraints.

Further, our data constraints made training from scratch impractical. The 2.5 TB dataset consists of screen recordings from Explore.org YouTube streams, introducing compression artifacts and resolution degradation. Bears and salmon are only visible in a subset of this footage, with high variability in camera angles, dynamic zoom transitions, and lighting conditions. Building a model from scratch would require a large, high-quality, consistently labeled dataset, in which we had volume but not quality. Thus fine-tuning a pretrained model that already understands "bear" as a visual concept was the only viable path.

## 6\. Consequences

What this enables:

* Rapid prototyping with pretrained weights: we had working detection within Week 2  
* Fine-tuning workflow using labeled Katmai frames with minimal custom code  
* Built-in video inference and tracking integration (ByteTrack compatible)  
* Consistent evaluation pipeline via Ultralytics' native mAP/precision/recall metrics  
* Leverages pretrained knowledge to compensate for limited high-quality training data

What this limits:

* Locked into Ultralytics ecosystem and its abstractions  
* Model architecture decisions constrained to YOLO variants (n/s/m/l/x)  
* Less flexibility for novel detection approaches compared to building from scratch  
* Performance ceiling bounded by how well pretrained COCO features transfer to low-quality Katmai footage

Revisit if:

* YOLOv8 fine-tuned performance plateaus below 75% mAP on validation set  
* Access to raw camera feeds (non-YouTube) becomes available, improving data quality  
* A pretrained wildlife-specific model becomes available with comparable tooling  
* Dataset grows significantly with consistent, high-quality labeled frames (1000+)

## 7\. Implementation Notes

The detection pipeline is implementing the detection module of the pipeline as follows:

src/detection/  
├── detector.py      \# BearDetector class \- training, inference, batch counting  
├── evaluate.py      \# Evaluation CLI with multiple modes  
├── metrics.py       \# IoU calculation, precision/recall/F1, counting accuracy  
├── predict.py       \# Single video inference  
├── train.py         \# Fine-tuning CLI  
└── bear\_count.py    \# Batch counting across multiple videos

## 8\. References

\[1\] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023\. \[Online\]. Available: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

\[2\] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779-788.

\[3\] N. Carion *et al.*, "End-to-End Object Detection with Transformers," in *Proc. European Conf. Computer Vision (ECCV)*, 2020, pp. 213-229.

\[4\] T.-Y. Lin *et al.*, "Microsoft COCO: Common Objects in Context," in *Proc. European Conf. Computer Vision (ECCV)*, 2014, pp. 740-755.

\[5\] Explore.org Katmai Bear Cams. \[Online\]. Available: [https://explore.org/livecams/brown-bears/brown-bear-salmon-cam-brooks-falls](https://explore.org/livecams/brown-bears/brown-bear-salmon-cam-brooks-falls)

