# Model Fine-Tuning & Inference Guide

This document provides detailed instructions on how to fine-tune the YOLOv8 model and use the trained model for inference.

---

## Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Model Fine-Tuning](#2-model-fine-tuning)
3. [Using Trained Models](#3-using-trained-models)
4. [Complete Pipeline](#4-complete-pipeline)
5. [Frequently Asked Questions](#5-frequently-asked-questions)

---

## 1. Data Preparation

### 1.1 Dataset Structure

Your dataset should be organized in the following structure:

```
data/annotation/bears/
├── bear.yaml              # Dataset configuration file
├── images/
│   ├── train/            # Training images
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/              # Validation images
│       ├── img_101.jpg
│       ├── img_102.jpg
│       └── ...
└── labels/
    ├── train/            # Training labels (YOLO format)
    │   ├── img_001.txt
    │   ├── img_002.txt
    │   └── ...
    └── val/              # Validation labels
        ├── img_101.txt
        ├── img_102.txt
        └── ...
```

### 1.2 Dataset Configuration File (bear.yaml)

```yaml
path: ./data/annotation/bears  # Dataset root directory
train: images/train             # Training images path (relative to path)
val: images/val                 # Validation images path (relative to path)

nc: 1                           # Number of classes
names:
  0: bear                       # Class name
```

### 1.3 Annotation Format (YOLO Format)

Each image has a corresponding `.txt` file with one object per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

- All coordinates are **normalized** (between 0-1)
- `x_center`, `y_center`: Center coordinates of bounding box (relative to image width/height)
- `width`, `height`: Bounding box dimensions (relative to image width/height)

**Example**:
```
0 0.512 0.345 0.234 0.156
0 0.678 0.521 0.145 0.098
```

---

## 2. Model Fine-Tuning

### 2.1 Basic Training

**Command**:
```bash
python -m src.main \
    --mode train \
    --data data/annotation/bears/bear.yaml \
    --epochs 50 \
    --model models/pretrained/yolov8n.pt
```

**Parameter Description**:
- `--mode train`: Training mode
- `--data`: Path to dataset configuration file
- `--epochs`: Number of training epochs (recommended: 50-100)
- `--model`: Path to pretrained model (optional, defaults to yolov8n)

### 2.2 Python API Training

```python
from src.detection.detector import BearDetector

# Initialize detector with pretrained model
detector = BearDetector(model_path='models/pretrained/yolov8n.pt')

# Start training
results = detector.train(
    data_yaml='data/annotation/bears/bear.yaml',
    epochs=50,
    imgsz=640,      # Image size
    batch=8,        # Batch size
    project='models/trained',  # Output directory
    name='bear_detector'       # Experiment name
)

# After training, model is automatically saved at:
# models/trained/bear_detector/weights/best.pt
```

### 2.3 Training Parameter Tuning

#### Common Parameters

| Parameter | Default | Description | Recommended |
|-----------|---------|-------------|-------------|
| `epochs` | 50 | Training epochs | 50-100 |
| `batch` | 8 | Batch size | 8-16 (depends on GPU memory) |
| `imgsz` | 640 | Image size | 640 (standard) or 1280 (more precise) |
| `patience` | 50 | Early stopping patience | 30-50 |
| `lr0` | 0.01 | Initial learning rate | 0.01 (default) |
| `augment` | True | Data augmentation | True |

#### Example: High Precision Training

```python
detector.train(
    data_yaml='data/annotation/bears/bear.yaml',
    epochs=100,
    imgsz=1280,     # Higher resolution
    batch=4,        # Reduce batch size for large images
    patience=30,
    name='bear_detector_high_res'
)
```

#### Example: Quick Test Training

```python
detector.train(
    data_yaml='data/annotation/bears/bear.yaml',
    epochs=3,       # Quick test
    batch=16,
    name='bear_detector_test'
)
```

### 2.4 View Training Results

After training, results are saved in:
```
models/trained/bear_detector/
├── weights/
│   ├── best.pt      # Best model on validation set ⭐
│   └── last.pt      # Last epoch model
├── results.csv      # Training metrics (loss, mAP, etc.)
├── confusion_matrix.png
├── results.png
└── ...
```

**Key Files**:
- `best.pt`: **This is the model you should use**
- `results.png`: Training curves (loss, mAP, etc.)
- `confusion_matrix.png`: Confusion matrix

---

## 3. Using Trained Models

### 3.1 Single Video Prediction

#### Method 1: Command Line

```bash
# Use fine-tuned model
python -m src.main \
    --mode predict \
    --video "bears/video1.mkv" \
    --model models/trained/bear_detector/weights/best.pt \
    --conf 0.25 \
    --classes 0
```

**Parameter Description**:
- `--model`: Path to fine-tuned model
- `--conf`: Confidence threshold (0-1)
- `--classes 0`: Detect class 0 (bear)

#### Method 2: Python API

```python
from src.detection.detector import BearDetector

# Load your trained model
detector = BearDetector(
    model_path='models/trained/bear_detector/weights/best.pt'
)

# Predict video
results, output_dir = detector.predict_video(
    video_path='data/bears/video1.mkv',
    conf=0.25,
    classes=0,      # Only detect bear class
    save=True       # Save annotated video
)

print(f"Results saved at: {output_dir}")
```

### 3.2 Batch Video Processing

#### Sequential Processing

```bash
python -m src.detection.bear_count \
    --video-dir data/bears/ \
    --pattern "*.mkv" \
    --model models/trained/bear_detector/weights/best.pt \
    --conf 0.25 \
    --frame-skip 30
```

#### Parallel Processing (Faster)

```bash
python -m src.detection.bear_count \
    --video-dir data/bears/ \
    --pattern "*.mkv" \
    --model models/trained/bear_detector/weights/best.pt \
    --conf 0.25 \
    --frame-skip 30 \
    --parallel \
    --max-workers 4
```

### 3.3 Counting Mode (Fast)

Count bears without saving annotated videos:

```python
detector = BearDetector(
    model_path='models/trained/bear_detector/weights/best.pt'
)

stats = detector.count_bears_in_video(
    video_path='data/bears/video1.mkv',
    conf=0.25,
    frame_skip=30,  # Process every 30th frame (~1fps)
    classes=0
)

print(f"Video name: {stats['video_name']}")
print(f"Max bears: {stats['max_bears_in_frame']}")
print(f"Avg bears: {stats['avg_bears_per_frame']:.2f}")
print(f"Processing time: {stats['processing_time']:.1f}s")
```

---

## 4. Complete Pipeline

### 4.1 Train + Predict + Evaluate

Complete workflow in one command:

```bash
python -m src.main \
    --mode full \
    --video "bears/video1.mkv" \
    --data data/annotation/bears/bear.yaml \
    --epochs 50 \
    --conf 0.25 \
    --ground-truth 5
```

**Workflow**:
1. Train model using `bear.yaml` (50 epochs)
2. Predict `video1.mkv` with trained model
3. Compare with ground truth (5 bears) and generate evaluation report

### 4.2 Skip Training (Use Existing Model)

```bash
python -m src.main \
    --mode full \
    --video "bears/video1.mkv" \
    --model models/trained/bear_detector/weights/best.pt \
    --skip-train \
    --conf 0.25 \
    --ground-truth 5
```

---

## 5. Frequently Asked Questions

### Q1: CUDA Out of Memory During Training

**Solution**:
```python
# Reduce batch size
detector.train(
    data_yaml='data/annotation/bears/bear.yaml',
    epochs=50,
    batch=4  # Reduce from 8 to 4
)

# Or reduce image size
detector.train(
    data_yaml='data/annotation/bears/bear.yaml',
    epochs=50,
    imgsz=320  # Reduce from 640 to 320
)
```

### Q2: How to Choose the Best Model?

During training, the system automatically saves the model with **highest mAP on validation set** as `best.pt`.

Use this model:
```python
detector = BearDetector(
    model_path='models/trained/bear_detector/weights/best.pt'
)
```

### Q3: No Performance Improvement After Training?

Possible reasons:
1. **Dataset too small**: Recommend at least 500+ annotated images
2. **Insufficient epochs**: Increase to 100 epochs
3. **Data quality issues**: Check annotation accuracy
4. **Parameter tuning needed**: Try adjusting learning rate, batch size

### Q4: How to Resume Interrupted Training?

```python
detector = BearDetector(
    model_path='models/trained/bear_detector/weights/last.pt'  # Use last.pt
)

detector.train(
    data_yaml='data/annotation/bears/bear.yaml',
    epochs=100,
    resume=True  # Enable resume
)
```

### Q5: What Should I Use for the classes Parameter?

**Depends on the model you're using**:

| Model Type | classes Parameter | Description |
|------------|------------------|-------------|
| COCO pretrained (yolov8n.pt) | `classes=21` | Bear is class 21 in COCO dataset |
| Fine-tuned model (best.pt) | `classes=0` | Bear is class 0 in your dataset |
| Fine-tuned model (best.pt) | `classes=None` | Detect all classes (same result if only one class) |

### Q6: How to Evaluate Model Performance?

```bash
python -m src.main \
    --mode evaluate \
    --video "bears/test_video.mkv" \
    --model models/trained/bear_detector/weights/best.pt \
    --ground-truth 5 \
    --conf 0.25
```

Generates:
- CSV report (detailed statistics)
- Visualization charts (detection comparison)

### Q7: Where Are Prediction Results Saved?

```
predictions/
├── <timestamp>_<video_name>/
│   ├── <video_name>.avi       # Annotated video
│   ├── metadata.json          # Detection metadata
│   └── labels/                # YOLO format labels
│       ├── frame_000.txt
│       ├── frame_001.txt
│       └── ...
```

---

## 6. Complete Example Scripts

### Example 1: Train and Test Model

```python
from src.detection.detector import BearDetector

# 1. Train model
print("Starting training...")
detector = BearDetector(model_path='models/pretrained/yolov8n.pt')

detector.train(
    data_yaml='data/annotation/bears/bear.yaml',
    epochs=50,
    batch=8,
    name='my_bear_detector'
)

print("Training complete!")
print(f"Model saved at: {detector.model_path}")

# 2. Predict with trained model
print("\nStarting prediction...")
results, output_dir = detector.predict_video(
    video_path='data/bears/test_video.mkv',
    conf=0.25,
    classes=0
)

print(f"Prediction complete! Results saved at: {output_dir}")

# 3. Statistics
total_detections = sum(len(r.boxes) for r in results)
print(f"\nTotal frames: {len(results)}")
print(f"Total detections: {total_detections}")
print(f"Average per frame: {total_detections/len(results):.2f} bears")
```

### Example 2: Batch Process Multiple Videos

```python
from src.detection.detector import BearDetector

# Load fine-tuned model
detector = BearDetector(
    model_path='models/trained/my_bear_detector/weights/best.pt'
)

# Batch processing
results = detector.batch_count_bears_parallel(
    video_dir='data/bears/',
    pattern='*.mkv',
    conf=0.25,
    frame_skip=30,
    max_workers=4,
    save_results=True
)

# View statistics
print(f"\nProcessing complete!")
print(f"Total videos: {results['aggregate']['total_videos']}")
print(f"Successful: {results['aggregate']['successful_videos']}")
print(f"Failed: {results['aggregate']['failed_videos']}")
print(f"Total bears: {results['aggregate']['total_unique_bears']}")
```

---

## 7. Recommended Workflow

### Starting a New Project

1. **Prepare dataset** (at least 500+ annotated images)
2. **Quick test training** (3 epochs to verify workflow)
   ```bash
   python -m src.main --mode train --data bear.yaml --epochs 3
   ```
3. **Full training** (50-100 epochs)
   ```bash
   python -m src.main --mode train --data bear.yaml --epochs 50
   ```
4. **Test model** (single video)
   ```bash
   python -m src.main --mode predict \
       --video test.mkv \
       --model models/trained/bear_detector/weights/best.pt
   ```
5. **Batch processing** (all videos)
   ```bash
   python -m src.detection.bear_count \
       --video-dir data/bears/ \
       --model models/trained/bear_detector/weights/best.pt \
       --parallel
   ```

---

## 8. References

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Project GitHub**: https://github.com/katmai-vision-lab
- **System Design Document**: `docs/SYSTEM_DESIGN.md`

---

**Last Updated**: 2026-01-30  
**Maintainer**: Team Katmai
