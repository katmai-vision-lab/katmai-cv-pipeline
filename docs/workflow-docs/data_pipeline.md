# Data Pipeline Documentation

## Overview

This document describes the workflow for converting raw video footage into training data for bear detection models.

## Directory Structure

```
data/
├── raw_videos/          # Original videos (place your videos here)
├── processed_videos/    # Cleaned clips after scene segmentation
├── frames/              # Extracted frames for annotation
├── annotations/         # Annotation files (COCO, YOLO format)
└── datasets/            # Final training datasets
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## Pipeline Steps

### Step 1: Place Raw Videos

Put your video files in `data/raw_videos/`.

### Step 2: Scene Segmentation (TODO)

Run scene detection to split videos with multiple scenes:

```bash
python scripts/scene_split.py --input data/raw_videos/ --output data/processed_videos/
```

### Step 3: Frame Extraction (TODO)

Extract frames from processed videos:

```bash
python scripts/extract_frames.py --input data/processed_videos/ --output data/frames/ --fps 1
```

### Step 4: Annotation

Use CVAT or Label Studio to annotate frames in `data/frames/`.

Export annotations to `data/annotations/`.

### Step 5: Dataset Preparation (TODO)

Convert annotations and split into train/val/test:

```bash
python scripts/prepare_dataset.py --frames data/frames/ --annotations data/annotations/ --output data/datasets/
```

## Notes

- Video and image files are excluded from git (see `data/.gitignore`)
- Use cloud storage (Google Drive, etc.) to share large files with team
