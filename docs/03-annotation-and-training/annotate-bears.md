# Annotate Bears

The bear annotation pipeline uses Grounding DINO as a single open-vocabulary detector to auto-generate YOLO-format labels. It achieved 89.3% precision and 99.8% recall on 341 validation images — good enough to skip manual labeling entirely for most footage.

For a deep technical treatment, see [Bear Annotation System →](../04-technical-reference/bear-annotation-system.md).

---

## Step 1 — Extract frames from video

```bash
python -m src.preprocessing.annotation_bear.frame_extractor \
    --input path/to/video.mp4 \
    --output data/frames/video_name/ \
    --fps 1
```

`--fps 1` extracts one frame per second — appropriate for slow-moving bears. For dense or fast scenes, try `--fps 2`.

---

## Step 2 — Run multi-model annotation

```bash
python -m src.preprocessing.annotation_bear.multi_model_annotator \
    --input data/frames/video_name/ \
    --output data/auto_labels/ \
    --prompt "bear" \
    --min-agreement 2 \
    --auto-approve
```

- `--min-agreement 2` — a detection must appear in at least 2 of the 3 models to be kept
- `--auto-approve` — skips the human review queue; remove this flag to manually review uncertain frames
- Labels are written in YOLO format: `<class_id> <cx> <cy> <width> <height>` (normalized)

---

## Step 3 — Visual verification (optional but recommended)

Overlay labels on frames to spot-check quality:

```bash
python -m src.preprocessing.annotation_bear.visualize_labels \
    --images data/frames/video_name/ \
    --labels data/auto_labels/ \
    --output data/visualization/
```

Look for: missed detections at edges of frame, false positives on rocks/logs, duplicate boxes on the same bear.

---

## Step 4 — Prepare the dataset split

```bash
python -m src.preprocessing.split_dataset \
    --input data/annotation/bears
```

Creates `train/` and `val/` subfolders with an 80/20 split and generates `bear.yaml`.

---

## Output format

```
data/auto_labels/
├── images/
│   ├── frame_0001.jpg
│   └── ...
└── labels/
    ├── frame_0001.txt      # one line per bear: 0 cx cy w h
    └── ...
```

---

## Adjusting the pipeline

**Too many false positives** (rocks, logs detected as bears):  
→ Raise `--min-agreement` to 3 (all models must agree)

**Too many missed detections** (distant bears not labeled):  
→ Lower `--min-agreement` to 1, then manually review uncertain frames

**Different species or context**:  
→ Change `--prompt` to match your target (e.g., `"brown bear"`, `"grizzly bear standing in water"`)

See [Adapt to New Dataset →](../05-extending/adapt-to-new-dataset.md) for larger-scale adaptation.
