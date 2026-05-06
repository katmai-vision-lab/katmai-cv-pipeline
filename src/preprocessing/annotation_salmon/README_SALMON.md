# Salmon Auto-Annotation System

This system is tuned specifically for **detecting salmon as they jump out of the water**.

## System architecture

### Three-model configuration (recommended optimum)

| Model | Variant | Weight | Specialty |
|-------|---------|--------|-----------|
| **Grounding DINO** | base | 0.40 | Text understanding, general-purpose detection |
| **OWL-ViT v2** | base-ensemble | 0.35 | Action understanding ("jumping", "leaping") |
| **Florence-2** | base | 0.25 | Robust on complex scenes (splash, reflections) |

**min_agreement**: 2 (at least 2 of the 3 models must agree)

### vs. the bear system (annotation_bear)

| Difference | Bear system | Salmon system |
|------------|-------------|---------------|
| Model combo | GDINO + DETR + MegaDetector | GDINO + OWL-ViT v2 + Florence-2 |
| MegaDetector | ✅ enabled (land animals) | ❌ disabled (not suited to aquatic species) |
| DETR | ✅ enabled | ❌ disabled (low precision, 35.4%) |
| Default prompt | "bear" | "salmon" / "salmon jumping out of water" |
| Model weights | 0.406 / 0.335 / 0.259 | 0.40 / 0.35 / 0.25 |
| Scene specialization | Land / forest environment | Above-water jumping |

### Why these models

#### 🚀 Strength of OWL-ViT v2
- **CLIP architecture**: good at action concepts ("jumping", "leaping")
- **Zero-shot**: understands "salmon jumping" without prior training
- **Scene fit**: responds well to prompts like "salmon jumping out of water"

#### 🧠 Strength of Florence-2
- **Latest VLM** (2024 release): vision-language multimodal model
- **Robust**: stable on splashes and reflections
- **Generalizes well**: consistent across lighting and angle

#### ❌ Why MegaDetector and DETR are disabled
- **MegaDetector v5**: trained for **land wildlife** (bears, deer, wolves); poor on fish morphology
- **DETR**: only 35.4% precision in the bear-system arena, with too many false positives

## Usage

### Basic flow

```bash
# 1. Extract video frames
python -m src.preprocessing.annotation_salmon.frame_extractor \
  --input salmon_jumping_video.mp4 \
  --output data/frames/salmon/ \
  --fps 0.5

# 2. Auto-annotate (3-model ensemble)
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon/ \
  --output data/auto_labels_salmon/ \
  --review-queue data/review_queue_salmon/ \
  --prompt "salmon jumping out of water" \
  --min-agreement 2 \
  --auto-approve

# 3. Visualize for sanity-check
python -m src.preprocessing.annotation_salmon.visualize_labels \
  --images data/frames/salmon/subfolder/ \
  --labels data/auto_labels_salmon/ \
  --output data/visualized_salmon/ \
  --limit 10
```

### Advanced: probability calibration

If you have a labeled salmon validation set:

```bash
# Train calibrators (using all 3 models)
python -m src.preprocessing.annotation_salmon.train_calibration \
  --images data/annotation/salmon/images/train/ \
  --labels data/annotation/salmon/labels/train/ \
  --output models/calibrators_salmon.pkl \
  --prompt "salmon jumping out of water" \
  --use-gdino \
  --use-owlvit \
  --use-florence2

# Use the calibrators when annotating
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon/ \
  --output data/auto_labels_salmon/ \
  --prompt "salmon jumping out of water" \
  --auto-approve \
  --calibrator models/calibrators_salmon.pkl
```

## Prompt tuning

**Recommended prompts (jumping scenes):**

```bash
# Best default
--prompt "jumping salmon"

# Alternative action words
--prompt "leaping salmon"
--prompt "salmon in mid-air"

# Species-specific
--prompt "jumping chinook salmon"
--prompt "leaping sockeye salmon"

# Concise generic
--prompt "salmon"
```

**Not recommended**:
- ❌ "salmon jumping out of water" (the word "water" causes Florence-2 to draw very large boxes)
- ❌ "salmon swimming" (underwater scene, model isn't tuned for this)
- ❌ "dead salmon" (no motion signal)
- ❌ "salmon in bear mouth" (composite scene; the bear system is a better fit)

## Performance expectations

With only the 3 models active, expect:
- **Recall**: possibly slightly lower than the bear system (one fewer model)
- **Precision**: depends on how well GDINO and OWL-ViT generalize to "salmon"
- **Speed**: faster (one fewer model loaded)

**Suggestions**:
1. First test on a small sample
2. If you see many false positives, raise `--min-agreement` to 2
3. If you miss many salmon, lower the confidence threshold or run a single model

## Future improvements

1. **Collect a labeled salmon validation set** (100–500 images)
   - Run the model arena
   - Compute salmon-specific optimal weights
   - Train a dedicated calibrator

2. **Try other models**
   - OWL-ViT: another strong zero-shot detector
   - Fine-tune YOLOv8 on the auto-labeled data for a specialist model

3. **Prompt engineering**
   - Try different text phrasings
   - Use species names to raise precision

4. **Post-processing**
   - Temporal smoothing (consistency across consecutive frames)
   - Size filtering (drop boxes that are too small or too large)

## File layout

```
annotation_salmon/
├── multi_model_annotator.py       # core annotator (salmon-tuned)
├── train_calibration.py           # calibrator training (MegaDetector off by default)
├── auto_annotator_gdino.py        # Grounding DINO wrapper (shared)
├── auto_annotator_detr.py         # DETR wrapper (shared)
├── auto_annotator_megadet.py      # MegaDetector (off by default here)
├── probability_calibrator.py      # calibration module (shared)
├── frame_extractor.py             # video-frame extractor (shared)
├── visualize_labels.py            # annotation visualizer (shared)
└── README_SALMON.md               # this doc
```

## Feedback and improvements

This is a quick adaptation of the bear system. If you notice:
- Detection quality is suboptimal
- Scene-specific issues
- Need for new features

please file an issue or contact the dev team. We can tune further once we have real data.
