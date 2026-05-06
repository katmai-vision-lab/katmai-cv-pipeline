# Stacking Meta-Learner — Smart Multi-Model Fusion

## 📖 What is stacking?

**Stacking (stacked generalization)** is an advanced ensemble-learning method that uses a **meta-learner** to learn the best way to combine the predictions of several base models.

### vs. traditional voting

| Method | Decision rule | Pros | Cons |
|--------|---------------|------|------|
| **Simple voting** | Fixed rule (e.g. 2/3 must agree) | Straightforward | Can't adapt to data; often too strict or too loose |
| **Weighted voting** | Hand-set weights | Flexible | Needs expert knowledge; hard to optimize |
| **Stacking** ⭐ | Learns the optimal rule from data | Adaptive, high accuracy | Requires a labeled validation set |

### How stacking works

```
[image]
   ↓
[GDINO] [OWL-ViT] [Florence-2]   ← base models
   ↓        ↓          ↓
 box1     box2       box3
 conf1    conf2      conf3
   ↓        ↓          ↓
   [feature extraction: 11-dim feature vector]
   ↓
[Random Forest meta-learner]      ← trained on validation data
   ↓
[TP / FP decision] + [confidence score]
   ↓
[final detections]
```

## 🎯 Feature design (11-d)

Stacking extracts the following features to learn the optimal fusion strategy:

1. **Model identity (3-d)**
   - One-hot for GDINO / OWL-ViT / Florence-2
   - Lets the meta-learner learn each model's reliability pattern

2. **Confidence (1-d)**
   - Raw confidence score from the base model
   - High confidence ≠ correct (context matters)

3. **Box size (2-d)**
   - Normalized width and height
   - Lets the meta-learner notice that very large or very small boxes tend to be false positives

4. **Box position (2-d)**
   - Normalized center (center_x, center_y)
   - Lets the meta-learner discover edge-of-frame vs center reliability differences

5. **Multi-model agreement (3-d)**
   - `max_iou`: max overlap with any other model's box
   - `num_overlaps`: how many other models also detected something here
   - `avg_overlap_conf`: average confidence of the overlapping detections
   - **Key features** — multi-model agreement is the strongest signal

## 🚀 Usage

### 1. Prepare a validation set

You need a **human-labeled validation set** to train the stacker.

```bash
data/annotation/salmon/
├── images/
│   └── train/
│       ├── salmon_001.jpg
│       ├── salmon_002.jpg
│       └── ...
└── labels/
    └── train/
        ├── salmon_001.txt  # YOLO format
        ├── salmon_002.txt
        └── ...
```

**Best practices**:
- At least 100–200 images
- Cover diverse scenes (lighting, angle, jump height)
- Annotation quality must be high

### 2. Train the stacking meta-learner

```bash
cd /home/katmai/katmai-cv-pipeline

python3 -m src.preprocessing.annotation_salmon.train_stacking \
  --images data/annotation/salmon/images/train/ \
  --labels data/annotation/salmon/labels/train/ \
  --output models/stacker_salmon.pkl \
  --prompt "jumping salmon" \
  --meta-learner rf
```

**Argument reference**:
- `--images`: validation image directory
- `--labels`: YOLO-format labels directory
- `--output`: output stacker file path
- `--prompt`: detection prompt (must match what you use at inference)
- `--meta-learner`: meta-learner type
  - `rf` (Random Forest) ⭐ recommended — robust, hard to overfit
  - `gb` (Gradient Boosting) — potentially higher accuracy but easier to overfit
  - `lr` (Logistic Regression) — simplest, suitable for very small datasets

**Example output**:
```
[1/4] Loading base models...
[2/4] Extracting features from validation set...
Processing images: 100%|████████| 150/150

Dataset collected:
  Total detections: 1247
  True Positives: 1089 (87.3%)
  False Positives: 158 (12.7%)
  Feature dimension: 11

[3/4] Training meta-learner...
[4/4] Evaluating meta-learner...

Validation Performance:
  Precision: 0.923
  Recall:    0.956
  F1 Score:  0.939
  AUC-ROC:   0.982

Feature Importances:
  num_overlaps: 0.287       ← most important!
  max_iou: 0.195
  avg_overlap_conf: 0.143
  confidence: 0.112
  model_gdino: 0.089

Saving stacking model to: models/stacker_salmon.pkl
Done!
```

### 3. Run inference with the stacker

```bash
python3 -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon/ \
  --output data/auto_labels_stacking/ \
  --review-queue data/review_queue/ \
  --stacker models/stacker_salmon.pkl
```

**Key differences**:
- Add `--stacker` to point at the trained model
- `--min-agreement` is ignored (stacking makes its own decisions)
- The human-review queue isn't strictly needed (the stacker already filters)

## 📊 Performance comparison

On our test set (85 frames of salmon video):

| Method | Coverage | # detections | Precision (est.) | Recall (est.) |
|--------|----------|-------------|------------------|---------------|
| **min-agreement=2** | 36.5% | 92 | ⭐⭐⭐⭐⭐ high | ⭐⭐ low |
| **min-agreement=1** | 100% | 394 | ⭐⭐ low | ⭐⭐⭐⭐⭐ high |
| **Stacking** ⭐ | ~85–95% | ~250 | ⭐⭐⭐⭐ high | ⭐⭐⭐⭐ high |

**Why stacking wins**:
- ✅ Balances precision and recall
- ✅ Learns the optimal decision boundary automatically
- ✅ Adapts to different scenes and model combos
- ✅ No manual tuning required

## 🔧 Advanced

### Adjust the IoU threshold

```bash
python3 -m src.preprocessing.annotation_salmon.train_stacking \
  --images ... \
  --labels ... \
  --output ... \
  --iou-threshold 0.7  # stricter TP definition
```

### Try a different meta-learner

```bash
# Random Forest (recommended)
--meta-learner rf

# Gradient Boosting (potentially higher accuracy)
--meta-learner gb

# Logistic Regression (fastest, small datasets)
--meta-learner lr
```

### Inspect feature importances

Training automatically prints feature importances so you can understand the model's decisions:

```
Feature Importances:
  num_overlaps: 0.287       # most important: do other models also detect this region?
  max_iou: 0.195            # important: overlap with other detections
  avg_overlap_conf: 0.143   # important: average confidence of overlapping detections
  confidence: 0.112         # moderate: raw confidence
  model_gdino: 0.089        # lower: which specific model is this
```

## ⚠️ Caveats

1. **You need labeled data**
   - At least 100 well-labeled images
   - Labeling errors propagate directly into stacker performance

2. **Train and test distributions must match**
   - If inference scenes differ a lot from training scenes, stacking may underperform
   - Recommendation: sample your validation set from the actual deployment scenes

3. **Overfitting risk**
   - With small datasets, prefer Random Forest
   - Avoid very deep Gradient Boosting

4. **Compute cost**
   - Stacking inference is ~10–20% slower than simple voting
   - Feature extraction adds extra work

## 💡 Best-practice recipes

### Scenario 1 — accuracy-first (paper, production)
```bash
# 1. Collect 200+ high-quality labeled images
# 2. Train a Random Forest stacker
--meta-learner rf

# 3. Verify precision > 90% before deploying
```

### Scenario 2 — fast prototyping (exploration)
```bash
# Use simple voting with min-agreement=2
--min-agreement 2

# No extra labels needed, iterate quickly
```

### Scenario 3 — recall-first (must not miss detections)
```bash
# Train stacking but adjust the decision threshold
# (you'll need to modify the code to use predict_proba with a low threshold)
```

## 📚 Technical references

- **Paper**: "Stacked Generalization" (Wolpert, 1992)
- **Related techniques**:
  - Weighted Boxes Fusion (WBF)
  - Non-Maximum Suppression (NMS)
  - Soft-NMS

**Why stacking instead of WBF?**
- WBF fuses **overlapping boxes** into an average box
- Stacking decides **whether each box is real or false**, filtering false positives
- Our problem is "too many false detections," not "boxes are inaccurate" — so stacking fits better

## 🎓 Further reading

For multi-model ensembles in general:
- Kaggle object-detection competition write-ups
- COCO Detection Challenge technical reports
- *Ensemble Methods in Machine Learning* (book)
