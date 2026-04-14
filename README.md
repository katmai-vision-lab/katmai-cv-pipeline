# Computer Vision Pipeline to Detect, Track & Quantify Feeding Habits of Katmai NPP Alaskan Brown Bears
This project focuses on building an open-source, Python-based computer vision pipeline that analyzes video data from Katmai National Park & Preserve. The system is designed to automatically detect and track individual Alaskan brown bears, count salmon attempting to jump Brooks Falls, and quantify feeding behavior over time. In addition to visual analysis, the pipeline will integrate environmental context such as water level, stream flow, weather, and time of day to support deeper ecological insight.

Video data is sourced primarily from Explore.org bear cams in the Brooks Falls and Brooks River region.

## Scope
The system is designed to run on a consumer-grade laptop or desktop and ingest short-form video clips (1–15 minutes). From these inputs, it will produce structured outputs including individual and total bear counts, bear movement trajectories, salmon jump counts, and feeding behavior metrics. These results will be paired with environmental data to enable analysis across time, conditions, and location.

More information: https://github.com/katmai-vision-lab

## Auto-Annotation with Multi-Model Consensus

The pipeline includes an advanced multi-model annotation system that combines **Grounding DINO**, **DETR**, and **MegaDetector v5** to automatically generate high-quality training labels.

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended 8GB+ VRAM)
- **CUDA**: 12.x (tested with CUDA 12.8)
- **Python**: 3.10
- **RAM**: 16GB+ recommended
- **Disk Space**: ~10GB for models and dependencies

### Installation

**1. Create Python environment**
```bash
conda create -n katmai python=3.10 -y
conda activate katmai
```

**2. Install PyTorch with CUDA support**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Install core dependencies (IMPORTANT: Version-specific)**
```bash
# Critical: transformers version incompatibility issues
pip install transformers==4.47.1
pip install huggingface-hub==0.36.2

# Model dependencies
pip install timm omegaconf pytorch-lightning lightning PytorchWildlife

# Other requirements
pip install -r requirements.txt
```

**4. Verify installation**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import transformers; print('transformers version:', transformers.__version__)"
```

### Features
- **Model Arena Evaluation**: Scientifically validated model weights based on 341 test images
  - Grounding DINO: 0.406 weight (89.3% precision, 99.8% recall)
  - MegaDetector v5: 0.335 weight (65.6% precision, 84.4% recall)
  - DETR: 0.259 weight (35.4% precision, 74.7% recall)
- **Weighted Consensus**: Uses model weights × confidence scores to select best detections
- **Two Modes**: Human review mode or fully automatic mode for training data generation

### Usage

**Step 1: Extract frames from video**
```bash
python -m src.preprocessing.annotation.frame_extractor \
  --input path/to/video.mp4 \
  --output data/frames/video_name/ \
  --fps 0.2
```

**Step 2: Generate training labels (auto-approve mode)**
```bash
# Fully automatic - only saves detections with model consensus (≥2/3 models agree)
python -m src.preprocessing.annotation.multi_model_annotator \
  --input data/frames/video_name/ \
  --output data/auto_labels/ \
  --review-queue data/review_queue/ \
  --prompt "bear" \
  --min-agreement 2 \
  --auto-approve
```

**Alternative: Human review mode**
```bash
# Saves uncertain cases to review queue for manual verification
python -m src.preprocessing.annotation.multi_model_annotator \
  --input data/frames/video_name/ \
  --output data/consensus_labels/ \
  --review-queue data/review_queue/ \
  --prompt "bear" \
  --min-agreement 2
```

**Step 3: Visualize results (optional)**
```bash
python -m src.preprocessing.annotation.visualize_labels \
  --images data/frames/video_name/subfolder/ \
  --labels data/auto_labels/ \
  --output data/visualized/ \
  --limit 10
```

### Output Format
- YOLO format labels (ready for training)
- One `.txt` file per image with format: `class_id x_center y_center width height`
- Only includes detections with consensus (weighted agreement ≥ min_agreement)

### Advanced: Probability Calibration (Recommended)

For improved accuracy, you can train probability calibrators that transform raw confidence scores into calibrated probabilities. This addresses the fact that different models have different confidence distributions (e.g., a 0.9 from DETR doesn't mean the same as 0.9 from Grounding DINO).

**Benefits of calibration:**
- More accurate combining of multi-model predictions
- Better confidence estimates for each detection
- Scientifically grounded probability scores
- Reference: [sklearn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

**Step 1: Train calibrators on validation set**
```bash
python -m src.preprocessing.annotation.train_calibration \
  --images data/annotation/bears/images/ \
  --labels data/annotation/bears/labels/ \
  --output models/calibrators.pkl \
  --prompt "bear" \
  --iou-threshold 0.5
```

This will:
- Run all 3 models on your validation images
- Match detections to ground truth using IoU
- Train isotonic regression calibrators
- Report Expected Calibration Error (ECE) before/after

**Step 2: Use calibrators during annotation**
```bash
python -m src.preprocessing.annotation.multi_model_annotator \
  --input data/frames/video_name/ \
  --output data/auto_labels/ \
  --review-queue data/review_queue/ \
  --prompt "bear" \
  --min-agreement 2 \
  --auto-approve \
  --calibrator models/calibrators.pkl  # Add this line
```

**What happens:**
- Raw confidence scores are transformed into calibrated probabilities
- Model weights are applied to calibrated probabilities
- Results in more accurate consensus detection selection

**Note**: Training calibrators requires:
- A validation set with ground truth labels (YOLO format)
- At least 100+ samples per model (more is better)
- ~30-60 minutes on GPU depending on dataset size

### Important Notes

⚠️ **Version Compatibility**
- Must use `transformers==4.47.1` (v5.x has breaking changes)
- `huggingface_hub` must be compatible with transformers 4.47.1
- Do not upgrade transformers automatically

⚠️ **First Run**
- Models will be downloaded automatically (~5GB total)
- Downloads cached in `~/.cache/huggingface/`
- First run may take 10-15 minutes for model downloads

⚠️ **Memory Management**
- Models load sequentially to avoid GPU OOM
- Expect ~6-8GB GPU memory usage
- CPU mode available but slow (add `--device cpu`)

⚠️ **File Paths**
- Frame extractor creates subdirectory structure
- Visualize tool requires the actual image subdirectory path
- Example: `data/frames/video_name/video_name_frames/` not `data/frames/video_name/`

### Troubleshooting

**Issue: ModuleNotFoundError for timm/omegaconf/PytorchWildlife**
```bash
pip install timm omegaconf pytorch-lightning lightning PytorchWildlife
```

**Issue: CUDA out of memory**
- Reduce batch processing (models run sequentially by default)
- Close other GPU applications
- Use smaller model with `--use-detr False --use-megadet False`

**Issue: transformers version conflict**
```bash
pip install transformers==4.47.1 --force-reinstall
```

**Issue: No detections generated**
- Check `--min-agreement` value (default: 2)
- Try lowering threshold: `--min-agreement 1`
- Verify input frames exist with correct path

---

## Salmon Detection with Stacking Meta-Learner

The salmon detection system uses a **Stacking ensemble approach** that combines three state-of-the-art zero-shot detection models with a trained meta-learner for optimal precision-recall balance.

### Architecture Overview

**Base Models (Zero-Shot Detectors)**:
1. **Grounding DINO** - Visual-linguistic grounding
2. **OWL-ViT v2** - Open-vocabulary object detection
3. **Florence-2** - Vision foundation model with grounding

**Meta-Learner**:
- **Random Forest Classifier** trained on 375 manually cleaned examples
- Extracts 11 features per detection (model ID, confidence, IoU overlaps, box size/position, etc.)
- Learns optimal fusion strategy instead of hard-coded voting rules

### Performance Comparison

| Method | Precision | Recall | F1 Score | Auto Decision |
|--------|-----------|--------|----------|---------------|
| **Voting (min-agreement=2)** | ~60-70% | ~40% | ~48% | ❌ 30% need review |
| **Stacking (trained)** | **97.5%** | **96.6%** | **97.1%** | ✅ 100% automatic |

**Key Advantages**:
- 🎯 **+133% detection rate** compared to voting method
- 🧠 **Smart fusion** based on learned feature importance
- ⚡ **Zero manual review** required (97.5% precision)
- 📊 **Near-perfect AUC-ROC** (99.9%)

### System Requirements

Same as bear detection system:
- GPU: 8GB+ VRAM (RTX 2080 or better)
- CUDA: 12.x
- Python: 3.10
- Disk: ~8GB for models

### Quick Start Guide

**1. Extract frames from salmon videos**
```bash
python -m src.preprocessing.frame_extractor \
  --input path/to/salmon_video.mp4 \
  --output data/frames/salmon_video/ \
  --fps 1
```

**2. Option A: Use pre-trained Stacking model (Recommended)**
```bash
python -m src.preprocessing.annotation_salmon.predict_stacking \
  --images data/frames/salmon_video/ \
  --stacker models/stacker_salmon_fish.pkl \
  --output data/salmon_results/ \
  --prompt "fish" \
  --confidence 0.5 \
  --device cuda \
  --visualize
```

**Output**:
- `data/salmon_results/labels/` - YOLO format labels
- `data/salmon_results/visualized/` - Detection visualizations
- Fully automatic, no manual review needed

**2. Option B: Traditional voting method (for comparison)**
```bash
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon_video/ \
  --output data/voting_results/ \
  --review-queue data/review_queue/ \
  --prompt "fish" \
  --min-agreement 2 \
  --gdino-threshold 0.37 \
  --owlvit-threshold 0.37 \
  --florence2-threshold 0.37
```

**Note**: Voting method may flag 30% of images for manual review.

### Training Your Own Stacking Model

If you have new salmon videos and want to retrain the meta-learner:

**Step 1: Generate initial annotations**
```bash
# Use voting method with optimized thresholds
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon_videos/ \
  --output data/auto_labels_salmon/ \
  --review-queue data/review_queue/ \
  --prompt "fish" \
  --min-agreement 2 \
  --gdino-threshold 0.37 \
  --owlvit-threshold 0.37 \
  --florence2-threshold 0.37
```

**Step 2: Visualize for manual review**
```bash
python -m src.preprocessing.annotation_salmon.visualize_nested \
  --images data/frames/salmon_videos/ \
  --labels data/auto_labels_salmon/ \
  --output data/visualized_salmon/
```

**Step 3: Manual cleanup**
- Browse `data/visualized_salmon/` folder
- Delete images with misdetections (bears, birds, water splashes labeled as fish)
- Keep images with correct fish detections

**Step 4: Sync labels with cleaned images**
```bash
python sync_labels_from_visualized.py \
  --visualized data/visualized_salmon/ \
  --labels data/auto_labels_salmon/
```

**Step 5: Prepare flat training directory**
```bash
mkdir -p data/training_salmon/images data/training_salmon/labels

# Copy cleaned data
for label in data/auto_labels_salmon/*.txt; do
  basename="${label##*/}"
  basename="${basename%.txt}"
  find data/frames/salmon_videos/ -name "${basename}.jpg" -exec cp {} data/training_salmon/images/ \;
  cp "$label" data/training_salmon/labels/
done
```

**Step 6: Train Stacking meta-learner**
```bash
python -m src.preprocessing.annotation_salmon.train_stacking \
  --images data/training_salmon/images/ \
  --labels data/training_salmon/labels/ \
  --output models/stacker_salmon_custom.pkl \
  --prompt "fish" \
  --meta-learner rf \
  --device cuda
```

**Training Output**:
```
Dataset collected:
  Total detections: 6393
  True Positives: 1637 (25.6%)
  False Positives: 4756 (74.4%)

Validation Performance:
  Precision: 0.975
  Recall:    0.966
  F1 Score:  0.971
  AUC-ROC:   0.999

Feature Importances:
  avg_overlap_conf: 0.385  ← Multi-model agreement is key!
  max_iou: 0.246
  num_overlaps: 0.178
  confidence: 0.088
```

**Expected Training Time**: ~10-15 minutes for 375 images on GPU

### Feature Importance Analysis

The Stacking model learns that **multi-model consensus is most important**:

1. **avg_overlap_conf** (38.5%) - Average confidence of overlapping detections from other models
2. **max_iou** (24.6%) - Maximum IoU overlap with other detections
3. **num_overlaps** (17.8%) - Number of other models detecting the same object
4. **confidence** (8.8%) - Raw model confidence (less reliable alone)
5. **model_owlvit** (5.7%) - Model-specific patterns

**Key Insight**: A detection is trustworthy when multiple models independently detect it at the same location with high confidence—not just because one model has high confidence.

### Prompt Optimization

Through empirical testing on 5 test images:

| Prompt | Avg Detections | Bear Misdetections |
|--------|----------------|-------------------|
| "jumping salmon" | 5.2 | ❌ Yes (2.4/image) |
| "salmon fish" | 3.4 | ❌ Yes (1.8/image) |
| "salmon" | 3.0 | ❌ Yes (1.6/image) |
| **"fish"** | **3.0** | ✅ **No (0/image)** |

**Recommendation**: Use `--prompt "fish"` for best generalization and fewer false positives.

### Threshold Optimization

Default thresholds for voting method:
- Grounding DINO: 0.37 (was 0.25)
- OWL-ViT v2: 0.37 (was 0.35)
- Florence-2: 0.37 (was 0.30)

Higher thresholds reduce false positives by 22.8% while maintaining good recall.

### Output Format

Both methods generate YOLO format labels:
```
# data/salmon_results/labels/frame_00123.txt
0 0.5234 0.3891 0.0823 0.1245
0 0.7123 0.5234 0.0912 0.1456
```

Format: `class_id center_x center_y width height` (normalized 0-1)

### Visualizations

Stacking visualizations show:
- Green bounding boxes
- Model name (gdino/owlvit/florence2)
- Stacking confidence (0-1)

Example: `gdino: 0.87` means Grounding DINO detected it, Stacking predicts 87% probability of true positive.

### Troubleshooting Salmon Detection

**Issue: Too many false positives (bears, birds, water)**
- ✅ Use Stacking method (already trained to filter these)
- ✅ Increase `--confidence` threshold (try 0.6-0.7)
- ❌ Don't use prompts like "jumping salmon" (too specific)

**Issue: Missing real salmon**
- Lower `--confidence` threshold (try 0.4)
- Check if salmon are very small or partially occluded
- Consider using `--min-agreement 1` for voting method (then review manually)

**Issue: Slow inference**
- Each frame takes ~1.7 seconds on RTX 2080
- Expect ~10-15 minutes for 375 images
- No way to speed up without better GPU (models run sequentially to fit in 8GB VRAM)

**Issue: Out of memory**
- Models already load sequentially
- Close other GPU applications
- Reduce image resolution if possible

**Issue: Model download timeout**
```bash
# Pre-download models manually
python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; \
AutoProcessor.from_pretrained('google/owlv2-large-patch14-ensemble'); \
AutoModelForZeroShotObjectDetection.from_pretrained('google/owlv2-large-patch14-ensemble')"
```

### Files and Directories

**Salmon annotation system files**:
```
src/preprocessing/annotation_salmon/
├── auto_annotator_gdino.py       # Grounding DINO wrapper
├── auto_annotator_owlvit.py      # OWL-ViT v2 wrapper
├── auto_annotator_florence2.py   # Florence-2 wrapper
├── multi_model_annotator.py      # Voting method
├── train_stacking.py             # Train meta-learner
├── predict_stacking.py           # Inference with stacking
└── visualize_nested.py           # Visualize labels
```

**Pre-trained models**:
```
models/
└── stacker_salmon_fish.pkl       # Trained on 375 cleaned samples
```

**Typical workflow outputs**:
```
data/
├── frames/salmon_videos/         # Extracted frames
├── auto_labels_salmon/           # Voting method output
├── visualized_salmon/            # For manual review
├── training_salmon/              # Cleaned training data
│   ├── images/
│   └── labels/
└── salmon_results/               # Stacking inference output
    ├── labels/
    └── visualized/
```

### Best Practices

1. **Use Stacking for production**: 97.5% precision, fully automatic
2. **Use voting for training data generation**: Then manually clean 
3. **Always visualize first batch**: Verify prompt and thresholds work for your data
4. **Prompt**: Stick with `"fish"` unless you have specific requirements
5. **Confidence threshold**: Start with 0.5, increase if too many FPs, decrease if missing detections
6. **Training frequency**: Retrain stacking model when:
   - Adding new camera angles
   - Different lighting conditions (sunrise/sunset vs. midday)
   - Different seasons (water clarity, background)

### Citation

If you use this salmon detection system in research:

```bibtex
@software{katmai_salmon_stacking,
  title={Salmon Detection with Stacking Meta-Learner},
  author={Katmai Vision Lab},
  year={2026},
  note={Multi-model ensemble with learned fusion weights},
  url={https://github.com/katmai-vision-lab}
}
```

---

## PR Process
- Do the local development on your own dev branch, eg. dev-yourname
- Once your code is ready, create a PR merge to main branch

You may use the following commands.
```
git checkout -b dev-xxx origin/main
git pull origin main
git push origin dev-xxx
```
### Split video
```
python -m src.preprocessing.split_dataset \
    --input data/annotation/bears
```
### Training process
```
python -m src.detection.train \
    --data data/annotation/bears/bear.yaml \
    --config configs/train_config.yaml
```

## Run Pipeline
Quick test with pretrained model.
```bash
python -m src.main \
    --mode full \
    --video "bears/2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv" \
    --skip-train \
    --epochs 3 \
    --conf 0.12 \
    --ground-truth 5
```

Train + predict + evaluate (original full pipeline).
```bash
python -m src.main \
    --mode full \
    --video "bears/2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv" \
    --data data/annotation/bears/bear.yaml \
    --epochs 3 \
    --conf 0.25 \
    --ground-truth 5
```

Use your fine-tuned model.
```bash
python -m src.main \
    --mode full \
    --video "bears/2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv" \
    --model models/trained/bear_detector/weights/best.pt \
    --skip-train \
    --classes 0 \
    --conf 0.25 \
    --ground-truth 5
```
```

Bear counting batch.
```bash
python -m src.detection.bear_count \
    --video-dir bears \
    --pattern "*.mp4"
```

```
python -m src.detection.bear_count \
    --video-dir bears \
    --model models/trained/bear_detector3/weights/best.pt \
    --pattern "*.mp4" \
    --classes 0 \
    --conf 0.5
```

```
python3 -m src.detection.bear_count \
    --video-dir bears \
    --pattern "*.mp4" \
    --model models/trained/bear_detector3/weights/best.pt \
    --classes 0 \
    --conf 0.7 \
    --tracking \
    --frame-skip 1 \
    --verbose
```

OutPut the Bytetrack video
```
python -m src.detection.track_video   --video "bears/2025-09-19 23-30-11_Brooks_Falls_Low_5_bears.mp4"   --model models/trained/bear_detector3/weights/best.pt   --classes 0   --conf 0.7   --frame-skip 1
```
## Useful Link
SharePoint:
https://uwnetid.sharepoint.com/sites/katmai-vision-lab/Shared%20Documents/Forms/AllItems.aspx
