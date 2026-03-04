# Salmon Detection System - Complete User Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [Method Comparison](#method-comparison)
4. [Detailed Workflows](#detailed-workflows)
5. [Understanding the Models](#understanding-the-models)
6. [Training Custom Stacking Model](#training-custom-stacking-model)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [FAQ](#faq)

---

## Overview

### What is this system?

The salmon detection system automatically identifies jumping salmon in video footage using three state-of-the-art AI models combined with a meta-learner. Think of it as having three expert annotators working together, with a fourth "supervisor" (the Stacking model) that learned from 375 examples to decide which detections to trust.

### Why three models?

- **Grounding DINO**: Best at understanding natural language prompts ("fish")
- **OWL-ViT v2**: Excellent at open-vocabulary detection (can detect anything described in text)
- **Florence-2**: Strong vision foundation model with grounding capabilities

Each model has different strengths and weaknesses. By combining them intelligently, we achieve 97.5% precision with 96.6% recall.

### Two Approaches Available

| Approach | Best For | Precision | Setup Time | Manual Work |
|----------|----------|-----------|------------|-------------|
| **Stacking (Recommended)** | Production use | 97.5% | 0 mins | None |
| **Voting + Manual Review** | Creating training data | 60-70% → Manual cleanup | 0 mins | High |

---

## Quick Start (5 Minutes)

### Prerequisites Check

1. **GPU Check**
   ```bash
   nvidia-smi
   # Should show GPU with CUDA 12.x
   ```

2. **Environment Check**
   ```bash
   conda activate katmai
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   # Should print: CUDA: True
   ```

3. **Disk Space Check**
   ```bash
   df -h
   # Ensure 10GB+ free space
   ```

### Step-by-Step: Detect Salmon in Your Video

**1. Prepare your video**
```bash
# Put your video in a folder
mkdir -p ~/salmon_videos
cp /path/to/your/salmon_video.mp4 ~/salmon_videos/
```

**2. Extract frames**
```bash
cd /home/katmai/katmai-cv-pipeline

python -m src.preprocessing.frame_extractor \
  --input ~/salmon_videos/salmon_video.mp4 \
  --output data/frames/my_salmon_video/ \
  --fps 1
```

**Expected output**: `Extracted 120 frames at 1.0 fps`

**3. Run Stacking detection**
```bash
python -m src.preprocessing.annotation_salmon.predict_stacking \
  --images data/frames/my_salmon_video/ \
  --stacker models/stacker_salmon_fish.pkl \
  --output data/results/my_salmon_video/ \
  --prompt "fish" \
  --confidence 0.5 \
  --device cuda \
  --visualize
```

**Expected time**: ~3 minutes for 120 frames (RTX 2080)

**4. View results**
```bash
# Open visualizations
nautilus data/results/my_salmon_video/visualized/ &

# Or use image viewer
eog data/results/my_salmon_video/visualized/*.jpg
```

**5. Check statistics**
```bash
# Count detections
echo "Total frames: $(ls data/results/my_salmon_video/labels/*.txt 2>/dev/null | wc -l)"
echo "Total detections: $(cat data/results/my_salmon_video/labels/*.txt 2>/dev/null | wc -l)"
```

**Done! 🎉** You now have:
- YOLO format labels in `data/results/my_salmon_video/labels/`
- Visualizations in `data/results/my_salmon_video/visualized/`

---

## Method Comparison

### Stacking Meta-Learner (Recommended)

**How it works:**
1. Three models detect objects independently
2. For each detection, extracts 11 features:
   - Which model detected it
   - Model's confidence
   - How many other models agree (IoU overlap)
   - Average confidence of agreeing models
   - Box size and position
3. Random Forest classifier predicts: "Is this really a fish?"
4. Only keeps detections with probability ≥ 0.5 (adjustable)

**Pros:**
- ✅ 97.5% precision (only 2.5% false positives)
- ✅ 96.6% recall (catches almost all salmon)
- ✅ Zero manual review required
- ✅ Learns from data (not hard-coded rules)
- ✅ Handles edge cases intelligently

**Cons:**
- ❌ Requires pre-trained model (we provide one)
- ❌ Slightly slower than voting (adds ~0.1s per frame)

**When to use:**
- ✅ Production annotation pipeline
- ✅ Need high accuracy without manual review
- ✅ Processing large video batches
- ✅ Consistency across different videos

### Voting Method (Multi-Model Consensus)

**How it works:**
1. Three models detect objects independently
2. Clusters detections by location (IoU-based)
3. Counts how many models agree
4. Keeps detection if ≥ 2 models agree (min_agreement=2)
5. Flags uncertain cases for manual review

**Pros:**
- ✅ No training required
- ✅ Transparent decision logic
- ✅ Good for creating training data (after manual cleanup)
- ✅ Adjustable agreement threshold

**Cons:**
- ❌ 60-70% precision (many false positives)
- ❌ ~30% of images need manual review
- ❌ Rigid rules (can't adapt to patterns)
- ❌ May miss good single-model detections

**When to use:**
- ✅ Creating initial training data for new domain
- ✅ You have time for manual review
- ✅ Want to understand individual model performance
- ✅ Debugging model behavior

---

## Detailed Workflows

### Workflow 1: Batch Processing Multiple Videos (Production)

**Scenario**: You have 50 salmon videos to annotate for a study.

```bash
#!/bin/bash
# batch_salmon_detection.sh

VIDEO_DIR="/path/to/salmon_videos"
OUTPUT_BASE="data/batch_results"

for video in "$VIDEO_DIR"/*.mp4; do
  video_name=$(basename "$video" .mp4)
  echo "Processing: $video_name"
  
  # Extract frames
  python -m src.preprocessing.frame_extractor \
    --input "$video" \
    --output "data/frames/$video_name/" \
    --fps 1
  
  # Run stacking detection
  python -m src.preprocessing.annotation_salmon.predict_stacking \
    --images "data/frames/$video_name/" \
    --stacker models/stacker_salmon_fish.pkl \
    --output "$OUTPUT_BASE/$video_name/" \
    --prompt "fish" \
    --confidence 0.5 \
    --device cuda \
    --visualize
  
  # Generate summary stats
  num_frames=$(ls "data/frames/$video_name/"*.jpg 2>/dev/null | wc -l)
  num_detections=$(cat "$OUTPUT_BASE/$video_name/labels/"*.txt 2>/dev/null | wc -l)
  
  echo "$video_name: $num_frames frames, $num_detections detections" >> batch_summary.txt
done

echo "Batch processing complete!"
cat batch_summary.txt
```

**Run it:**
```bash
chmod +x batch_salmon_detection.sh
./batch_salmon_detection.sh
```

### Workflow 2: Creating Training Data for Custom Domain

**Scenario**: You're working with a new camera angle or different salmon species and want to train a custom Stacking model.

**Step 1: Generate candidate labels (30 min for 20 videos)**
```bash
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/new_domain_videos/ \
  --output data/auto_labels_custom/ \
  --review-queue data/review_queue_custom/ \
  --prompt "fish" \
  --min-agreement 2 \
  --gdino-threshold 0.37 \
  --owlvit-threshold 0.37 \
  --florence2-threshold 0.37
```

**Step 2: Visualize all results (15 min)**
```bash
python -m src.preprocessing.annotation_salmon.visualize_nested \
  --images data/frames/new_domain_videos/ \
  --labels data/auto_labels_custom/ \
  --output data/visualized_custom/
```

**Step 3: Manual review (1-2 hours)**
```bash
# Open folder
nautilus data/visualized_custom/

# Instructions for reviewer:
# 1. Browse all images
# 2. DELETE images with wrong detections:
#    - Bears labeled as fish
#    - Birds labeled as fish
#    - Water splashes labeled as fish
#    - Rocks/logs labeled as fish
# 3. KEEP images with correct fish detections
# 4. When done, close file browser
```

**Step 4: Sync labels with reviewed images (1 min)**
```bash
python sync_labels_from_visualized.py \
  --visualized data/visualized_custom/ \
  --labels data/auto_labels_custom/

# Output will show:
# "Deleted 47 labels, kept 328 labels"
```

**Step 5: Prepare training directory (2 min)**
```bash
mkdir -p data/training_custom/{images,labels}

for label in data/auto_labels_custom/*.txt; do
  basename="${label##*/}"
  basename="${basename%.txt}"
  find data/frames/new_domain_videos/ -name "${basename}.jpg" \
    -exec cp {} data/training_custom/images/ \;
  cp "$label" data/training_custom/labels/
done

echo "Training set: $(ls data/training_custom/images/*.jpg | wc -l) images"
```

**Step 6: Train custom Stacking model (10-15 min)**
```bash
python -m src.preprocessing.annotation_salmon.train_stacking \
  --images data/training_custom/images/ \
  --labels data/training_custom/labels/ \
  --output models/stacker_custom_domain.pkl \
  --prompt "fish" \
  --meta-learner rf \
  --device cuda
```

**Expected output:**
```
======================================================================
Stacking Meta-Learner Training
======================================================================

[1/4] Loading base models...
[2/4] Extracting features from validation set...
Processing images: 100%|██████████| 328/328 [09:12<00:00,  1.69s/it]

Dataset collected:
  Total detections: 5521
  True Positives: 1423 (25.8%)
  False Positives: 4098 (74.2%)
  Feature dimension: 11

[3/4] Training meta-learner...
[4/4] Evaluating meta-learner...

Validation Performance:
  Precision: 0.978
  Recall:    0.969
  F1 Score:  0.973
  AUC-ROC:   0.998

Feature Importances:
  avg_overlap_conf: 0.392
  max_iou: 0.241
  num_overlaps: 0.174
  confidence: 0.091

Saving stacking model to: models/stacker_custom_domain.pkl
Done!
```

**Step 7: Test custom model (5 min)**
```bash
# Test on held-out videos
python -m src.preprocessing.annotation_salmon.predict_stacking \
  --images data/frames/test_videos/ \
  --stacker models/stacker_custom_domain.pkl \
  --output data/test_results/ \
  --prompt "fish" \
  --confidence 0.5 \
  --device cuda \
  --visualize

# Manually check visualizations
nautilus data/test_results/visualized/
```

**Total time**: ~3-4 hours (mostly waiting + 1-2 hours manual review)

### Workflow 3: Comparing Stacking vs Voting

**Scenario**: You want to understand the difference between methods on your specific videos.

```bash
# 1. Run both methods on same test set
TEST_FRAMES="data/frames/comparison_test/"

# Stacking
python -m src.preprocessing.annotation_salmon.predict_stacking \
  --images "$TEST_FRAMES" \
  --stacker models/stacker_salmon_fish.pkl \
  --output data/comparison/stacking/ \
  --prompt "fish" \
  --confidence 0.5 \
  --device cuda \
  --visualize

# Voting
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input "$TEST_FRAMES" \
  --output data/comparison/voting/labels/ \
  --review-queue data/comparison/voting/review/ \
  --prompt "fish" \
  --min-agreement 2 \
  --gdino-threshold 0.37 \
  --owlvit-threshold 0.37 \
  --florence2-threshold 0.37

# Visualize voting results
python -m src.preprocessing.annotation_salmon.visualize_nested \
  --images "$TEST_FRAMES" \
  --labels data/comparison/voting/labels/ \
  --output data/comparison/voting/visualized/

# 2. Generate comparison report
cat << 'EOF' > comparison_report.sh
#!/bin/bash
echo "=== Detection Comparison Report ==="
echo ""
echo "Test set: $TEST_FRAMES"
echo "Total frames: $(ls $TEST_FRAMES/*.jpg | wc -l)"
echo ""
echo "STACKING METHOD:"
stacking_frames=$(ls data/comparison/stacking/visualized/*.jpg 2>/dev/null | wc -l)
stacking_dets=$(cat data/comparison/stacking/labels/*.txt 2>/dev/null | wc -l)
echo "  Frames with detections: $stacking_frames"
echo "  Total detections: $stacking_dets"
echo "  Avg detections per frame: $(echo "scale=2; $stacking_dets / $stacking_frames" | bc)"
echo ""
echo "VOTING METHOD:"
voting_frames=$(ls data/comparison/voting/visualized/*.jpg 2>/dev/null | wc -l)
voting_dets=$(cat data/comparison/voting/labels/*.txt 2>/dev/null | wc -l)
echo "  Frames with detections: $voting_frames"
echo "  Total detections: $voting_dets"
echo "  Avg detections per frame: $(echo "scale=2; $voting_dets / $voting_frames" | bc)"
echo ""
review_needed=$(ls data/comparison/voting/review/*.txt 2>/dev/null | wc -l)
echo "  Frames needing review: $review_needed"
echo ""
echo "DIFFERENCE:"
frame_diff=$((stacking_frames - voting_frames))
det_diff=$((stacking_dets - voting_dets))
echo "  Additional frames detected: $frame_diff (+$(echo "scale=1; 100*$frame_diff/$voting_frames" | bc)%)"
echo "  Additional detections: $det_diff (+$(echo "scale=1; 100*$det_diff/$voting_dets" | bc)%)"
echo "  Manual review eliminated: $review_needed cases"
EOF

chmod +x comparison_report.sh
./comparison_report.sh > comparison_results.txt
cat comparison_results.txt
```

---

## Understanding the Models

### Model Performance Characteristics

| Model | Strengths | Weaknesses | Confidence Distribution |
|-------|-----------|------------|------------------------|
| **Grounding DINO** | - Best text grounding<br>- Handles complex prompts<br>- Good localization | - Slower inference<br>- Sensitive to prompt wording | 0.3-0.9 range |
| **OWL-ViT v2** | - Fast inference<br>- Open vocabulary<br>- Consistent scores | - Lower precision<br>- More false positives | 0.1-0.6 range |
| **Florence-2** | - Strong vision model<br>- Good with small objects<br>- Contextual understanding | - Sometimes too large boxes<br>- Needs box filtering | 0.2-0.8 range |

### What Makes a Detection "Trustworthy"?

The Stacking model learned these patterns from 375 examples:

**High Trust Scenarios** (Probability > 0.8):
- ✅ All 3 models detect at same location (IoU > 0.5)
- ✅ All have confidence > 0.5
- ✅ Box size reasonable (2-10% of image)
- ✅ Box in typical location (middle/upper area for jumping salmon)

**Medium Trust Scenarios** (Probability 0.5-0.8):
- ⚠️ 2 models agree strongly (IoU > 0.7)
- ⚠️ One model very confident (>0.8), others moderate
- ⚠️ Reasonable box size and position

**Low Trust Scenarios** (Probability < 0.5):
- ❌ Only 1 model detects
- ❌ Large disagreement in box locations (IoU < 0.3)
- ❌ Very large or very small boxes
- ❌ Unusual positions (edges, corners)

### Feature Importance Breakdown

From training on 375 images, the Random Forest learned:

1. **avg_overlap_conf (38.5%)** - *"Do multiple models confidently agree on this location?"*
   - If 2+ models overlap with confidence >0.6 → Very trustworthy
   - Single model alone → Suspicious

2. **max_iou (24.6%)** - *"How precisely do the boxes align?"*
   - IoU >0.7 → Models see same object
   - IoU <0.3 → Models see different things (or noise)

3. **num_overlaps (17.8%)** - *"How many other models detected something here?"*
   - 2-3 overlaps → Strong consensus
   - 0 overlaps → Outlier detection

4. **confidence (8.8%)** - *"What's the raw model confidence?"*
   - Less important than consensus!
   - High confidence alone doesn't mean true positive

5. **model_owlvit (5.7%)** - *"Which model detected this?"*
   - OWL-ViT has slightly different patterns
   - GDINO + Florence are more similar

**Key Insight**: The model prioritizes *agreement between models* over *individual model confidence*. This is why it outperforms voting—it's not just counting votes, it's assessing the quality of agreement.

---

## Training Custom Stacking Model

### When to Retrain

Consider training a custom model when:

✅ **New camera setup**
- Different angle (aerial vs. ground level)
- Different zoom level
- Different camera quality

✅ **New environment**
- Different water color/clarity
- Different lighting (perpetual dawn vs. midday)
- Different background (waterfall vs. calm river)

✅ **New salmon species/behavior**
- Different size (sockeye vs. chinook)
- Different jumping height/arc
- Different coloration

❌ **Don't retrain if:**
- Same setup, just more videos → Use existing model
- Only time/date changed → Existing model handles this
- Small dataset (<100 examples) → Not enough data

### Training Requirements

**Minimum**:
- 100+ manually verified fish detections
- Mix of positive and negative examples
- 10+ different video frames

**Recommended**:
- 300+ verified detections
- 200+ false positives for hard negative mining
- 30+ video frames covering various conditions

**Ideal**:
- 500+ verified detections
- Diverse conditions (morning/evening, calm/rough water, clear/cloudy)
- Multiple videos from different days

### Training Hyperparameters

Default settings (already optimized):
```python
RandomForestClassifier(
    n_estimators=100,      # 100 trees in forest
    max_depth=10,          # Maximum tree depth
    min_samples_split=10,  # Minimum samples to split node
    random_state=42,       # Reproducibility
    n_jobs=-1             # Use all CPU cores
)
```

**When to adjust**:
- **Overfitting** (train=0.99, val=0.85): Decrease `max_depth` to 5-7
- **Underfitting** (train=0.75, val=0.73): Increase `max_depth` to 15, `n_estimators` to 200
- **Going well**: Don't change anything!

### Expected Performance Ranges

| Dataset Size | Expected Precision | Expected Recall | Training Time |
|--------------|-------------------|-----------------|---------------|
| 100-200 samples | 90-94% | 88-92% | 5-8 min |
| 200-400 samples | 95-97% | 93-96% | 10-15 min |
| 400+ samples | 97-99% | 95-98% | 15-20 min |

If you're getting significantly worse results:
1. Check label quality (visualize and review)
2. Ensure diverse examples (not all from same video)
3. Verify balanced dataset (not 90% positive examples)

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
1. **Close other GPU applications**
   ```bash
   nvidia-smi
   # Kill processes using GPU
   kill <PID>
   ```

2. **Clear CUDA cache**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Use smaller batch (already default)**
   - Models already load sequentially
   - No batch processing options to reduce

4. **Use CPU mode (slow!)**
   ```bash
   # Add --device cpu
   python -m src.preprocessing.annotation_salmon.predict_stacking \
     --device cpu \
     # ... other args
   ```

#### Issue 2: Models download very slowly

**Symptoms:**
```
Downloading: 2% |▌ | 234MB/10GB [15:32<8:23:11, 321kB/s]
```

**Solutions:**
1. **Pre-download models**
   ```python
   from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
   from transformers import AutoModelForCausalLM
   
   # Grounding DINO
   AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
   AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
   
   # OWL-ViT v2
   AutoProcessor.from_pretrained('google/owlv2-large-patch14-ensemble')
   AutoModelForZeroShotObjectDetection.from_pretrained('google/owlv2-large-patch14-ensemble')
   
   # Florence-2
   AutoProcessor.from_pretrained('microsoft/Florence-2-base')
   AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
   ```

2. **Use faster mirror** (if in China)
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **Download overnight**
   ```bash
   nohup python download_models.py > download.log 2>&1 &
   ```

#### Issue 3: Too many false positives (bears, birds, water)

**Symptoms:**
- Visualizations show bears labeled as fish
- Water splashes detected as fish
- Rocks/logs misclassified

**Solutions (in order of preference):**

1. **Increase Stacking confidence threshold**
   ```bash
   --confidence 0.7  # More conservative (was 0.5)
   ```

2. **Verify you're using "fish" prompt, not "salmon"**
   ```bash
   --prompt "fish"  # NOT "jumping salmon"
   ```

3. **Retrain with local data**
   - Follow "Training Custom Stacking Model" workflow
   - Include examples of bears/birds as negative samples

4. **For voting method: increase base thresholds**
   ```bash
   --gdino-threshold 0.45 \
   --owlvit-threshold 0.45 \
   --florence2-threshold 0.45
   ```

#### Issue 4: Missing real salmon (low recall)

**Symptoms:**
- Visualizations show obvious salmon not detected
- Detection count much lower than expected

**Solutions:**

1. **Lower Stacking confidence threshold**
   ```bash
   --confidence 0.3  # More permissive (was 0.5)
   ```

2. **For voting: lower min-agreement**
   ```bash
   --min-agreement 1  # Accept single strong detection
   # Then manually review outputs
   ```

3. **Check image quality**
   - Are salmon visible to human eye?
   - Too small (<20 pixels)?
   - Too blurry?
   - Heavily obscured?

4. **For voting: lower base thresholds**
   ```bash
   --gdino-threshold 0.25 \
   --owlvit-threshold 0.25 \
   --florence2-threshold 0.25
   ```

5. **Consider retraining if salmon appearance different**

#### Issue 5: Slow inference (>3s per frame)

**Expected**: ~1.7s per frame on RTX 2080
**If slower**: 

1. **Check GPU utilization**
   ```bash
   watch -n 1 nvidia-smi
   # Should show 90-100% GPU usage during inference
   ```

2. **Verify CUDA version**
   ```bash
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   # Should match (e.g., both CUDA 12.1)
   ```

3. **Check for CPU bottlenecks**
   ```bash
   htop
   # CPU shouldn't be maxed out
   ```

4. **Reduce visualization overhead (if using --visualize)**
   ```bash
   # Remove --visualize flag for faster processing
   # Visualize only sample after complete
   ```

#### Issue 6: "ModuleNotFoundError"

**Symptoms:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solutions:**
```bash
# Activate environment first
conda activate katmai

# Install missing packages
pip install transformers==4.47.1
pip install timm omegaconf pytorch-lightning

# Verify
python -c "import transformers; print(transformers.__version__)"
```

#### Issue 7: Inconsistent results across runs

**Symptoms:**
- Same video, different detection counts each time
- Results don't match previous runs

**Explanation:**
- This shouldn't happen with Stacking (deterministic)
- Check if using different confidence thresholds

**Solutions:**
1. **Verify same parameters**
   ```bash
   # Log parameters to file
   echo "Run $(date): --confidence 0.5 --prompt fish" >> run_log.txt
   ```

2. **Check model version**
   ```bash
   ls -lh models/stacker_salmon_fish.pkl
   # Verify modification date
   ```

3. **For voting method**: Some randomness in clustering
   - Set random seed if needed (not exposed in CLI currently)

---

## Performance Optimization

### Speed Optimization

**Current Performance** (RTX 2080, 1920x1080 images):
- Grounding DINO: ~0.6s/frame
- OWL-ViT v2: ~0.4s/frame
- Florence-2: ~0.5s/frame
- Stacking inference: ~0.1s/frame
- **Total**: ~1.7s/frame

**Optimization Strategies**:

1. **Use lower resolution** (if acceptable)
   ```bash
   # Resize images before detection
   python resize_images.py \
     --input data/frames/original/ \
     --output data/frames/resized/ \
     --size 1280x720
   
   # Then detect on resized
   python -m src.preprocessing.annotation_salmon.predict_stacking \
     --images data/frames/resized/ \
     ...
   ```
   **Speedup**: ~40% faster (1.7s → 1.0s/frame)
   **Trade-off**: May miss small/distant salmon

2. **Skip visualization during batch processing**
   ```bash
   # Remove --visualize flag
   # Visualize sample afterwards
   ```
   **Speedup**: ~10% faster
   **Trade-off**: Can't immediately inspect results

3. **Process multiple videos in parallel** (if multiple GPUs)
   ```bash
   # Terminal 1
   CUDA_VISIBLE_DEVICES=0 python -m src.preprocessing.annotation_salmon.predict_stacking \
     --images data/frames/video1/ ...
   
   # Terminal 2
   CUDA_VISIBLE_DEVICES=1 python -m src.preprocessing.annotation_salmon.predict_stacking \
     --images data/frames/video2/ ...
   ```

4. **Use queue system for large batches** (enterprise setup)
   - Deploy model as REST API with FastAPI
   - Use job queue (Celery + Redis)
   - Multi-GPU load balancing

### Memory Optimization

**Current Memory Usage**:
- Peak VRAM: ~7.5GB
- RAM: ~4GB
- Models already load sequentially

**If hitting memory limits**:

1. **Clear cache between videos**
   ```python
   import torch
   torch.cuda.empty_cache()
   import gc
   gc.collect()
   ```

2. **Process in smaller batches**
   ```bash
   # Split large frame directory
   python split_directory.py \
     --input data/frames/huge_video/ \
     --batch-size 100
   
   # Process each batch separately
   for batch in data/frames/huge_video_batch_*/; do
     python -m src.preprocessing.annotation_salmon.predict_stacking \
       --images "$batch" ...
   done
   ```

### Accuracy vs Speed Trade-offs

| Configuration | Speed | Precision | Recall | Use Case |
|---------------|-------|-----------|--------|----------|
| **All 3 models + Stacking** | 1.7s | 97.5% | 96.6% | Production quality |
| **GDINO + Florence + Stacking** | 1.1s | 95% | 94% | Faster, still good |
| **GDINO only + threshold** | 0.6s | 85% | 92% | Quick exploration |
| **Voting 2/3, high thresh** | 1.5s | 75% | 60% | Conservative |

---

## FAQ

### General Questions

**Q: Which method should I use for my project?**

A: 
- **Production pipeline**: Stacking (97.5% precision, automatic)
- **Creating training data**: Voting → Manual review → Retrain Stacking
- **Quick exploration**: Grounding DINO only with high threshold
- **Understanding models**: Run all methods and compare

**Q: Can I use this system for other fish species?**

A: Yes! Just use generic prompt "fish" or specific species name:
```bash
--prompt "trout"
--prompt "steelhead"
--prompt "fish"  # Generic, works for all
```

For best results with new species:
1. Run voting method on 20-30 videos
2. Manually review (1-2 hours)
3. Train custom Stacking model (10 min)

**Q: How much data do I need to train custom Stacking model?**

A:
- **Minimum**: 100 verified detections (quick test)
- **Recommended**: 300+ detections (good performance)
- **Ideal**: 500+ detections (best performance)

**Q: Can this detect swimming salmon (not jumping)?**

A: Partially. The system detects any "fish" object, but:
- ✅ Jumping salmon: 97% accuracy
- ⚠️ Surface swimming: 70-80% (reflection, occlusion)
- ❌ Underwater: Cannot detect (models trained on visible objects)

For underwater, you'd need specialized models.

**Q: Does this count unique salmon or total detections?**

A: **Total detections**. Each frame analyzed independently.

For unique salmon counts, you need:
1. This system (detect per frame)
2. Tracking algorithm (link detections across frames)
3. De-duplication (identify same salmon in multiple frames)

We recommend SORT or DeepSORT for tracking.

**Q: How accurate is the stacking model?**

A: On validated test set:
- **Precision**: 97.5% (of detected salmon, 97.5% are real)
- **Recall**: 96.6% (of real salmon, 96.6% are detected)
- **F1 Score**: 97.1% (harmonic mean)
- **AUC-ROC**: 99.9% (near-perfect classification)

**Q: Can I run this without GPU?**

A: Yes, but slow:
```bash
--device cpu
```
- GPU: 1.7s/frame
- CPU: 15-25s/frame (10-15x slower)

For large batches, GPU strongly recommended.

### Technical Questions

**Q: What's the difference between Stacking and voting?**

A: **Voting**:
- Simple rule: "If ≥2 models agree, keep detection"
- Treats all models equally
- Doesn't learn from data

**Stacking**:
- Extracts 11 features per detection
- Trains ML model to predict "is this real?"
- Learns model strengths/weaknesses
- Learns what "agreement" means (IoU, confidence patterns)

**Q: Why Random Forest instead of neural network?**

A:
- ✅ Fast training (10 min vs hours)
- ✅ No overfitting on small data (375 samples)
- ✅ Interpretable (feature importances)
- ✅ No hyperparameter tuning needed
- ✅ Works with tabular features

Neural network would need 10,000+ samples to outperform RF.

**Q: Can I use my own YOLOv8 model instead of base detectors?**

A: Not directly. The system is designed for zero-shot detectors because:
- No need for training data to start
- Works across different domains
- Can use natural language prompts

If you have trained YOLOv8:
1. Generate predictions on frames
2. Convert to YOLO format
3. Skip this system entirely (you already have labels!)

**Q: How does the system handle video vs images?**

A: **System only processes images**. Workflow:
1. Extract frames from video (`frame_extractor.py`)
2. Detect on frames (`predict_stacking.py`)
3. Optionally: Link frames with tracking algorithm

We don't process video directly because:
- Frames are easier to parallelize
- Can sample at different fps
- Easier to review/debug

**Q: What's the YOLO format output?**

A: Each frame gets a `.txt` file:
```
# frame_00123.txt
0 0.5234 0.3891 0.0823 0.1245
0 0.7123 0.5234 0.0912 0.1456
```
Format: `class_id center_x center_y width height`
- All values normalized to 0-1
- `class_id` always 0 (fish)
- Can load directly in YOLOv8/v11 for training

**Q: Can I combine this with manual annotations?**

A: Yes! Workflow:
1. Run Stacking on 80% of data (automatic)
2. Manually annotate 20% (high-value frames)
3. Combine both datasets
4. Train final YOLO model

Mix of auto+manual gives best results for training.

### Troubleshooting Questions

**Q: Why am I getting different results than the README examples?**

A: Check:
1. **Same prompt?** ("fish" vs "salmon" gives different results)
2. **Same confidence threshold?** (0.5 vs 0.7 is significant)
3. **Same model version?** (`ls -lh models/stacker_salmon_fish.pkl`)
4. **Different videos?** (Performance varies by video quality)

**Q: My stacking model has lower precision than 97.5%. What's wrong?**

A: Possible reasons:
1. **Insufficient training data** (<200 samples)
   - Solution: Collect more examples
2. **Imbalanced dataset** (95% positive examples)
   - Solution: Include more false positives in training
3. **Different domain** (your videos have different characteristics)
   - Solution: This is expected! 97.5% was on specific test set
4. **Label errors** (some training labels are wrong)
   - Solution: Re-review visualizations carefully

**Q: System says "Model loaded successfully!" but then crashes**

A: This means:
- ✅ Model downloads worked
- ✅ Model initialization worked
- ❌ Inference failed

Check:
1. **Input images exist and are valid**
   ```bash
   file data/frames/test/*.jpg
   # Should show: JPEG image data
   ```

2. **Sufficient GPU memory during inference**
   ```bash
   nvidia-smi -l 1
   # Watch memory usage during run
   ```

3. **Image format compatible**
   ```bash
   python -c "from PIL import Image; img = Image.open('test.jpg'); print(img.size)"
   # Should print dimensions without error
   ```

---

## Additional Resources

### Directory Structure Reference

```
katmai-cv-pipeline/
├── src/preprocessing/annotation_salmon/   # Salmon detection code
│   ├── auto_annotator_gdino.py            # Grounding DINO
│   ├── auto_annotator_owlvit.py           # OWL-ViT v2
│   ├── auto_annotator_florence2.py        # Florence-2
│   ├── multi_model_annotator.py           # Voting method
│   ├── train_stacking.py                  # Train meta-learner
│   ├── predict_stacking.py                # Inference
│   └── visualize_nested.py                # Visualization
├── models/
│   └── stacker_salmon_fish.pkl            # Pre-trained Stacking (539KB)
├── data/
│   ├── frames/                            # Extracted video frames
│   ├── results/                           # Detection outputs
│   └── training_salmon/                   # Training data for custom model
└── docs/
    └── SALMON_DETECTION_GUIDE.md          # This guide
```

### Command Reference Cheatsheet

```bash
# Extract frames
python -m src.preprocessing.frame_extractor \
  --input video.mp4 --output data/frames/video/ --fps 1

# Detect (Stacking)
python -m src.preprocessing.annotation_salmon.predict_stacking \
  --images data/frames/video/ --stacker models/stacker_salmon_fish.pkl \
  --output data/results/ --prompt "fish" --confidence 0.5 --device cuda --visualize

# Detect (Voting)
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/video/ --output data/voting/ --review-queue data/review/ \
  --prompt "fish" --min-agreement 2 --gdino-threshold 0.37 \
  --owlvit-threshold 0.37 --florence2-threshold 0.37

# Visualize
python -m src.preprocessing.annotation_salmon.visualize_nested \
  --images data/frames/video/ --labels data/results/labels/ \
  --output data/visualizations/

# Train custom
python -m src.preprocessing.annotation_salmon.train_stacking \
  --images data/training/images/ --labels data/training/labels/ \
  --output models/custom.pkl --prompt "fish" --meta-learner rf --device cuda

# Count detections
cat data/results/labels/*.txt | wc -l

# Check GPU
nvidia-smi
```

### Related Documentation

- [Main README](../README.md) - Project overview and bear detection
- [Frame Extractor Usage](frame_extraction.md) - Video preprocessing
- [YOLO Format Specification](yolo_format.md) - Label format details
- [Model Architecture Deep Dive](model_architecture.md) - Technical details

### Support and Contact

- **GitHub Issues**: https://github.com/katmai-vision-lab/issues
- **Documentation**: https://github.com/katmai-vision-lab/docs
- **SharePoint**: [UW Katmai Vision Lab](https://uwnetid.sharepoint.com/sites/katmai-vision-lab)

---

**Last Updated**: March 4, 2026
**Version**: 1.0
**Tested with**: CUDA 12.1, PyTorch 2.1, transformers 4.47.1
