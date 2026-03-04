# Salmon Detection Quick Reference Card
> **New User?** Start with [🔰 First Time Setup](#-first-time-setup-10-minutes) → Then try [🚀 Quick Commands](#-quick-commands)

## 📋 Quick Index

- **[🔰 First Time Setup](#-first-time-setup-10-minutes)** - Install, download models, test (10 min)
- **[🚀 Quick Commands](#-quick-commands)** - Standard workflow (5 min)
- **[📊 Method Comparison](#-method-comparison)** - Stacking vs Voting
- **[⚙️ Key Parameters](#️-key-parameters)** - Confidence, prompts, thresholds
- **[🔧 Common Adjustments](#-common-adjustments)** - Tune for your needs
- **[📁 Output Files](#-output-files)** - Understanding results
- **[❗ Troubleshooting](#-troubleshooting)** - Fix common issues
- **[🗂️ Model Management](#️-model-management)** - Download, cache, train
- **[💡 Pro Tips](#-pro-tips)** - Best practices
- **[🆘 Quick Help](#-quick-help)** - Diagnostic commands

---
## 🔰 First Time Setup (10 minutes)

### Prerequisites
- ✅ NVIDIA GPU with 8GB+ VRAM (e.g., RTX 2080, 3060, 4060 or better)
- ✅ CUDA 11.8+ or 12.x installed
- ✅ Ubuntu 20.04+ / Linux (or WSL2 on Windows)
- ✅ Anaconda or Miniconda installed
- ✅ 10GB free disk space (for models and data)
- ✅ Internet connection (first run only, to download models)

### Quick Status Check
Run these commands to verify your system is ready:
```bash
nvidia-smi                    # ✓ Should show your GPU
nvcc --version                # ✓ Should show CUDA 11.8+
conda --version               # ✓ Should show conda installed
df -h .                       # ✓ Should show 10GB+ available
```

### 1. System Requirements
```bash
# Check GPU (requires NVIDIA GPU with 8GB+ VRAM)
nvidia-smi

# Verify CUDA 11.8+ or 12.x
nvcc --version
```

### 2. Install Environment
```bash
# Navigate to project directory
cd katmai-cv-pipeline/

# Create conda environment
conda create -n katmai python=3.10 -y
conda activate katmai

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Download Pre-trained Model
```bash
# Option A: Download from project SharePoint/Google Drive
# (Get link from team - models/stacker_salmon_fish.pkl)

# Option B: Use existing model (if already in repo)
ls -lh models/stacker_salmon_fish.pkl  # Should show ~539KB

# If missing, contact team or train new model (see guide)
```

### 4. Test Installation
```bash
# Download test video or use sample
# Extract frames from test video
python -m src.preprocessing.frame_extractor \
  --input test_video.mp4 \
  --output data/test_frames/ \
  --fps 1

# Run detection on 5 frames (quick test)
python -m src.preprocessing.annotation_salmon.predict_stacking \
  --images data/test_frames/ \
  --stacker models/stacker_salmon_fish.pkl \
  --output data/test_results/ \
  --prompt "fish" \
  --confidence 0.5 \
  --device cuda \
  --visualize

# Check results
ls data/test_results/visualized/*.jpg
# ✓ If you see annotated images, system is ready!
```

---

## 🚀 Quick Commands

> **After completing First Time Setup**, use these commands for daily work.

### Standard Workflow (5 minutes)
```bash
# 1. Extract frames (fps=1 for salmon)
python -m src.preprocessing.frame_extractor \
  --input video.mp4 --output data/frames/my_video/ --fps 1

# 2. Detect with Stacking (recommended)
python -m src.preprocessing.annotation_salmon.predict_stacking \
  --images data/frames/my_video/ \
  --stacker models/stacker_salmon_fish.pkl \
  --output data/results/my_video/ \
  --prompt "fish" \
  --confidence 0.5 \
  --device cuda \
  --visualize

# 3. View results
nautilus data/results/my_video/visualized/
```

---

## 📊 Method Comparison

| Feature | Stacking | Voting |
|---------|----------|--------|
| **Precision** | 97.5% ⭐ | ~65% |
| **Recall** | 96.6% ⭐ | ~40% |
| **Manual review** | 0% ⭐ | 30% |
| **Speed** | 1.7s/frame | 1.5s/frame |
| **Use case** | Production | Training data |

---

## ⚙️ Key Parameters

### Confidence Threshold (Stacking)
- `0.3` - High recall (more detections, some false positives)
- **`0.5` - Balanced (default)** ⭐
- `0.7` - High precision (fewer detections, very accurate)

### Prompt Selection
- **`"fish"` - Best for salmon (recommended)** ⭐
- `"salmon"` - More specific, more false positives
- `"jumping salmon"` - Too specific, many false positives

### Agreement Threshold (Voting)
- `1` - Accept single strong detection (manual review needed)
- **`2` - Require 2+ models agree (default)** ⭐
- `3` - Very conservative (may miss detections)

---

## 🔧 Common Adjustments

### Too Many False Positives?
```bash
# Increase confidence threshold
--confidence 0.7

# Or for voting, increase base thresholds
--gdino-threshold 0.45 --owlvit-threshold 0.45 --florence2-threshold 0.45
```

### Missing Real Salmon?
```bash
# Decrease confidence threshold
--confidence 0.3

# Or for voting, lower agreement
--min-agreement 1
```

### Processing Too Slow?
```bash
# Remove visualization during batch processing
# (remove --visualize flag)

# Or reduce image resolution before detection
python resize_images.py --input data/frames/ --size 1280x720
```

---

## 📁 Output Files

```
data/results/my_video/
├── labels/                    # YOLO format detections
│   ├── frame_00001.txt       # 0 0.523 0.389 0.082 0.124
│   └── frame_00002.txt
└── visualized/               # Images with bounding boxes
    ├── frame_00001.jpg
    └── frame_00002.jpg
```

---

## ❗ Troubleshooting

### Configuration Issues

| Problem | Solution |
|---------|----------|
| **Model file not found** | Download from SharePoint or train new: `python -m src.preprocessing.annotation_salmon.train_stacking ...` |
| **CUDA not available** | Install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| **Module not found** | Activate environment: `conda activate katmai`, reinstall: `pip install -r requirements.txt` |
| **Permission denied** | Check file paths are accessible: `ls -l models/stacker_salmon_fish.pkl` |
| **Import error (transformers)** | Upgrade transformers: `pip install --upgrade transformers` (need 4.30+) |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| **CUDA OOM** | Close other GPU apps (`nvidia-smi`), reduce batch size, use smaller models |
| **No detections found** | Check input path exists, lower `--confidence 0.3`, verify images are valid |
| **Slow model downloads** | Pre-cache models: Run once with internet, models cache to `~/.cache/huggingface/` |
| **Wrong detections** | Verify `--prompt "fish"` (not "salmon"), check `--confidence 0.5`, ensure image quality |
| **Visualize folder empty** | Check terminal for errors, ensure `--visualize` flag is set, verify write permissions |

### Model Configuration

| Problem | Solution |
|---------|----------|
| **Stacker.pkl corrupted** | Re-download or retrain. Check file size: `ls -lh models/stacker_salmon_fish.pkl` (should be ~539KB) |
| **Base models not downloading** | Check internet connection, HuggingFace accessible, or download manually to cache |
| **Different model versions** | Update transformers: `pip install --upgrade transformers`, may need to retrain stacker |
| **Custom model not working** | Verify training completed successfully, check training log for errors, ensure 200+ training examples |

### Quick Diagnostics

```bash
# Check environment
conda activate katmai
python -c "import torch, transformers; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check model file
ls -lh models/stacker_salmon_fish.pkl
python -c "import pickle; print(pickle.load(open('models/stacker_salmon_fish.pkl', 'rb')).keys())"

# Test base models (should download/cache automatically)
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('google/owlv2-base-patch16-ensemble')"

# Check GPU memory
nvidia-smi
```

---

## 📚 Full Documentation

- **Complete Guide**: `docs/SALMON_DETECTION_GUIDE.md`
- **README Section**: Search "Salmon Detection" in `README.md`
- **Training Custom Model**: See guide section "Training Your Own"

---

## 🎯 Decision Tree

```
Do you have manual labels?
├─ YES → Skip this system, use your labels
└─ NO → Need auto-annotations
    │
    ├─ For production use?
    │  └─ YES → Use Stacking (predict_stacking.py) ⭐
    │
    └─ Creating training data?
       └─ YES → Use Voting (multi_model_annotator.py) 
                → Manual review → Train custom Stacking
```

---

## �️ Model Management

### Pre-trained Stacking Model
```
Location: models/stacker_salmon_fish.pkl
Size: ~539KB
Performance: P=97.5%, R=96.6%, F1=97.1%
Training data: 375 cleaned salmon images
```

### Base Models (Auto-downloaded)
```
Cached at: ~/.cache/huggingface/hub/
Models:
- IDEA-Research/grounding-dino-base       (~700MB)
- google/owlv2-base-patch16-ensemble      (~1.5GB)
- microsoft/Florence-2-large              (~1.8GB)

First run: 5-10 min download time
Subsequent runs: Instant (uses cache)
```

### Getting Models

**Option 1: From Team Resources** (Recommended)
```bash
# Get stacker_salmon_fish.pkl from:
# - Project SharePoint: Models folder
# - Google Drive: https://drive.google.com/... (ask team)
# - Or train your own (see Training section)

# Place in project
mv ~/Downloads/stacker_salmon_fish.pkl models/
```

**Option 2: Train Your Own**
```bash
# Requires: 200+ manually labeled images
python -m src.preprocessing.annotation_salmon.train_stacking \
  --images data/training_images/ \
  --labels data/training_labels/ \
  --output models/my_stacker.pkl \
  --prompt "fish" \
  --device cuda
# Training time: ~10-15 minutes for 300 images
```

**Option 3: Pre-cache Base Models**
```bash
# Download all base models before first run (optional)
python -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# GDINO
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')

# OWL-ViT v2
Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')

print('✓ All models cached!')
"
```

---

## 💡 Pro Tips

1. **First time setup**: Run test on 5 images to verify everything works
2. **Always visualize first batch** before processing hundreds of videos
3. **Use "fish" prompt** unless you have specific requirements
4. **Start with default settings** (confidence=0.5, prompt="fish")
5. **For new domains**: Train custom model (300+ examples)
6. **Batch processing**: Remove `--visualize` flag for speed
7. **Model sharing**: Share trained stacker.pkl file with team (only ~539KB)

---

## 🆘 Quick Help

### System Check (First Time)
```bash
# 1. Check GPU
nvidia-smi                                          # GPU available? Check memory

# 2. Check environment
conda activate katmai                               # Activate environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"  # CUDA working?
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" # Version 4.30+?

# 3. Check model file
ls -lh models/stacker_salmon_fish.pkl              # File exists? Size ~539KB?

# 4. Check dependencies
pip list | grep -E "torch|transformers|opencv|PIL" # All installed?

# 5. Run end-to-end test
python -m src.preprocessing.annotation_salmon.predict_stacking \
  --images data/test_frames/ \
  --stacker models/stacker_salmon_fish.pkl \
  --output data/test_results/ \
  --prompt "fish" --confidence 0.5 --device cuda --visualize
# ✓ If successful: System ready for production!
```

### Count Results
```bash
# Frames with detections
ls data/results/my_video/visualized/*.jpg | wc -l

# Total detections across all frames
cat data/results/my_video/labels/*.txt | wc -l

# Detection rate
echo "scale=2; $(ls data/results/my_video/visualized/*.jpg | wc -l) / $(ls data/frames/my_video/*.jpg | wc -l) * 100" | bc
# Output: % of frames with salmon
```

### Common Commands
```bash
# Quick test on 10 frames
head -n 10 <(ls data/frames/video/*.jpg) | xargs -I {} cp {} data/test_frames/
python -m src.preprocessing.annotation_salmon.predict_stacking \
  --images data/test_frames/ --stacker models/stacker_salmon_fish.pkl \
  --output data/quick_test/ --prompt "fish" --confidence 0.5 --device cuda --visualize

# Batch process multiple videos
for video in videos/*.mp4; do
  name=$(basename "$video" .mp4)
  python -m src.preprocessing.frame_extractor --input "$video" --output "data/frames/$name/" --fps 1
  python -m src.preprocessing.annotation_salmon.predict_stacking \
    --images "data/frames/$name/" --stacker models/stacker_salmon_fish.pkl \
    --output "data/results/$name/" --prompt "fish" --confidence 0.5 --device cuda
done

# View random samples
ls data/results/my_video/visualized/*.jpg | shuf -n 10 | xargs eog
```

---

## 📞 Getting More Help

- **Complete guide**: [docs/SALMON_DETECTION_GUIDE.md](SALMON_DETECTION_GUIDE.md)
- **Technical details**: [docs/salmon_auto_annotation_system_report.md](salmon_auto_annotation_system_report.md)
- **Main README**: [../README.md](../README.md) - Search "Salmon Detection"
- **GitHub Issues**: Report bugs or ask questions
- **Team SharePoint**: Internal documentation and resources

---

**Last Updated**: March 4, 2026  
**Version**: 1.1  
**Maintained by**: Katmai Vision Lab
```

---

**Version**: 1.0 | **Updated**: March 4, 2026 | **Contact**: GitHub Issues
