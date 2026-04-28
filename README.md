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

## Salmon Jump
### Extract one frame for calibration
python -c "
import cv2
cap = cv2.VideoCapture('data/raw/salmons/salmon_jump_2.mkv')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
_, frame = cap.read()
cv2.imwrite('calibration_frame.jpg', frame)
cap.release()
"

```python
python src/detection/salmons/salmon_jump_counter_cv.py data/raw/salmons/salmon_jump_2.mkv
```

Debug mode
```python
python src/detection/salmons/salmon_jump_counter_cv.py data/raw/salmons/salmon_jump_2.mov ./debug_salmon_frames/
```

Diagnose.
```python
python src/detection/salmons/diagnose_blobs.py data/raw/salmons/salmon_jump_0.mp4
```
## Useful Link
SharePoint:
https://uwnetid.sharepoint.com/sites/katmai-vision-lab/Shared%20Documents/Forms/AllItems.aspx
