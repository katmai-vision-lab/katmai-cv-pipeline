# Katmai CV Pipeline — Bear Detection, Tracking & Feeding Behavior Analysis

Open-source Python computer vision pipeline for detecting, tracking, and quantifying the feeding behavior of Alaskan brown bears at Katmai National Park & Preserve. Built as a UW ENGINE capstone project (2025–2026).

The system automatically detects and tracks bears, counts salmon jumping Brooks Falls, classifies feeding events using vision-language models, and integrates environmental context (river flow, weather, precipitation) for ecological analysis. Video data is sourced from Explore.org bear cams in the Brooks Falls and Brooks River region.

**Deepwiki:** https://deepwiki.com/katmai-vision-lab/katmai-cv-pipeline  
**Full documentation:** [`docs/`](docs/README.md)

## Scope

Runs on consumer-grade hardware and ingests short video clips (1–15 minutes). Outputs include bear counts, movement trajectories, salmon jump counts, feeding event timestamps, and environmental summaries paired with video timestamps.

## Performance

| Component | Metric | Value |
|---|---|---|
| Bear detector (fine-tuned YOLOv8n) | mAP@0.5 | 95.1% |
| Bear detector | Precision / Recall / F1 | 92.2% / 90.6% / 91.4% |
| Bear detector (pretrained, baseline) | mAP@0.5 | 17.9% |
| Salmon annotation (stacking meta-learner) | Precision / Recall | 97.5% / 96.6% |
| Bear annotation (multi-model consensus) | Precision / Recall | 89.3% / 99.8% |

---

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended 8GB+ VRAM)
- **CUDA**: 12.x (tested with CUDA 12.8)
- **Python**: 3.10
- **RAM**: 16GB+ recommended
- **Disk Space**: ~10GB for models and dependencies

---

## Installation

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

### macOS / Windows installation notes

The pipeline is developed and tested on **Linux + NVIDIA GPU**, but every module is written to be cross-platform — `src/config.get_device()` auto-detects CUDA / MPS / CPU at runtime and `torch.cuda.empty_cache()` calls are guarded against non-CUDA environments.

#### macOS

- **Apple Silicon (M1 / M2 / M3 / M4)** — supported via PyTorch MPS:
  ```bash
  pip install torch torchvision
  ```
  Auto-detected at runtime. Molmo2-8B runs on MPS at roughly the speed of a mid-range NVIDIA card, but needs ~14 GB of unified memory for FP16 (8 GB Macs should use cloud VLM backends instead).
- **Intel Mac** — CPU only. YOLOv8 + ByteTrack + pixel detector work but slowly; recommend `--backend anthropic` or `--backend gemini` for behavior classification.
- **FFmpeg** (needed by the `--render` flag of the pixel eating detector): `brew install ffmpeg`.
- **`PytorchWildlife`** occasionally has dependency conflicts on macOS:
  ```bash
  pip install --no-deps PytorchWildlife
  ```

#### Windows

- **Recommended: WSL2 + Ubuntu 22.04**. Install WSL2 with `wsl --install -d Ubuntu-22.04` and follow the Linux steps inside the WSL shell.
- **Native Windows (PowerShell)** also works — install PyTorch with CUDA from the same `--index-url`, convert multi-line commands to PowerShell backtick continuations, and install FFmpeg from <https://www.gyan.dev/ffmpeg/builds/>.
- **No NVIDIA GPU**: use a cloud VLM backend (`--backend gemini` is cheapest); the rest of the pipeline runs on CPU.

#### Device-selection summary

Every CLI that runs a model exposes a `--device` flag. Passing nothing auto-detects: **CUDA → MPS → CPU**. Pass `--device cpu` to force CPU.

⚠️ **Version Compatibility**
- Must use `transformers==4.47.1` (v5.x has breaking changes)
- Do not upgrade transformers automatically

---

## Interactive TUI

The primary entry point for running the full pipeline interactively:

```bash
python -m src.cli
```

The TUI presents a menu with tab-completion for file paths and step-by-step prompts for all parameters:

| Key | Function | Description |
|-----|----------|-------------|
| 1 | Detect bears | Raw YOLO inference → annotated output video with bounding boxes |
| 2 | Track bears | ByteTrack → annotated video with persistent bear IDs |
| 3 | Batch count bears | Frame-sampled counts across many videos, no output video |
| 4 | Detect feeding events | VLM frame analysis → timestamped behavior JSON |
| 5 | Count salmon jumps | CV-based jump detection → count + timestamps |
| 6 | Fetch environmental data | RAWS / NADP / USGS — weather, precipitation, hydrology |
| 7 | Evaluate model | mAP, precision, recall, counting accuracy vs. ground truth |
| 8 | Train model | Fine-tune YOLOv8n on a new labeled dataset |

---

## Bear Detection & Tracking

### Run Pipeline (quick test with pretrained model)
```bash
python -m src.main \
    --mode full \
    --video "bears/2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv" \
    --skip-train \
    --epochs 3 \
    --conf 0.12 \
    --ground-truth 5
```

### Train + predict + evaluate (full pipeline)
```bash
python -m src.main \
    --mode full \
    --video "bears/2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv" \
    --data data/annotation/bears/bear.yaml \
    --epochs 3 \
    --conf 0.25 \
    --ground-truth 5
```

### Train model
```bash
python -m src.preprocessing.split_dataset \
    --input data/annotation/bears

python -m src.detection.train \
    --data data/annotation/bears/bear.yaml \
    --config configs/train_config.yaml
```

### Bear counting (batch)
```bash
python -m src.detection.bear_count \
    --video-dir bears \
    --model models/trained/bear_detector3/weights/best.pt \
    --pattern "*.mp4" \
    --classes 0 \
    --conf 0.7 \
    --tracking \
    --frame-skip 1 \
    --verbose
```

### ByteTrack annotated video
```bash
python -m src.detection.track_video \
  --video "bears/0505.mp4" \
  --model models/trained/bear_detector3/weights/best.pt \
  --classes 0 \
  --conf 0.7 \
  --imgsz 1280 \
  --frame-skip 1
```

### Trajectory overlay video

Renders each bear's fading movement trail from a saved `trajectories.json`:
```bash
python -m src.detection.trajectory_video \
    --trajectories "predictions/<run>/trajectories.json" \
    --trail-frames 300 \
    --thickness 3
```

### Track dump (debug)

Dumps per-frame ByteTrack IDs to stdout or JSON without generating a video:
```bash
python -m src.detection.track_dump \
    --video bears/xxx.mp4 \
    --model models/trained/bear_detector3/weights/best.pt \
    --classes 0 \
    --conf 0.7 \
    --json-out predictions/track_debug.json
```

---

## Feeding Behavior Detection

### VLM-based feeding analysis (recommended)

Runs YOLO + ByteTrack, samples one frame every N seconds, and sends each frame to a vision model with per-bear position context. Produces timestamped behavior descriptions in JSON.

**Step 1: Analyze**
```bash
python -m src.behavior.analyze_feeding \
    --video path/to/video.mp4 \
    --interval 0.5 \
    --backend molmo2          # molmo2 | anthropic | gemini | openai
```

Available backends: `molmo2` (local, free), `anthropic` (Claude), `gemini` (cheapest cloud), `openai` (GPT-4o). Use `--vision-model` to override the default model per backend.

**Step 2: Render viewer with closed-caption overlay**
```bash
python -m src.behavior.feeding_viewer \
    --video path/to/video.mp4 \
    --analysis predictions/<stem>_feeding_analysis/analysis.json
```

### Pixel-based eating detector (research artifact)

A CPU-only detector (`src/behavior/pixel_eating_detector.py`) using HSV color analysis and posture heuristics was developed and tested. It does not work reliably on Brooks Falls footage — early-season salmon are silver-bodied and flesh is not visible inside the bear's bounding box. It is preserved for reference. See [`docs/04-technical-reference/pixel-eating-detection-research.md`](docs/04-technical-reference/pixel-eating-detection-research.md) for the full experiment report.

### Bear identity augmentation

Assigns cross-video bear identities using PoseSwin embeddings matched against a persistent gallery. Reads an existing `analysis.json` and writes `id_mapping.json`.

```bash
python -m src.identity.identify_bears \
    --video path/to/video.mp4 \
    --analysis predictions/<run>/analysis.json \
    --gallery data/identity/bear_gallery.json
```

**Build a named gallery** from labeled head-crop folders (one folder per bear name):
```bash
python -m src.identity.build_named_gallery \
    --image-root data/identity/gallery_images \
    --output data/identity/named_bear_gallery.json
```

---

## Salmon Jump Counting

### CV-based jump counter (HSV + blob detection)

```bash
# Load saved config
python -m src.detection.salmons.salmon_jump_counter_cv \
    data/raw/salmons/salmon_jump_9.mov \
    --config configs/salmon/config.json

# Override ROI for a new video
python -m src.detection.salmons.salmon_jump_counter_cv \
    video2.mov --config config.json --roi 100 200 400 300

# Tune blob sizes and gap without editing any file
python -m src.detection.salmons.salmon_jump_counter_cv \
    video.mov --min-blob-area 600 --max-blob-area 5000 --min-jump-gap-sec 1.0

# Try new HSV values on a different video
python -m src.detection.salmons.salmon_jump_counter_cv \
    ocean_video.mov --salmon-hsv-lower 5 40 60 --salmon-hsv-upper 25 255 255

# Save a new config once happy with parameters
python -m src.detection.salmons.salmon_jump_counter_cv \
    video.mov --roi 0 400 1280 300 --min-blob-area 500 --save-config river_config.json
```

### Visualization
```bash
# Simplest — ROI from embedded config in result.json
python -m src.detection.salmons.visualize_salmon_jumps video.mov result.json

# Custom output path
python -m src.detection.salmons.visualize_salmon_jumps \
    video.mov result.json --output review/jump9_annotated.mp4
```

Debug frames are saved as 2×2 grids:
```
┌─────────────────────┬─────────────────────┐
│  annotated original │   fg mask (motion)  │
│  ROI dim + cyan box │   white = movement  │
├─────────────────────┼─────────────────────┤
│   colour mask       │  combined mask      │
│   white = salmon    │  fg AND colour      │
└─────────────────────┴─────────────────────┘
│frame 00438 | 7.30s / 15.4s | blobs: 2 | min_area=800 max_area=8000│
```

### Background subtraction counter (interactive)
```bash
# Interactive setup (recommended first run)
python -m src.detection.salmons.salmon_jump_counter_bg \
    --video data/raw/salmons/salmon_jump_9.mov

# Re-run with saved parameters (headless)
python -m src.detection.salmons.salmon_jump_counter_bg \
    --video data/raw/salmons/salmon_jump_9.mov \
    --roi 434,720,710,1062 \
    --line-y 850 \
    --var-threshold 40 \
    --min-area 300 \
    --output data/raw/salmons/salmon_jump_9_result.mp4
```

### Parameters reference
![Salmon parameters](/docs/images/salmon-params.png)

### Count salmon jumps via behavior module (VLM-based)
```bash
python -m src.behavior.count_salmon_jumps \
    --video path/to/video.mp4
```

---

## Environmental Data

All modules fetch public APIs — no API keys required.

### USGS Hydrological Data (water level, stream flow, water temperature)

Uses the nearest USGS gauge on the Brooks River drainage (Kvichak River at Igiugig, site 15300500):
```bash
python -m src.environment.usgs_hydro --date 2023-07-15
python -m src.environment.usgs_hydro --date 2023-07-15 --format csv
python -m src.environment.usgs_hydro --date 2023-07-15 --output my_data.json
python -m src.environment.usgs_hydro --start 2023-07-01 --end 2023-07-31
```

### RAWS Weather (wind, temperature, humidity, precipitation)

Uses NPS RAWS stations near Brooks Falls (Three Forks ~22 km, Coville ~46 km, Pfaff Mine ~71 km):
```bash
python -m src.environment.raws_weather --date 2023-07-15
python -m src.environment.raws_weather --date 2023-07-15 --station ATHF
python -m src.environment.raws_weather --date 2023-07-15 --all-stations
python -m src.environment.raws_weather --date 2023-07-15 --output my_data.json
```

### NADP Precipitation (daily totals)

Uses NADP NTN station AK97 (nearest Brooks Falls station):
```bash
python -m src.environment.nadp_precip --date 2023-07-15
python -m src.environment.nadp_precip --date 2023-07-15 --output my_data.csv --format csv
python -m src.environment.nadp_precip --start 2023-07-01 --end 2023-07-31
```

### Video context resolution

Resolves the datetime and GPS location of a video before making environmental API calls. Camera recordings extract metadata automatically; screen recordings prompt the user:
```bash
python -m src.environment.video_context --video path/to/video.mp4
```

---

## Auto-Annotation with Multi-Model Consensus

### Bear annotation

Combines **Grounding DINO**, **DETR**, and **MegaDetector v5** to generate training labels.

**Extract frames**
```bash
python -m src.preprocessing.annotation_bear.frame_extractor \
  --input path/to/video.mp4 \
  --output data/frames/video_name/ \
  --fps 0.2
```

**Generate labels (auto-approve)**
```bash
python -m src.preprocessing.annotation_bear.multi_model_annotator \
  --input data/frames/video_name/ \
  --output data/auto_labels/ \
  --review-queue data/review_queue/ \
  --prompt "bear" \
  --min-agreement 2 \
  --auto-approve
```

**Visualize results**
```bash
python -m src.preprocessing.annotation_bear.visualize_labels \
  --images data/frames/video_name/subfolder/ \
  --labels data/auto_labels/ \
  --output data/visualized/ \
  --limit 10
```

**Probability calibration (optional, recommended)**
```bash
# Train calibrators
python -m src.preprocessing.annotation_bear.train_calibration \
  --images data/annotation/bears/images/ \
  --labels data/annotation/bears/labels/ \
  --output models/calibrators.pkl \
  --prompt "bear" \
  --iou-threshold 0.5

# Use during annotation
python -m src.preprocessing.annotation_bear.multi_model_annotator \
  ... \
  --calibrator models/calibrators.pkl
```

**Model Arena weights (341 test images)**:
- Grounding DINO: 0.406 weight (89.3% precision, 99.8% recall)
- MegaDetector v5: 0.335 weight (65.6% precision, 84.4% recall)
- DETR: 0.259 weight (35.4% precision, 74.7% recall)

---

### Salmon annotation

Uses a **Stacking meta-learner** combining Grounding DINO, OWL-ViT v2, and Florence-2. Trained on 375 manually cleaned examples.

| Method | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| Voting (min-agreement=2) | ~60-70% | ~40% | ~48% |
| **Stacking (trained)** | **97.5%** | **96.6%** | **97.1%** |

**Extract frames**
```bash
python -m src.preprocessing.annotation_salmon.frame_extractor \
  --input path/to/salmon_video.mp4 \
  --output data/frames/salmon_video/ \
  --fps 1
```

**Option A: Use pre-trained stacking model (recommended)**
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

**Option B: Voting method (for comparison / retraining data)**
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

**Train your own stacking model**
```bash
# Step 1: generate initial annotations (voting method)
# Step 2: visualize for manual review
python -m src.preprocessing.annotation_salmon.visualize_nested \
  --images data/frames/salmon_videos/ \
  --labels data/auto_labels_salmon/ \
  --output data/visualized_salmon/

# Step 3: sync labels after manual cleanup
python sync_labels_from_visualized.py \
  --visualized data/visualized_salmon/ \
  --labels data/auto_labels_salmon/

# Step 4: train meta-learner
python -m src.preprocessing.annotation_salmon.train_stacking \
  --images data/training_salmon/images/ \
  --labels data/training_salmon/labels/ \
  --output models/stacker_salmon_custom.pkl \
  --prompt "fish" \
  --meta-learner rf \
  --device cuda
```

**Prompt recommendation**: Use `--prompt "fish"` — avoids bear misdetections that occur with "salmon" or "jumping salmon".

**Salmon annotation module files**:
```
src/preprocessing/annotation_salmon/
├── auto_annotator_gdino.py
├── auto_annotator_owlvit.py
├── auto_annotator_florence2.py
├── multi_model_annotator.py
├── train_stacking.py
├── predict_stacking.py
└── visualize_nested.py
```

---

## Image Enhancement

Preprocessing utilities to improve detection on low-quality or small-object footage.

```bash
python -m src.preprocessing.enhancement.image_enhancer \
    --input data/frames/video_name/ \
    --output data/enhanced/
```

Techniques: CLAHE contrast enhancement, sharpening, denoising, motion blur reduction. Also see `super_resolution.py` and `upscale_realesrgan.py` for RealESRGAN-based upscaling.

---

## Documentation

Full documentation is in [`docs/`](docs/README.md), organized into:

- [`docs/01-getting-started/`](docs/01-getting-started/installation.md) — installation, first run, hardware guide
- [`docs/02-how-to-guides/`](docs/02-how-to-guides/detect-and-count-bears.md) — task-oriented guides per module
- [`docs/03-annotation-and-training/`](docs/03-annotation-and-training/annotate-bears.md) — labeling, fine-tuning, evaluation
- [`docs/04-technical-reference/`](docs/04-technical-reference/architecture.md) — architecture and module deep-dives
- [`docs/05-extending/`](docs/05-extending/add-vlm-backend.md) — adding backends, adapting to new datasets

**Deepwiki:** https://deepwiki.com/katmai-vision-lab/katmai-cv-pipeline

## Useful Links

- **SharePoint (video data):** https://uwnetid.sharepoint.com/sites/katmai-vision-lab/Shared%20Documents/Forms/AllItems.aspx
- **PoseSwin dataset (identity module):** https://zenodo.org/records/17822054
- **Explore.org cameras:** https://explore.org/livecams/brown-bears
