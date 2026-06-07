# Architecture Overview

The Katmai CV Pipeline is a modular command-line system organized in four layers. Each layer can be used independently — you don't have to run the full pipeline to get useful output from any individual module.

---

## System layers

```
┌─────────────────────────────────────────────────────────┐
│                  User Interface Layer                    │
│        CLI (src/cli.py)  ·  main.py  ·  module CLIs    │
└──────────┬────────────────────────────────┬─────────────┘
           │                                │
┌──────────▼────────────┐      ┌────────────▼────────────┐
│   AI Perception Layer  │      │  Context & Analytics    │
│                        │      │  Layer                  │
│  ┌─────────────────┐  │      │  ┌──────────────────┐   │
│  │ Bear Detection  │  │      │  │ Environmental    │   │
│  │ (YOLOv8n)       │  │      │  │ Data Module      │   │
│  └────────┬────────┘  │      │  │ (USGS/RAWS/NADP) │   │
│           │            │      │  └──────────────────┘   │
│  ┌────────▼────────┐  │      │  ┌──────────────────┐   │
│  │ Bear Tracking   │  │      │  │ Integration      │   │
│  │ (ByteTrack)     │  │      │  │ (merge all       │   │
│  └────────┬────────┘  │      │  │  module outputs) │   │
│           │            │      │  └──────────────────┘   │
│  ┌────────▼────────┐  │      └────────────┬────────────┘
│  │ Feeding Event   │  │                   │
│  │ Detection (VLM) │  │      ┌────────────▼────────────┐
│  └─────────────────┘  │      │      Output Layer        │
│  ┌─────────────────┐  │      │  JSON · CSV · MP4        │
│  │ Salmon Jump     │  │      │  Annotated video         │
│  │ Estimation (CV) │  │      │  Trajectory data         │
│  └─────────────────┘  │      └─────────────────────────┘
└───────────────────────┘
```

---

## Module inventory

### `src/detection/` — Bear detection and counting

| File | Role |
|---|---|
| `detector.py` | `BearDetector` class — wraps YOLOv8 for train/predict/track/batch |
| `bear_count.py` | Batch counting CLI across multiple videos |
| `track_video.py` | ByteTrack integration, annotated video output |
| `track_dump.py` | Debug output for raw track data |
| `trajectory_video.py` | Renders movement trails from `trajectories.json` |
| `evaluate.py` | mAP / precision / recall / counting accuracy |
| `train.py` | Fine-tuning wrapper around Ultralytics training loop |
| `salmons/` | Salmon jump counting (two approaches) |

### `src/behavior/` — Feeding behavior analysis

| File | Role |
|---|---|
| `analyze_feeding.py` | Main entry point — YOLO + ByteTrack + VLM, writes `analysis.json` |
| `feeding_viewer.py` | Renders demo video overlaying behavior labels on each bear |
| `count_salmon_jumps.py` | Background-subtraction jump counter (MOG2 + tripwire) |
| `backends/` | Pluggable VLM backends: `anthropic_claude.py`, `openai_gpt4o.py`, `gemini.py`, `molmo2.py` |
| `pixel_eating_detector.py` | HSV color + posture heuristic (research artifact — does not work reliably) |
| `xclip_behavior_recognition.py` | X-CLIP video classification (experimental) |

### `src/identity/` — Bear individual identification (optional)

| File | Role |
|---|---|
| `poseswin_identifier.py` | PoseSwin Re-ID model wrapper — produces 512-d face embeddings |
| `face_detector.py` | Faster-RCNN bear head detector (crops for Re-ID) |
| `identify_bears.py` | Matches crop embeddings against gallery, writes `id_mapping.json` |
| `build_named_gallery.py` | Builds a named gallery JSON from reference images |
| `add_to_gallery.py` | Adds new bears to an existing gallery |

### `src/environment/` — Environmental data

| File | Role |
|---|---|
| `usgs_hydro.py` | Fetches USGS river gauge data (Kvichak River, site 15300500) |
| `raws_weather.py` | Fetches RAWS weather station data (Three Forks, Coville, Pfaff Mine) |
| `nadp_precip.py` | Fetches NADP daily precipitation (station AK97) |
| `video_context.py` | Resolves recording datetime from video metadata; prompts for screen recordings |

### `src/preprocessing/` — Annotation pipelines

| Folder | Role |
|---|---|
| `annotation_bear/` | Multi-model consensus bear labeling (gDINO + DETR + MegaDet) |
| `annotation_salmon/` | Stacking meta-learner salmon labeling (gDINO + OWL-ViT + Florence-2) |

---

## Configuration

Central paths and device detection live in `src/config.py`:

```python
PROJECT_ROOT   # repo root
DATA_DIR       # data/
MODELS_DIR     # models/
PREDICTIONS_DIR  # predictions/

get_device()   # returns "cuda" | "mps" | "cpu"
```

Model hyperparameters are in `configs/train_config.yaml`. ByteTrack parameters are in `configs/trackers/bytetrack.yaml`.

---

## Data flow for a full pipeline run

```
Video file
    │
    ▼ frame extraction (OpenCV VideoCapture, configurable stride)
Frames
    │
    ▼ YOLO inference (BearDetector.predict_video)
Bounding boxes + confidence scores
    │
    ▼ ByteTrack (two-stage IoU + Kalman)
Persistent track IDs + trajectories
    │
    ├──▶ VLM backend (analyze_feeding.py)
    │         │
    │         ▼ analysis.json (timestamped behavior per bear)
    │
    ├──▶ Salmon jump counter (count_salmon_jumps.py)
    │         │
    │         ▼ jump counts + timestamps
    │
    └──▶ Environmental data (video_context.py + env modules)
              │
              ▼ env_context block appended to analysis.json
```

---

## Output layer

All outputs land in `predictions/`:

```
predictions/
├── batch_counting/
│   └── batch_<timestamp>/
│       ├── batch_results.json
│       └── batch_summary.csv
├── <video_stem>_tracked.mp4
├── <video_stem>/
│   └── trajectories.json
└── <video_stem>_feeding_analysis/
    ├── analysis.json
    ├── id_mapping.json          # if identity module ran
    └── summary.txt
```

`analysis.json` schema:
```json
{
  "video": "path/to/clip.mp4",
  "fps": 60.0,
  "interval_sec": 0.5,
  "backend": "molmo2",
  "summary": "Natural-language video summary",
  "environmental_context": { ... },
  "entries": [
    {
      "timestamp_sec": 1.5,
      "frame_idx": 90,
      "bears": {
        "1": {
          "bbox": [12, 473, 524, 892],
          "conf": 0.96,
          "behavior": "[CATCHING] The bear is clamping its jaws around a salmon..."
        }
      }
    }
  ]
}
```
