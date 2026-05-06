# Bear Feeding Behavior Viewer — Design Plan

## Goal

Build a demo tool that analyzes brown bear feeding behavior in video footage.
The output is a side-by-side display:

- **Left panel**: ByteTrack-annotated video with per-bear bounding boxes and IDs (Bear 1, Bear 2, ...)
- **Right panel**: Pre-generated behavior descriptions synced to video timestamps

Example right panel output at t=00:12:
```
Bear 1: Standing at the lip of the falls, watching for fish
Bear 2: Actively catching a salmon mid-air at the waterfall crest
Bear 3: Retreating to the bank, salmon in mouth
```

---

## Architecture: Two-Step Pipeline

Real-time inference is not required. Quality takes priority over speed.
Analysis is pre-computed and saved; the viewer replays it in sync.

```
Step 1 — Analyze                          Step 2 — View
─────────────────────────────             ──────────────────────────────
raw video                                 raw video + analysis.json
    │                                          │              │
    ├─ YOLO + ByteTrack                        ├─ ByteTrack   └─ timestamp lookup
    │   → bear IDs + bboxes per frame          │   (re-run inline,
    │                                          │    same settings)
    ├─ sample every N seconds                  │
    │                                          └─ left panel + right panel
    └─ Molmo2-8B vision model                      side-by-side OpenCV window
        → behavior description per bear
        → saved to analysis.json
```

---

## Step 1: `src/behavior/analyze_feeding.py`

### Inputs
| Argument | Default | Description |
|---|---|---|
| `--video` | required | Path to raw video file |
| `--output` | `predictions/<stem>_feeding_analysis/analysis.json` | Output JSON path |
| `--interval` | `2.0` | Analyze every N seconds |
| `--model` | bear_detector3 | YOLO model path |
| `--conf` | `0.25` | Detection confidence threshold |
| `--vision-model` | `allenai/Molmo2-8B` | HuggingFace vision model name |
| `--save-frames` | off | Save sampled frames for debugging |

### Process
1. Run YOLO + ByteTrack on the full video (streaming)
   - Collect per-frame: `{raw_track_id: (x1, y1, x2, y2, conf)}`
2. Apply `_merge_fragmented_tracks` — same logic as `track_and_save_video`
   - Produces stable display IDs (Bear 1, Bear 2, ...) matching the viewer
3. Sample one frame every N seconds
4. For each sampled frame:
   - Build a prompt with human-readable position hints (e.g. "Bear 1 — top-left of frame")
   - Send full frame + prompt to Molmo2-8B
   - Parse response into per-bear behavior strings
5. Save `analysis.json`

### Output: `analysis.json`
```json
{
  "video": "path/to/video.mp4",
  "fps": 60.0,
  "total_frames": 702,
  "frame_size": [732, 1292],
  "interval_sec": 2,
  "vision_model": "allenai/Molmo2-8B",
  "created": "2026-04-13T...",
  "entries": [
    {
      "timestamp_sec": 0.0,
      "frame_idx": 0,
      "bears": {
        "1": {
          "bbox": [120, 80, 340, 290],
          "conf": 0.87,
          "position_hint": "left side of frame",
          "behavior": "Standing at the bank, scanning the water surface"
        }
      },
      "raw_response": "Bear 1: Standing at the bank..."
    }
  ]
}
```

### Vision Model Prompt
```
You are analyzing a wildlife video frame of brown bears fishing for salmon
at a river (such as Brooks Falls in Katmai National Park, Alaska).

Detected bears in this frame:
  Bear 1 — left side of frame
  Bear 2 — center-right of frame

For each detected bear, write one concise sentence (under 15 words) describing
their specific feeding behavior. Focus on posture, water interaction, and salmon activity.

Respond strictly in this format:
Bear 1: <behavior>
Bear 2: <behavior>

Only describe bears listed above. Do not add extra commentary.
```

---

## Step 2: `src/behavior/feeding_viewer.py`

### Inputs
| Argument | Default | Description |
|---|---|---|
| `--video` | required | Path to raw video (same as used in analysis) |
| `--analysis` | required | Path to `analysis.json` |
| `--model` | bear_detector3 | YOLO model path |
| `--conf` | `0.25` | Detection confidence threshold |
| `--speed` | `1.0` | Playback speed multiplier |
| `--panel-width` | `420` | Right panel width in pixels |

### Display Layout
```
┌─────────────────────────────┬────────────────────────┐
│                             │  BEAR FEEDING ANALYSIS │
│   ByteTrack video           │  t = 00:14             │
│   with bounding boxes       │  ─────────────────── │
│   and bear IDs              │  ● Bear 1              │
│                             │    Standing at the lip │
│   ┌──────────┐              │    watching for fish   │
│   │ Bear 1   │              │                        │
│   └──────────┘              │  ● Bear 2              │
│             ┌──────────┐    │    Catching salmon      │
│             │ Bear 2   │    │    mid-air at falls    │
│             └──────────┘    │                        │
│                             │  SPACE pause  ←→ seek  │
└─────────────────────────────┴────────────────────────┘
```

### Sync Logic
- Pre-run ByteTrack on the video before playback (same merge logic as analysis step)
- At each displayed frame, compute `current_sec = frame_idx / fps`
- Find the `analysis.json` entry with the nearest `timestamp_sec`
- If `dist > interval * 0.6`, mark the entry as stale (text dimmed)

### Controls
| Key | Action |
|---|---|
| `Space` | Pause / Resume |
| `← / a` | Seek back 10 seconds |
| `→ / d` | Seek forward 10 seconds |
| `+` | Speed up (×1.5, max 8×) |
| `-` | Slow down (÷1.5, min 0.25×) |
| `Q / Esc` | Quit |

---

## Vision Model: Molmo2-8B

### Why Molmo2-8B
- Best open-weight video understanding benchmark (beats Qwen3-VL-8B, GPT-5, Gemini 2.5 Pro on VideoPoint)
- Native spatial pointing — identifies exactly where in the frame each bear is acting
- float16 ~18 GB, fits across 2× RTX 2080 Ti (22 GB) via `device_map="auto"`
- No bitsandbytes required
- Fully open weights and training data (Allen Institute for AI)

### Loading
```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

processor = AutoProcessor.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo2-8B",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
```

### Estimated throughput
- ~3–8 seconds per frame on 2× RTX 2080 Ti
- 2-second interval → ~1.5–4× real-time analysis speed
- A 10-minute video with 2s interval = 300 frames → ~15–40 min to analyze

---

## File Structure

```
src/behavior/
├── analyze_feeding.py          NEW: Step 1 — pre-compute analysis
├── feeding_viewer.py           NEW: Step 2 — demo viewer
├── xclip_behavior_recognition.py   (existing, unused here)
├── detect_eating_fish.py           (existing)
└── simple_eating_detector.py       (existing)

predictions/
└── <video_stem>_feeding_analysis/
    ├── analysis.json
    └── sampled_frames/   (optional, --save-frames)
```

---

## Usage

```bash
# Step 1 — analyze (run overnight or before demo)
python -m src.behavior.analyze_feeding \
    --video feed/data_video/salmon_jump_2.mov \
    --interval 2 \
    --save-frames

# Step 2 — demo viewer
python -m src.behavior.feeding_viewer \
    --video feed/data_video/salmon_jump_2.mov \
    --analysis predictions/salmon_jump_2_feeding_analysis/analysis.json
```

---

## Dependencies

Already installed: `transformers==4.47.1`, `accelerate`, `opencv-python`, `torch`, `Pillow`

No new packages required.

---

## Open Questions

1. **Interval tuning**: 2s may be too coarse for fast action. Worth testing 1s vs 2s.
2. **Frame sampling strategy**: Pick the frame with the highest average detection confidence within each interval, rather than a fixed stride?
3. **Molmo2 API**: Molmo2-8B is Qwen3-based and may use standard HF `generate()` rather than the original Molmo `generate_from_batch()`. The code tries the Molmo API first, then falls back to standard HF API automatically.
