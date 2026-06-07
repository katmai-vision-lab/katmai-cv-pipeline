# Your First Run

After [installing](installation.md), this guide walks you through the interactive TUI and your first end-to-end detection.

---

## The TUI (recommended entry point)

Launch the interactive menu:

```bash
python -m src.cli
```

You'll see a menu like:

```
Katmai CV Pipeline
──────────────────────────────────────
  1  Detect bears
  2  Track bears
  3  Batch count bears
  4  Detect feeding events
  5  Count salmon jumps
  6  Fetch environmental data
  7  Evaluate model
  8  Train model
  q  Quit
──────────────────────────────────────
Select option:
```

Navigate with single keypresses. Each option prompts you for inputs step-by-step.

---

## First detection (5 minutes)

Select **option 1 → Detect bears**. When prompted:

- **Video path:** path to any `.mp4` or `.mkv` file with bears
- **Model path:** press Enter to use the default fine-tuned weights (`models/trained/bear_detector3/weights/best.pt`)
- **Confidence threshold:** `0.25` is a good default; raise to `0.5` to reduce false positives

The pipeline runs YOLO inference and saves an annotated video to `predictions/`. It also prints per-frame bear counts to the terminal.

---

## First full pipeline run (command-line)

If you prefer CLI over TUI:

```bash
python -m src.main \
    --mode full \
    --video path/to/your/video.mp4 \
    --skip-train \
    --conf 0.25
```

This runs detection → tracking → counting and saves outputs to `predictions/`.

---

## Understanding the output

After any detection run, outputs land in `predictions/`:

```
predictions/
└── batch_counting/
    └── batch_20260606_143022/
        ├── batch_results.json      # full structured output per video
        └── batch_summary.csv       # flat table: video, bear count, etc.
```

For tracking runs, you also get:
```
predictions/
├── <video_stem>_tracked.mp4       # annotated video with track IDs
└── <video_stem>/
    └── trajectories.json           # per-bear trajectory coordinates
```

For feeding behavior runs:
```
predictions/
└── <video_stem>_feeding_analysis/
    ├── analysis.json               # timestamped VLM outputs per bear per frame
    └── summary.txt                 # natural-language video summary
```

---

## What to try next

| Goal | Guide |
|---|---|
| Process many videos at once | [Detect & Count Bears → Batch counting](../02-how-to-guides/detect-and-count-bears.md#batch-counting) |
| Get per-bear behavior labels | [Feeding Behavior](../02-how-to-guides/feeding-behavior-and-identity.md) |
| Understand the output JSON schema | [Architecture → Output Layer](../04-technical-reference/architecture.md#output-layer) |
