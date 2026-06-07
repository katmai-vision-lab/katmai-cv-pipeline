# Detect and Count Bears

This guide covers single-video detection, batch counting across many videos, and reading the output.

---

## Single video — detection only

Run YOLO inference on one video and save an annotated output:

```bash
python -m src.detection.predict \
    --video path/to/clip.mp4 \
    --conf 0.25
```

Output: annotated video saved to `predictions/`.

---

## Single video — with tracking

Assign persistent bear IDs across frames using ByteTrack:

```bash
python -m src.detection.track_video \
    --video path/to/clip.mp4 \
    --model models/trained/bear_detector3/weights/best.pt \
    --classes 0 \
    --conf 0.7 \
    --imgsz 1280 \
    --frame-skip 1
```

Key parameters:
- `--conf 0.7` — higher threshold reduces false positives in tracking mode
- `--imgsz 1280` — larger input catches distant bears; use `640` on CPU for speed
- `--frame-skip 1` — process every frame; increase to 2–5 for faster (less accurate) runs

---

## Batch counting

Count bears across an entire folder of videos. Produces a CSV summary suitable for spreadsheet analysis:

```bash
python -m src.detection.bear_count \
    --video-dir path/to/videos/ \
    --model models/trained/bear_detector3/weights/best.pt \
    --pattern "*.mkv" \
    --classes 0 \
    --conf 0.7 \
    --tracking
```

- `--pattern` accepts glob patterns (`*.mp4`, `*.mkv`, `2025-07-*.mp4`, etc.)
- `--tracking` adds ByteTrack and outputs unique bear estimates in addition to peak counts
- Remove `--tracking` for faster counting-only mode

**Output files** in `predictions/batch_counting/batch_<timestamp>/`:

| File | Contents |
|---|---|
| `batch_results.json` | Full structured output: per-video frame arrays, aggregate stats, run config |
| `batch_summary.csv` | Flat table — one row per video with peak count, avg per frame, unique estimate, processing time |

---

## Reading the output

`batch_summary.csv` columns:

| Column | Description |
|---|---|
| `video_name` | Filename |
| `unique_bear_estimate` | Estimated distinct individuals (max frame count in detection mode; unique ByteTrack IDs in tracking mode) |
| `max_bears_in_frame` | Peak simultaneous count |
| `avg_bears_per_frame` | Mean across all sampled frames |
| `total_detections` | Sum of per-frame counts |
| `frames_analysed` | Number of frames sampled |
| `processing_time_sec` | Wall-clock seconds |

`batch_results.json` structure:

```json
{
  "videos": {
    "clip.mp4": {
      "max_bears_in_frame": 5,
      "unique_bear_estimate": 7,
      "avg_bears_per_frame": 3.1,
      "frame_data": [
        {"frame_id": 0, "bear_count": 3, "track_ids": [1, 2, 3]},
        ...
      ]
    }
  },
  "aggregate": {
    "total_unique_bears": 12,
    "mean_bears_per_video": 4.2
  }
}
```

---

## Trajectory visualization

After a tracking run, visualize movement trails:

```bash
python -m src.detection.trajectory_video \
    --trajectories "predictions/<run>/trajectories.json" \
    --trail-frames 300 \
    --thickness 3
```

---

## Tuning tips

- **Too many false positives** → raise `--conf` (try 0.5–0.7)
- **Missing distant bears** → raise `--imgsz` to 1280 or 1920
- **Slow on CPU** → raise `--frame-skip` to 5–10; you'll still get accurate peak counts
- **ID switching in tracking** → lower `--conf` slightly (0.4); ByteTrack's second stage recovers low-confidence detections during occlusions

See [ByteTrack Pipeline →](../04-technical-reference/bytetrack-pipeline.md) for tracker parameter details.
