# Bear Tracking Pipeline: From Training the Model to Generating Trajectories

This document records the full workflow of the bear tracking system — from training the YOLO detector, configuring ByteTrack, doing post-processing merge, to producing the final visualizations — and how to debug it when something goes wrong.

---

## 1. Overall Pipeline

```
Raw video
   ↓
[ YOLO detector ]      ← Per-frame, predicts bbox + confidence
   ↓
[ ByteTrack tracker ]  ← Associates bboxes across frames, assigns track IDs
   ↓
[ Post-merge ]         ← Stitches raw tracks that belong to the same bear
   ↓
[ Post-filter ]        ← Drops short or low-confidence pseudo-trajectories
   ↓
Output: trajectories.json + merge_report.json
   ↓
Visualization: bbox_video.mp4 + trajectories_overlay.mp4
```

Details, parameters, and common pitfalls for each step follow.

---

## 2. Step 1: Train a Good Detection Model

**Key insight**: ByteTrack does not recognize anything — it only associates bboxes across frames. If the detector misses or hallucinates bears, no tracker can save you. **Detector quality is the foundation of everything else.**

### 2.1 Training Data Requirements

We fine-tune YOLOv8 (`bear_detector3`). Quality depends on:

- **Sample diversity**: different lighting (dawn / midday / dusk), angles (top-down / side / close-up), and bears (adult / cub / different fur colors)
- **Hard samples**:
  - **Light-colored bears** — easily confused with rocks or water foam
  - **Partial occlusion** — only the upper body visible in water, or partially behind branches
  - **Close-ups** — bear occupies half the frame, easy to be split into "head" + "body" bboxes
  - **Distant small targets** — tens of pixels in size
- **Negative samples** — moose, deer, otters, humans, and other "looks-like-a-bear-but-isn't" objects must be explicitly labeled **non-bear**. Without negatives, the model tends to flag them as bears (this is why we initially had moose detected as bears).

### 2.2 Annotation Conventions

- bbox tight to the **visible silhouette** of the bear, including head, body, legs (skip underwater parts — the model can't learn from what isn't there)
- One bbox per bear; if two bears overlap closely, label each individually
- Annotation consistency matters — the same bear in successive frames should be labeled by the same rule, otherwise the model learns to output jittery bboxes

### 2.3 Training Command

```bash
python -m src.detection.train \
  --data configs/datasets/bears.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --name bear_detector_v4
```

Trained weights land in `models/trained/bear_detector_v4/weights/best.pt`.

---

## 3. Step 2: Detection Inference Parameters

Once the model is trained, these parameters at inference time directly determine **what quality of detections feed into ByteTrack**.

### 3.1 `--conf` (Confidence Threshold)

The detector outputs a 0–1 confidence per candidate bbox. Anything below this threshold is dropped.

| Value | Behavior |
|---|---|
| `0.7` (recommended start) | Keeps most real bears; filters obvious false positives |
| `0.5` | Higher recall; light / distant bears more likely detected; more false positives |
| `0.9+` | Only super-confident detections; may miss real bears |

**Tuning approach**: Start at `0.7`. If light / distant bears are missed, drop to `0.5`. If you see false positives (rocks / water foam tagged as bears), raise the threshold.

### 3.2 `--imgsz` (Inference Image Size)

Before inference, every frame is resized to `imgsz × imgsz` (square, with padding). **This is the single biggest knob for detection quality.**

| `--imgsz` | Speed (relative to 640) | Use case |
|---|---|---|
| `640` (YOLO default) | 1× | Video is already 640p; quick smoke test |
| `1280` | 3–4× slower | **Sweet spot** for 1080p / 4K video |
| `1792` | 5–6× slower | When light / distant bear recall is still insufficient |
| `2048` | 7–8× slower, ~10 GB VRAM | Practical ceiling on RTX 2080 Ti (11 GB) |

**Must be a multiple of 32.** Common legal values: `640, 832, 960, 1280, 1408, 1536, 1664, 1792, 1920, 2048`.

**Key insight**: The original video doesn't change. What changes is how compressed the "snapshot" the model sees becomes. `imgsz=640` squashes a 1080p frame to 640×360 (with padding) — many details blur out. `imgsz=1280` only halves it.

### 3.3 `--frame-skip` (Frame Subsampling)

Run inference every N frames. `1` = every frame, `2` = every other frame.

| `--frame-skip` | Use case |
|---|---|
| `1` | Final render after tuning is done |
| `2` | Parameter iteration on 60 fps video — 2× faster, no visible jitter |
| `3-5` | Fast verification on long videos |
| `≥10` | Not recommended — fast-moving bears lose tracking |

**Important**: If you raise `frame-skip`, all the frame-count-based thresholds below (ByteTrack and merge) must scale down too. Example: "merge within 5 seconds" is `300` frames at 60 fps full-frame, but becomes `150` at `frame-skip=2`.

---

## 4. Step 3: ByteTrack Tracker

ByteTrack config lives in `configs/trackers/bytetrack.yaml`:

```yaml
tracker_type: bytetrack

# Confidence thresholds
track_high_thresh: 0.25     # High-conf threshold
track_low_thresh: 0.08      # Low-conf (used for recovering occluded targets)
new_track_thresh: 0.5       # Detections must exceed this to start a new track

# Track management
track_buffer: 450           # Frames a lost track is kept before deletion (450 ≈ 15s @ 30fps)
match_thresh: 0.7           # IoU threshold for association

# Misc
min_box_area: 10
mot20: False
fuse_score: True
```

### 4.1 The Two-Stage Association

ByteTrack's distinguishing feature versus older trackers is **two-stage association**:

**Stage 1**: high-confidence detections (≥ `track_high_thresh`) match against all existing tracks via IoU + Kalman motion prediction
- Most clearly visible bears get re-associated to their original track here

**Stage 2**: tracks that didn't match in Stage 1 try again, this time against **low-confidence** detections (between `track_low_thresh` and `track_high_thresh`)
- This is ByteTrack's killer feature — a bear partially occluded by branches or water may drop to conf 0.1; older methods would discard it, but ByteTrack uses it to save the track

After both stages:
- High-conf detections with no match (≥ `new_track_thresh`) → start a new track
- Tracks with no match → enter the buffer for `track_buffer` frames; if they don't reappear, they're deleted

### 4.2 Tuning the Core Parameters

**`track_buffer`** — how long lost tracks are kept
- We use `450` (15s @ 30 fps, 7.5s @ 60 fps). Bears briefly out-of-frame or fully occluded for under 15s keep their ID
- Too large: two similar bears that appear back-to-back can be mistakenly linked
- Too small: a bear leaving the frame for slightly too long gets a new ID

**`match_thresh`** — IoU required for association
- `0.7` is strict, requiring high bbox overlap
- When a stationary bear's bbox shrinks (due to occlusion), IoU drops below 0.7 → association fails → new ID
- Lower to `0.5` for more lenient matching, but two close bears may then get their IDs swapped

**`new_track_thresh`** — threshold to start a new track
- We set this fairly high at `0.5` to prevent noisy low-conf detections from spawning ghost tracks
- Combined with `track_low_thresh: 0.08`, low-conf detections still contribute to Stage 2 association but cannot create new tracks on their own

### 4.3 Even With Tuned Parameters, ByteTrack Still Fragments

Causes:
- The bear is out of frame longer than `track_buffer`
- Camera cut or major zoom: the new bbox shares IoU 0 with the old one
- Long occlusion behind branches / water foam

These cases are beyond ByteTrack's reach — so we added a **post-merge** step.

---

## 5. Step 4: Post-Merge

After ByteTrack finishes, we have a bunch of raw tracks each with its own ID. The post-merge step stitches together raw tracks that **actually belong to the same bear** but were split.

Implementation: `BearDetector._merge_fragmented_tracks` in `src/detection/detector.py`.

### 5.1 Merge Criteria

For each pair of raw tracks `(A, B)`, **all four conditions must hold** to merge:

1. **They never truly co-occurred** — if A and B appeared in the same frame, and this co-occurrence lasted ≥ `cooccurrence_tolerance_frames` (default 60), they are deemed different bears
   - Exception: if their bboxes have **high IoU** (≥ `cooccurrence_artifact_iou`, default 0.3), they're treated as the same bear detected twice by mistake (e.g. "head" + "body" split detection in close-ups), and allowed to merge
2. **Temporally sequential** — A ends before B begins (or vice versa)
3. **Gap within `max_gap_frames`** (default 3600 frames ≈ 2 min @ 30 fps)
4. **Spatial distance within `max_dist_px`** (default 150 pixels) — Euclidean distance from A's last position to B's first position

### 5.2 Candidates Sorted by "Confidence"

All pairs satisfying the four criteria enter a candidate list, sorted ascending by `(gap, dist)`. The tightest merges happen first. This reduces wrong transitive merges (merging A↔B, then merging B↔C even though A↔C should not be merged).

### 5.3 Key Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `max_gap_frames` | 3600 | Max temporal gap |
| `max_dist_px` | 150 | Max spatial distance |
| `cooccurrence_tolerance_frames` | 60 | Co-occurrence shorter than this is treated as detector artifact, not real two bears |
| `cooccurrence_artifact_iou` | 0.3 | Co-occurring bboxes with IoU ≥ this are treated as artifacts |
| `min_duration` | 30 | Drop groups whose total span is shorter than N frames |
| `min_mean_conf` | 0.80 | Drop groups with mean conf below this (long-lived groups ≥ 500f are kept regardless) |

### 5.4 IoU Artifact Detection (Special Note)

In close-up shots, the detector often produces two overlapping bboxes for one bear (one around head/shoulders, one around the body). These two bboxes co-occur in many frames → criterion 1 would label them as "different bears" → merge rejected → one bear becomes two.

The IoU exception fixes this:

```
if mean_iou(A, B) >= 0.3:    # bboxes clearly overlap
    → treat as the same bear, two erroneous bboxes (artifact)
    → allow merge
```

This rule resolved most of the "close-up split detection" cases we encountered.

---

## 6. Step 5: Output Formats

### 6.1 `trajectories.json`

```json
{
  "video": "0507.mov",
  "total_frames": 13953,
  "fps": 60.0,
  "bears": {
    "bear_1": {
      "raw_track_ids": [1, 6, 7, 8],
      "num_detections": 10798,
      "first_frame": 133,
      "last_frame": 13329,
      "trajectory": [
        {"frame": 133, "cx": 678.5, "cy": 513.0, "w": 393, "h": 142, "conf": 0.866, "raw_id": 1},
        ...
      ]
    },
    "bear_2": { ... }
  }
}
```

Each final `Bear N` contains:
- Which `raw_track_id`s were merged to form it
- Per-frame centroid coordinates, bbox dimensions, and confidence

### 6.2 `merge_report.json`

Records the "why" behind every merge decision:

```json
{
  "params": {...},
  "summary": {
    "raw_track_count": 20,
    "final_bear_count": 8,
    "merged_pairs": 9,
    "rejected_pairs": 180
  },
  "raw_tracks": {
    "1": {"first_frame": 0, "last_frame": 298, "duration": 299, "mean_conf": 0.915, "final_bear": "Bear 1"}
  },
  "final_bears": {
    "Bear 1": {
      "raw_ids": [1, 14, 20, 22],
      "merge_chain": [
        {"from": 14, "to": 20, "gap": 25, "dist_px": 86.9}
      ]
    }
  },
  "decisions": [
    {"a": 8, "b": 11, "gap": 0, "dist_px": 240.5, "result": "rejected",
     "reason": "dist_px (240.5) > max_dist_px (150)"}
  ]
}
```

**How to use it**:
- During tuning, look at borderline `decisions[]` (rejected pairs with gap/dist close to threshold) to decide whether to loosen a parameter
- `final_bears[].merge_chain` shows exactly why a given bear is made of certain `raw_id`s

---

## 7. Step 6: Visualization

### 7.1 BBox Video

Per-frame bbox + `Bear N (conf)` label. Rendered by `track_video.py`'s Pass 2.

### 7.2 Trajectory Video (`trajectory_video.py`)

Overlays each bear's history of centroids as a colored polyline. Parameters:

- `--reset-gap-frames` (default 30) — When a bear is absent for more than N frames, clear its old trail. Handles camera zooms / cuts where old pixel coordinates no longer correspond to the new view.
- `--anchor` (default `top`) — Use the top edge of the bbox (the bear's head) as the trail anchor, not the centroid. **Key defense against "person walks past the bear, occludes the lower body, and the bbox-centroid jumps".**
- `--smooth-frames` (default 10) — Moving average over the last N frames to reduce jitter.
- `--fade` (default off) — When off, the entire trail stays at full brightness. When on, older points are dimmed and thinned.

### 7.3 Full Pipeline Commands

```bash
# Recommended workflow (three steps)
python -m src.detection.track_analyze \
  --video "bears/0507.mov" \
  --model models/trained/bear_detector3/weights/best.pt \
  --conf 0.7 --imgsz 1792 --classes 0
# → trajectories.json + merge_report.json, ~2–4 min

python -m src.detection.bbox_video \
  --trajectories "predictions/<run>/trajectories.json"
# → bbox_video.mp4, no YOLO inference needed

python -m src.detection.trajectory_video \
  --trajectories "predictions/<run>/trajectories.json"
# → trajectories_overlay.mp4, no YOLO inference needed
```

Or do everything in one command:

```bash
python -m src.detection.track_video \
  --video "bears/0507.mov" \
  --model models/trained/bear_detector3/weights/best.pt \
  --conf 0.7 --imgsz 1792 --classes 0
```

---

## 8. Lessons Learned

Notes from debugging:

1. **`min_duration` applies to the final group, not individual raw tracks.** An earlier version cut raw tracks shorter than 5s, which deleted "the first 2 seconds of a long-lived bear's track". Now it applies to the merged group's total span.

2. **`frame_skip` rescales all "frame-count" thresholds.** With `--frame-skip 2`, `--min-duration 300` is effectively 10 seconds (@ 60 fps), not 5. Every frame-count threshold must scale with `frame_skip`.

3. **IoU is not "distance".** Two heavily overlapping bboxes (one on the head, one on the body) can have centroids 250 pixels apart. The merge distance check can't rely only on centroid distance — it needs an IoU bypass.

4. **Trail anchor should be the bbox top, not the centroid.** The centroid is highly sensitive to occlusion; the top edge (bear's head) is much more stable.

5. **ByteTrack's "two stages" refers to association, not Kalman filtering.** Each track has one Kalman filter; the two stages are: high-confidence detections match first, then low-confidence detections recover the rest.

---

## 9. Parameter Quick Reference

| Parameter | Default | Affects | Effect of increasing | Effect of decreasing |
|---|---|---|---|---|
| `--conf` | 0.7 | Detection sensitivity | More misses | More false positives |
| `--imgsz` | 640 | Inference detail | Better recall on light/distant bears, slower | Faster, but small targets missed |
| `--frame-skip` | 1 | Frame subsampling | Faster, jittery tracking | More accurate but slow |
| `--max-gap-frames` | 3600 | Merge max temporal gap | Merges across long gaps too | More fragmented bears |
| `--max-dist-px` | 150 | Merge max spatial distance | Distant pairs merged too (risk of wrong merge) | More fragmented bears |
| `--cooccur-artifact-iou` | 0.3 | Detection artifact threshold | More co-occurring pairs treated as artifacts and merged | Stricter, fewer merges |
| `--min-duration` | 30 | Final group minimum length | More short groups dropped | More short groups kept |
| `--min-mean-conf` | 0.80 | Final group minimum mean conf | More low-conf groups dropped | More low-conf groups kept |

---

## Appendix: Related Files

- Detector core: `src/detection/detector.py`
- Main entry points: `src/detection/track_video.py`, `src/detection/track_analyze.py`
- Visualization: `src/detection/trajectory_video.py`, `src/detection/bbox_video.py`
- Debugging: `src/detection/track_dump.py`
- ByteTrack config: `configs/trackers/bytetrack.yaml`
- ByteTrack paper summary: `docs/design-docs/tracking.md`
- Model fine-tuning: `docs/design-docs/MODEL_FINE_TUNING_EN.md`
