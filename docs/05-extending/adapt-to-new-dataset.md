# Adapt to a New Camera or Species

The pipeline was built for Brooks Falls brown bears but every component can be retargeted. This guide covers the most common adaptation: a new camera angle, a different park, or a different wildlife species.

---

## How much adaptation do you need?

| Scenario | What to redo |
|---|---|
| New Katmai camera angle (same bears) | Re-annotate 500–1000 frames, fine-tune |
| Different National Park (same species) | Re-annotate 1000–2000 frames, fine-tune |
| Different species (similar size) | Re-annotate, fine-tune, update `--prompt` |
| Very small or very fast animals | Consider a larger YOLO variant (YOLOv8s/m); re-annotate |

---

## Step 1 — Collect and trim video

Get representative clips from your target camera. If they are screen recordings (YouTube downloads), trim out UI elements with a video editor or `ffmpeg`:

```bash
ffmpeg -i raw_recording.mp4 -vf "crop=1920:1080:0:50" -c:v libx264 trimmed.mp4
```

Aim for diversity: different times of day, different bear densities, different weather/lighting. 500–2000 frames is enough to fine-tune from the existing weights.

---

## Step 2 — Extract frames

```bash
python -m src.preprocessing.annotation_bear.frame_extractor \
    --input path/to/new_video.mp4 \
    --output data/frames/new_camera/ \
    --fps 1
```

---

## Step 3 — Auto-annotate with Grounding DINO

The annotation pipeline is prompt-driven. Change the prompt to match your target:

```bash
python -m src.preprocessing.annotation_bear.multi_model_annotator \
    --input data/frames/new_camera/ \
    --output data/new_labels/ \
    --prompt "brown bear"       # or "polar bear", "elk", "bison", etc.
    --min-agreement 2
```

Spot-check with the visualizer. If the annotation quality is poor (< 80% of animals labeled), try:
- Changing `--prompt` to be more specific
- Lowering `--min-agreement` to 1 and doing more manual review
- Adding MegaDetector as a base model (it generalizes to many wildlife species)

---

## Step 4 — Review and clean labels

```bash
python -m src.preprocessing.annotation_bear.review_app \
    --images data/frames/new_camera/ \
    --labels data/new_labels/
```

The review app shows each frame with overlaid boxes. Mark frames to delete or fix. For a well-annotated dataset, target < 5% false positive rate before training.

---

## Step 5 — Create the dataset config

Copy and edit `data/annotation/bears/bear.yaml`:

```yaml
path: data/annotation/new_camera
train: images/train
val: images/val
nc: 1
names: ['bear']   # or ['elk'], ['bison'], etc.
```

Split the dataset:

```bash
python -m src.preprocessing.split_dataset \
    --input data/annotation/new_camera
```

---

## Step 6 — Fine-tune

```bash
python -m src.detection.train \
    --data data/annotation/new_camera/new_camera.yaml \
    --config configs/train_config.yaml
```

Key settings in `configs/train_config.yaml`:
- `model`: start from `models/trained/bear_detector3/weights/best.pt` (transfer from existing weights) rather than `yolov8n.pt` — this converges much faster for bear-adjacent tasks
- `epochs`: 10–30 for a new camera angle; up to 100 for a very different species
- `imgsz`: 640 for speed, 1280 for detecting distant/small animals

---

## Step 7 — Point the pipeline at new weights

All modules accept a `--model` flag:

```bash
python -m src.detection.bear_count \
    --video-dir new_footage/ \
    --model models/trained/new_camera/weights/best.pt \
    --classes 0 --conf 0.5

python -m src.behavior.analyze_feeding \
    --video new_footage/clip.mp4 \
    --model models/trained/new_camera/weights/best.pt \
    --backend gemini
```

---

## ByteTrack parameters for new scenes

If tracking performance degrades (frequent ID switches), tune `configs/trackers/bytetrack.yaml`:

```yaml
track_high_thresh: 0.4   # raise if too many false tracks
track_low_thresh: 0.15   # raise if splashes/water cause noise
new_track_thresh: 0.85   # raise if new animals are assigned IDs too readily
track_buffer: 300        # reduce if animals leave/re-enter frame frequently
```

For fast-moving animals, lower `track_buffer` (100–150); for large, slow animals that frequently enter/exit, increase it (500+).
