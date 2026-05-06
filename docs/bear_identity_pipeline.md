# Brown Bear Identity Pipeline — Technical Document

**Author:** Darian Ding
**Date:** May 4, 2026
**Audience:** team members, technical mentors, future maintainers
**Code branch:** `feature/auto-annotation`

---

## 1. Project background

### 1.1 Motivation

In the Winter Quarter design report, the requirement "identify the same bear across videos" (cross-video bear identification) was descoped because ByteTrack only does motion-based per-frame association — there is no appearance-feature Re-ID network. The direct consequences:

- Each video numbers its bears starting from 1 again (`Bear 1, Bear 2, ...`)
- The same physical bear in 5 videos gets 5 different IDs
- Population-level questions like "how many salmon did Otis eat in total" cannot be answered

In February 2026, the EPFL Mathis Lab and Alaska Pacific University co-published the **PoseSwin** paper (Rosenberg et al., *Current Biology*) — the first open-source individual-identification (Re-ID) model for Alaskan coastal brown bears. Alex forwarded the paper and asked whether we could integrate it.

### 1.2 Design goals

1. **Persistence across videos**: the same physical bear gets the same label in different videos
2. **Real names**: not just anonymous `Bear A/B/C` but well-known names like `Plunger`, `Bony_Butt`
3. **Don't break the existing pipeline**: identity is an **add-on** to `analyze_feeding.py` + `feeding_viewer.py`, leaving their core logic untouched
4. **Don't depend on the cloud**: model and gallery live on disk; once an inference run is done, everything works offline

### 1.3 Overall strategy

```
┌─────────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐
│ YOLOv8n     │  │ Molmo2-8B    │  │ Faster-RCNN   │  │ PoseSwin         │
│ + ByteTrack │  │ (behavior)   │  │ (face det.)   │  │ (identity emb.)  │
└──────┬──────┘  └──────┬───────┘  └───────┬───────┘  └────────┬─────────┘
       │ bbox+ID         │ behavior         │ head bbox          │ 512-d embed
       └─────────┬───────┴───────┬──────────┴────────┬───────────┘
                 │               │                   │
                 ▼               ▼                   ▼
              analysis.json  →  id_mapping.json  →  Gallery (persistent JSON)
                          │                          │
                          └──────────┬───────────────┘
                                     ▼
                            feeding_viewer rendering
                            "Plunger [CATCHING] ..."
```

---

## 2. Module breakdown

### 2.1 `PoseSwinIdentifier` — model wrapper

**File:** [`src/identity/poseswin_identifier.py`](../src/identity/poseswin_identifier.py)

Wraps the EPFL-trained pose-aware Swin Transformer Re-ID model.

**Core API:**

```python
identifier = PoseSwinIdentifier(device="cuda:0")
emb_512d = identifier.embed(head_crop_bgr)          # (512,) L2-normalized
embs     = identifier.embed_batch(list_of_crops)    # (N, 512) batched
```

**Key implementation details:**

1. **Swin-Base + custom projection head**:
   - Backbone: `embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32]` (standard Swin-Base)
   - Pose integration: HRNet-W48 produces 13 facial keypoints, injected stage-by-stage into Swin (see paper §3.2)
   - Output projection: 1024 → 512

2. **Config override**: the upstream YAML (`swin_base_patch4_window7_224_22k.yaml`) ships with `EMBED_DIM=512`, which is wrong; we explicitly override it to `128` or the checkpoint shape doesn't match.

3. **L2 normalization**: every embedding leaving the wrapper is normalized to unit length so cosine similarity equals dot product downstream.

### 2.2 `BearFaceDetector` — Faster-RCNN head detection

**File:** [`src/identity/face_detector.py`](../src/identity/face_detector.py)

Converts the upstream mmdetection 2.x Faster-RCNN (`latest.pth`, 330 MB) **and loads it into torchvision's FasterRCNN**, sidestepping the mmdet 2.x + mmcv 1.3.17 vs PyTorch 2.6 compatibility hell.

**Why convert instead of installing mmdet?**

- mmdet 2.22 / mmcv 1.3.17 is from 2021, and has multiple ABI conflicts with CUDA 12.4 + PyTorch 2.6
- Installing mmdet pulls in roughly 2 GB of dependencies (mmcv-full, mmengine, etc.)
- The Faster-RCNN architecture is **identical** in both frameworks (same ResNet-50 backbone, same FPN, same anchor config)
- The only real differences are naming conventions and a couple of conventions (e.g. class-index ordering)

**Weight conversion mapping (the core of it):**

| mmdet 2.x name | torchvision name | Note |
|---|---|---|
| `backbone.{conv1,bn1,layer1-4}.*` | `backbone.body.{conv1,bn1,layer1-4}.*` | prefix `body.` |
| `neck.lateral_convs.{i}.conv.*` | `backbone.fpn.inner_blocks.{i}.0.*` | torchvision wraps in `Conv2dNormActivation` |
| `neck.fpn_convs.{0..3}.conv.*` | `backbone.fpn.layer_blocks.{i}.0.*` | same |
| `neck.fpn_convs.4.*` | *(dropped)* | torchvision uses parameterless `LastLevelMaxPool` to produce P6 |
| `rpn_head.rpn_conv.*` | `rpn.head.conv.0.0.*` | torchvision RPN head is also wrapped |
| `rpn_head.rpn_cls.*` | `rpn.head.cls_logits.*` | direct copy |
| `rpn_head.rpn_reg.*` | `rpn.head.bbox_pred.*` | direct copy |
| `roi_head.bbox_head.shared_fcs.{0,1}.*` | `roi_heads.box_head.fc{6,7}.*` | rename |
| `roi_head.bbox_head.fc_cls.*` | `roi_heads.box_predictor.cls_score.*` | **must swap rows 0/1** (see below) |
| `roi_head.bbox_head.fc_reg.*` | `roi_heads.box_predictor.bbox_pred.*` | **shape (4,) → (8,), only fill the bear-head slot** |

**Two critical gotchas:**

1. **Class-index convention is reversed**
   - mmdet 2.x: `cls_score` row 0 = bear_head, row 1 = background
   - torchvision: `cls_score` row 0 = background, row 1 = bear_head
   - Loading without a row swap classifies **everything as foreground** (every proposal scores 1.000, you get 100 false positives)

2. **Regression-head dimension mismatch**
   - Our mmdet config has `reg_class_agnostic=True` → `fc_reg.weight: (4, 1024)`, a single bbox-delta head
   - torchvision forces class-specific regression → `bbox_pred.weight: (8, 1024)`, one set per class
   - Conversion: zero out the bg slot (filtered at inference anyway), fill the bear_head slot with the mmdet weights

**Debug experience**: the conversion bug looks like "100 fake detections, every score = 1.000". Debug by hooking `roi_heads.box_predictor`'s forward pass and inspecting the raw cls logits — if the bg logits are uniformly very negative and the fg logits uniformly very positive, the class indices are swapped.

**Inference API:**

```python
detector = BearFaceDetector(device="cuda:0", score_threshold=0.3)
heads = detector(frame_bgr)              # [(x1,y1,x2,y2,score), ...]
best  = detector.best_head_crop(frame)   # take the highest-scoring head crop, with padding
```

### 2.3 `Gallery` — persistent embedding store

**File:** the `Gallery` class in `src/identity/poseswin_identifier.py`

JSON-serialized `name → embedding` data structure. Supports:

```python
gallery = Gallery.load("data/identity/named_bear_gallery.json")
name, sim = gallery.match(query_emb, threshold=0.6)  # nearest neighbor
gallery.add_anonymous(query_emb)                     # auto-name "Bear A/B/C/..."
gallery.reinforce(name, query_emb)                   # add another exemplar to a known bear
gallery.save()
```

**Schema:**

```jsonc
{
  "next_anon_idx": 3,
  "entries": [
    {
      "name": "Plunger",
      "embeddings": [[0.123, -0.045, ...]],  // (512,) L2-normalized
      "n_observations": 15
    },
    ...
  ]
}
```

**Multi-shot averaging**: each known bear stores up to 5 head-crop embeddings; matching uses the mean vector (re-normalized). New observations are added through `reinforce()` in a rolling buffer; old ones get evicted, letting the gallery adapt to seasonal/age-related appearance shifts.

### 2.4 `head_crop_from_face_detector` — smart crop selection

**File:** the helper inside [`src/identity/identify_bears.py`](../src/identity/identify_bears.py)

The glue that wires the face detector into the identity pipeline. Logic:

```python
def head_crop_from_face_detector(frame, bear_bbox, face_detector):
    if face_detector is not None:
        # Run Faster-RCNN on the FULL frame (not on the YOLO crop —
        # full-frame resolution is higher and detection is more accurate)
        face_dets = face_detector(frame)
        # Keep faces whose center sits inside the bear bbox (≥ 70% containment)
        candidates = [(box, score, frac) for box, score, frac in face_dets
                      if bbox_contains(bear_bbox, box) >= 0.7]
        if candidates:
            return crop, "face_detector", best_score
    # Fallback: heuristic — top 50% × center 60% of the bbox
    return heuristic_crop(frame, bear_bbox), "heuristic", None
```

**Why detect on the full frame instead of inside the YOLO crop?**

Measured:
- Full frame (1426×794): 2 bear faces detected, scores 0.99
- YOLO bbox crop (433×557): 1 false face detected, score 0.54

Reason: Faster-RCNN was trained at ~1000–2000 px resolution. After YOLO cropping, resolution is too low, the target is too large, and surrounding context is lost.

---

## 3. Data flow: one full run

### 3.1 Inputs

- A video file (MP4/MOV, any resolution)
- An `analysis.json` previously produced by `analyze_feeding.py` (with bbox per bear per frame)

### 3.2 Processing steps

```
                    [1] best_frames_per_bear()
analysis.json  ───►  for each ByteTrack ID, take the top-K=10 highest-conf frames
                    │
                    ▼
                    [2] for those K frames:
                        - run face detector on the full frame
                        - find the face inside the bear bbox
                        - found → use the face crop
                        - not found → fall back to the heuristic crop
                    │
                    ▼
                    [3] PoseSwinIdentifier.embed_batch()
                        K head crops → K × 512-dim embeddings
                    │
                    ▼
                    [4] mean + L2 normalize → 1 representative embedding
                    │
                    ▼
                    [5] Gallery.match()
                        nearest neighbor over 98 named bears + previously
                        accumulated anonymous bears
                        cos sim ≥ 0.45 → reuse existing name
                        else            → gallery.add_anonymous()
                    │
                    ▼
                    [6] write id_mapping.json
                        + persist gallery.json
```

### 3.3 Output

`predictions/<video>/id_mapping.json`:

```json
{
  "video": "/path/to/video.mp4",
  "gallery_path": "data/identity/named_bear_gallery.json",
  "threshold": 0.45,
  "mapping": {
    "1": {
      "name": "Plunger",
      "similarity": 0.851,
      "is_new": false,
      "n_shots": 10,
      "n_face_crops": 2,
      "n_heuristic_crops": 8,
      "max_conf": 0.97
    }
  }
}
```

`feeding_viewer.py` reads this through its `--id-mapping` flag and replaces "Bear 1" in the right-hand panel with "Plunger".

---

## 4. Building the named gallery

**File:** [`src/identity/build_named_gallery.py`](../src/identity/build_named_gallery.py)

A one-shot script that builds a "named bears" gallery from PoseSwin's training set.

### 4.1 Data source

- **Source**: `data/reid_annotations/test_on_2022/train_iid.csv` from `Public_release.zip` (35,986 rows × 98 unique bears)
- **Images**: `Public_release/images/{2017-2021}_heads/images/*.JPG` (already cropped head images by their face detector)
- **Bear-name encoding**: the CSV's `id` column is literally the name ("Plunger", "Bony_Butt", "Simba", ...)
- **Sampling strategy**: 15 images per bear, spread across years; 1468 in total
- **Total GPU time**: ~7 minutes (98 bears × 15 images / batch_size=8)

### 4.2 Important caveat

**The 98 named bears are mostly from the McNeil River Bear Sanctuary, NOT Brooks Falls / Brooks River.**

- McNeil researchers use descriptive names (Plunger, Hotlips, Aardvark, ...)
- Brooks Falls bears get NPS numeric IDs + nicknames (480 Otis, 128 Grazer, 747, 856, ...)
- The two populations may share a few individuals (bears do roam between drainages), but **most are different**

Practical implication: when we run the identifier on a Brooks Falls video, **the name "Plunger" the model returns means "the bear in the PoseSwin training set whose face most resembles the bear in this video" — not the actual identity**. To recognize real Brooks Falls individuals, build a Brooks-Falls-specific gallery using the NPS *Bears of Brooks River* eBook as ground truth.

---

## 5. Empirical results

### 5.1 Heuristic vs face-detector crop

Cosine similarity for 3 test bears (across 2 videos), higher = more confident match:

| Bear ID | Heuristic crop | Face detector + fallback | Improvement |
|---|---|---|---|
| Gully clip → Plunger | 0.677 | **0.851** | **+0.174** |
| salmon_jump_2 #1 → Bony_Butt | 0.745 | **0.850** | **+0.105** |
| salmon_jump_2 #2 → Simba | 0.792 | 0.760 | -0.032 |

The 3 final matched names didn't change, but 2 out of 3 confidences improved substantially while 1 dropped slightly. Gully going from 0.677 → 0.851 lifted it from "barely above 0.6 threshold" to "high confidence".

### 5.2 Face-detector coverage

On 48 sampled frames of the Gully clip:

- 19% of frames have a detectable bear face (score > 0.3)
- Top score 0.89
- Frames where detection fails are mostly: bear with head down chasing fish, bear with back to camera, bear in spray / backlight

Strategy: top-K=10 frame sampling guarantees each bear gets ≥ 1 face-detector crop, with the rest filled by the heuristic.

### 5.3 Performance overhead

| Stage | GPU time (single RTX 2080 Ti) | Note |
|---|---|---|
| Load PoseSwin model | ~7 s | one-time |
| Load face detector | ~2 s | one-time |
| Face detection (full frame 794×1426) | ~0.15 s/frame | K=10 frames per bear = 1.5 s |
| PoseSwin embedding (batch 10) | ~0.4 s | once per bear |
| Gallery match | ~1 ms | NumPy matmul |
| **Total**: ~12 s per bear, per video | ~10–15 s | |

`identify_bears.py` uses GPU but **runs only once** per video, caching the mapping to JSON. Subsequent `feeding_viewer.py` rendering only reads the JSON — no GPU needed.

---

## 6. Limitations

| Limitation | Detail | Mitigation |
|---|---|---|
| **Training-set mismatch** | PoseSwin training set is mostly McNeil River bears; Brooks Falls bears (Otis, Grazer, etc.) are not in the gallery | Build a Brooks Falls gallery using the NPS *Bears of Brooks River* eBook |
| **CC BY-NC 4.0 license** | Both model weights and training data are non-commercial | Wait for Alex's commercial-scope answer, or use only the method (Swin + metric learning) and re-train on permissive data |
| **Low recall in extreme poses** | Face detector fails when the bear has its head down eating fish or its back to the camera | Heuristic fallback covers it; or sample more frames via top-k |
| **Doesn't separate co-located bears within a single video** | If ByteTrack accidentally merges two bears into one track, the identifier will hand them one name | Upstream issue — needs ByteTrack tuning or a stronger tracker |
| **Match threshold is hand-tuned** | 0.45 was set based on a 3-bear sample — not statistically validated | ROC analysis on a 100+ frame human-labeled ground-truth set |

---

## 7. Usage examples

### 7.1 Full 3-step pipeline

```bash
cd /home/katmai/katmai-cv-pipeline

# Step 1 — behavior analysis (per-bear bbox + 5-stage labels per frame)
WANDB_MODE=disabled venv/bin/python3 -m src.behavior.analyze_feeding \
    --video feed/data_video/<clip>.mp4 \
    --interval 0.25

# Step 2 — identity assignment (Faster-RCNN head detection + PoseSwin matching)
WANDB_MODE=disabled venv/bin/python3 -m src.identity.identify_bears \
    --video feed/data_video/<clip>.mp4 \
    --analysis predictions/<clip>_feeding_analysis/analysis.json \
    --gallery data/identity/named_bear_gallery.json \
    --threshold 0.45

# Step 3 — render the identity-aware demo video
WANDB_MODE=disabled venv/bin/python3 -m src.behavior.feeding_viewer \
    --video feed/data_video/<clip>.mp4 \
    --analysis predictions/<clip>_feeding_analysis/analysis.json \
    --id-mapping predictions/<clip>_feeding_analysis/id_mapping.json
```

### 7.2 Common CLI options

```bash
# Heuristic crop only — fast, no face detector
... identify_bears --no-face-detector ...

# Stricter face detection threshold (fewer false positives)
... identify_bears --face-score-threshold 0.5 ...

# Dry run — don't update the gallery
... identify_bears --dry-run ...

# Use the anonymous gallery instead of the named one
# (suitable when you only want cross-video persistence, not real names)
... identify_bears --gallery data/identity/bear_gallery.json ...

# Change the matching threshold
... identify_bears --threshold 0.55 ...
```

### 7.3 Rebuilding the named gallery

If the training data is updated or you switch source data:

```bash
# 1. Drop your per-bear head crops in data/identity/gallery_images/<bear_name>/*.JPG
# 2. Recompute embeddings
WANDB_MODE=disabled venv/bin/python3 -m src.identity.build_named_gallery \
    --image-root data/identity/gallery_images \
    --output     data/identity/named_bear_gallery.json \
    --max-per-bear 15
```

---

## 8. File listing

| File | Purpose | LOC |
|---|---|---|
| [`src/identity/__init__.py`](../src/identity/__init__.py) | package marker | 0 |
| [`src/identity/poseswin_identifier.py`](../src/identity/poseswin_identifier.py) | PoseSwin model wrapper + Gallery class + heuristic head crop | ~210 |
| [`src/identity/face_detector.py`](../src/identity/face_detector.py) | Faster-RCNN head detector + mmdet→torchvision weight conversion | ~160 |
| [`src/identity/identify_bears.py`](../src/identity/identify_bears.py) | CLI entry point: analysis.json + video → id_mapping.json | ~210 |
| [`src/identity/build_named_gallery.py`](../src/identity/build_named_gallery.py) | one-shot: build named gallery from training head crops | ~80 |
| [`src/behavior/feeding_viewer.py`](../src/behavior/feeding_viewer.py) | modified: added `--id-mapping` flag for rendering real names | ~400 |
| [`data/identity/named_bear_gallery.json`](../data/identity/named_bear_gallery.json) | embedding library for 98 named bears (~200 KB) | — |
| [`data/identity/gallery_images/`](../data/identity/gallery_images/) | 1468 training head crops, organized by bear name | — |
| [`external/BrownBear_ReID/`](../external/BrownBear_ReID/) | upstream repo + 4.2 GB checkpoints (gitignored) | — |

---

## 9. Future work (in priority order)

1. **Brooks Falls dedicated gallery** — use the NPS *Bears of Brooks River eBook* to build a gallery for Otis, Grazer, and other famous bears so the model can output real names instead of "looks like Plunger"
2. **Threshold calibration** — ROC analysis on 100–200 human-labeled ground-truth frames to find the optimal threshold per bear
3. **License question** — pending Alex's response on whether CC BY-NC 4.0 affects the project deliverable; if it does, prepare a method-only re-training plan
4. **Better detection accuracy** — explore DeepLabCut or MMPose AP10K for more accurate keypoint localization (replacing the heuristic fallback)
5. **Cross-video demo** — run the full pipeline on a 5–10 clip set of the same bear and verify the gallery keeps the identity stable across all of them
6. **Integration test** — add a pytest harness so future PoseSwin / face-detector refactors don't silently break the matching results

---

## 10. References and resources

1. **Rosenberg, B., Zhou, M., Wolf, N., Mathis, M.W., Harris, B.P., Mathis, A.** (2026). *Individual identification of brown bears using pose-aware metric learning.* Current Biology.
2. **Liu, Z., Lin, Y., Cao, Y., et al.** (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV.
3. **Ren, S., He, K., Girshick, R., Sun, J.** (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS.
4. **PoseSwin GitHub**: https://github.com/amathislab/BrownBear_ReID
5. **Swin Transformer GitHub**: https://github.com/microsoft/Swin-Transformer
6. **Bears of Brooks River eBook (NPS)**: https://www.nps.gov/katm/learn/photosmultimedia/bears-of-brooks-river-ebook.htm
7. **Public_release Zenodo dataset**: https://zenodo.org/records/17822054 (32.9 GB)
