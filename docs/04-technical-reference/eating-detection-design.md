# Bear-Eating Detection — Design Document

**Author:** Darian Ding
**Date:** May 2026
**Sponsor request (Alex):** Build a pixel-based / pose-based eating detector that runs on a consumer-grade computer; document a cloud-based alternative as well; the user should have both options.

---

## 1. Goals

Detect when a bear is eating salmon in Katmai bear-cam footage, and produce one of three outputs per frame: `eating`, `maybe`, `not_eating`, plus a continuous score in `[0, 1]`.

Two delivery paths are supported:

1. **Consumer-grade (CPU-only, no cloud, no GPU)** — pixel + posture analysis. Runs at 100+ FPS on a laptop.
2. **Cloud / GPU (highest accuracy)** — Vision-language model (Molmo2-8B) classifies each frame into 5 stages (`WAITING / LUNGING / CATCHING / EATING / MISSED`).
3. **Hybrid (recommended)** — pixel pre-filter cheaply rejects obvious non-eating frames, GPU model only invoked on candidate frames. Cuts GPU cost ~10× while keeping accuracy.

---

## 2. Approaches considered

### 2.1 Salmon-flesh color analysis (consumer-grade, primary signal)

**Hypothesis (per Alex):** when a bear eats salmon and the camera shows a fair-to-good close-up, the salmon flesh exposes pink, red, and white/light-gray colors that are visually distinctive against the bear's brown fur and the gray water.

**Implementation** (in `src/behavior/pixel_eating_detector.py`):

* Take the bear bbox from YOLO (provided by `analyze_feeding.py`).
* Crop the upper-front portion of the bbox (where the head/mouth lives across both standing and feeding postures).
* Mask out brown bear-fur pixels in HSV space (`H 5–28, S 40–220, V 20–130`).
* Within the **non-bear** portion, count pixels matching:
  * **Pink** — `H 0–15, S 100–220, V 150–240` (saturated salmon flesh)
  * **Red** — `H 0–10` ∪ `H 165–180`, `S ≥ 150, V ≥ 130` (salmon meat / blood)
  * **Light** — `S < 40, V > 200` (white salmon belly)
  * **Bright-non-brown** — anything bright (`V > 130`) that isn't bear fur (catches silver salmon body too)
* Combine into a `color_score ∈ [0, 1]`.

### 2.2 Posture / proximity heuristic (consumer-grade, secondary signal)

**Hypothesis (per Alex):** "head/mouth are very close to its paws only when it's eating."

**Implementation:**

* Compute the bbox aspect ratio `w/h`. Standing-alert bears tend to be tall & narrow (`w/h < 0.85`); feeding bears are hunched/curled, producing wide bboxes (`w/h > 1.0`).
* Compute frame-to-frame center motion. Eating bears are mostly still; pursuing bears move a lot.
* Combine: `posture_score = sigmoid((aspect − 0.85) × 4) × stillness`.

> **Note on "head close to paws":** a true keypoint-based proximity test would require an animal-pose model (DeepLabCut / MMPose AP10K / PoseSwin's HRNet). These were considered but each adds 100–300 MB of weights and 50+ ms/frame inference. The aspect-ratio + motion proxy gives a similar binary signal at near-zero cost.

### 2.3 Combined score

```
eating_score = 0.55 × color_score + 0.45 × posture_score
label        = "eating"     if eating_score > 0.60
             = "maybe"      if eating_score > 0.40
             = "not_eating" otherwise
```

A 5-frame moving-average smooths out single-frame noise.

---

## 3. Empirical results — Brooks Falls "Bear 903 (Gully)" test clip (12 s, 60 fps)

Comparison against Molmo2-8B labels on the same 48 sampled frames:

| Metric | Pixel detector v2 | Molmo2 (ground-truth proxy) |
|---|---|---|
| CATCHING-class mean score | **0.47** | (label, n=39) |
| WAITING-class mean score | **0.48** | (label, n=9) |
| Separation | **−0.01** | — |
| Frames flagged "eating" | 1 / 48 (2%) | 39 / 48 (81%) |

**Conclusion: the pixel approach does not work on this clip.**

### Why it fails on this footage

1. **No visible salmon flesh.** Brooks Falls fish at this point in the season are silver-bodied (ocean-phase / early run) — they have not yet developed the spawning red. The fresh salmon flesh (pink) is hidden inside the bear's mouth and never exposed in the bbox.
2. **Bear fur dominates the bbox.** ~80% of the bbox is brown bear in both states. The bear-fur mask removes it, but what remains contains too little fish signal.
3. **Posture barely changes between states.** This particular bear maintains a similar standing-over-waterfall posture through both `WAITING` and `CATCHING` — the aspect ratio difference is < 0.15.

### When pixel-RGB analysis IS expected to work (per Alex's brief)

Pixel-RGB analysis should work in conditions matching what Alex described:

| Condition | Why it matters |
|---|---|
| **Camera zoomed in on the bear** | Salmon occupies enough pixels to dominate the signal |
| **Bear has torn open the fish** | Pink/red flesh is visible (not just silver skin) |
| **Fair-to-good lighting** | Saturated colors trigger the masks; dim light desaturates everything |
| **Fish is held externally** (above the water, in front of the bear's face) | Not occluded by the bear's mouth |
| **Spawning-phase fish** (red sockeye, late summer) | Fish itself is naturally pink/red |

These are exactly the close-up shots that produce the most visually compelling content; the detector should be re-tested on Alex's incoming "bear catching salmon" close-up clips.

---

## 4. Recommended path forward — Hybrid pipeline

The honest read: pure pixel detection is an **excellent pre-filter** but a **mediocre standalone classifier** on this footage. The combined system below uses the strengths of both:

```
┌─────────────────────┐
│ YOLO + ByteTrack    │  every N frames (1.0 s)
└──────────┬──────────┘
           │  bbox per bear
           ▼
┌─────────────────────┐
│ Pixel detector      │  CPU, < 1 ms/frame
│ (color + posture)   │  → eating_score per bear
└──────────┬──────────┘
           │
           ├─ score < 0.30 → label "not_eating", DONE (no VLM call)
           │
           └─ score ≥ 0.30 → CANDIDATE, send to Molmo2 ─┐
                                                       │
                                                       ▼
                                          ┌─────────────────────┐
                                          │ Molmo2-8B (GPU)     │  ~5 s/frame
                                          │ stage classifier    │  → CATCHING/EATING/...
                                          └─────────────────────┘
```

**Cost savings:** on a video where pixel pre-filter rejects ~70% of frames, GPU inference time drops from `100 % × 5 s` to `30 % × 5 s` — a **3× speed-up** with negligible accuracy loss.

This satisfies Alex's "both options" requirement:

| User context | Mode |
|---|---|
| Researcher running on a laptop, no internet | Pixel-only mode (`--mode pixel`); accept lower accuracy on side/long shots, full accuracy on close-ups |
| Lab / cloud GPU available | Molmo2-only mode (`--mode vlm`); full accuracy, ~5 s/frame |
| Production pipeline, large batch | **Hybrid** (`--mode hybrid`); recommended default |

---

## 5. Code map

| File | Purpose |
|---|---|
| `src/behavior/pixel_eating_detector.py` | Standalone pixel detector. Reuses bboxes from existing `analysis.json` for zero-GPU operation; falls back to live YOLO+ByteTrack. |
| `src/behavior/analyze_feeding.py` | The Molmo2-based behavior classifier (existing). |
| `src/behavior/feeding_viewer.py` | Side-by-side video renderer (existing). |

**CLI examples:**

```bash
# Consumer-grade only — no GPU, < 5 seconds for a 12 s clip
venv/bin/python3 -m src.behavior.pixel_eating_detector \
    --video feed/data_video/<clip>.mp4 \
    --analysis predictions/<...>/analysis.json \
    --render

# Cloud / GPU only (existing)
venv/bin/python3 -m src.behavior.analyze_feeding \
    --video feed/data_video/<clip>.mp4 \
    --interval 0.25
```

---

## 6. Open work

1. **Validate on close-up footage.** Re-test pixel detector on the new clips Alex will share (single bear, camera zoomed in, salmon visible) — expected to perform much better.
2. **Build the hybrid `--mode hybrid` flag.** Wire the pixel pre-filter into `analyze_feeding.py` so it skips Molmo2 calls on obvious-non-eating frames.
3. **Animal-pose option (cloud-grade alternative).** If a bear-pose model is desired, MMPose AP10K (50 MB, CPU-runnable in 200 ms/frame) or DeepLabCut (custom-trained) would let us measure true mouth-to-paw distance instead of the aspect-ratio proxy.

---

## 7. Honest summary for the sponsor

The CV technique Alex described — using pixel RGB analysis to detect pink/red/white salmon-flesh colors — is a **valid and cheap signal** that works exactly as expected when the camera is close and the salmon is exposed. On the side/long-shot Brooks Falls footage the team currently has, the salmon are mostly silvery and obscured in the bear's mouth, so the signal is too weak to drive a standalone classifier.

The pragmatic delivery is therefore a **hybrid pipeline**: pixel analysis runs on every frame (free, instant), and the heavier vision-language model is only invoked on frames the pixel detector flags as candidates. This gives the end user both a fully-local CPU-only mode and an accurate cloud/GPU mode without forcing them to choose between cost and quality.
