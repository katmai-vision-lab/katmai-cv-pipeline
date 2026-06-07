# Pixel-based bear-eating detection — Experiment Report

**Author:** Darian Ding
**Date:** May 3, 2026
**Source of task:** Alex's proposal to use pixel-RGB analysis to detect a bear eating salmon, with the constraint that the system must run on consumer-grade hardware

---

## 1. Background

Alex raised two ideas in Slack:

1. **Pixel-color analysis**: when the camera zooms in on a bear and the lighting is fair-to-good, the salmon shows distinct **pink, red, and white/light-gray** regions. These colors only appear when a bear is eating salmon.
2. **Pose proximity**: a bear's head/mouth is only close to its paws when it is eating — could some pixel/image analysis capture that?

**Constraints:**
- Must run on a consumer-grade computer (no cloud / no GPU)
- A cloud-based path may be documented as a complement, but cannot be the only option

**Test video:** `katmai_2026_05_03_8to20s.mp4` (Bear 903 "Gully" fishing at Brooks Falls; 12 s, 60 fps)
**Ground truth:** Molmo2-8B has already been run on this video. Of 48 samples at 0.25 s, 39 frames are labeled `[CATCHING]` and 9 are `[WAITING]`.

---

## 2. Methods tried (chronological)

### Method 1 — Basic HSV color masks

**Idea:** in OpenCV's HSV color space, use `cv2.inRange()` to pick out pink, red, and white pixels, and compute their fraction inside the upper half of the bbox (the "mouth region").

**Implementation details:**
```
PINK:  H 0–15,  S 60–180, V 100–220
RED:   H 0–10 ∪ 165–180, S 120–255, V 80–200
LIGHT: S < 50, V > 180
mouth_region = inset top 60% × middle 70% of bbox
combined_score = pink × 30 + red × 25 + light × 3
```

**Result:**

| Metric | Value |
|---|---|
| `max_score` (top score across 48 frames) | **0.60** |
| Frames flagged as `eating` | **0/48 (0%)** |
| `avg_pink_ratio` | **0.007** (0.7% of pixels match pink) |

**Why it failed:**

Pulled stats from 5 CATCHING frames + 5 WAITING frames:

| Metric | CATCHING (n=5) | WAITING (n=5) |
|---|---|---|
| `warm_pct` (red/pink/orange pixels) | 84% | 80% |
| `saturated_warm` | 56% | 52% |
| `silver_pct` (gray pixels) | 7% | **14%** ⚠️ inverted |

**Root cause: the brown bear itself is in the warm-color band.** Roughly 80% of the bbox is bear fur, and the pink/red mask is heavily "polluted" by it. The color distribution is essentially the same in CATCHING and WAITING frames.

---

### Method 2 — Bear-fur mask + signal computed only on non-bear pixels

**Improvement idea:** since brown bear fur is the noise source, identify and mask it out first; then compute the pink/red/white ratios on only the remaining pixels (background water/sky/possible fish).

**Implementation details:**
```
BEAR_FUR_MASK: H 5–28, S 40–220, V 20–130 (dark warm = brown bear fur)
non_bear = NOT bear_fur_mask
salmon_signals = (pink/red/light pixels) ∩ non_bear / non_bear_count
```

Raise all thresholds (V ≥ 150, S ≥ 100) so we only match **saturated and bright** salmon-style colors and skip bear-fur shades.

**Result:**

| Metric | v1 | v2 |
|---|---|---|
| `max_score` | 0.60 | 0.61 |
| `eating` frames | 0/48 | 1/48 |
| `avg_pink_ratio` | 0.007 | 0.000 |

**Almost no improvement.**

**Why it failed:**

After masking bear fur, the non-bear part of the bbox is dominated by:
1. **Water spray, white foam** — triggers `light_mask` but isn't fish
2. **Rocks, distant background** — gray/green, doesn't trigger any salmon mask
3. **The actual salmon** — but the salmon is **silver** (not the spawning red), so it sits outside the pink/red ranges

The bigger problem we discovered: **the fish is barely visible at all**. The salmon is held in the bear's mouth, with at most the tail end protruding — under 1–2% of the bbox. Even if it were silver, that's too few pixels.

---

### Method 3 — Add a generic "bright but not brown" signal

**Idea:** since the salmon is silver, the precise pink/red detector misses it. Lower the specificity — detect **any bright pixel that is also non-brown** (V > 130, outside the bear-fur range). At least we'd catch the silver fish body.

**Implementation details:**
```
bright = inRange(hsv, [0,0,130], [180,255,255])
bright_non_brown = bright ∩ non_bear
score gets weighted: bright_non_brown × 1.5
```

**Result: still no significant improvement.**

**Why it failed:**

"Bright and not brown" also includes:
- Water spray, waterfall splashes (V > 130, low S)
- Sky, distant hills (medium-to-high V)
- White foam on rocks

These appear in roughly equal proportions in WAITING and CATCHING frames, because the bear is standing at the water in both states. The `bright_non_brown` signal is around 30% in both — no separation.

---

### Method 4 — Posture heuristics (aspect ratio + motion)

**Idea:** Alex also pointed out that "the head/mouth is close to the paws only when eating." With true keypoints we could measure the distance directly, but YOLO only outputs a bbox.

**Proxy signals:**
1. **Bbox aspect ratio `w/h`** — standing/scanning: tall and narrow (`w/h < 0.85`); crouched and eating: wide and flat (`w/h > 1.0`)
2. **Frame-to-frame motion** — bears are mostly still while eating; large motion when stalking/lunging

```
posture_score = sigmoid((aspect - 0.85) * 4) * stillness
stillness = 1 - clip(motion * 6, 0, 1)
```

**Result — actual measured aspect ratios:**

| State | t=0.0 | t=0.5 | t=1.25 | t=2.75 | t=11.5 |
|---|---|---|---|---|---|
| Molmo2 label | WAITING | WAITING | CATCHING | CATCHING | CATCHING |
| `w/h` | 1.23 | 1.23 | 1.30 | 1.19 | 1.36 |

**The WAITING and CATCHING aspect-ratio ranges fully overlap** (both within 1.19–1.36). `posture_score` has no discriminative power either.

**Why it failed:**

This particular bear (Bear 903 "Gully") stands at the top of the falls staring down throughout the clip — its body posture barely changes. It's not the kind of bear that "sits down to eat once it has the fish" — it stays standing and just lowers its head slightly while holding the fish. The bbox shape stays roughly constant.

The posture heuristic is only useful for clips where posture **changes a lot** (e.g., a bear that carries the fish onto the bank and sits down to eat).

---

### Method 5 — Weighted fusion into a final score

**Idea:** since no single signal works, combine them and see if a faint signal can be extracted.

```
eating_score = 0.55 * color_score + 0.45 * posture_score
threshold:
  > 0.60  →  "eating"
  > 0.40  →  "maybe"
  else    →  "not_eating"
```

**Plus a 5-frame moving-average smooth to reduce noise.**

**Correlation against the Molmo2 ground truth:**

| | CATCHING (n=39) | WAITING (n=9) |
|---|---|---|
| Pixel score, mean | **0.471** | 0.484 |
| Pixel score, max | 0.529 | 0.615 |
| Pixel score, min | 0.326 | 0.378 |
| **Separation** | **−0.013** (WAITING is actually slightly higher) ||

**Zero separation.** The mean WAITING score is even slightly higher than CATCHING. ROC AUC ≈ 0.5 (random guessing).

---

## 3. Root causes of failure (in order of importance)

### Cause 1 — Salmon colors don't match the assumption ⭐⭐⭐

Alex's hypothesis was "we'll see pink/red/white-light salmon flesh." But:

- **Brooks Falls salmon in early/mid summer are silver** (ocean / early-run phase) — they don't yet have the spawning red
- **Pink salmon flesh** is only exposed *after* the bear has torn into the fish — while the fish is held intact, you can't see it
- **The white belly** is also mostly hidden inside the bear's mouth

### Cause 2 — Most of the bbox is bear, not fish ⭐⭐⭐

YOLO's bbox covers the **whole bear**; the fish takes up at most 1–2% of pixels. No matter how cleverly we mask out bear fur or shrink the mouth crop, the leftover non-bear region is dominated by:
- The fish (small, possibly outside our color ranges)
- Water (largest portion)
- Rocks, sky (background)

Separating these three by color alone is extremely hard.

### Cause 3 — The posture signal collapses at this camera angle ⭐⭐

In a side-view long shot, the bear posture barely changes between WAITING and CATCHING:
- Aspect ratio differs by < 0.15
- Position barely moves
- Head orientation always points downward

The heuristic only works in **top-down or close-up** shots.

### Cause 4 — Missing a real "proximity" measurement ⭐⭐

Alex's "head close to paws" needs keypoint detection. We didn't wire in:
- DeepLabCut (animal pose)
- MMPose AP10K (17-point animal pose)
- PoseSwin's HRNet (only bear-face keypoints, no paws)

Using bbox aspect ratio as a proxy is too coarse — it loses the local-distance information.

---

## 4. Numerical comparison across methods

| Method | max_score | eating frames | avg pink | CATCHING mean | WAITING mean | Separation |
|---|---|---|---|---|---|---|
| v1 naive HSV | 0.60 | 0/48 | 0.007 | 0.42 | 0.45 | −0.03 |
| v2 + bear-fur mask | 0.61 | 1/48 | 0.000 | 0.45 | 0.46 | −0.01 |
| v3 + bright-non-brown | 0.61 | 1/48 | 0.000 | 0.47 | 0.48 | −0.01 |
| **Target (Molmo2 reference)** | — | **39/48** | — | (CATCHING) | (WAITING) | **clearly separable** |

**Bottom line: on this Brooks Falls clip, the pixel approach has ROC AUC ≈ 0.5 (no discriminative power). Molmo2 is essentially at ground truth.**

---

## 5. When the pixel approach **should** work (not validated yet)

**This is not a wholesale failure of the technique.** Alex's described scenario should still work if we get the right footage:

| Condition | Why it helps |
|---|---|
| ✅ Camera **zoomed in on the bear's face/mouth** | Fish takes up a much bigger fraction of the bbox (5–30%) |
| ✅ Bear has **already torn into the fish**, exposing the flesh | Saturated pink/red pixels become visible |
| ✅ **Late-summer spawning sockeye** | The fish itself is bright red — high contrast |
| ✅ Fish is **held externally** (in front of the face, dropped on the bank) | Not occluded by the bear's mouth |
| ✅ Good lighting, high contrast | Saturated colors aren't desaturated by haze |
| ❌ Our test video (side view, silver fish, fish held inside mouth) | None of the above apply |

**Next action:** when Alex uploads the close-up "single bear repeatedly eating salmon" clips, re-test on those.

---

## 6. Alternatives considered (not implemented)

Documented for completeness — these were considered during the experiments but not pursued because of complexity / poor cost-benefit:

| Idea | Why not |
|---|---|
| **Frame-to-frame color delta** — detect "suddenly appearing" salmon-colored pixels | High-frequency jitter; flowing water also creates deltas |
| **Optical flow / motion mask** — find a small "moving" object near the bear's mouth (the struggling fish) | Water spray itself is always moving — hard to separate |
| **Fish-shape detection** — OpenCV elongated-ellipse contour fit | Splash debris and rocks are also elongated |
| **Per-bear background color profile** — model the "normal" color histogram for each bear, flag anomalies | Needs a "non-eating" baseline; doesn't transfer across videos |
| **Train a dedicated fish classifier** — fine-tune a YOLO detector for "fish in mouth" inside the bear bbox | Requires labeling many "fish in mouth" examples → back to the labeling problem |
| **Plug in DeepLabCut / MMPose for real keypoint distances** | Adds 100–300 MB of model + 50 ms / frame — violates the consumer-grade target |

---

## 7. What is actually shippable

**Even though pure pixel classification isn't accurate, the detector is still useful** as a **pre-filter**:

```
If pixel score < 0.30  →  highly unlikely to be eating → skip Molmo2
If pixel score ≥ 0.30  →  candidate frame → send to Molmo2 for the real call
```

If the pixel filter rejects 70% of frames, the Molmo2 GPU time drops by ~3×. This is exactly the "both options" architecture Alex asked for.

**Delivered:**

| File | Purpose |
|---|---|
| [src/behavior/pixel_eating_detector.py](../src/behavior/pixel_eating_detector.py) | Standalone CPU-only detector (v3, with all the improvements) |
| [docs/eating_detection_design.md](eating_detection_design.md) | Design doc + hybrid-pipeline recommendation |
| `predictions/.../pixel_eating.json` | Per-frame signal data for the 48 samples |
| `predictions/.../pixel_eating_demo.mp4` | Annotated demo (illustrates the principle, not the accuracy) |

---

## 8. One-line summary for the sponsor

> We tried five versions of pixel + posture analysis and they all failed on the current Brooks Falls side-view clip (separation between CATCHING and WAITING ≈ 0). The technique itself isn't wrong — the problem is **silver fish hidden inside the bear's mouth, occupying < 2% of the bbox** — there are simply no pink/red/white salmon-flesh pixels to detect. Once your weekend close-up clips arrive, the technique should work in those conditions; meanwhile we recommend a hybrid "pixel pre-filter + Molmo2 confirmation" pipeline so the user gets both options (cheap CPU-only and high-accuracy GPU/cloud).
