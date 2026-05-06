# Auto-annotation systems — Bear vs. Salmon

This document covers the two independent auto-annotation systems in the
`src/preprocessing/` directory, each tuned for a different target species.

## System overview

| System | Directory | Target | # models | Status |
|--------|-----------|--------|----------|--------|
| 🐻 **Bear system** | `annotation_bear/` | Brown bear detection | 3 | ✅ Production |
| 🐟 **Salmon system** | `annotation_salmon/` | Salmon jumping out of the water | 3 | 🧪 Experimental |

## Bear system (`annotation_bear/`)

### Highlights
- **3-model consensus**: Grounding DINO + DETR + MegaDetector v5
- **Scientifically validated**: model-arena evaluation on 341 test images
- **Probability calibration**: isotonic regression on confidence scores
- **High performance**: 89.3% precision, 99.8% recall

### Use cases
- Bear monitoring at Katmai National Park
- Other brown-bear habitats
- Land-based wildlife detection in general (by changing the text prompt)

### Quick start
```bash
python -m src.preprocessing.annotation_bear.multi_model_annotator \
  --input data/frames/bear_video/ \
  --output data/auto_labels_bear/ \
  --prompt "bear" \
  --min-agreement 2 \
  --auto-approve
```

### Documentation
- Full technical report: [docs/bear_auto_annotation_system_report.md](bear_auto_annotation_system_report.md)
- Usage instructions: [README.md](../README.md)

---

## Salmon system (`annotation_salmon/`)

### Highlights
- **3-model consensus**: Grounding DINO + OWL-ViT v2 + Florence-2
- **Scene-tuned**: designed for "salmon jumping out of the water"
- **Action understanding**: OWL-ViT v2 is good at action concepts like "jumping"
- **Robust to complex scenes**: Florence-2 handles splashes and reflections well

### Model selection rationale
✅ **Enabled**:
- Grounding DINO: strong text understanding
- OWL-ViT v2: CLIP-based, understands action concepts ("jumping")
- Florence-2: latest VLM (2024), robust to complex scenes

❌ **Disabled**:
- MegaDetector v5: only trained on land animals (bears, deer, wolves)
- DETR: low precision (35.4%), high false-positive rate

### Use cases
- ✅ Salmon-jumping scenes (migration, going up the falls)
- ✅ Dynamic fish above the water surface
- ⚠️ Underwater swimming (not optimized for this)

### Quick start
```bash
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon_jumping/ \
  --output data/auto_labels_salmon/ \
  --prompt "salmon jumping out of water" \
  --min-agreement 2 \
  --auto-approve
```

### Documentation
- Salmon system docs: [annotation_salmon/README_SALMON.md](../src/preprocessing/annotation_salmon/README_SALMON.md)

---

## Technical comparison

### Model configuration

| Property | Bear system | Salmon system |
|----------|-------------|---------------|
| **Grounding DINO** | ✅ base, threshold 0.25 | ✅ base, threshold 0.25 |
| **DETR ResNet-101** | ✅ threshold 0.5 | ❌ disabled |
| **MegaDetector v5** | ✅ threshold 0.3 | ❌ disabled |
| **OWL-ViT v2** | ❌ not used | ✅ ensemble, threshold 0.3 |
| **Florence-2** | ❌ not used | ✅ base, threshold 0.3 |
| **Model weights** | gdino:0.406, detr:0.259, megadet:0.335 | gdino:0.40, owlvit:0.35, florence2:0.25 |
| **Default min_agreement** | 2/3 | 2/3 |

### Performance characteristics

| Dimension | Bear system | Salmon system |
|-----------|-------------|---------------|
| **Recall** | High (validated 99.8%) | TBD |
| **Precision** | High (validated 89.3%) | TBD |
| **Speed** | Medium (~4–5 s / image) | Medium (~4–5 s / image) |
| **GPU memory** | 6–8 GB | 7–9 GB (Florence-2 is larger) |
| **Scene specialization** | Land / forest environments | Above-water jumping scenes |
| **Production-ready** | ✅ Yes | 🧪 Experimental |

---

## Selection guide

### Use the **bear system** if you need:
- ✅ Detection of land wildlife (bears, deer, wolves, etc.)
- ✅ Maximum detection quality and reliability
- ✅ A validated production system (89.3% P / 99.8% R)
- ✅ Full probability-calibration support

### Use the **salmon system** if you need:
- ✅ Detection of **salmon jumping out of the water** (leaping fish, going up falls)
- ✅ Action-aware detection (OWL-ViT's CLIP backbone)
- ✅ Robustness to complex water scenes (splashes, reflections)
- ⚠️ You're OK with an experimental system (needs validation on real data)

### ❌ Neither system fits:
- Underwater swimming fish (no underwater training)
- Stationary or dead fish (no action signal)
- Mixed scenes like "bear catching fish" (would need a custom model combination)

---

## Extending to a new species

Want to build an auto-annotation system for another species? Follow these steps:

### 1. Copy an existing system
```bash
cd src/preprocessing
cp -r annotation_bear annotation_newspecies
```

### 2. Tune the key parameters

**a. Decide whether to use MegaDetector**
- Land mammals (deer, fox, raccoon, ...): keep MegaDetector ✅
- Birds, fish, insects, ...: disable MegaDetector ❌

**b. Update the default prompt**
```python
# in multi_model_annotator.py and train_calibration.py
default="newspecies"  # or a more specific phrase
```

**c. Adjust model weights**
- If you disable MegaDetector, redistribute the weights (see the salmon system)
- If you have validation data, run a model arena to compute the optimal weights

**d. Update `min_agreement`**
- 3 models: default = 2
- 2 models: default = 1

### 3. Test and tune
```bash
# Small-batch sanity test
python -m src.preprocessing.annotation_newspecies.multi_model_annotator \
  --input test_frames/ \
  --output test_labels/ \
  --prompt "newspecies" \
  --limit 20

# Visual sanity check
python -m src.preprocessing.annotation_newspecies.visualize_labels \
  --images test_frames/ \
  --labels test_labels/ \
  --output test_visualized/ \
  --limit 10
```

### 4. Optional: train species-specific calibrators
If you have ground-truth labels (100+ samples recommended):
```bash
python -m src.preprocessing.annotation_newspecies.train_calibration \
  --images data/annotation/newspecies/images/ \
  --labels data/annotation/newspecies/labels/ \
  --output models/calibrators_newspecies.pkl \
  --prompt "newspecies"
```

---

## Maintenance notes

### File organization
Each system should stay independent:
- ✅ Its own README
- ✅ Its own calibrator file
- ✅ Its own output directory
- ✅ Its own commits

### Shared components
The following modules are shared across systems (don't duplicate):
- `frame_extractor.py` — video frame extraction
- `visualize_labels.py` — annotation visualization
- `probability_calibrator.py` — calibration algorithm (data is kept separate per system)

### Version control
```bash
# Bear system update
git add src/preprocessing/annotation_bear/
git commit -m "feat(bear): ..."

# Salmon system update
git add src/preprocessing/annotation_salmon/
git commit -m "feat(salmon): ..."
```

---

## FAQ

**Q: Can I use the bear system to detect salmon?**
A: Yes, but the dedicated salmon system works much better. Just changing the prompt to "salmon" works, but MegaDetector tends to produce false positives.

**Q: Do I need to retrain models for each species?**
A: No. We use zero-shot models (Grounding DINO et al.) — just change the text prompt.

**Q: How do I know which system fits my species?**
A: Look at the category:
- Land mammals → use the bear system
- Aquatic / flying / small animals → adapt the salmon system (disable MegaDetector)

**Q: Can I detect multiple species at once?**
A: Yes — Grounding DINO supports multi-prompt input:
```bash
--prompt "bear. salmon. eagle."
```

---

## References

- Bear system technical report: [docs/bear_auto_annotation_system_report.md](bear_auto_annotation_system_report.md)
- Main README: [README.md](../README.md)
- Grounding DINO paper: https://arxiv.org/abs/2303.05499
- DETR paper: https://arxiv.org/abs/2005.12872
- MegaDetector: https://github.com/microsoft/CameraTraps

---

**Last updated**: 2026-03-04
**Maintained by**: Katmai Vision Lab
