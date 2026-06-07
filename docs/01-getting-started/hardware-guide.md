# Hardware Guide

The pipeline is designed to run on consumer hardware, but different modules have very different compute requirements. This page helps you pick the right configuration.

---

## At a glance

| Module | CPU-only | Apple Silicon (MPS) | NVIDIA GPU |
|---|---|---|---|
| Bear detection (YOLO) | ✅ Slow (~2 fps) | ✅ Good | ✅ Fast (~30+ fps) |
| Bear tracking (ByteTrack) | ✅ | ✅ | ✅ |
| Feeding behavior (Molmo2-8B local) | ⚠️ Very slow | ⚠️ Slow (~30s/frame) | ✅ 5–8s/frame |
| Feeding behavior (cloud VLM) | ✅ Network-bound | ✅ | ✅ |
| Salmon jump counter (CV) | ✅ 100+ fps | ✅ | ✅ |
| Bear annotation (Grounding DINO) | ⚠️ Slow | ✅ | ✅ |
| Salmon annotation (stacking) | ✅ | ✅ | ✅ |
| Environmental data fetch | ✅ | ✅ | ✅ |

---

## Configurations

### Laptop / no dedicated GPU

Use the cloud VLM backend for feeding behavior. Everything else runs fine on CPU.

```bash
# Use Gemini (cheapest) for feeding analysis
python -m src.behavior.analyze_feeding \
    --video clip.mp4 \
    --backend gemini
```

Annotation runs will be slow (~10–30s per image for Grounding DINO). For large datasets, run overnight or use the ENGINE Lab machine.

### Apple Silicon (M1/M2/M3/M4)

PyTorch automatically uses the MPS backend. Detection and tracking are fast. Molmo2 works but is slow (no Flash Attention on MPS); Claude or Gemini is a better choice for behavior.

Some models will silently fall back to CPU — the pipeline handles this automatically via `src/config.py`'s device detection.

### NVIDIA GPU (recommended for training and annotation)

The ENGINE Lab machine (dual RTX 2080 Ti, 128 GB RAM) is available for heavy workloads. SSH in and run training or bulk annotation there.

For personal machines: any GPU with 8 GB+ VRAM handles all modules. 6 GB can work for inference-only with `--imgsz 640`.

---

## Cloud VLM cost estimates

If using a paid API backend for feeding behavior:

| Backend | Cost per frame | 1-minute clip at 0.5s interval |
|---|---|---|
| Claude Sonnet | ~$0.02 | ~$2.40 |
| GPT-4o | ~$0.02 | ~$2.40 |
| Gemini 2.0 Flash | ~$0.004 | ~$0.48 |
| Molmo2-8B (local) | $0 | — |

For research use, Gemini is the cheapest paid option. Molmo2 local is free if you have GPU.

---

## Device auto-detection

The pipeline detects your device automatically via `src/config.py`:

```python
# Priority order: CUDA → MPS → CPU
device = config.get_device()
```

You can override with `--device cpu` or `--device cuda` on most commands.
