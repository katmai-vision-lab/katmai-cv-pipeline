# Evaluation Metrics Quick Reference

## 🎯 Choose Your Evaluation Mode

```
Do you have...
│
├─ Annotated validation dataset (YOLO format)?
│  └─> Use: --mode dataset
│      📊 Outputs: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95
│
├─ Video + known bear count (no bounding boxes)?
│  └─> Use: --mode counting
│      📊 Outputs: Counting accuracy, MAE, RMSE, temporal analysis
│
└─ Video with bounding box annotations?
   └─> Use Python API: evaluate_with_ground_truth_boxes()
       📊 Outputs: IoU-based metrics, confusion matrix, AP
```

---

## ⚡ Quick Commands

### Validate on Dataset (Most Important!)
```bash
# Basic - using default model
python -m src.detection.evaluate --mode dataset --data data/annotation/bears/bear.yaml

# With your trained model
python -m src.detection.evaluate --mode dataset \
    --data data/annotation/bears/bear.yaml \
    --model runs/detect/train/weights/best.pt

# With custom confidence threshold
python -m src.detection.evaluate --mode dataset \
    --data data/annotation/bears/bear.yaml \
    --model runs/detect/train/weights/best.pt \
    --conf 0.5
```

### Count Bears in Video
```bash
# Basic (5 bears, process every 30 frames)
python -m src.detection.evaluate --mode counting \
    --video path/to/video.mkv \
    --ground-truth 5 \
    --frame-skip 30

# Every frame (slower, more accurate)
python -m src.detection.evaluate --mode counting \
    --video path/to/video.mkv \
    --ground-truth 5 \
    --frame-skip 1
```

### Simple Analysis (No Ground Truth)
```bash
python -m src.detection.evaluate --mode simple \
    --video path/to/video.mkv \
    --plot
```

---

## 📊 Metrics Explained (Simple)

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Precision** | "When I say there's a bear, how often am I right?" | > 0.8 |
| **Recall** | "Of all the bears, how many did I find?" | > 0.7 |
| **F1 Score** | "Overall detection quality (balance of P & R)" | > 0.75 |
| **mAP@0.5** | "Average precision across all images" | > 0.7 |
| **mAP@0.5:0.95** | "Strict mAP (requires precise boxes)" | > 0.5 |
| **Counting Accuracy** | "% frames with exact bear count" | > 80% |
| **MAE** | "Average counting error" | < 0.5 |

---

## 🎨 Output Files

All saved to `predictions/evaluations/`:

- `metrics_*.json` - Full metrics in JSON
- `yolo_val_metrics_*.json` - YOLO validation results
- `counting_eval_*.csv` - Per-frame counting data
- `*_metrics_*.png` - Visualization plots
- `frame_metrics_*.csv` - Detailed frame-level analysis

---

## 🔧 Common Workflows

### After Training
```bash
# 1. Train
python -m src.main --mode train --data bear.yaml --epochs 50

# 2. Evaluate
python -m src.detection.evaluate --mode dataset \
    --data bear.yaml \
    --model runs/detect/train/weights/best.pt
```

### Compare Models
```bash
# Model 1
python -m src.detection.evaluate --mode dataset \
    --data bear.yaml --model model1.pt

# Model 2  
python -m src.detection.evaluate --mode dataset \
    --data bear.yaml --model model2.pt

# Compare JSON outputs
```

### Find Best Confidence Threshold
```bash
for conf in 0.1 0.25 0.5 0.75; do
    echo "Testing conf=$conf"
    python -m src.detection.evaluate --mode counting \
        --video test.mkv --ground-truth 5 \
        --conf $conf --frame-skip 30
done
```

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Video not found` | Use absolute path or place video in `data/raw/` |
| `--data is required for dataset mode` | Provide path to YAML: `--data data/annotation/bears/bear.yaml` |
| Low mAP score | Expected with only 122 images - need 1000+ for good scores |
| Counts too high | Lower `--conf` threshold |
| Counts too low | Raise `--conf` threshold or train more |

---

## 💡 Pro Tips

1. **Always start with `dataset` mode** - it's the ML standard
2. **Save your results** - all metrics are auto-saved as JSON/CSV
3. **Use multiple metrics** - don't rely on just one number
4. **Test on diverse videos** - different lighting, angles, bear counts
5. **Document your threshold choices** - record what conf/IoU you used

---

## 📝 Example Python Usage

```python
from src.detection.detector import BearDetector
from src.detection.metrics import VideoEvaluator

# Initialize
detector = BearDetector(model_path='your_model.pt')
evaluator = VideoEvaluator(detector, conf_threshold=0.25)

# Option 1: Dataset validation
metrics = evaluator.evaluate_dataset_with_yolo(
    data_yaml='data/annotation/bears/bear.yaml',
    save_dir='results/eval'
)
print(f"mAP@0.5: {metrics['map50']:.4f}")

# Option 2: Counting
df = evaluator.evaluate_counting_accuracy(
    video_path='video.mkv',
    ground_truth_counts=5,
    save_dir='results/counting'
)
accuracy = (df['is_correct'].sum() / len(df)) * 100
print(f"Counting accuracy: {accuracy:.2f}%")
```

---

## 📚 More Info

- Full guide: `docs/EVALUATION_GUIDE.md`
- Examples: `examples/evaluate_comprehensive.py`
- Code: `src/detection/metrics.py`

---

**Quick Start:** `python -m src.detection.evaluate --mode dataset --data data/annotation/bears/bear.yaml`
