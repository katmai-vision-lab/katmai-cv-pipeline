# Count Salmon Jumps

Two independent methods are available for estimating salmon jump activity at Brooks Falls. Use the interactive background-subtraction counter for most cases; use the color-blob method if you need headless/batch processing with a config file.

---

## Method 1 — Background subtraction with tripwire (recommended)

`salmon_jump_counter_bg.py` uses MOG2 background subtraction to detect motion, then counts objects crossing a user-placed horizontal tripwire line. Best for Brooks Falls waterfall footage where the water texture is consistent enough to model as background.

### Interactive first run (set your parameters)

```bash
python -m src.detection.salmons.salmon_jump_counter_bg \
    --video path/to/salmon_clip.mp4
```

On launch:
1. **Draw ROI** — click and drag to select the region of interest (the falls)
2. **Place tripwire** — click to set the horizontal counting line
3. **Tune parameters** — live trackbars let you adjust blur, threshold, and min area in real time
4. Press `q` to quit; area stats are printed on exit to guide your `--min-area` setting

### Non-interactive run (after you know your parameters)

```bash
python -m src.detection.salmons.salmon_jump_counter_bg \
    --video path/to/salmon_clip.mp4 \
    --line-y 650 \
    --min-area 1200 \
    --roi 120,80,1800,900 \
    --var-threshold 90 \
    --blur-size 7 \
    --output out.mp4
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `--line-y` | interactive | Y-pixel position of the tripwire |
| `--roi` | interactive | `x1,y1,x2,y2` bounding box of detection zone |
| `--min-area` | 800 | Minimum contour area to count as a salmon |
| `--var-threshold` | 80 | MOG2 sensitivity — lower = more sensitive to small motion |
| `--blur-size` | 7 | Gaussian blur kernel (odd number) to suppress water texture |
| `--history` | 300 | MOG2 background model frames |
| `--skip-frames` | 2 | Process every Nth frame |
| `--no-display` | — | Headless mode (no GUI window) |

---

## Method 2 — Color blob detection with peak analysis

`salmon_jump_counter_cv.py` isolates silver-colored blobs matching salmon HSV ranges, tracks them frame-by-frame, and uses `scipy.signal.find_peaks` to count jump events from the resulting blob-count signal. Good for footage with high turbulence where MOG2 struggles, or when you want a config-file-driven workflow.

```bash
python -m src.detection.salmons.salmon_jump_counter_cv \
    path/to/salmon_clip.mp4 \
    --config config.json \
    --roi 100 200 400 300 \
    --min-blob-area 600 \
    --max-blob-area 5000 \
    --min-jump-gap-sec 1.0
```

Config file format (saves and reloads parameters):
```json
{
  "salmon_hsv_lower": [0, 0, 40],
  "salmon_hsv_upper": [180, 60, 160],
  "silver_hsv_lower": [0, 0, 160],
  "silver_hsv_upper": [180, 40, 255],
  "min_blob_area": 800,
  "max_blob_area": 8000,
  "roi": null,
  "min_jump_gap_sec": 0.5,
  "sample_rate": 2
}
```

---

## Choosing between methods

| Situation | Recommended |
|---|---|
| Brooks Falls standard footage | Method 1 (background subtraction) |
| Very turbulent water / constant splash | Method 2 (color blob) |
| First time on a clip | Method 1 (interactive setup) |
| Batch processing with known params | Either — both support non-interactive |
| Need real-time display | Method 1 |

---

## Tips

- **Too many false positives** — raise `--min-area`; water splashes are small and irregular
- **Missing large jumps** — lower `--var-threshold` or `--min-area`
- **Brooks Falls Low camera** — the falls are closer; use a smaller ROI and lower `--min-area` (~400–600)
- **Riffles camera** — not designed for jump counting; salmon are too dispersed
