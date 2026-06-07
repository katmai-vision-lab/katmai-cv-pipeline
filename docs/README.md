# Katmai CV Pipeline — Documentation

Open-source computer vision pipeline for detecting, tracking, and quantifying the feeding behavior of Alaskan brown bears at Katmai National Park. Built as a UW ENGINE capstone project (2025–2026) in partnership with the UW ECE department.

**Deepwiki:** https://deepwiki.com/katmai-vision-lab/katmai-cv-pipeline  
**GitHub:** https://github.com/katmai-vision-lab/katmai-cv-pipeline

---

## Start here

| I want to… | Go to |
|---|---|
| Install and run the pipeline for the first time | [Getting Started →](01-getting-started/installation.md) |
| Detect and count bears in a video | [Detect & Count Bears →](02-how-to-guides/detect-and-count-bears.md) |
| Track bears with persistent IDs | [Track Bears →](02-how-to-guides/track-bears.md) |
| Detect feeding events and classify behavior | [Feeding Behavior →](02-how-to-guides/feeding-behavior-and-identity.md) |
| Count salmon jumps at Brooks Falls | [Count Salmon Jumps →](02-how-to-guides/count-salmon-jumps.md) |
| Fetch weather and river data for a recording date | [Environmental Data →](02-how-to-guides/fetch-environmental-data.md) |
| Annotate a new bear video dataset | [Annotate Bears →](03-annotation-and-training/annotate-bears.md) |
| Annotate a new salmon video dataset | [Annotate Salmon →](03-annotation-and-training/annotate-salmon.md) |
| Fine-tune the bear detector on new footage | [Fine-Tune →](03-annotation-and-training/fine-tune-bear-detector.md) |
| Understand how the system is architected | [Architecture →](04-technical-reference/architecture.md) |
| Add a new VLM backend for behavior classification | [Add VLM Backend →](05-extending/add-vlm-backend.md) |
| Adapt the pipeline to a new camera or species | [Adapt to New Dataset →](05-extending/adapt-to-new-dataset.md) |

---

## Folder structure

```
docs/
├── 01-getting-started/         # Installation, first run, hardware guide
├── 02-how-to-guides/           # Task-oriented usage guides per module
├── 03-annotation-and-training/ # Labeling, fine-tuning, evaluation
├── 04-technical-reference/     # Deep-dives into each module's design
├── 05-extending/               # How to add backends, datasets, APIs
├── demos/                      # Demo videos from the pipeline
├── images/                     # Figures used in documentation
└── _archive/                   # Internal team materials (meeting notes, design templates)
```

---

## Module map

```
User (CLI / TUI)
      │
      ├── Bear Detection & Tracking   src/detection/
      │     ├── YOLOv8n fine-tuned    detector.py
      │     ├── ByteTrack             track_video.py
      │     └── Batch counting        bear_count.py
      │
      ├── Feeding Behavior Analysis   src/behavior/
      │     ├── VLM backends          backends/ (molmo2, claude, gpt4o, gemini)
      │     ├── Feeding viewer        feeding_viewer.py
      │     └── Salmon jump counter   count_salmon_jumps.py
      │
      ├── Bear Identity (optional)    src/identity/
      │     ├── PoseSwin Re-ID        poseswin_identifier.py
      │     └── Gallery               build_named_gallery.py
      │
      ├── Environmental Data          src/environment/
      │     ├── USGS hydrology        usgs_hydro.py
      │     ├── RAWS weather          raws_weather.py
      │     └── NADP precipitation    nadp_precip.py
      │
      └── Annotation Pipelines        src/preprocessing/
            ├── Bear (multi-model)    annotation_bear/
            └── Salmon (stacking)     annotation_salmon/
```

---

## Key performance numbers

| Component | Metric | Value |
|---|---|---|
| Bear detector (fine-tuned YOLOv8n) | mAP@0.5 | 95.1% |
| Bear detector | Precision / Recall / F1 | 92.2% / 90.6% / 91.4% |
| Bear detector (pretrained, no fine-tuning) | mAP@0.5 | 17.9% |
| Salmon annotation (stacking meta-learner) | Precision / Recall | 97.5% / 96.6% |
| Bear annotation (multi-model consensus) | Precision / Recall | 89.3% / 99.8% |

---

## External links

- **SharePoint (video data):** https://uwnetid.sharepoint.com/sites/katmai-vision-lab
- **PoseSwin dataset (identity):** https://zenodo.org/records/17822054
- **Explore.org cameras:** https://explore.org/livecams/brown-bears
