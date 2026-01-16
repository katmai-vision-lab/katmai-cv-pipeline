# Computer Vision Pipeline to Detect, Track & Quantify Feeding Habits of Katmai NPP Alaskan Brown Bears
This project focuses on building an open-source, Python-based computer vision pipeline that analyzes video data from Katmai National Park & Preserve. The system is designed to automatically detect and track individual Alaskan brown bears, count salmon attempting to jump Brooks Falls, and quantify feeding behavior over time. In addition to visual analysis, the pipeline will integrate environmental context such as water level, stream flow, weather, and time of day to support deeper ecological insight.

Video data is sourced primarily from Explore.org bear cams in the Brooks Falls and Brooks River region.

## Scope
The system is designed to run on a consumer-grade laptop or desktop and ingest short-form video clips (1–15 minutes). From these inputs, it will produce structured outputs including individual and total bear counts, bear movement trajectories, salmon jump counts, and feeding behavior metrics. These results will be paired with environmental data to enable analysis across time, conditions, and location.

More information: https://github.com/katmai-vision-lab

## Project Structure
```arduino
katmai-cv-pipeline/
│
├── README.md
├── requirements.txt
|
├── docs
│   ├── images
│   │   └── 20260113-requirement.png
│   ├── meeting-notes.md
│
├── data/
│   ├── raw/
│   │   ├── videos/
│   │   │   ├── bears_cam1.mp4
│   │   │   └── bears_cam2.mp4
│   │   └── images/              # extracted frames (optional)
│   │
│   ├── annotations/             # only if you fine-tune
│   │   ├── images/
│   │   └── labels/
│   │
│   └── outputs/
│       ├── annotated_videos/
│       ├── frames/
│       └── logs/
│
├── models/
│   ├── yolov8/
│   │   └── yolov8n.pt
│   └── trackers/
│       ├── bytetrack.yaml
│       └── deepsort.yaml
│
├── src/
│   ├── config/
│   │   └── config.yaml
│   │
│   ├── detection/
│   │   └── yolo_detector.py
│   │
│   ├── tracking/
│   │   ├── bytetrack.py
│   │   └── deepsort.py
│   │
│   ├── counting/
│   │   └── bear_counter.py
│   │
│   ├── utils/
│   │   ├── video_io.py
│   │   ├── visualization.py
│   │   └── logger.py
│   │
│   └── main.py
│
├── notebooks/
│   ├── 01_explore_video.ipynb
│   ├── 02_test_yolo.ipynb
│   └── 03_debug_tracking.ipynb
│
└── scripts/
    ├── run_detection.py
    ├── run_tracking.py
    └── run_counting.py
```
## Local Dev Setup
Use conda to manage the virtual environment.
```
conda create -n katmai python=3.10 -y
conda activate katmai
pip install -r requirements.txt
```

## PR Process
- Do the local development on your own dev branch, eg. dev-yourname
- Once your code is ready, create a PR merge to main branch

You may use the following commands.
```
git checkout -b dev-xxx origin/main
git pull origin main
git push origin dev-xxx
```