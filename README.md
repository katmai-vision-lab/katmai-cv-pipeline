# Computer Vision Pipeline to Detect, Track & Quantify Feeding Habits of Katmai NPP Alaskan Brown Bears
This project focuses on building an open-source, Python-based computer vision pipeline that analyzes video data from Katmai National Park & Preserve. The system is designed to automatically detect and track individual Alaskan brown bears, count salmon attempting to jump Brooks Falls, and quantify feeding behavior over time. In addition to visual analysis, the pipeline will integrate environmental context such as water level, stream flow, weather, and time of day to support deeper ecological insight.

Video data is sourced primarily from Explore.org bear cams in the Brooks Falls and Brooks River region.

## Scope
The system is designed to run on a consumer-grade laptop or desktop and ingest short-form video clips (1–15 minutes). From these inputs, it will produce structured outputs including individual and total bear counts, bear movement trajectories, salmon jump counts, and feeding behavior metrics. These results will be paired with environmental data to enable analysis across time, conditions, and location.

More information: https://github.com/katmai-vision-lab

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
## Run Pipeline
Quick test with pretrained model.
```bash
python -m src.main \
    --mode full \
    --video "bears/2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv" \
    --skip-train \
    --epochs 3 \
    --conf 0.12 \
    --ground-truth 5
```

Train + predict + evaluate (original full pipeline).
```bash
python -m src.main \
    --mode full \
    --video "data/raw/bears/2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv" \
    --model models/trained/bear_detector/weights/best.pt \
    --skip-train \
    --classes 0 \
    --conf 0.25 \
    --ground-truth 5
```

Use your fine-tuned model.
```bash
python -m src.main \
    --mode full \
    --video "bears/2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv" \
    --model models/trained/pipeline_trained_model/weights/best.pt \
    --skip-train \
    --epochs 3 \
    --conf 0.12 \
    --ground-truth 5
```

## Useful Link
SharePoint:
https://uwnetid.sharepoint.com/sites/katmai-vision-lab/Shared%20Documents/Forms/AllItems.aspx
