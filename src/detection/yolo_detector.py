from ultralytics import YOLO
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    YOLOV8N_PATH,
    OUTPUTS_DIR,
    TRAINED_MODELS_DIR
)
video_path = str(RAW_DATA_DIR) + '/' + '2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv'
# Load pre-trained model
model = YOLO(str(YOLOV8N_PATH))

# TODO: waiting for CVAT frames
# model.train(
#     data = 'mock.yaml',
#     epochs=30,
#     imgsz=640,
#     batch=8,
#     project=str(TRAINED_MODELS_DIR),
#     name='bear_detector_finetuned',
#     resume=True  # continue from previous training
# )

# Test on your video
results = model.predict(
    source=video_path,
    classes=[21],  # 21 is 'bear' in COCO dataset
    save=True,
    conf=0.25,
    show_labels=True,
    show_conf=True,
    project=str(OUTPUTS_DIR), # Base directory
    name='detect_bear', # Subdirectory name
    exist_ok=True # overwrite
)