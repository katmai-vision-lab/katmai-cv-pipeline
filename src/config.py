"""
src/config.py
Central configuration for all paths in the project
"""
from pathlib import Path

# Project root: go up one level from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
ANNOTATION_DIR = DATA_DIR / 'annotation'

# Model directories
MODELS_DIR = PROJECT_ROOT / 'models'
PRETRAINED_MODELS_DIR = MODELS_DIR / 'pretrained'
TRAINED_MODELS_DIR = MODELS_DIR / 'trained'

# Model paths (commonly used models)
YOLOV8N_PATH = PRETRAINED_MODELS_DIR / 'yolov8n.pt'

# Output directories
PREDICTIONS_DIR = PROJECT_ROOT / 'predictions'

# Default YOLO classes
COCO_BEAR_CLASS = 21