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
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
DATASETS_DIR = DATA_DIR / 'datasets'
FRAMES_DIR = DATA_DIR / 'frames'

# Model directories
MODELS_DIR = PROJECT_ROOT / 'models'
PRETRAINED_MODELS_DIR = MODELS_DIR / 'pretrained'
TRAINED_MODELS_DIR = MODELS_DIR / 'trained'

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

# Config directory
CONFIG_DIR = PROJECT_ROOT / 'config'

# Docs directory
DOCS_DIR = PROJECT_ROOT / 'docs'

# Model paths (commonly used models)
YOLOV8N_PATH = PRETRAINED_MODELS_DIR / 'yolov8n.pt'

# Default YOLO classes
COCO_BEAR_CLASS = 21