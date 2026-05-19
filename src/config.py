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
TRAINED_BEAR_DETECTOR_PATH = TRAINED_MODELS_DIR / 'bear_detector3' / 'weights' / 'best.pt'

# Output directories
PREDICTIONS_DIR = PROJECT_ROOT / 'predictions'

# Tracker configurations
TRACKERS_CONFIG_DIR = PROJECT_ROOT / 'configs' / 'trackers'

# Default YOLO classes
COCO_BEAR_CLASS = 21
FINETUNED_BEAR_CLASS = 0  # Class ID in the fine-tuned bear_detector3 model


# ---------------------------------------------------------------------------
# Cross-platform device helpers
# ---------------------------------------------------------------------------
def get_device(prefer: str | None = None) -> str:
    """Return the best available torch device string.

    Resolution order (when `prefer` is None):
      1. CUDA (NVIDIA on Linux/Windows, or container with --gpus all)
      2. MPS  (Apple Silicon: M1/M2/M3/M4)
      3. CPU  (fallback everywhere)

    If `prefer` is given ("cuda" / "mps" / "cpu") it is honored when
    available; otherwise the helper falls back to the next option.
    """
    import torch  # local import so config.py stays lightweight

    if prefer == "cpu":
        return "cpu"
    if prefer in (None, "cuda") and torch.cuda.is_available():
        return "cuda"
    if prefer in (None, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cuda_empty_cache() -> None:
    """torch.cuda.empty_cache() that's a no-op outside CUDA.

    Some PyTorch versions raise on Mac/MPS when empty_cache is called
    unconditionally; this guard keeps cross-platform code safe.
    """
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()