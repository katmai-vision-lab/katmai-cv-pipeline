import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, TRAINED_MODELS_DIR

def evaluate_video(model, video_path, ground_truth_count=None):
    """Evaluate model on video and compare with manual count"""
    results = model.predict(source=video_path, stream=True)
    
    detections_per_frame = []
    
    for frame_id, result in enumerate(results):
        boxes = result.boxes
        num_bears = len(boxes)
        confidences = boxes.conf.cpu().numpy()
        
        detections_per_frame.append({
            'frame': frame_id,
            'num_bears': num_bears,
            'avg_confidence': confidences.mean() if len(confidences) > 0 else 0,
            'max_confidence': confidences.max() if len(confidences) > 0 else 0
        })
    
    df = pd.DataFrame(detections_per_frame)
    
    print(f"Average bears per frame: {df['num_bears'].mean():.2f}")
    print(f"Max bears in frame: {df['num_bears'].max()}")
    print(f"Average confidence: {df['avg_confidence'].mean():.2f}")
    
    if ground_truth_count:
        print(f"Ground truth: {ground_truth_count} bears")
    
    return df

# Use it
video_path = str(RAW_DATA_DIR) + '/' + '2025-09-19 23-30-11_Brooks_Falls_Low_really_0630PM _MDT_5_bears.mkv'
tuned_model_path = str(TRAINED_MODELS_DIR) + '/' + 'bear_detector_finetuned.pt'
model = YOLO(tuned_model_path)
stats = evaluate_video(model, video_path, ground_truth_count=3)
