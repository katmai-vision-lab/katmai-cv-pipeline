from ultralytics import YOLO
from pathlib import Path
import sys
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    PRETRAINED_MODELS_DIR,
    TRAINED_MODELS_DIR,
    PREDICTIONS_DIR,
    YOLOV8M_PATH
)

class BearDetector:
    """YOLO-based bear detector with training, prediction, and evaluation"""

    def __init__(self, model_path=None):
        """
        Initialize detector
        
        Args:
            model_path: Path to model file. If None, uses latest trained model with best mAP50
        """
        if model_path is None:
            # 用最佳的自训练模型（按mAP50评分）
            trained_dir = TRAINED_MODELS_DIR
            model_dirs = [d for d in trained_dir.iterdir() if d.is_dir()]
            
            best_model = None
            best_mAP50 = -1
            
            # 检查每个模型的结果
            for model_dir in model_dirs:
                results_csv = model_dir / 'results.csv'
                if results_csv.exists():
                    try:
                        with open(results_csv, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                # 读取最后一行（最好的epoch）
                                last_line = lines[-1].strip()
                                values = last_line.split(',')
                                # mAP50(B) 通常在第8列 (0-indexed: 7)
                                if len(values) > 7:
                                    try:
                                        mAP50 = float(values[7])
                                        if mAP50 > best_mAP50:
                                            best_mAP50 = mAP50
                                            best_model = model_dir
                                    except (ValueError, IndexError):
                                        pass
                    except Exception:
                        pass
            
            if best_model:
                model_path = best_model / 'weights' / 'best.pt'
                print(f"🏆 Auto-selected best model: {best_model.name} (mAP50={best_mAP50:.4f})")
            else:
                model_path = YOLOV8M_PATH
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.model = YOLO(str(model_path))
        print(f"✓ Loaded model: {model_path.parent.parent.name} ({model_path})")


    def train(self, data_yaml, epochs=50, imgsz=640, batch=8, 
              project=None, name='bear_detector', resume=False, **kwargs):
        """
        Train/fine-tune the model
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            project: Output directory for training results
            name: Experiment name
            resume: Resume from last checkpoint
            **kwargs: Additional YOLO training parameters
        
        Returns:
            Training results
        """
        if project is None:
            project = TRAINED_MODELS_DIR

        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

        print(f"\n{'='*60}")
        print(f"Training Bear Detector")
        print(f"{'='*60}")
        print(f"Data: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch}")
        print(f"Output: {project}/{name}")
        print(f"{'='*60}\n")

        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(project),
            name=name,
            resume=resume,
            **kwargs
        )

        # Update model path to trained weights
        self.model_path = Path(project) / name / 'weights' / 'best.pt'
        self.model = YOLO(str(self.model_path))

        print(f"\n✓ Training complete!")
        print(f"Best weights: {self.model_path}")

        return results

    def predict_video(self, video_path, output_name=None, conf=0.25, 
                      classes=None, save=True, **kwargs):
        """
        Run detection on video
        
        Args:
            video_path: Path to video file or filename in RAW_DATA_DIR
            output_name: Output directory name
            conf: Confidence threshold
            classes: List of class IDs to detect (None = all classes)
            save: Save annotated video
            **kwargs: Additional YOLO prediction parameters
        
        Returns:
            results: YOLO results object
            output_dir: Path to output directory
        """
        # Handle video path
        video_path = Path(video_path)
        if not video_path.is_absolute():
            video_path = RAW_DATA_DIR / video_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Generate output name
        if output_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_name = f"{timestamp}_{video_path.stem}"

        print(f"\n{'='*60}")
        print(f"Bear Detection")
        print(f"{'='*60}")
        print(f"Video: {video_path.name}")
        print(f"Model: {self.model_path.name}")
        print(f"Confidence: {conf}")
        print(f"Output: {PREDICTIONS_DIR / output_name}")
        print(f"{'='*60}\n")

        # Run prediction with streaming to avoid memory overload
        results = self.model.predict(
            source=str(video_path),
            conf=conf,
            classes=classes,
            save=save,
            show_labels=True,
            show_conf=True,
            project=str(PREDICTIONS_DIR),
            name=output_name,
            exist_ok=True,
            stream=True,
            **kwargs
        )

        output_dir = PREDICTIONS_DIR / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process streaming results and collect metadata
        total_frames = 0
        total_detections = 0
        
        for result in results:
            total_frames += 1
            boxes = result.boxes
            if boxes is not None:
                total_detections += len(boxes)

        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'video': str(video_path),
            'model': str(self.model_path),
            'confidence_threshold': conf,
            'total_frames': total_frames,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Detection complete!")
        print(f"Results: {output_dir}")

        return results, output_dir