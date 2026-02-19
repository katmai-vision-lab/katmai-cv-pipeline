"""
X-CLIP Video Action Recognition for Bear Behavior Analysis

Uses Microsoft's X-CLIP model to understand video content and identify
bear behaviors like catching salmon, eating fish, etc.

Zero-shot capability: No training needed, just describe the action in text.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Tuple
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from transformers import AutoProcessor, AutoModel
    import decord
    decord.bridge.set_bridge('torch')
except ImportError as e:
    print(f"Error: {e}")
    print("Please install: pip install transformers decord")
    sys.exit(1)


class XCLIPBehaviorRecognizer:
    """Video action recognition using X-CLIP"""
    
    def __init__(self, model_name="microsoft/xclip-base-patch32", device=None):
        """
        Initialize X-CLIP model.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cuda:0', 'cuda:1', or 'cpu'
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading X-CLIP model: {model_name}")
        print(f"Device: {device}")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        print("Model loaded successfully!\n")
    
    def load_video_frames(self, video_path: str, num_frames: int = 8) -> torch.Tensor:
        """
        Load video and sample frames uniformly.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            
        Returns:
            Tensor of shape (num_frames, H, W, 3)
        """
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        
        return frames
    
    def load_frame_sequence(self, frame_paths: List[Path], num_frames: int = 8) -> np.ndarray:
        """
        Load a sequence of image frames.
        
        Args:
            frame_paths: List of paths to image files
            num_frames: Number of frames to sample
            
        Returns:
            Numpy array of shape (num_frames, H, W, 3)
        """
        # Sample frames uniformly if we have more than needed
        if len(frame_paths) > num_frames:
            indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
            sampled_paths = [frame_paths[i] for i in indices]
        else:
            sampled_paths = frame_paths
        
        frames = []
        for path in sampled_paths:
            img = Image.open(path).convert('RGB')
            frames.append(np.array(img))
        
        return np.array(frames)
    
    def recognize_actions(
        self,
        frames: np.ndarray,
        action_descriptions: List[str],
        return_scores: bool = True
    ) -> Dict:
        """
        Recognize actions in video frames.
        
        Args:
            frames: Numpy array of shape (num_frames, H, W, 3)
            action_descriptions: List of text descriptions of actions
            return_scores: Whether to return confidence scores
            
        Returns:
            Dictionary with predictions and scores
        """
        # Prepare inputs
        inputs = self.processor(
            text=action_descriptions,
            videos=list(frames),
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_video = outputs.logits_per_video
            probs = logits_per_video.softmax(dim=1)
        
        # Get results
        probs_np = probs.cpu().numpy()[0]
        best_idx = np.argmax(probs_np)
        
        result = {
            'predicted_action': action_descriptions[best_idx],
            'confidence': float(probs_np[best_idx])
        }
        
        if return_scores:
            result['all_scores'] = {
                action: float(score)
                for action, score in zip(action_descriptions, probs_np)
            }
        
        return result
    
    def analyze_video_segments(
        self,
        frame_dir: Path,
        action_descriptions: List[str],
        segment_size: int = 16,
        stride: int = 8,
        min_confidence: float = 0.3
    ) -> List[Dict]:
        """
        Analyze video by sliding window of frames.
        
        Args:
            frame_dir: Directory containing sequential frames
            action_descriptions: List of actions to detect
            segment_size: Number of frames per segment
            stride: Number of frames to slide forward
            min_confidence: Minimum confidence to include result
            
        Returns:
            List of detected actions with timestamps
        """
        # Get all frames
        frame_files = sorted(frame_dir.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(frame_dir.glob("*.png"))
        
        if not frame_files:
            print(f"No frames found in {frame_dir}")
            return []
        
        print(f"Found {len(frame_files)} frames")
        print(f"Analyzing with segment_size={segment_size}, stride={stride}")
        print(f"Action descriptions: {action_descriptions}\n")
        
        results = []
        
        # Sliding window
        for start_idx in range(0, len(frame_files) - segment_size + 1, stride):
            end_idx = start_idx + segment_size
            segment_frames = frame_files[start_idx:end_idx]
            
            # Extract timestamp from first frame name
            first_frame = segment_frames[0].stem
            if '_t' in first_frame:
                timestamp_str = first_frame.split('_t')[1].replace('s', '')
                timestamp = float(timestamp_str)
            else:
                timestamp = start_idx / 2.0  # Assuming 2 FPS
            
            print(f"[{start_idx:3d}-{end_idx:3d}] Time: {timestamp:.1f}s", end='')
            
            # Load frames
            frames = self.load_frame_sequence(segment_frames, num_frames=8)
            
            # Recognize action
            result = self.recognize_actions(frames, action_descriptions)
            
            print(f" -> {result['predicted_action']} ({result['confidence']:.3f})")
            
            # Only include if confidence is high enough
            if result['confidence'] >= min_confidence:
                results.append({
                    'timestamp': timestamp,
                    'frame_range': (start_idx, end_idx),
                    'action': result['predicted_action'],
                    'confidence': result['confidence'],
                    'all_scores': result['all_scores']
                })
        
        return results


def analyze_bear_behavior(
    frame_dir: str,
    output_json: str = None,
    segment_size: int = 16,
    stride: int = 8,
    min_confidence: float = 0.3,
    device: str = None
):
    """
    Analyze bear behavior in video frames.
    
    Args:
        frame_dir: Directory with sequential frames
        output_json: Path to save results (optional)
        segment_size: Frames per segment
        stride: Frames to skip between segments
        min_confidence: Minimum confidence threshold
        device: GPU device to use
    """
    frame_dir = Path(frame_dir)
    
    # Define bear behaviors to detect
    action_descriptions = [
        "a bear catching salmon in a river",
        "a bear eating fish",
        "a bear standing in water",
        "a bear walking",
        "a bear sitting",
        "salmon jumping in water near a bear"
    ]
    
    print("="*60)
    print("X-CLIP Bear Behavior Recognition")
    print("="*60)
    print(f"Frame directory: {frame_dir}")
    print(f"Segment size: {segment_size} frames")
    print(f"Stride: {stride} frames")
    print(f"Min confidence: {min_confidence}")
    print("="*60)
    print()
    
    # Initialize recognizer
    recognizer = XCLIPBehaviorRecognizer(device=device)
    
    # Analyze video
    results = recognizer.analyze_video_segments(
        frame_dir=frame_dir,
        action_descriptions=action_descriptions,
        segment_size=segment_size,
        stride=stride,
        min_confidence=min_confidence
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total segments analyzed: {len(results)}")
    
    # Count actions
    action_counts = {}
    for r in results:
        action = r['action']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print("\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")
    
    # Find salmon-related segments
    salmon_segments = [
        r for r in results 
        if 'catching salmon' in r['action'] or 'eating fish' in r['action'] or 'salmon jumping' in r['action']
    ]
    
    print(f"\nSegments with salmon activity: {len(salmon_segments)}")
    if salmon_segments:
        print("Timestamps with salmon:")
        for seg in salmon_segments:
            print(f"  {seg['timestamp']:.1f}s: {seg['action']} (conf: {seg['confidence']:.3f})")
    
    # Save results
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'config': {
                'segment_size': segment_size,
                'stride': stride,
                'min_confidence': min_confidence,
                'action_descriptions': action_descriptions
            },
            'results': results,
            'summary': {
                'total_segments': len(results),
                'action_counts': action_counts,
                'salmon_segments': len(salmon_segments)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="X-CLIP bear behavior recognition")
    parser.add_argument("--frames", required=True, help="Directory with sequential frames")
    parser.add_argument("--output", help="JSON file to save results")
    parser.add_argument("--segment-size", type=int, default=16, help="Frames per segment")
    parser.add_argument("--stride", type=int, default=8, help="Frames between segments")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Min confidence")
    parser.add_argument("--device", default=None, help="Device: cuda, cuda:0, cuda:1, cpu")
    
    args = parser.parse_args()
    
    analyze_bear_behavior(
        frame_dir=args.frames,
        output_json=args.output,
        segment_size=args.segment_size,
        stride=args.stride,
        min_confidence=args.min_confidence,
        device=args.device
    )
