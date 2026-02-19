"""
Simple binary detector: Is the bear eating fish? Yes or No.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List, Dict
import json
from tqdm import tqdm
import argparse


class SimpleEatingDetector:
    def __init__(self, model_name="microsoft/xclip-base-patch32", device=None):
        """Initialize detector"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("Model loaded!")
    
    def load_frames(self, frame_paths: List[Path]) -> List[np.ndarray]:
        """Load frames"""
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert('RGB')
            frames.append(np.array(img))
        return frames
    
    def is_eating_fish(self, frames: List[np.ndarray]) -> Dict:
        """
        Simple binary question: Is bear eating fish?
        
        Returns dict with: is_eating, confidence, all_scores
        """
        # Use only 2 very clear, contrastive prompts
        prompts = [
            "a bear eating fish",  # Positive
            "a bear swimming in water"  # Negative (common alternative action)
        ]
        
        # Process
        text_inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        pixel_values = self.processor.image_processor(
            frames,
            return_tensors="pt"
        )["pixel_values"].to(self.device)
        
        # Inference
        with torch.no_grad():
            inputs = {**text_inputs, "pixel_values": pixel_values}
            outputs = self.model(**inputs)
            logits = outputs.logits_per_video[0]  # Shape: (num_prompts,)
            probs = torch.softmax(logits, dim=0)
        
        probs_np = probs.cpu().numpy()
        eating_score = float(probs_np[0])
        not_eating_score = float(probs_np[1])
        
        return {
            'is_eating': eating_score > not_eating_score,
            'eating_confidence': eating_score,
            'not_eating_confidence': not_eating_score,
            'confidence_margin': eating_score - not_eating_score,
            'all_scores': {p: float(s) for p, s in zip(prompts, probs_np)}
        }
    
    def analyze_video(
        self,
        frame_dir: Path,
        stride: int = 4
    ) -> List[Dict]:
        """
        Analyze video with fixed 8-frame segments
        """
        segment_size = 8  # Fixed for X-CLIP
        
        frame_files = sorted(frame_dir.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(frame_dir.glob("*.png"))
        
        if not frame_files:
            raise ValueError(f"No frames in {frame_dir}")
        
        total_frames = len(frame_files)
        print(f"Found {total_frames} frames")
        print(f"Using {segment_size} frames per segment, stride {stride}")
        
        results = []
        fps = 2.0
        
        for start_idx in tqdm(range(0, total_frames - segment_size + 1, stride),
                             desc="Analyzing"):
            end_idx = start_idx + segment_size
            segment_frames = frame_files[start_idx:end_idx]
            
            frames = self.load_frames(segment_frames)
            detection = self.is_eating_fish(frames)
            
            middle_frame = (start_idx + end_idx) // 2
            timestamp = middle_frame / fps
            
            results.append({
                'frame_range': [start_idx, end_idx - 1],
                'middle_frame': middle_frame,
                'timestamp': timestamp,
                **detection
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Simple eating detector')
    parser.add_argument('--frames', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stride', type=int, default=4)
    
    args = parser.parse_args()
    
    frame_dir = Path(args.frames)
    if not frame_dir.exists():
        print(f"Error: {frame_dir} not found")
        return
    
    detector = SimpleEatingDetector()
    
    print("\n" + "="*60)
    print("Simple Eating Detection: Bear eating fish?")
    print("="*60)
    
    results = detector.analyze_video(frame_dir, stride=args.stride)
    
    # Stats
    eating_segments = [r for r in results if r['is_eating']]
    high_conf = [r for r in eating_segments if r['eating_confidence'] > 0.7]
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total segments: {len(results)}")
    print(f"Eating detected: {len(eating_segments)} ({len(eating_segments)/len(results)*100:.1f}%)")
    print(f"High confidence (>70%): {len(high_conf)}")
    
    if eating_segments:
        avg_conf = np.mean([r['eating_confidence'] for r in eating_segments])
        max_conf = max(r['eating_confidence'] for r in eating_segments)
        print(f"Average eating confidence: {avg_conf:.2%}")
        print(f"Max eating confidence: {max_conf:.2%}")
        
        # Top 10
        top = sorted(eating_segments, key=lambda x: x['eating_confidence'], reverse=True)[:10]
        print("\nTop 10 eating moments:")
        for i, seg in enumerate(top, 1):
            print(f"  {i}. Time {seg['timestamp']:.1f}s, Confidence: {seg['eating_confidence']:.1%}")
    
    # Save
    output_data = {
        'config': {
            'frame_dir': str(frame_dir),
            'stride': args.stride,
            'model': 'microsoft/xclip-base-patch32'
        },
        'statistics': {
            'total_segments': len(results),
            'eating_segments': len(eating_segments),
            'high_confidence': len(high_conf),
            'eating_percentage': len(eating_segments)/len(results)*100 if results else 0
        },
        'results': results
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Saved to: {output_path}")


if __name__ == '__main__':
    main()
