"""
Detect fish-eating behavior using X-CLIP as a binary classifier
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


class FishEatingDetector:
    def __init__(self, model_name="microsoft/xclip-base-patch32", device=None):
        """
        Initialize the fish-eating detector
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def load_frames(self, frame_paths: List[Path]) -> np.ndarray:
        """Load frames as numpy array"""
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert('RGB')
            frames.append(np.array(img))
        return frames
    
    def detect_eating(
        self,
        frames: List[np.ndarray],
        return_all_scores: bool = False
    ) -> Dict:
        """
        Detect if bear is eating fish
        
        Args:
            frames: List of frames (numpy arrays)
            return_all_scores: Return scores for all actions
            
        Returns:
            Dictionary with eating detection result
        """
        # Binary classification prompts - much more direct and specific
        positive_prompts = [
            "a bear with a fish in its mouth",
            "a bear chewing on a salmon",
            "a bear holding and eating a fish",
            "a bear biting a salmon",
            "a bear tearing apart a fish"
        ]
        
        negative_prompts = [
            "a bear in water looking for fish",
            "a bear standing without food",
            "a bear walking in a river"
        ]
        
        all_prompts = positive_prompts + negative_prompts
        
        # Process text
        text_inputs = self.processor(
            text=all_prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Process frames
        pixel_values = self.processor.image_processor(
            frames,
            return_tensors="pt"
        )["pixel_values"].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            inputs = {**text_inputs, "pixel_values": pixel_values}
            outputs = self.model(**inputs)
            logits = outputs.logits_per_video  # Shape: (1, num_prompts)
            probs = torch.softmax(logits, dim=1)[0]
        
        probs_np = probs.cpu().numpy()
        
        # Calculate eating probability (average of positive prompts)
        eating_prob = float(np.mean(probs_np[:len(positive_prompts)]))
        not_eating_prob = float(np.mean(probs_np[len(positive_prompts):]))
        
        # Get best matching prompt
        best_idx = int(np.argmax(probs_np))
        best_prompt = all_prompts[best_idx]
        best_conf = float(probs_np[best_idx])
        
        result = {
            'is_eating': eating_prob > not_eating_prob,
            'eating_confidence': eating_prob,
            'not_eating_confidence': not_eating_prob,
            'best_match': best_prompt,
            'best_confidence': best_conf,
            'confidence_margin': abs(eating_prob - not_eating_prob)
        }
        
        if return_all_scores:
            result['all_scores'] = {
                prompt: float(score)
                for prompt, score in zip(all_prompts, probs_np)
            }
        
        return result
    
    def analyze_video(
        self,
        frame_dir: Path,
        segment_size: int = 8,
        stride: int = 4
    ) -> List[Dict]:
        """
        Analyze video segments for fish eating
        
        Args:
            frame_dir: Directory with frames
            segment_size: Frames per segment
            stride: Stride between segments
            
        Returns:
            List of detection results
        """
        # Get frames
        frame_files = sorted(frame_dir.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(frame_dir.glob("*.png"))
        
        if not frame_files:
            raise ValueError(f"No frames found in {frame_dir}")
        
        total_frames = len(frame_files)
        print(f"Found {total_frames} frames")
        print(f"Segment size: {segment_size}, Stride: {stride}")
        
        # Calculate FPS (assume frame filenames contain frame numbers)
        fps = 2.0  # Default
        
        results = []
        
        # Sliding window
        for start_idx in tqdm(range(0, total_frames - segment_size + 1, stride),
                             desc="Analyzing segments"):
            end_idx = start_idx + segment_size
            segment_frames = frame_files[start_idx:end_idx]
            
            # Load frames
            frames = self.load_frames(segment_frames)
            
            # Detect eating
            detection = self.detect_eating(frames, return_all_scores=True)
            
            # Calculate timestamp (middle of segment)
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
    parser = argparse.ArgumentParser(
        description='Detect fish-eating behavior in video frames'
    )
    parser.add_argument('--frames', type=str, required=True,
                       help='Directory containing video frames')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--segment-size', type=int, default=8,
                       help='Number of frames per segment (default: 8)')
    parser.add_argument('--stride', type=int, default=4,
                       help='Stride between segments (default: 4)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate paths
    frame_dir = Path(args.frames)
    if not frame_dir.exists():
        print(f"Error: Frame directory not found: {frame_dir}")
        return
    
    # Initialize detector
    detector = FishEatingDetector(device=args.device)
    
    # Analyze video
    print("\n" + "="*60)
    print("Fish-Eating Behavior Detection")
    print("="*60)
    
    results = detector.analyze_video(
        frame_dir,
        segment_size=args.segment_size,
        stride=args.stride
    )
    
    # Calculate statistics
    eating_segments = [r for r in results if r['is_eating']]
    high_conf_eating = [r for r in eating_segments if r['eating_confidence'] > 0.6]
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Total segments analyzed: {len(results)}")
    print(f"Segments with eating detected: {len(eating_segments)} ({len(eating_segments)/len(results)*100:.1f}%)")
    print(f"High-confidence eating (>60%): {len(high_conf_eating)} ({len(high_conf_eating)/len(results)*100:.1f}%)")
    
    if eating_segments:
        avg_conf = np.mean([r['eating_confidence'] for r in eating_segments])
        print(f"Average eating confidence: {avg_conf:.2%}")
        
        # Show top 5 eating segments
        top_eating = sorted(eating_segments, key=lambda x: x['eating_confidence'], reverse=True)[:5]
        print("\nTop 5 eating segments:")
        for i, seg in enumerate(top_eating, 1):
            print(f"  {i}. Time {seg['timestamp']:.1f}s, "
                  f"Confidence: {seg['eating_confidence']:.2%}, "
                  f"Match: {seg['best_match']}")
    
    # Save results
    output_data = {
        'config': {
            'frame_dir': str(frame_dir),
            'segment_size': args.segment_size,
            'stride': args.stride,
            'model': 'microsoft/xclip-base-patch32'
        },
        'statistics': {
            'total_segments': len(results),
            'eating_segments': len(eating_segments),
            'high_confidence_eating': len(high_conf_eating),
            'eating_percentage': len(eating_segments)/len(results)*100 if results else 0
        },
        'results': results
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"📊 File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
