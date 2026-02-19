"""
Annotate eating fish moments on frames
"""
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def annotate_eating_frame(frame, confidence, timestamp, frame_idx):
    """
    Annotate frame with eating detection
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    # Color: green for high conf, yellow for medium, orange for low
    if confidence > 0.7:
        color = (0, 255, 0)  # Green
        label = "BEAR EATING FISH - HIGH CONFIDENCE"
    elif confidence > 0.6:
        color = (0, 255, 255)  # Yellow
        label = "BEAR EATING FISH - MEDIUM CONFIDENCE"
    else:
        color = (0, 165, 255)  # Orange
        label = "BEAR EATING FISH - LOW CONFIDENCE"
    
    # Draw border
    border_thickness = max(8, int(h * 0.015))
    cv2.rectangle(annotated, (0, 0), (w, h), color, border_thickness)
    
    # Semi-transparent overlay at top
    overlay = annotated.copy()
    panel_height = int(h * 0.15)
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(1.5, w / 700)
    thickness = max(3, int(font_scale * 2.5))
    
    # Main label
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 50
    
    # Shadow
    cv2.putText(annotated, label, (text_x + 3, text_y + 3), font,
               font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Text
    cv2.putText(annotated, label, (text_x, text_y), font,
               font_scale, color, thickness, cv2.LINE_AA)
    
    # Confidence bar
    bar_width = int(w * 0.5)
    bar_height = 30
    bar_x = (w - bar_width) // 2
    bar_y = text_y + 30
    
    # Background
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                 (40, 40, 40), -1)
    # Fill
    filled_width = int(bar_width * confidence)
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                 color, -1)
    # Border
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                 color, 3)
    
    # Confidence text
    conf_text = f"Confidence: {confidence:.1%}"
    conf_size = cv2.getTextSize(conf_text, font, font_scale * 0.7, thickness - 1)[0]
    conf_x = (w - conf_size[0]) // 2
    cv2.putText(annotated, conf_text, (conf_x, bar_y + bar_height + 35), font,
               font_scale * 0.7, (255, 255, 255), thickness - 1, cv2.LINE_AA)
    
    # Timestamp at bottom
    time_text = f"Time: {timestamp:.1f}s | Frame: {frame_idx}"
    time_size = cv2.getTextSize(time_text, font, font_scale * 0.6, 2)[0]
    time_x = (w - time_size[0]) // 2
    cv2.putText(annotated, time_text, (time_x, h - 30), font,
               font_scale * 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='Detection results JSON')
    parser.add_argument('--frames', required=True, help='Frames directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                       help='Min confidence to annotate')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.json, 'r') as f:
        data = json.load(f)
    
    eating_segments = [r for r in data['results'] 
                       if r['is_eating'] and r['eating_confidence'] >= args.min_confidence]
    
    # Get frames
    frames_dir = Path(args.frames)
    frame_files = sorted(frames_dir.glob('*.jpg')) or sorted(frames_dir.glob('*.png'))
    
    if not frame_files:
        print(f"No frames in {frames_dir}")
        return
    
    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(frame_files)} frames")
    print(f"Will annotate {len(eating_segments)} eating segments")
    print(f"Min confidence: {args.min_confidence:.0%}")
    print()
    
    # Annotate
    annotated_count = 0
    
    for seg in tqdm(eating_segments, desc="Annotating"):
        # Use middle frame
        middle_frame = seg['middle_frame']
        
        if middle_frame >= len(frame_files):
            continue
        
        frame_path = frame_files[middle_frame]
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            continue
        
        # Annotate
        annotated = annotate_eating_frame(
            frame,
            seg['eating_confidence'],
            seg['timestamp'],
            middle_frame
        )
        
        # Save
        output_file = output_dir / f"eating_{seg['timestamp']:.1f}s_frame{middle_frame:04d}_conf{seg['eating_confidence']:.0%}.jpg"
        cv2.imwrite(str(output_file), annotated)
        annotated_count += 1
    
    print(f"\n✅ Annotated {annotated_count} frames")
    print(f"📁 Saved to: {output_dir}")


if __name__ == '__main__':
    main()
