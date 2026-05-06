"""
Rescale YOLO format labels to match target image size.

Use this when you detect objects on upscaled images but need labels
for the original resolution.
"""

import argparse
from pathlib import Path


def rescale_yolo_labels(
    input_dir: str,
    output_dir: str,
    scale_factor: float,
):
    """
    Rescale YOLO format labels by a scale factor.
    
    Since YOLO format uses normalized coordinates (0-1), we don't need
    to rescale the coordinates themselves - they stay the same!
    We just copy the files.
    
    Args:
        input_dir: Directory with original labels
        output_dir: Directory to save rescaled labels
        scale_factor: Scale factor (e.g., 0.5 for 2x upscaled images)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(input_dir.glob("*.txt"))
    
    print(f"\n{'='*60}")
    print(f"Copying YOLO Labels (normalized coordinates)")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Note: YOLO coordinates are normalized, no rescaling needed!")
    print(f"Labels: {len(label_files)}")
    print(f"{'='*60}\n")
    
    for label_file in label_files:
        # YOLO format uses normalized coordinates (0-1)
        # So they work for any resolution!
        output_file = output_dir / label_file.name
        
        with open(label_file, 'r') as f:
            content = f.read()
        
        with open(output_file, 'w') as f:
            f.write(content)
    
    print(f"Copied {len(label_files)} label files")
    print(f"These labels work for ANY image resolution!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescale YOLO labels")
    parser.add_argument("--input", required=True, help="Input label directory")
    parser.add_argument("--output", required=True, help="Output label directory")
    parser.add_argument("--scale", type=float, default=0.5, 
                       help="Scale factor (0.5 = half size, 2.0 = double size)")
    
    args = parser.parse_args()
    
    rescale_yolo_labels(
        input_dir=args.input,
        output_dir=args.output,
        scale_factor=args.scale,
    )
