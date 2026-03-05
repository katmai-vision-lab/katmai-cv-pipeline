"""
Simple Real-ESRGAN 4x upscaling script
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from basicsr.utils.download_util import load_file_from_url
    import torch
    import cv2
except ImportError as e:
    print(f"Error importing: {e}")
    print("Please install: pip install realesrgan basicsr torch opencv-python")
    sys.exit(1)

def upscale_images(input_dir, output_dir, limit=None):
    """Upscale images 4x using Real-ESRGAN"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(input_dir.rglob(f"*{ext}"))
    
    image_files = sorted(image_files)[:limit] if limit else sorted(image_files)
    
    print(f"Found {len(image_files)} images")
    print(f"Loading Real-ESRGAN x4plus model...")
    
    # Setup model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    
    model_path = load_file_from_url(
        url=model_url,
        model_dir='models/Real-ESRGAN',
        progress=True,
        file_name=None
    )
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=256,  # Smaller tile to avoid OOM
        tile_pad=10,
        pre_pad=0,
        half=True,
    )
    
    print("Model loaded!\n")
    
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_path.name}")
        
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        print(f"  Input: {w}x{h}")
        
        # Upscale
        try:
            output, _ = upsampler.enhance(img, outscale=4)
            h_out, w_out = output.shape[:2]
            print(f"  Output: {w_out}x{h_out}")
            
            # Save
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Saved: {output_path}\n")
        except Exception as e:
            print(f"  Error: {e}\n")
            continue
    
    print(f"\nDone! Upscaled images saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    upscale_images(args.input, args.output, args.limit)
