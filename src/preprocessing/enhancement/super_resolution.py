"""
Super-Resolution Image Enhancement using Real-ESRGAN

Upscales images 2x or 4x using deep learning models, significantly improving
quality for small object detection.

Models:
- RealESRGAN_x4plus: General purpose 4x upscaling
- RealESRGAN_x2plus: Faster 2x upscaling
- RealESRGAN_x4plus_anime_6B: For anime/cartoon-style images

Requires: pip install realesrgan basicsr facexlib gfpgan
"""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
import torch

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from basicsr.utils.download_util import load_file_from_url
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("Warning: Real-ESRGAN not installed")
    print("Install with: pip install realesrgan basicsr")


class SuperResolutionEnhancer:
    """Enhance images using Real-ESRGAN super-resolution"""
    
    def __init__(self, model_name='RealESRGAN_x4plus', scale=4, tile=400, tile_pad=10):
        """
        Args:
            model_name: Model to use
                - 'RealESRGAN_x4plus': Best quality, 4x upscale
                - 'RealESRGAN_x2plus': Faster, 2x upscale
                - 'RealESRNet_x4plus': Sharper but more artifacts
            scale: Upscaling factor (2 or 4)
            tile: Tile size for processing (smaller = less VRAM)
            tile_pad: Padding for tiles
        """
        if not REALESRGAN_AVAILABLE:
            raise ImportError("Real-ESRGAN not installed. Run: pip install realesrgan basicsr")
        
        self.model_name = model_name
        self.scale = scale
        
        # Model configurations
        model_configs = {
            'RealESRGAN_x4plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                'scale': 4,
            },
            'RealESRNet_x4plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
                'scale': 4,
            },
            'RealESRGAN_x2plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                'scale': 2,
            },
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_configs.keys())}")
        
        config = model_configs[model_name]
        
        print(f"Loading Real-ESRGAN model: {model_name}")
        print(f"Scale: {config['scale']}x")
        
        # Download model weights first
        model_path = load_file_from_url(
            url=config['url'],
            model_dir='models/Real-ESRGAN',
            progress=True,
            file_name=None
        )
        
        # Initialize upsampler
        self.upsampler = RealESRGANer(
            scale=config['scale'],
            model_path=model_path,
            model=config['model'],
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=0,
            half=True,  # Use FP16 for faster processing
        )
        
        print("Model loaded successfully!")
    
    def enhance(self, image):
        """
        Enhance image using super-resolution.
        
        Args:
            image: PIL Image or numpy array (RGB)
            
        Returns:
            Enhanced PIL Image
        """
        # Convert to numpy if PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply super-resolution
        output, _ = self.upsampler.enhance(img_bgr, outscale=self.scale)
        
        # Convert back to RGB
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(output_rgb)


def enhance_batch_sr(
    input_dir: str,
    output_dir: str,
    model_name: str = 'RealESRGAN_x2plus',
    scale: int = 2,
    tile: int = 400,
    max_size: int = None,
    limit: int = None,
):
    """
    Enhance a batch of images using super-resolution.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save enhanced images
        model_name: Real-ESRGAN model name
        scale: Upscaling factor
        tile: Tile size for processing (reduce if OOM)
        max_size: Max dimension of output image (to avoid huge files)
        limit: Max number of images to process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []
    
    for f in input_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in image_extensions:
            image_files.append(f)
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    image_files = sorted(image_files, key=lambda x: x.name)
    
    if limit:
        image_files = image_files[:limit]
    
    print(f"\n{'='*60}")
    print(f"Super-Resolution Image Enhancement")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_name}")
    print(f"Scale: {scale}x")
    print(f"Tile size: {tile}")
    if max_size:
        print(f"Max output size: {max_size}px")
    print(f"Images: {len(image_files)}")
    print(f"{'='*60}\n")
    
    # Create enhancer
    enhancer = SuperResolutionEnhancer(
        model_name=model_name,
        scale=scale,
        tile=tile,
    )
    
    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {image_path.name}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size
        print(f"  Original size: {orig_size[0]}x{orig_size[1]}")
        
        # Enhance
        try:
            enhanced = enhancer.enhance(image)
            enhanced_size = enhanced.size
            print(f"  Enhanced size: {enhanced_size[0]}x{enhanced_size[1]}")
            
            # Resize if too large
            if max_size and max(enhanced_size) > max_size:
                ratio = max_size / max(enhanced_size)
                new_size = (int(enhanced_size[0] * ratio), int(enhanced_size[1] * ratio))
                enhanced = enhanced.resize(new_size, Image.LANCZOS)
                print(f"  Resized to: {new_size[0]}x{new_size[1]} (max_size={max_size})")
            
            # Save
            output_path = output_dir / image_path.name
            enhanced.save(output_path, quality=95)
            print(f"  Saved: {output_path}\n")
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue
    
    print(f"\n{'='*60}")
    print(f"Enhancement complete!")
    print(f"Enhanced images saved to: {output_dir}")
    print(f"{'='*60}\n")


# Alternative: Use BSRGAN (simpler, no dependencies)
def enhance_with_opencv_sr(
    input_dir: str,
    output_dir: str,
    scale: int = 2,
    limit: int = None,
):
    """
    Simple super-resolution using OpenCV's DNN module.
    Less quality than Real-ESRGAN but no external dependencies.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []
    
    for f in input_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in image_extensions:
            image_files.append(f)
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    image_files = sorted(image_files, key=lambda x: x.name)
    
    if limit:
        image_files = image_files[:limit]
    
    print(f"\n{'='*60}")
    print(f"OpenCV Super-Resolution Enhancement")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Scale: {scale}x (using EDSR)")
    print(f"Images: {len(image_files)}")
    print(f"{'='*60}\n")
    
    # Download EDSR model if needed
    model_path = Path("models/EDSR_x4.pb")
    if not model_path.exists():
        print("Downloading EDSR model...")
        import urllib.request
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded!\n")
    
    # Load super-resolution model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel("edsr", scale)
    
    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {image_path.name}")
        
        # Load image
        img = cv2.imread(str(image_path))
        orig_h, orig_w = img.shape[:2]
        print(f"  Original size: {orig_w}x{orig_h}")
        
        # Upscale
        result = sr.upsample(img)
        new_h, new_w = result.shape[:2]
        print(f"  Enhanced size: {new_w}x{new_h}")
        
        # Save
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved: {output_path}\n")
    
    print(f"\n{'='*60}")
    print(f"Enhancement complete!")
    print(f"Enhanced images saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super-resolution image enhancement")
    parser.add_argument("--input", required=True, help="Input directory with images")
    parser.add_argument("--output", required=True, help="Output directory for enhanced images")
    parser.add_argument("--method", default="opencv", choices=["realesrgan", "opencv"], 
                       help="Enhancement method")
    parser.add_argument("--model", default="RealESRGAN_x2plus", 
                       choices=["RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRNet_x4plus"],
                       help="Real-ESRGAN model (only for method=realesrgan)")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4], help="Upscaling factor")
    parser.add_argument("--tile", type=int, default=400, help="Tile size for processing")
    parser.add_argument("--max-size", type=int, help="Max output dimension (to avoid huge files)")
    parser.add_argument("--limit", type=int, help="Limit number of images")
    
    args = parser.parse_args()
    
    if args.method == "realesrgan":
        if not REALESRGAN_AVAILABLE:
            print("Error: Real-ESRGAN not installed")
            print("Install with: pip install realesrgan basicsr")
            print("\nFalling back to OpenCV method...")
            args.method = "opencv"
    
    if args.method == "realesrgan":
        enhance_batch_sr(
            input_dir=args.input,
            output_dir=args.output,
            model_name=args.model,
            scale=args.scale,
            tile=args.tile,
            max_size=args.max_size,
            limit=args.limit,
        )
    else:
        enhance_with_opencv_sr(
            input_dir=args.input,
            output_dir=args.output,
            scale=args.scale,
            limit=args.limit,
        )
