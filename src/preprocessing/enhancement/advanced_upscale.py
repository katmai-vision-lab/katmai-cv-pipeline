"""
Advanced Image Upscaling for Small Object Detection

Uses high-quality interpolation + sharpening + denoising to improve image quality
for better small object detection (like salmon in video).

No external model dependencies - uses PIL and OpenCV.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def advanced_upscale(image, scale=2, sharpen=True, denoise=True):
    """
    Advanced upscaling using Lanczos interpolation + post-processing.
    
    Args:
        image: PIL Image
        scale: Upscaling factor (2, 3, or 4)
        sharpen: Apply sharpening after upscale
        denoise: Apply denoising before upscale
        
    Returns:
        Upscaled PIL Image
    """
    # Convert to numpy for OpenCV processing
    img_array = np.array(image)
    
    # 1. Denoise first (reduces noise amplification during upscaling)
    if denoise:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_denoised = cv2.fastNlMeansDenoisingColored(
            img_bgr, None, h=10, hColor=10,
            templateWindowSize=7, searchWindowSize=21
        )
        img_array = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_array)
    
    # 2. Upscale using Lanczos (best quality interpolation)
    new_size = (image.width * scale, image.height * scale)
    upscaled = image.resize(new_size, Image.LANCZOS)
    
    # 3. Post-processing to enhance details
    if sharpen:
        # Apply unsharp mask for natural sharpening
        upscaled_array = np.array(upscaled)
        upscaled_bgr = cv2.cvtColor(upscaled_array, cv2.COLOR_RGB2BGR)
        
        # Gaussian blur
        gaussian = cv2.GaussianBlur(upscaled_bgr, (0, 0), 2.0)
        
        # Unsharp mask: original + (original - blurred) * amount
        unsharp_mask = cv2.addWeighted(upscaled_bgr, 1.5, gaussian, -0.5, 0)
        
        upscaled_rgb = cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2RGB)
        upscaled = Image.fromarray(np.clip(upscaled_rgb, 0, 255).astype(np.uint8))
    
    # 4. Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(upscaled)
    upscaled = enhancer.enhance(1.1)
    
    # 5. Enhance sharpness
    enhancer = ImageEnhance.Sharpness(upscaled)
    upscaled = enhancer.enhance(1.2)
    
    return upscaled


def multi_scale_detection_prep(image, scales=[1.0, 1.5, 2.0]):
    """
    Create multiple scales of an image for multi-scale object detection.
    Helps detect objects at different sizes.
    
    Args:
        image: PIL Image
        scales: List of scale factors
        
    Returns:
        List of (scale, image) tuples
    """
    results = []
    
    for scale in scales:
        if scale == 1.0:
            results.append((scale, image))
        else:
            new_size = (int(image.width * scale), int(image.height * scale))
            scaled = image.resize(new_size, Image.LANCZOS)
            results.append((scale, scaled))
    
    return results


def enhance_batch_advanced(
    input_dir: str,
    output_dir: str,
    scale: int = 2,
    sharpen: bool = True,
    denoise: bool = True,
    limit: int = None,
):
    """
    Enhance a batch of images using advanced upscaling.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save enhanced images
        scale: Upscaling factor (2, 3, or 4)
        sharpen: Apply sharpening
        denoise: Apply denoising
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
    print(f"Advanced Image Upscaling")
    print(f"{'='*60}")
    print(f"Method: Lanczos + Unsharp Mask + Contrast Enhancement")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Scale: {scale}x")
    print(f"Sharpen: {sharpen}")
    print(f"Denoise: {denoise}")
    print(f"Images: {len(image_files)}")
    print(f"{'='*60}\n")
    
    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {image_path.name}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size
        print(f"  Original: {orig_size[0]}x{orig_size[1]}")
        
        # Upscale
        enhanced = advanced_upscale(image, scale=scale, sharpen=sharpen, denoise=denoise)
        new_size = enhanced.size
        print(f"  Enhanced: {new_size[0]}x{new_size[1]}")
        
        # Save
        output_path = output_dir / image_path.name
        enhanced.save(output_path, quality=95)
        print(f"  Saved: {output_path}\n")
    
    print(f"\n{'='*60}")
    print(f"Enhancement complete!")
    print(f"Enhanced images saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced image upscaling")
    parser.add_argument("--input", required=True, help="Input directory with images")
    parser.add_argument("--output", required=True, help="Output directory for enhanced images")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="Upscaling factor")
    parser.add_argument("--no-sharpen", action="store_true", help="Disable sharpening")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    parser.add_argument("--limit", type=int, help="Limit number of images")
    
    args = parser.parse_args()
    
    enhance_batch_advanced(
        input_dir=args.input,
        output_dir=args.output,
        scale=args.scale,
        sharpen=not args.no_sharpen,
        denoise=not args.no_denoise,
        limit=args.limit,
    )
