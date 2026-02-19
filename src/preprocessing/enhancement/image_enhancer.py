"""
Image Enhancement for Small Object Detection

Enhances images to improve detection of small, fast-moving objects like salmon.
Techniques:
- Contrast enhancement (CLAHE)
- Sharpening
- Denoising
- Motion blur reduction
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


class ImageEnhancer:
    """Enhance images for better object detection"""
    
    def __init__(
        self,
        clahe_clip_limit=3.0,
        clahe_tile_size=8,
        sharpen_strength=1.5,
        denoise_strength=10,
    ):
        """
        Args:
            clahe_clip_limit: Contrast limiting (higher = more contrast)
            clahe_tile_size: Grid size for CLAHE
            sharpen_strength: Sharpening intensity (1.0 = mild, 2.0 = strong)
            denoise_strength: Denoising strength (higher = more smoothing)
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.sharpen_strength = sharpen_strength
        self.denoise_strength = denoise_strength
        
        # Create CLAHE object for adaptive histogram equalization
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
    
    def enhance(self, image):
        """
        Apply all enhancement techniques.
        
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
        
        # 1. Denoise to reduce noise before enhancement
        img_denoised = cv2.fastNlMeansDenoisingColored(
            img_bgr,
            None,
            h=self.denoise_strength,
            hColor=self.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # 2. Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 3. Apply CLAHE to L channel (lightness)
        l_enhanced = self.clahe.apply(l)
        
        # 4. Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        img_contrast = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 5. Sharpen to enhance edges (helpful for small objects)
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        kernel = kernel / kernel.sum() * self.sharpen_strength
        kernel[1, 1] = kernel[1, 1] + (1 - self.sharpen_strength)
        
        img_sharpened = cv2.filter2D(img_contrast, -1, kernel)
        
        # 6. Convert back to RGB for PIL
        img_rgb = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB)
        
        # Clip values to valid range
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_rgb)
    
    def enhance_for_motion(self, image):
        """
        Enhanced version specifically for motion (jumping salmon).
        Uses edge detection and temporal information.
        
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
        
        # 1. Apply bilateral filter (edge-preserving smoothing)
        img_bilateral = cv2.bilateralFilter(img_bgr, 9, 75, 75)
        
        # 2. Enhance contrast with CLAHE
        lab = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        img_contrast = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Strong sharpening for motion objects
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        img_sharpened = cv2.filter2D(img_contrast, -1, kernel)
        
        # 4. Enhance saturation to make salmon more visible
        hsv = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.3)  # Increase saturation by 30%
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv_enhanced = cv2.merge([h, s, v])
        img_final = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_rgb)


def enhance_batch(
    input_dir: str,
    output_dir: str,
    method: str = "standard",
    clahe_clip_limit: float = 3.0,
    sharpen_strength: float = 1.5,
    denoise_strength: int = 10,
    limit: int = None,
):
    """
    Enhance a batch of images.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save enhanced images
        method: "standard" or "motion"
        clahe_clip_limit: Contrast limiting
        sharpen_strength: Sharpening intensity
        denoise_strength: Denoising strength
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
    print(f"Image Enhancement for Object Detection")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Method: {method}")
    print(f"CLAHE clip limit: {clahe_clip_limit}")
    print(f"Sharpen strength: {sharpen_strength}")
    print(f"Denoise strength: {denoise_strength}")
    print(f"Images: {len(image_files)}")
    print(f"{'='*60}\n")
    
    # Create enhancer
    enhancer = ImageEnhancer(
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=8,
        sharpen_strength=sharpen_strength,
        denoise_strength=denoise_strength,
    )
    
    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {image_path.name}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Enhance
        if method == "motion":
            enhanced = enhancer.enhance_for_motion(image)
        else:
            enhanced = enhancer.enhance(image)
        
        # Save with same filename
        output_path = output_dir / image_path.name
        enhanced.save(output_path, quality=95)
        print(f"  Saved: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Enhancement complete! Enhanced images saved to:")
    print(f"  {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance images for better object detection")
    parser.add_argument("--input", required=True, help="Input directory with images")
    parser.add_argument("--output", required=True, help="Output directory for enhanced images")
    parser.add_argument("--method", default="standard", choices=["standard", "motion"], 
                       help="Enhancement method: standard or motion (for jumping objects)")
    parser.add_argument("--clahe-clip", type=float, default=3.0, help="CLAHE contrast limit")
    parser.add_argument("--sharpen", type=float, default=1.5, help="Sharpening strength")
    parser.add_argument("--denoise", type=int, default=10, help="Denoising strength")
    parser.add_argument("--limit", type=int, help="Limit number of images")
    
    args = parser.parse_args()
    
    enhance_batch(
        input_dir=args.input,
        output_dir=args.output,
        method=args.method,
        clahe_clip_limit=args.clahe_clip,
        sharpen_strength=args.sharpen,
        denoise_strength=args.denoise,
        limit=args.limit,
    )
