"""
Two-Stage Detection for Small Objects (Salmon in Bear Video)

Stage 1: Detect large objects (bears) at full resolution
Stage 2: Crop regions around bears and detect small objects (salmon) at higher resolution

This approach improves small object detection by:
- Focusing computational resources on relevant regions
- Increasing effective resolution for small targets
- Reducing false positives from background
"""

import argparse
from pathlib import Path
import sys
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.auto_annotator_gdino import GroundingDINOAnnotator


def expand_box(box, expansion_factor, image_width, image_height):
    """
    Expand a bounding box by a factor, staying within image bounds.
    
    Args:
        box: [x_min, y_min, x_max, y_max] in pixels
        expansion_factor: How much to expand (1.5 = 50% expansion)
        image_width, image_height: Image dimensions
    
    Returns:
        Expanded box [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = box
    
    # Calculate current center and size
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Expand
    new_width = width * expansion_factor
    new_height = height * expansion_factor
    
    # New box
    new_x_min = max(0, center_x - new_width / 2)
    new_y_min = max(0, center_y - new_height / 2)
    new_x_max = min(image_width, center_x + new_width / 2)
    new_y_max = min(image_height, center_y + new_height / 2)
    
    return [new_x_min, new_y_min, new_x_max, new_y_max]


def two_stage_detect(
    image_path: Path,
    output_dir: Path,
    stage1_prompt: str = "brown bear",
    stage2_prompt: str = "salmon. fish",
    stage1_box_threshold: float = 0.30,
    stage1_text_threshold: float = 0.25,
    stage2_box_threshold: float = 0.20,  # Lower for small objects
    stage2_text_threshold: float = 0.20,
    expansion_factor: float = 1.5,  # Expand bear regions by 50%
    min_crop_size: int = 400,  # Minimum crop size in pixels
    model_size: str = "base",
):
    """
    Two-stage detection: First detect bears, then search for salmon around them.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save labels
        stage1_prompt: Text prompt for large objects (bears)
        stage2_prompt: Text prompt for small objects (salmon)
        stage1_box_threshold: Confidence threshold for stage 1
        stage1_text_threshold: Text matching threshold for stage 1
        stage2_box_threshold: Confidence threshold for stage 2 (lower for small objects)
        stage2_text_threshold: Text matching threshold for stage 2
        expansion_factor: How much to expand bear regions (1.5 = search 50% beyond bear box)
        min_crop_size: Minimum crop size to ensure enough context
        model_size: "tiny", "base", or "large"
    
    Returns:
        Dict with detection counts
    """
    # Initialize model
    model_id = f"IDEA-Research/grounding-dino-{model_size}"
    annotator = GroundingDINOAnnotator(model_id=model_id)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    all_detections = []
    
    # STAGE 1: Detect bears at full resolution
    print(f"  Stage 1: Detecting bears...")
    bear_detections = annotator.detect(
        image=image,
        text_prompt=stage1_prompt,
        box_threshold=stage1_box_threshold,
        text_threshold=stage1_text_threshold,
    )
    
    print(f"    Found {len(bear_detections)} bear(s)")
    
    # Save bear detections as class 0
    for det in bear_detections:
        det['class_id'] = 0
        det['class_name'] = 'bear'
        all_detections.append(det)
    
    # STAGE 2: Search for salmon around each bear
    salmon_count = 0
    
    if len(bear_detections) > 0:
        print(f"  Stage 2: Searching for salmon around bears...")
        
        for i, bear_det in enumerate(bear_detections):
            bear_box = bear_det['box']
            
            # Expand bear region
            search_region = expand_box(bear_box, expansion_factor, width, height)
            
            # Ensure minimum crop size
            crop_width = search_region[2] - search_region[0]
            crop_height = search_region[3] - search_region[1]
            
            if crop_width < min_crop_size or crop_height < min_crop_size:
                # Expand to minimum size
                center_x = (search_region[0] + search_region[2]) / 2
                center_y = (search_region[1] + search_region[3]) / 2
                half_size = max(min_crop_size / 2, crop_width / 2, crop_height / 2)
                
                search_region = [
                    max(0, center_x - half_size),
                    max(0, center_y - half_size),
                    min(width, center_x + half_size),
                    min(height, center_y + half_size),
                ]
            
            # Crop image
            crop_box = [int(x) for x in search_region]
            cropped_image = image.crop(crop_box)
            
            # Detect salmon in cropped region with LOWER threshold
            salmon_detections = annotator.detect(
                image=cropped_image,
                text_prompt=stage2_prompt,
                box_threshold=stage2_box_threshold,
                text_threshold=stage2_text_threshold,
            )
            
            # Convert coordinates back to full image space
            for det in salmon_detections:
                crop_box_local = det['box']
                
                # Transform to full image coordinates
                det['box'] = [
                    crop_box_local[0] + crop_box[0],
                    crop_box_local[1] + crop_box[1],
                    crop_box_local[2] + crop_box[0],
                    crop_box_local[3] + crop_box[1],
                ]
                det['class_id'] = 1
                det['class_name'] = 'salmon'
                all_detections.append(det)
                salmon_count += 1
            
            if len(salmon_detections) > 0:
                print(f"    Bear {i+1}: Found {len(salmon_detections)} salmon")
    else:
        print(f"  Stage 2: No bears found, skipping salmon search")
    
    # Convert to YOLO format and save
    output_dir.mkdir(parents=True, exist_ok=True)
    label_filename = image_path.stem + ".txt"
    label_path = output_dir / label_filename
    
    yolo_labels = []
    for det in all_detections:
        box = det['box']
        class_id = det['class_id']
        
        # Convert to YOLO format
        center_x = ((box[0] + box[2]) / 2) / width
        center_y = ((box[1] + box[3]) / 2) / height
        box_width = (box[2] - box[0]) / width
        box_height = (box[3] - box[1]) / height
        
        # Clamp to [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        box_width = max(0, min(1, box_width))
        box_height = max(0, min(1, box_height))
        
        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
    
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_labels))
    
    return {
        'bears': len(bear_detections),
        'salmon': salmon_count,
        'total': len(all_detections)
    }


def batch_two_stage_detect(
    input_dir: str,
    output_dir: str,
    stage1_prompt: str = "brown bear",
    stage2_prompt: str = "salmon. fish",
    stage1_box_threshold: float = 0.30,
    stage1_text_threshold: float = 0.25,
    stage2_box_threshold: float = 0.20,
    stage2_text_threshold: float = 0.20,
    expansion_factor: float = 1.5,
    min_crop_size: int = 400,
    limit: int = None,
    model_size: str = "base",
):
    """
    Run two-stage detection on a batch of images.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
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
    print(f"Two-Stage Detection for Small Objects")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(image_files)}")
    print(f"\nStage 1 (Bears):")
    print(f"  Prompt: {stage1_prompt}")
    print(f"  Box threshold: {stage1_box_threshold}")
    print(f"  Text threshold: {stage1_text_threshold}")
    print(f"\nStage 2 (Salmon):")
    print(f"  Prompt: {stage2_prompt}")
    print(f"  Box threshold: {stage2_box_threshold} (lower for small objects)")
    print(f"  Text threshold: {stage2_text_threshold}")
    print(f"  Expansion factor: {expansion_factor}x")
    print(f"  Min crop size: {min_crop_size}px")
    print(f"{'='*60}\n")
    
    total_bears = 0
    total_salmon = 0
    images_with_salmon = 0
    
    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {image_path.name}")
        
        result = two_stage_detect(
            image_path=image_path,
            output_dir=output_dir,
            stage1_prompt=stage1_prompt,
            stage2_prompt=stage2_prompt,
            stage1_box_threshold=stage1_box_threshold,
            stage1_text_threshold=stage1_text_threshold,
            stage2_box_threshold=stage2_box_threshold,
            stage2_text_threshold=stage2_text_threshold,
            expansion_factor=expansion_factor,
            min_crop_size=min_crop_size,
            model_size=model_size,
        )
        
        total_bears += result['bears']
        total_salmon += result['salmon']
        if result['salmon'] > 0:
            images_with_salmon += 1
        
        print(f"  Result: {result['bears']} bears, {result['salmon']} salmon\n")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Total bears: {total_bears}")
    print(f"  Total salmon: {total_salmon}")
    print(f"  Images with salmon: {images_with_salmon}")
    print(f"  Avg salmon per image: {total_salmon/len(image_files):.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage detection for small objects")
    parser.add_argument("--input", required=True, help="Input directory with images")
    parser.add_argument("--output", required=True, help="Output directory for labels")
    parser.add_argument("--stage1-prompt", default="brown bear", help="Stage 1 prompt (large objects)")
    parser.add_argument("--stage2-prompt", default="salmon. fish", help="Stage 2 prompt (small objects)")
    parser.add_argument("--stage1-box-threshold", type=float, default=0.30)
    parser.add_argument("--stage1-text-threshold", type=float, default=0.25)
    parser.add_argument("--stage2-box-threshold", type=float, default=0.20, help="Lower threshold for small objects")
    parser.add_argument("--stage2-text-threshold", type=float, default=0.20)
    parser.add_argument("--expansion-factor", type=float, default=1.5, help="Expand search regions by this factor")
    parser.add_argument("--min-crop-size", type=int, default=400, help="Minimum crop size in pixels")
    parser.add_argument("--limit", type=int, help="Limit number of images to process")
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "large"])
    
    args = parser.parse_args()
    
    batch_two_stage_detect(
        input_dir=args.input,
        output_dir=args.output,
        stage1_prompt=args.stage1_prompt,
        stage2_prompt=args.stage2_prompt,
        stage1_box_threshold=args.stage1_box_threshold,
        stage1_text_threshold=args.stage1_text_threshold,
        stage2_box_threshold=args.stage2_box_threshold,
        stage2_text_threshold=args.stage2_text_threshold,
        expansion_factor=args.expansion_factor,
        min_crop_size=args.min_crop_size,
        limit=args.limit,
        model_size=args.model_size,
    )
