"""
Split annotated dataset into train/val sets
Usage:
    python -m src.preprocessing.split_dataset --input data/annotation/bears --train-ratio 0.8
"""

import argparse
import shutil
from pathlib import Path
import random

def split_dataset(input_dir, train_ratio=0.8, val_ratio=0.2, seed=42):
    """Split dataset into train/val sets"""
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    images_dir = input_dir / 'images'
    labels_dir = input_dir / 'labels'
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Recursively get all image files (including subdirectories)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(images_dir.rglob(ext))  # rglob instead of glob
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    print(f"Found {len(image_files)} images (including subdirectories)")
    
    # Check for corresponding labels
    valid_files = []
    for img_file in image_files:
        # Get relative path from images_dir
        rel_path = img_file.relative_to(images_dir)
        
        # Try strategy 1: Same structure (labels/folder/subfolder/file.txt)
        label_file = labels_dir / rel_path.parent / f"{img_file.stem}.txt"
        
        if label_file.exists():
            valid_files.append((img_file, label_file))
        else:
            # Try strategy 2: Flattened structure with parent folder name in filename
            # e.g., images/cat1/video1/frame001.jpg -> labels/cat1/video1_frame001.txt
            if rel_path.parent != Path('.'):
                parts = list(rel_path.parent.parts)
                if len(parts) >= 2:
                    # Try: labels/cat1/video1_frame001.txt
                    flattened_name = f"{parts[-1]}_{img_file.stem}.txt"
                    label_file = labels_dir / parts[0] / flattened_name
                    
                    if label_file.exists():
                        valid_files.append((img_file, label_file))
                    else:
                        print(f"⚠️  Warning: No label file for {rel_path}")
                else:
                    print(f"⚠️  Warning: No label file for {rel_path}")
            else:
                print(f"⚠️  Warning: No label file for {rel_path}")
    
    print(f"Found {len(valid_files)} image-label pairs")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(valid_files)
    
    train_count = int(len(valid_files) * train_ratio)
    train_files = valid_files[:train_count]
    val_files = valid_files[train_count:]
    
    print(f"\nSplit:")
    print(f"  Training: {len(train_files)} samples ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_files)} samples ({val_ratio*100:.0f}%)")
    
    # Create base directories
    train_img_dir = input_dir / 'images' / 'train'
    train_lbl_dir = input_dir / 'labels' / 'train'
    val_img_dir = input_dir / 'images' / 'val'
    val_lbl_dir = input_dir / 'labels' / 'val'
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Move files (flatten structure - all files go directly into train/val)
    print("\nMoving files...")
    for img_file, label_file in train_files:
        # Move to train (flatten - use only filename)
        shutil.move(str(img_file), str(train_img_dir / img_file.name))
        shutil.move(str(label_file), str(train_lbl_dir / label_file.name))
    
    for img_file, label_file in val_files:
        # Move to val (flatten - use only filename)
        shutil.move(str(img_file), str(val_img_dir / img_file.name))
        shutil.move(str(label_file), str(val_lbl_dir / label_file.name))
    
    print(f"\n✓ Dataset split complete!")
    print(f"  Train images: {train_img_dir}")
    print(f"  Train labels: {train_lbl_dir}")
    print(f"  Val images: {val_img_dir}")
    print(f"  Val labels: {val_lbl_dir}")
    print(f"\n📝 Note: All files have been flattened into train/val folders")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train/val sets')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing images/ and labels/')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.train_ratio + args.val_ratio != 1.0:
        print(f"⚠️  Adjusting val_ratio to {1.0 - args.train_ratio}")
        args.val_ratio = 1.0 - args.train_ratio
    
    split_dataset(args.input, args.train_ratio, args.val_ratio, args.seed)
