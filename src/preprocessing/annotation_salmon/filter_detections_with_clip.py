#!/usr/bin/env python3
"""
使用CLIP过滤检测结果 - 移除特定类别（如熊）

原理：
    1. 读取YOLO标签文件
    2. 裁剪每个检测框
    3. 用CLIP判断是目标类（salmon）还是排除类（bear）
    4. 只保留目标类的检测

用法：
    python filter_detections_with_clip.py \\
        --images data/frames/salmon_validation/ \\
        --labels data/auto_labels/salmon_validation/ \\
        --output data/auto_labels/salmon_filtered/ \\
        --target-class "salmon fish" \\
        --exclude-classes "bear" "rock" "water splash" \\
        --threshold 0.6
"""

import argparse
import cv2
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_clip_model():
    """加载CLIP模型"""
    try:
        from transformers import CLIPProcessor, CLIPModel
        
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        return model, processor, device
    except ImportError:
        print("❌ 需要安装transformers: pip install transformers")
        exit(1)


def classify_detection(image_crop, model, processor, device, target_class, exclude_classes):
    """
    使用CLIP分类检测框
    
    Returns:
        (is_target, confidence, predicted_class)
    """
    # 准备文本
    all_classes = [target_class] + exclude_classes
    text_inputs = [f"a photo of {cls}" for cls in all_classes]
    
    # 转换为PIL
    if isinstance(image_crop, np.ndarray):
        image_crop = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    
    # CLIP推理
    inputs = processor(
        text=text_inputs,
        images=image_crop,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
    
    # 判断
    target_prob = probs[0].item()  # 第一个是target_class
    max_exclude_prob = probs[1:].max().item() if len(exclude_classes) > 0 else 0.0
    
    predicted_idx = probs.argmax().item()
    predicted_class = all_classes[predicted_idx]
    
    is_target = predicted_idx == 0
    confidence = target_prob
    
    return is_target, confidence, predicted_class


def filter_labels_file(image_path: Path, label_path: Path, output_path: Path,
                       model, processor, device, target_class, exclude_classes, threshold):
    """
    过滤单个标签文件
    
    Args:
        threshold: target_class的置信度阈值（高于此值才保留）
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        return 0, 0
    
    h, w = img.shape[:2]
    
    # 读取标签
    if not label_path.exists() or label_path.stat().st_size == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return 0, 0
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    kept_lines = []
    total_detections = len(lines)
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        
        # 转换为像素坐标
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        
        # 裁剪检测框
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
        
        # CLIP分类
        is_target, confidence, predicted_class = classify_detection(
            crop, model, processor, device, target_class, exclude_classes
        )
        
        # 判断是否保留
        if is_target and confidence >= threshold:
            kept_lines.append(line)
        # else:
        #     print(f"  过滤: {predicted_class} (conf={confidence:.2f})")
    
    # 保存过滤后的标签
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(kept_lines)
    
    return total_detections, len(kept_lines)


def filter_dataset(images_dir: Path, labels_dir: Path, output_dir: Path,
                   target_class: str, exclude_classes: list, threshold: float):
    """过滤整个数据集"""
    
    print("\n[1/3] 加载CLIP模型...")
    model, processor, device = load_clip_model()
    print(f"  设备: {device}")
    
    print(f"\n[2/3] 扫描标签文件...")
    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"  找到 {len(label_files)} 个标签文件")
    
    print(f"\n[3/3] 过滤检测...")
    print(f"  目标类别: '{target_class}'")
    print(f"  排除类别: {exclude_classes}")
    print(f"  置信度阈值: {threshold}")
    print()
    
    total_original = 0
    total_kept = 0
    non_empty_files = 0
    
    for label_path in tqdm(label_files, desc="处理"):
        # 查找对应的图像
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_path = None
        
        for ext in image_extensions:
            for img_path in images_dir.rglob(f"{label_path.stem}{ext}"):
                image_path = img_path
                break
            if image_path:
                break
        
        if not image_path or not image_path.exists():
            continue
        
        output_path = output_dir / label_path.name
        
        original, kept = filter_labels_file(
            image_path, label_path, output_path,
            model, processor, device, target_class, exclude_classes, threshold
        )
        
        total_original += original
        total_kept += kept
        
        if kept > 0:
            non_empty_files += 1
    
    print(f"\n✅ 完成！")
    print(f"  原始检测数: {total_original}")
    print(f"  保留检测数: {total_kept} ({100*total_kept/max(1,total_original):.1f}%)")
    print(f"  过滤掉: {total_original - total_kept} ({100*(total_original-total_kept)/max(1,total_original):.1f}%)")
    print(f"  非空文件数: {non_empty_files}")
    print(f"\n  输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="使用CLIP过滤检测结果，移除特定类别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 过滤掉熊，只保留三文鱼
  python filter_detections_with_clip.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --output data/auto_labels/salmon_filtered/ \\
      --target-class "salmon fish" \\
      --exclude-classes "bear" "rock" "water" \\
      --threshold 0.6
  
  # 更严格的阈值
  python filter_detections_with_clip.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --output data/auto_labels/salmon_filtered/ \\
      --target-class "jumping salmon" \\
      --exclude-classes "bear" "brown bear" "grizzly bear" \\
      --threshold 0.7
        """
    )
    
    parser.add_argument("--images", required=True, help="图像目录（支持嵌套）")
    parser.add_argument("--labels", required=True, help="标签目录（YOLO格式）")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--target-class", required=True, help="目标类别（如'salmon fish'）")
    parser.add_argument("--exclude-classes", nargs="+", default=["bear"], 
                       help="要排除的类别列表（默认: bear）")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="目标类别的置信度阈值（默认: 0.6）")
    
    args = parser.parse_args()
    
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)
    
    if not images_dir.exists():
        print(f"❌ 图像目录不存在: {images_dir}")
        return 1
    
    if not labels_dir.exists():
        print(f"❌ 标签目录不存在: {labels_dir}")
        return 1
    
    print("=" * 70)
    print("🔍 CLIP检测过滤器")
    print("=" * 70)
    
    filter_dataset(
        images_dir, labels_dir, output_dir,
        args.target_class, args.exclude_classes, args.threshold
    )
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
