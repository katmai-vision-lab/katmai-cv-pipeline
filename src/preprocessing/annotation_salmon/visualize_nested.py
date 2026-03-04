#!/usr/bin/env python3
"""
可视化嵌套目录中的YOLO标注

处理图像在子文件夹中，标签在单一目录中的情况
"""

import argparse
import cv2
from pathlib import Path
import random


def draw_yolo_boxes(image_path: Path, label_path: Path, output_path: Path, class_names: list = None):
    """在图像上绘制YOLO格式的边界框"""
    if class_names is None:
        class_names = ["salmon"]

    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return False

    h, w = img.shape[:2]

    # 读取标签
    if not label_path.exists():
        print(f"⚠️  标签文件不存在: {label_path.name}")
        return False
    
    # 检查标签是否为空
    if label_path.stat().st_size == 0:
        print(f"⚠️  空标签（无检测）: {image_path.name}")
        return False

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return False

    # 绘制每个检测框
    box_count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        # 转换归一化坐标到像素坐标
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # 绘制标签
        label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # 添加背景使文字更清晰
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # 绘制文字背景
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        # 绘制文字
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        box_count += 1

    # 保存图像
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"✓ {image_path.name} ({box_count} 检测)")

    return True


def visualize_nested_directory(images_dir: Path, labels_dir: Path, output_dir: Path, limit: int = None):
    """
    可视化嵌套目录中的标注
    
    Args:
        images_dir: 图像根目录（可能包含子文件夹）
        labels_dir: 标签目录（所有txt在同一层）
        output_dir: 输出目录
        limit: 最多处理多少张（None=全部）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 递归查找所有图像
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(images_dir.rglob(f"*{ext}"))
    
    image_files = sorted(image_files)
    
    print(f"\n找到 {len(image_files)} 张图像")
    
    # 过滤：只保留有对应标签的
    images_with_labels = []
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists() and label_path.stat().st_size > 0:
            images_with_labels.append(img_path)
    
    print(f"其中 {len(images_with_labels)} 张有标注（非空）")
    
    # 如果指定了limit，随机采样
    if limit and len(images_with_labels) > limit:
        images_with_labels = random.sample(images_with_labels, limit)
        print(f"随机采样 {limit} 张进行可视化")
    
    print(f"\n开始可视化...")
    
    visualized_count = 0
    for img_path in images_with_labels:
        label_path = labels_dir / (img_path.stem + ".txt")
        output_path = output_dir / img_path.name
        
        if draw_yolo_boxes(img_path, label_path, output_path):
            visualized_count += 1
    
    print(f"\n✅ 完成！成功可视化 {visualized_count} 张图像")
    print(f"   输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化嵌套目录中的YOLO标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化前50张有标注的图像
  python visualize_nested.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --output data/visualized/salmon_validation/ \\
      --limit 50
  
  # 可视化所有
  python visualize_nested.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --output data/visualized/salmon_validation/
        """
    )
    
    parser.add_argument("--images", required=True, help="图像根目录（支持嵌套子文件夹）")
    parser.add_argument("--labels", required=True, help="标签目录（YOLO格式txt文件）")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--limit", type=int, help="最多可视化多少张（随机采样）")
    
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
    print("🎨 YOLO标注可视化工具")
    print("=" * 70)
    
    visualize_nested_directory(images_dir, labels_dir, output_dir, args.limit)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    exit(main())
