#!/usr/bin/env python3
"""
清理孤立的标签文件 - 删除那些没有对应图像的标签

用法：
    python cleanup_orphaned_labels.py --images <图像目录> --labels <标签目录> [--dry-run]

例如：
    # 预览（不实际删除）
    python cleanup_orphaned_labels.py \
        --images data/frames/salmon_validation/ \
        --labels data/auto_labels/salmon_validation/ \
        --dry-run

    # 实际执行删除
    python cleanup_orphaned_labels.py \
        --images data/frames/salmon_validation/ \
        --labels data/auto_labels/salmon_validation/
"""

import os
import argparse
from pathlib import Path
from typing import Set, List


def get_image_stems(image_dir: Path) -> Set[str]:
    """
    获取所有图像文件的stem（不含扩展名的文件名）
    
    Args:
        image_dir: 图像目录路径
    
    Returns:
        所有图像文件的stem集合
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    stems = set()
    
    if image_dir.is_dir():
        for img_path in image_dir.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                stems.add(img_path.stem)
    else:
        for img_path in image_dir.parent.rglob('*'):
            if img_path.suffix.lower() in image_extensions and img_path.is_relative_to(image_dir.parent):
                stems.add(img_path.stem)
    
    return stems


def find_orphaned_labels(image_dir: Path, label_dir: Path) -> List[Path]:
    """
    查找孤立的标签文件（没有对应图像的标签）
    
    Args:
        image_dir: 图像目录
        label_dir: 标签目录
    
    Returns:
        孤立标签文件的路径列表
    """
    print(f"\n[1/3] 扫描图像目录: {image_dir}")
    image_stems = get_image_stems(image_dir)
    print(f"  找到 {len(image_stems)} 个图像文件")
    
    print(f"\n[2/3] 扫描标签目录: {label_dir}")
    orphaned_labels = []
    total_labels = 0
    
    for label_path in label_dir.rglob('*.txt'):
        total_labels += 1
        label_stem = label_path.stem
        
        # 检查是否有对应的图像
        if label_stem not in image_stems:
            orphaned_labels.append(label_path)
    
    print(f"  找到 {total_labels} 个标签文件")
    print(f"  其中 {len(orphaned_labels)} 个是孤立标签（无对应图像）")
    
    return orphaned_labels


def delete_orphaned_labels(orphaned_labels: List[Path], dry_run: bool = True):
    """
    删除孤立的标签文件
    
    Args:
        orphaned_labels: 孤立标签文件列表
        dry_run: 如果为True，只预览不实际删除
    """
    if not orphaned_labels:
        print("\n✅ 没有孤立的标签文件，数据集已经同步！")
        return
    
    print(f"\n[3/3] {'预览' if dry_run else '删除'}孤立标签:")
    
    if dry_run:
        print("\n  以下文件将被删除（当前为预览模式）：")
        for label_path in orphaned_labels[:20]:  # 只显示前20个
            print(f"    - {label_path.name}")
        
        if len(orphaned_labels) > 20:
            print(f"    ... 还有 {len(orphaned_labels) - 20} 个文件")
        
        print(f"\n  ⚠️  预览模式：没有实际删除任何文件")
        print(f"  💡 要执行删除，请移除 --dry-run 参数")
    else:
        deleted_count = 0
        failed_count = 0
        
        for label_path in orphaned_labels:
            try:
                label_path.unlink()
                deleted_count += 1
                if deleted_count <= 10 or deleted_count % 50 == 0:
                    print(f"  ✓ 已删除: {label_path.name}")
            except Exception as e:
                failed_count += 1
                print(f"  ✗ 删除失败: {label_path.name} - {e}")
        
        print(f"\n✅ 完成：")
        print(f"  成功删除: {deleted_count} 个文件")
        if failed_count > 0:
            print(f"  失败: {failed_count} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description='清理孤立的标签文件（删除那些没有对应图像的标签）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览将要删除的文件（推荐先运行）
  python cleanup_orphaned_labels.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --dry-run

  # 实际执行删除
  python cleanup_orphaned_labels.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/
        """
    )
    
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='图像目录路径（包含你审核后保留的图像）'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='标签目录路径（YOLO格式txt文件）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式：只显示将要删除的文件，不实际删除'
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    image_dir = Path(args.images).resolve()
    label_dir = Path(args.labels).resolve()
    
    # 验证目录存在
    if not image_dir.exists():
        print(f"❌ 错误：图像目录不存在: {image_dir}")
        return 1
    
    if not label_dir.exists():
        print(f"❌ 错误：标签目录不存在: {label_dir}")
        return 1
    
    print("=" * 70)
    print("🗑️  清理孤立标签工具")
    print("=" * 70)
    print(f"\n图像目录: {image_dir}")
    print(f"标签目录: {label_dir}")
    print(f"模式: {'预览（不删除）' if args.dry_run else '实际删除'}")
    
    # 查找孤立标签
    orphaned_labels = find_orphaned_labels(image_dir, label_dir)
    
    # 删除孤立标签
    delete_orphaned_labels(orphaned_labels, dry_run=args.dry_run)
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())
