#!/usr/bin/env python3
"""
根据标记列表删除原始图像

用法：
    python delete_marked_images.py --list marked_for_deletion.txt \\
        --frames data/frames/salmon_validation/ \\
        --dry-run  # 预览模式

    # 实际删除
    python delete_marked_images.py --list marked_for_deletion.txt \\
        --frames data/frames/salmon_validation/
"""

import argparse
from pathlib import Path


def delete_marked_images(marked_file: Path, frames_dir: Path, dry_run: bool = True):
    """
    根据标记列表删除原始图像
    
    Args:
        marked_file: 标记文件列表（每行一个文件名stem）
        frames_dir: 原始图像目录（递归搜索）
        dry_run: 预览模式
    """
    # 读取标记列表
    with open(marked_file, 'r') as f:
        marked_stems = {line.strip() for line in f if line.strip()}
    
    print(f"\n读取标记列表: {len(marked_stems)} 个文件")
    
    # 查找原始图像
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    found_images = []
    
    for ext in image_extensions:
        for img_path in frames_dir.rglob(f"*{ext}"):
            if img_path.stem in marked_stems:
                found_images.append(img_path)
    
    print(f"找到匹配图像: {len(found_images)} 个")
    
    if not found_images:
        print("\n⚠️  没有找到匹配的图像文件")
        return 0
    
    # 删除或预览
    if dry_run:
        print(f"\n📋 预览模式 - 以下文件将被删除:")
        for img_path in sorted(found_images)[:20]:
            print(f"  - {img_path.relative_to(frames_dir.parent)}")
        if len(found_images) > 20:
            print(f"  ... 还有 {len(found_images) - 20} 个文件")
        print(f"\n⚠️  要实际删除，请移除 --dry-run 参数")
        return len(found_images)
    else:
        print(f"\n🗑️  开始删除...")
        deleted_count = 0
        failed_count = 0
        
        for img_path in found_images:
            try:
                img_path.unlink()
                deleted_count += 1
                if deleted_count <= 10 or deleted_count % 50 == 0:
                    print(f"  ✓ 已删除: {img_path.name}")
            except Exception as e:
                failed_count += 1
                print(f"  ✗ 删除失败: {img_path.name} - {e}")
        
        print(f"\n✅ 完成:")
        print(f"  成功删除: {deleted_count} 个图像")
        if failed_count > 0:
            print(f"  失败: {failed_count} 个")
        
        return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description="根据标记列表删除原始图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览
  python delete_marked_images.py \\
      --list marked_for_deletion.txt \\
      --frames data/frames/salmon_validation/ \\
      --dry-run
  
  # 实际删除
  python delete_marked_images.py \\
      --list marked_for_deletion.txt \\
      --frames data/frames/salmon_validation/
        """
    )
    
    parser.add_argument("--list", required=True, help="标记列表文件")
    parser.add_argument("--frames", required=True, help="原始图像目录")
    parser.add_argument("--dry-run", action="store_true", help="预览模式（不实际删除）")
    
    args = parser.parse_args()
    
    marked_file = Path(args.list)
    frames_dir = Path(args.frames)
    
    if not marked_file.exists():
        print(f"❌ 标记列表不存在: {marked_file}")
        return 1
    
    if not frames_dir.exists():
        print(f"❌ 图像目录不存在: {frames_dir}")
        return 1
    
    print("=" * 70)
    print("🗑️  批量删除标记的图像")
    print("=" * 70)
    print(f"标记列表: {marked_file}")
    print(f"图像目录: {frames_dir}")
    print(f"模式: {'预览（不删除）' if args.dry_run else '实际删除'}")
    
    deleted = delete_marked_images(marked_file, frames_dir, args.dry_run)
    
    if not args.dry_run and deleted > 0:
        print(f"\n💡 接下来运行清理脚本删除对应标签:")
        print(f"  python -m src.preprocessing.annotation_salmon.cleanup_orphaned_labels \\")
        print(f"      --images data/frames/salmon_validation/ \\")
        print(f"      --labels data/auto_labels/salmon_validation/")
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
