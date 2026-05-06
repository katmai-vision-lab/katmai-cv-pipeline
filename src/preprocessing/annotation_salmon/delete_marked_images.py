#!/usr/bin/env python3
"""
Delete original images according to a marker list.

Usage:
    python delete_marked_images.py --list marked_for_deletion.txt \\
        --frames data/frames/salmon_validation/ \\
        --dry-run  # preview mode

    # Actually delete
    python delete_marked_images.py --list marked_for_deletion.txt \\
        --frames data/frames/salmon_validation/
"""

import argparse
from pathlib import Path


def delete_marked_images(marked_file: Path, frames_dir: Path, dry_run: bool = True):
    """
    Delete original images according to the marker list.

    Args:
        marked_file: marker file (one filename stem per line)
        frames_dir: original image directory (recursively searched)
        dry_run: preview mode
    """
    # Load the marker list
    with open(marked_file, 'r') as f:
        marked_stems = {line.strip() for line in f if line.strip()}

    print(f"\nLoaded marker list: {len(marked_stems)} files")

    # Find matching original images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    found_images = []

    for ext in image_extensions:
        for img_path in frames_dir.rglob(f"*{ext}"):
            if img_path.stem in marked_stems:
                found_images.append(img_path)

    print(f"Matching images found: {len(found_images)}")

    if not found_images:
        print("\n⚠️  No matching image files found")
        return 0

    # Delete or preview
    if dry_run:
        print(f"\n📋 Preview mode — the following files would be deleted:")
        for img_path in sorted(found_images)[:20]:
            print(f"  - {img_path.relative_to(frames_dir.parent)}")
        if len(found_images) > 20:
            print(f"  ... and {len(found_images) - 20} more")
        print(f"\n⚠️  To actually delete, drop the --dry-run flag")
        return len(found_images)
    else:
        print(f"\n🗑️  Starting deletion...")
        deleted_count = 0
        failed_count = 0

        for img_path in found_images:
            try:
                img_path.unlink()
                deleted_count += 1
                if deleted_count <= 10 or deleted_count % 50 == 0:
                    print(f"  ✓ Deleted: {img_path.name}")
            except Exception as e:
                failed_count += 1
                print(f"  ✗ Failed to delete: {img_path.name} - {e}")

        print(f"\n✅ Done:")
        print(f"  Deleted: {deleted_count} image(s)")
        if failed_count > 0:
            print(f"  Failed: {failed_count}")

        return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description="Delete original images according to a marker list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview
  python delete_marked_images.py \\
      --list marked_for_deletion.txt \\
      --frames data/frames/salmon_validation/ \\
      --dry-run

  # Actually delete
  python delete_marked_images.py \\
      --list marked_for_deletion.txt \\
      --frames data/frames/salmon_validation/
        """
    )

    parser.add_argument("--list", required=True, help="Marker list file")
    parser.add_argument("--frames", required=True, help="Original image directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview mode (no actual deletion)")

    args = parser.parse_args()

    marked_file = Path(args.list)
    frames_dir = Path(args.frames)

    if not marked_file.exists():
        print(f"❌ Marker list not found: {marked_file}")
        return 1

    if not frames_dir.exists():
        print(f"❌ Image directory not found: {frames_dir}")
        return 1

    print("=" * 70)
    print("🗑️  Batch-delete marked images")
    print("=" * 70)
    print(f"Marker list: {marked_file}")
    print(f"Image dir  : {frames_dir}")
    print(f"Mode       : {'preview (no deletion)' if args.dry_run else 'actual deletion'}")

    deleted = delete_marked_images(marked_file, frames_dir, args.dry_run)

    if not args.dry_run and deleted > 0:
        print(f"\n💡 Next: run the cleanup script to delete the orphaned labels:")
        print(f"  python -m src.preprocessing.annotation_salmon.cleanup_orphaned_labels \\")
        print(f"      --images data/frames/salmon_validation/ \\")
        print(f"      --labels data/auto_labels/salmon_validation/")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
