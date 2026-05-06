#!/usr/bin/env python3
"""
Clean up orphaned label files — delete labels that have no matching image.

Usage:
    python cleanup_orphaned_labels.py --images <image_dir> --labels <label_dir> [--dry-run]

Examples:
    # Preview (no actual deletion)
    python cleanup_orphaned_labels.py \
        --images data/frames/salmon_validation/ \
        --labels data/auto_labels/salmon_validation/ \
        --dry-run

    # Actually delete
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
    Collect the stem (filename without extension) of every image file.

    Args:
        image_dir: image directory path

    Returns:
        a set of all image-file stems
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
    Find orphaned label files (labels with no matching image).

    Args:
        image_dir: image directory
        label_dir: labels directory

    Returns:
        list of orphaned label paths
    """
    print(f"\n[1/3] Scanning image directory: {image_dir}")
    image_stems = get_image_stems(image_dir)
    print(f"  Found {len(image_stems)} image file(s)")

    print(f"\n[2/3] Scanning label directory: {label_dir}")
    orphaned_labels = []
    total_labels = 0

    for label_path in label_dir.rglob('*.txt'):
        total_labels += 1
        label_stem = label_path.stem

        # Check whether a matching image exists
        if label_stem not in image_stems:
            orphaned_labels.append(label_path)

    print(f"  Found {total_labels} label file(s)")
    print(f"  {len(orphaned_labels)} of them are orphaned (no matching image)")

    return orphaned_labels


def delete_orphaned_labels(orphaned_labels: List[Path], dry_run: bool = True):
    """
    Delete orphaned label files.

    Args:
        orphaned_labels: list of orphaned label files
        dry_run: if True, only preview (no deletion)
    """
    if not orphaned_labels:
        print("\n✅ No orphaned label files — the dataset is already in sync!")
        return

    print(f"\n[3/3] {'Previewing' if dry_run else 'Deleting'} orphaned labels:")

    if dry_run:
        print("\n  The following files would be deleted (preview only):")
        for label_path in orphaned_labels[:20]:  # only show the first 20
            print(f"    - {label_path.name}")

        if len(orphaned_labels) > 20:
            print(f"    ... and {len(orphaned_labels) - 20} more")

        print(f"\n  ⚠️  Preview mode: nothing was actually deleted")
        print(f"  💡 To delete for real, drop the --dry-run flag")
    else:
        deleted_count = 0
        failed_count = 0

        for label_path in orphaned_labels:
            try:
                label_path.unlink()
                deleted_count += 1
                if deleted_count <= 10 or deleted_count % 50 == 0:
                    print(f"  ✓ Deleted: {label_path.name}")
            except Exception as e:
                failed_count += 1
                print(f"  ✗ Failed to delete: {label_path.name} - {e}")

        print(f"\n✅ Done:")
        print(f"  Deleted: {deleted_count} file(s)")
        if failed_count > 0:
            print(f"  Failed: {failed_count} file(s)")


def main():
    parser = argparse.ArgumentParser(
        description='Clean up orphaned label files (delete labels without a matching image)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview which files would be deleted (recommended first)
  python cleanup_orphaned_labels.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/ \\
      --dry-run

  # Actually delete
  python cleanup_orphaned_labels.py \\
      --images data/frames/salmon_validation/ \\
      --labels data/auto_labels/salmon_validation/
        """
    )

    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to the image directory (the images you decided to keep after review)'
    )

    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to the labels directory (YOLO-format .txt files)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview mode: only list files that would be deleted, do not actually delete'
    )

    args = parser.parse_args()

    # Convert to Path
    image_dir = Path(args.images).resolve()
    label_dir = Path(args.labels).resolve()

    # Validate
    if not image_dir.exists():
        print(f"❌ Error: image directory not found: {image_dir}")
        return 1

    if not label_dir.exists():
        print(f"❌ Error: label directory not found: {label_dir}")
        return 1

    print("=" * 70)
    print("🗑️  Orphaned-label cleanup tool")
    print("=" * 70)
    print(f"\nImage dir : {image_dir}")
    print(f"Label dir : {label_dir}")
    print(f"Mode      : {'preview (no deletion)' if args.dry_run else 'actual deletion'}")

    # Find orphaned labels
    orphaned_labels = find_orphaned_labels(image_dir, label_dir)

    # Delete orphaned labels
    delete_orphaned_labels(orphaned_labels, dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("✅ Done!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
