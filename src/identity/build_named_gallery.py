"""
Build a NAMED bear gallery from PoseSwin training head crops.

For each subfolder under --image-root (one folder per bear name), compute PoseSwin
embeddings for all images, average per bear, store as a Gallery JSON entry under
that bear's real name (e.g. "Aardvark", "Hotlips").

The resulting gallery file is the same format as the auto-generated one used by
identify_bears.py — so the existing identifier code can match against named bears
out of the box.

Usage
-----
    venv/bin/python3 -m src.identity.build_named_gallery \
        --image-root data/identity/gallery_images \
        --output     data/identity/named_bear_gallery.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.identity.poseswin_identifier import PoseSwinIdentifier, Gallery, GalleryEntry


def main():
    parser = argparse.ArgumentParser(description="Build PoseSwin gallery from named-bear folders")
    parser.add_argument("--image-root", required=True,
                        help="Root folder; one subfolder per bear name (e.g. .../Aardvark/*.JPG)")
    parser.add_argument("--output", required=True, help="Output gallery JSON path")
    parser.add_argument("--max-per-bear", type=int, default=15,
                        help="Cap on how many images to embed per bear (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size (default: 8 on dual 2080 Ti)")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    args = parser.parse_args()

    image_root = Path(args.image_root).resolve()
    out_path = Path(args.output).resolve()

    bear_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])
    print(f"Found {len(bear_dirs)} named bears under {image_root}")

    print("\nLoading PoseSwin model...")
    identifier = PoseSwinIdentifier(device=args.device)

    gallery = Gallery(out_path)

    total_embedded = 0
    for i, bear_dir in enumerate(bear_dirs, 1):
        bear_name = bear_dir.name
        imgs = sorted(bear_dir.glob("*.JPG")) + sorted(bear_dir.glob("*.jpg"))
        imgs = imgs[: args.max_per_bear]
        if not imgs:
            print(f"[{i:3d}/{len(bear_dirs)}] {bear_name}: no images, skipping")
            continue

        # Load + batch
        bgrs = []
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None or bgr.size == 0:
                continue
            bgrs.append(bgr)

        if not bgrs:
            print(f"[{i:3d}/{len(bear_dirs)}] {bear_name}: 0 readable images")
            continue

        # Batch through GPU
        all_emb = []
        for j in range(0, len(bgrs), args.batch_size):
            chunk = bgrs[j : j + args.batch_size]
            embs = identifier.embed_batch(chunk)
            all_emb.append(embs)
        all_emb = np.concatenate(all_emb, axis=0)

        # Mean (PoseSwin embeddings already L2-normalized; renormalize after avg)
        mean_emb = all_emb.mean(axis=0)
        mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)

        gallery.entries.append(GalleryEntry(
            name=bear_name,
            embeddings=[mean_emb.astype(np.float32)],
            n_observations=len(bgrs),
        ))
        total_embedded += len(bgrs)
        print(f"[{i:3d}/{len(bear_dirs)}] {bear_name}: averaged {len(bgrs)} images → 1 gallery entry")

    gallery.save()
    print(f"\n✓ Wrote {len(gallery.entries)} named bears to {out_path}")
    print(f"  Total embeddings computed: {total_embedded}")


if __name__ == "__main__":
    main()
