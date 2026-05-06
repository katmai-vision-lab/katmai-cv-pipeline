"""
Add user-supplied bear photos to the identity gallery.

Two ways to add a bear:

  A) Pass a folder of pre-cropped head shots (recommended):

         my_bears/
           Otis_480/
             photo1.jpg
             photo2.jpg
             ...
           Grazer_128/
             photo1.jpg
             ...

     Each subfolder name becomes the bear's identity in the gallery.
     The face detector is OPTIONAL here — if your photos are already head
     crops you can pass --no-face-detector to skip it.

  B) Pass a folder of full bear photos and let the face detector find heads:

         photos/
           known_otis/
             *.jpg            # any photo containing Otis (full body, scenery, etc.)

     The face detector finds + crops the head before embedding.

Usage
-----
    # Add new named bears (auto-runs face detector by default)
    venv/bin/python3 -m src.identity.add_to_gallery \
        --image-root my_bears \
        --gallery    data/identity/named_bear_gallery.json

    # Skip face detector (your photos are already head crops)
    venv/bin/python3 -m src.identity.add_to_gallery \
        --image-root my_bears \
        --no-face-detector

    # Replace existing entries instead of merging
    venv/bin/python3 -m src.identity.add_to_gallery --image-root my_bears --replace
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

DEFAULT_GALLERY = PROJECT_ROOT / "data" / "identity" / "named_bear_gallery.json"


def main():
    parser = argparse.ArgumentParser(
        description="Add user-supplied bear photos to the PoseSwin identity gallery."
    )
    parser.add_argument("--image-root", required=True,
                        help="Root folder; one subfolder per bear name (e.g. .../Otis_480/*.jpg)")
    parser.add_argument("--gallery", default=str(DEFAULT_GALLERY),
                        help=f"Path to gallery JSON (default: {DEFAULT_GALLERY})")
    parser.add_argument("--no-face-detector", action="store_true",
                        help="Skip the bear-face Faster-RCNN. Use if your photos are already head crops.")
    parser.add_argument("--max-per-bear", type=int, default=15,
                        help="Cap on photos per bear (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size (default: 8)")
    parser.add_argument("--replace", action="store_true",
                        help="If a bear with this name already exists in the gallery, "
                             "REPLACE its embedding (default: merge — average old & new exemplars)")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    args = parser.parse_args()

    image_root = Path(args.image_root).resolve()
    bear_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])
    if not bear_dirs:
        print(f"No subfolders found under {image_root} — nothing to add.")
        return

    print(f"Found {len(bear_dirs)} bear folders under {image_root}")
    for d in bear_dirs:
        n = sum(1 for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png") for _ in d.glob(ext))
        print(f"  - {d.name}: {n} photos")

    print("\nLoading PoseSwin model...")
    identifier = PoseSwinIdentifier(device=args.device)

    face_detector = None
    if not args.no_face_detector:
        print("Loading bear-face detector...")
        from src.identity.face_detector import BearFaceDetector
        face_detector = BearFaceDetector(device=args.device, score_threshold=0.3)

    gallery = Gallery.load(args.gallery)
    print(f"Loaded gallery from {args.gallery} — {len(gallery.entries)} existing bears")
    print()

    added = updated = skipped = 0

    for bear_dir in bear_dirs:
        bear_name = bear_dir.name
        photos = []
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.PNG"):
            photos.extend(sorted(bear_dir.glob(ext)))
        photos = photos[: args.max_per_bear]
        if not photos:
            print(f"[skip ] {bear_name}: no photos")
            skipped += 1
            continue

        head_crops = []
        for p in photos:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            if face_detector is None:
                # Treat the whole image as a head crop
                head_crops.append(bgr)
            else:
                head = face_detector.best_head_crop(bgr)
                if head is not None and head.size > 0:
                    head_crops.append(head)
                else:
                    # Fall back to the original image as last resort
                    head_crops.append(bgr)

        if not head_crops:
            print(f"[skip ] {bear_name}: no usable head crops")
            skipped += 1
            continue

        # Batch-embed
        embeddings = []
        for j in range(0, len(head_crops), args.batch_size):
            chunk = head_crops[j : j + args.batch_size]
            embeddings.append(identifier.embed_batch(chunk))
        embeddings = np.concatenate(embeddings, axis=0)
        mean_emb = embeddings.mean(axis=0)
        mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)

        # Insert / update
        existing = next((e for e in gallery.entries if e.name == bear_name), None)
        if existing is None:
            gallery.entries.append(GalleryEntry(
                name=bear_name,
                embeddings=[mean_emb.astype(np.float32)],
                n_observations=len(head_crops),
            ))
            print(f"[new  ] {bear_name}: added with {len(head_crops)} photos")
            added += 1
        else:
            if args.replace:
                existing.embeddings = [mean_emb.astype(np.float32)]
                existing.n_observations = len(head_crops)
                print(f"[repl ] {bear_name}: replaced with {len(head_crops)} new photos")
            else:
                existing.embeddings.append(mean_emb.astype(np.float32))
                existing.embeddings = existing.embeddings[-5:]  # cap at 5 exemplars
                existing.n_observations += len(head_crops)
                print(f"[merge] {bear_name}: added {len(head_crops)} new photos "
                      f"(now {existing.n_observations} total observations)")
            updated += 1

    gallery.save()
    print()
    print(f"✓ Gallery saved to {args.gallery}")
    print(f"  Added new: {added}   Updated: {updated}   Skipped: {skipped}")
    print(f"  Total bears in gallery: {len(gallery.entries)}")


if __name__ == "__main__":
    main()
