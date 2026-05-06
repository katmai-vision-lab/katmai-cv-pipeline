#!/usr/bin/env python3
"""
Interactive annotation reviewer — quickly mark misdetected images.

Usage:
    python review_annotations.py --visualized data/visualized/salmon_validation/

Keyboard:
    → or Space  : next image (keep)
    ←           : previous image
    d or Delete : mark for deletion
    u           : unmark (undo)
    q           : quit and save the marker list
"""

import cv2
import argparse
from pathlib import Path
import json


class AnnotationReviewer:
    def __init__(self, visualized_dir: Path):
        self.visualized_dir = visualized_dir
        self.images = sorted(list(visualized_dir.glob("*.jpg")))
        self.current_idx = 0
        self.marked_for_deletion = set()
        self.window_name = "Review [→/Space=next | ←=prev | D=delete | U=undo | Q=quit]"

        print(f"\nFound {len(self.images)} images")
        print(f"\nKeyboard:")
        print(f"  → or Space  : next image (keep)")
        print(f"  ←           : previous image")
        print(f"  D or Delete : mark for deletion")
        print(f"  U           : unmark (undo)")
        print(f"  Q           : quit and save")
        print(f"\nStarting review...\n")

    def show_current_image(self):
        if self.current_idx >= len(self.images):
            print("\n✅ Review complete — all images processed!")
            return False

        img_path = self.images[self.current_idx]
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"❌ Could not read: {img_path.name}")
            self.current_idx += 1
            return True

        # Resize image to fit the screen
        h, w = img.shape[:2]
        max_h, max_w = 1000, 1800
        if h > max_h or w > max_w:
            scale = min(max_h/h, max_w/w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        # Add status info
        status_text = f"[{self.current_idx + 1}/{len(self.images)}] {img_path.name}"
        if img_path in self.marked_for_deletion:
            status_text += " [marked for deletion]"
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)

        cv2.putText(img, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Marked: {len(self.marked_for_deletion)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(self.window_name, img)
        return True

    def run(self):
        while self.current_idx < len(self.images):
            if not self.show_current_image():
                break

            key = cv2.waitKey(0) & 0xFF

            # Next
            if key == 83 or key == ord(' ') or key == ord('n'):  # → or Space
                self.current_idx += 1

            # Previous
            elif key == 81 or key == ord('p'):  # ←
                if self.current_idx > 0:
                    self.current_idx -= 1

            # Mark for deletion
            elif key == ord('d') or key == 127:  # D or Delete
                img_path = self.images[self.current_idx]
                self.marked_for_deletion.add(img_path)
                print(f"🗑️  Marked for deletion: {img_path.name}")
                self.current_idx += 1

            # Undo mark
            elif key == ord('u'):
                img_path = self.images[self.current_idx]
                if img_path in self.marked_for_deletion:
                    self.marked_for_deletion.remove(img_path)
                    print(f"↩️  Unmarked: {img_path.name}")

            # Quit
            elif key == ord('q') or key == 27:  # Q or ESC
                break

        cv2.destroyAllWindows()
        return self.marked_for_deletion


def main():
    parser = argparse.ArgumentParser(description="Interactive annotation reviewer")
    parser.add_argument("--visualized", required=True, help="Visualized image directory")
    parser.add_argument("--output", default="marked_for_deletion.txt",
                       help="Output marker-list file")

    args = parser.parse_args()

    visualized_dir = Path(args.visualized)
    if not visualized_dir.exists():
        print(f"❌ Directory not found: {visualized_dir}")
        return 1

    reviewer = AnnotationReviewer(visualized_dir)
    marked_images = reviewer.run()

    if marked_images:
        print(f"\n📝 Saving marker list...")
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            for img_path in sorted(marked_images):
                # Write the original filename (no path)
                f.write(img_path.stem + '\n')

        print(f"\n✅ Saved {len(marked_images)} marker(s) to: {output_file}")
        print(f"\nNext: run the deletion script:")
        print(f"  python delete_marked_images.py --list {output_file}")
    else:
        print(f"\n✅ No images were marked")

    return 0


if __name__ == "__main__":
    exit(main())
