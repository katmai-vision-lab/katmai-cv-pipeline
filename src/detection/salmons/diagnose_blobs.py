# diagnose_blobs.py — run on one video to print blob size distribution
import cv2
import numpy as np
from salmon_jump_counter_cv import (
    get_foreground_mask, get_salmon_color_mask, SALMON_HSV_LOWER, SALMON_HSV_UPPER
)

def diagnose(video_path, num_frames=60):
    cap = cv2.VideoCapture(video_path)
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    all_areas = []
    idx = 0

    while idx < num_frames * 3:
        ret, frame = cap.read()
        if not ret: break
        if idx % 3 == 0:
            fg = get_foreground_mask(frame, bg)
            color = get_salmon_color_mask(frame)
            combined = cv2.bitwise_and(fg, color)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = max(w,h) / (min(w,h) + 1e-5)
                    all_areas.append((area, aspect))
        idx += 1
    cap.release()

    all_areas.sort()
    print(f"\n{'Area':>8}  {'Aspect':>6}")
    print("-" * 18)
    for area, asp in all_areas[-40:]:   # show 40 largest blobs
        label = " <-- likely fish?" if 500 < area < 10000 and 1.5 < asp < 5 else ""
        print(f"{area:>8.0f}  {asp:>6.2f}{label}")

if __name__ == "__main__":
    import sys
    diagnose(sys.argv[1])