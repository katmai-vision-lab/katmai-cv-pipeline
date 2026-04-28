# salmon_jump_counter_cv.py
import cv2
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import json

# ── HSV color range for salmon (tune these for your water/lighting) ──────────
# Salmon in air = orange/silver flash. Water = blue/green background.
SALMON_HSV_LOWER = np.array([0,   0,  40])   # near-achromatic, dark
SALMON_HSV_UPPER = np.array([180, 60, 160])  # low saturation, mid brightness

# 4 sample point(s) collected
# H range: 115–138  S: 7–35  V: 44–207

# Silver/white variant (overcast light)
SILVER_HSV_LOWER = np.array([0,  0,  160])
SILVER_HSV_UPPER = np.array([180, 40, 255])

# Minimum blob area in pixels^2 — rejects splashes, keeps fish
MIN_BLOB_AREA = 800
MAX_BLOB_AREA = 8000

# Region of interest: (x, y, width, height) — set to None to use full frame
# Crop to the water surface area to reduce false positives
ROI = None  # e.g. (0, 100, 1280, 400)


def extract_frames(video_path: str, sample_rate: int = 3):
    """Yield (frame_index, frame) every `sample_rate` frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps:.1f} fps, {total} frames ({total/fps:.1f}s)")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            yield idx, frame
        idx += 1
    cap.release()
    return fps


def get_foreground_mask(frame, bg_subtractor):
    """Use MOG2 background subtraction to isolate moving objects."""
    fg_mask = bg_subtractor.apply(frame)
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)
    return fg_mask


def get_salmon_color_mask(frame):
    """Create a mask selecting salmon-colored pixels (orange + silver)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_orange = cv2.inRange(hsv, SALMON_HSV_LOWER, SALMON_HSV_UPPER)
    mask_silver = cv2.inRange(hsv, SILVER_HSV_LOWER, SILVER_HSV_UPPER)
    combined = cv2.bitwise_or(mask_orange, mask_silver)

    # Smooth out salt-and-pepper noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return combined


# def detect_salmon_blobs(frame, fg_mask, color_mask):
#     """
#     Combine foreground motion + color to find salmon candidates.
#     Returns list of (centroid_x, centroid_y, area) for each blob.
#     """
#     # Salmon pixel = moving AND salmon-colored
#     combined = cv2.bitwise_and(fg_mask, color_mask)

#     if ROI:
#         x, y, w, h = ROI
#         roi_mask = np.zeros_like(combined)
#         roi_mask[y:y+h, x:x+w] = combined[y:y+h, x:x+w]
#         combined = roi_mask

#     contours, _ = cv2.findContours(
#         combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     blobs = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if MIN_BLOB_AREA < area < MAX_BLOB_AREA:
#             M = cv2.moments(cnt)
#             if M["m00"] > 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#                 blobs.append((cx, cy, area))
#     return blobs


def detect_salmon_blobs(frame, fg_mask, color_mask):
    combined = cv2.bitwise_and(fg_mask, color_mask)

    if ROI:
        x, y, w, h = ROI
        roi_mask = np.zeros_like(combined)
        roi_mask[y:y+h, x:x+w] = combined[y:y+h, x:x+w]
        combined = roi_mask

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_BLOB_AREA < area < MAX_BLOB_AREA):
            continue

        # ── NEW: aspect ratio filter ──────────────────────────────
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1e-5)
        if aspect < 1.5 or aspect > 6.0:
            continue   # reject near-square blobs (water) and extreme slivers

        # ── NEW: solidity filter ──────────────────────────────────
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / (hull_area + 1e-5)
        if solidity < 0.5:
            continue   # reject jagged/irregular blobs (wave edges)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            blobs.append((cx, cy, area))

    return blobs


def count_jumps_from_trajectory(blob_presence_signal: list, fps: float,
                                sample_rate: int, min_jump_gap_sec: float = 1.0):
    """
    Detect jump events from the blob presence signal over time.
    A 'jump' = blob appears above water line (low y = high in frame),
    then disappears (re-enters water).

    blob_presence_signal: list of (frame_idx, max_blob_area) per sampled frame
    Returns: jump count and timestamps
    """
    if not blob_presence_signal:
        return 0, []

    frames, areas = zip(*blob_presence_signal)
    areas = np.array(areas, dtype=float)

    # Smooth the signal
    window = max(3, int(fps / sample_rate))
    kernel = np.ones(window) / window
    smoothed = np.convolve(areas, kernel, mode='same')

    # Find peaks = moments of maximum salmon visibility (apex of jump)
    min_gap_frames = int(min_jump_gap_sec * fps / sample_rate)
    peaks, props = find_peaks(
        smoothed,
        height=MIN_BLOB_AREA * 0.5,
        distance=min_gap_frames,
        prominence=MIN_BLOB_AREA * 0.3
    )

    timestamps = [frames[p] / fps for p in peaks]
    return len(peaks), timestamps, smoothed


def count_salmon_jumps(video_path: str, sample_rate: int = 3,
                       debug_output: str = None):
    """
    Main function: count salmon jumps in a video.
    
    Args:
        video_path:    Path to video file
        sample_rate:   Process every Nth frame (3 = fast, 1 = precise)
        debug_output:  If set, save annotated frames to this directory
    
    Returns:
        dict with jump_count, timestamps, and signal data
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # MOG2: learns background over first ~50 frames automatically
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,        # longer memory = water surface becomes "background"
        varThreshold=60,    # higher = only fast/distinct motion triggers
        detectShadows=False
    )

    blob_signal = []  # (frame_idx, total_salmon_area)

    for frame_idx, frame in extract_frames(video_path, sample_rate):
        fg_mask = get_foreground_mask(frame, bg_subtractor)
        color_mask = get_salmon_color_mask(frame)
        blobs = detect_salmon_blobs(frame, fg_mask, color_mask)

        total_area = sum(a for _, _, a in blobs)
        blob_signal.append((frame_idx, total_area))

        if debug_output and blobs:
            _save_debug_frame(frame, blobs, fg_mask, color_mask,
                              frame_idx, debug_output)

    jump_count, timestamps, signal = count_jumps_from_trajectory(
        blob_signal, fps, sample_rate
    )

    return {
        "video": video_path,
        "fps": fps,
        "jump_count": jump_count,
        "jump_timestamps_sec": [round(t, 2) for t in timestamps],
        "sample_rate": sample_rate,
    }


def _save_debug_frame(frame, blobs, fg_mask, color_mask, idx, out_dir):
    """Save annotated debug frame showing detected blobs."""
    Path(out_dir).mkdir(exist_ok=True)
    vis = frame.copy()
    for cx, cy, area in blobs:
        r = int(np.sqrt(area / np.pi))
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
        cv2.putText(vis, f"area={area:.0f}", (cx - 30, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(f"{out_dir}/frame_{idx:05d}.jpg", vis)


# ── HSV calibration helper ─────────────────────────────────────────────────
def calibrate_hsv(image_path: str):
    """
    Interactive HSV tuner — open an image of a salmon mid-jump,
    drag sliders until only the fish is highlighted, then note the values.
    Run this once before processing videos.
    """
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def nothing(_): pass
    cv2.namedWindow("HSV Calibration")
    for name, val in [("H_lo",5),("H_hi",30),("S_lo",60),("S_hi",255),
                       ("V_lo",80),("V_hi",255)]:
        cv2.createTrackbar(name, "HSV Calibration", val, 255, nothing)

    while True:
        lo = np.array([cv2.getTrackbarPos("H_lo","HSV Calibration"),
                       cv2.getTrackbarPos("S_lo","HSV Calibration"),
                       cv2.getTrackbarPos("V_lo","HSV Calibration")])
        hi = np.array([cv2.getTrackbarPos("H_hi","HSV Calibration"),
                       cv2.getTrackbarPos("S_hi","HSV Calibration"),
                       cv2.getTrackbarPos("V_hi","HSV Calibration")])
        mask = cv2.inRange(hsv, lo, hi)
        cv2.imshow("HSV Calibration", cv2.bitwise_and(img, img, mask=mask))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"SALMON_HSV_LOWER = np.array({lo.tolist()})")
            print(f"SALMON_HSV_UPPER = np.array({hi.tolist()})")
            break
    cv2.destroyAllWindows()


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python salmon_jump_counter_cv.py <video.mp4> [debug_dir]")
        sys.exit(1)

    video = sys.argv[1]
    debug = sys.argv[2] if len(sys.argv) > 2 else None

    result = count_salmon_jumps(video, sample_rate=3, debug_output=debug)
    print(json.dumps(result, indent=2))