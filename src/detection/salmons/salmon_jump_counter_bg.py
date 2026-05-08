"""
Salmon Jump Counter — Background Subtraction + Contour Tracking
================================================================
NEW in this version:
  • Interactive ROI selector  — draw the detection zone with your mouse
  • Interactive tripwire      — click to place the counting line
  • Live trackbars            — tune all parameters in real time
  • Gaussian blur             — suppresses water texture before MOG2
  • Stronger morphology       — dual-kernel strategy for cleaner blobs
  • Area stats on exit        — guides you to the right --min-area value

Usage:
    # Interactive setup (recommended first run)
    python salmon_counter.py --video path/to/video.mp4

    # After you know your parameters — skip interactive steps
    python salmon_counter.py --video path/to/video.mp4 \
        --line-y 650 --min-area 1200 --roi 120,80,1800,900 \
        --var-threshold 90 --blur-size 7 --output out.mp4

Optional flags:
    --line-y          Y-position of tripwire (default: interactive click)
    --min-area        Minimum contour area (default: 800)
    --roi             x1,y1,x2,y2  Fixed ROI, skips mouse-draw step
    --var-threshold   MOG2 varThreshold (default: 80)
    --history         MOG2 history frames (default: 300)
    --blur-size       Gaussian blur kernel size, odd number (default: 7)
    --skip-frames     Process every Nth frame (default: 2)
    --output          Save annotated video to this path
    --no-display      Headless / no GUI
    --debug-area      Print every contour area to terminal
    --no-trackbars    Disable the live-tuning sidebar
"""

import cv2
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys


# ══════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════

@dataclass
class Track:
    track_id: int
    cx: int
    cy: int
    history: list = field(default_factory=list)
    counted: bool = False
    frames_missing: int = 0

    def update(self, cx: int, cy: int):
        self.cx = cx
        self.cy = cy
        self.history.append((cx, cy))
        self.frames_missing = 0


# ══════════════════════════════════════════════════════════
#  Centroid tracker
# ══════════════════════════════════════════════════════════

class CentroidTracker:
    def __init__(self, max_distance: int = 80, max_missing: int = 20):
        self.next_id = 0
        self.tracks: dict[int, Track] = {}
        self.max_distance = max_distance
        self.max_missing = max_missing

    def update(self, centroids: list[tuple[int, int]]) -> list[Track]:
        for t in self.tracks.values():
            t.frames_missing += 1

        unmatched = list(range(len(centroids)))

        if self.tracks:
            track_ids  = list(self.tracks.keys())
            track_ctrs = [(self.tracks[tid].cx, self.tracks[tid].cy)
                          for tid in track_ids]
            matched_tids: set[int] = set()

            for i, (cx, cy) in enumerate(centroids):
                best_dist, best_tid = self.max_distance, None
                for tid, (tx, ty) in zip(track_ids, track_ctrs):
                    if tid in matched_tids:
                        continue
                    d = np.hypot(cx - tx, cy - ty)
                    if d < best_dist:
                        best_dist, best_tid = d, tid
                if best_tid is not None:
                    self.tracks[best_tid].update(cx, cy)
                    matched_tids.add(best_tid)
                    if i in unmatched:
                        unmatched.remove(i)

        for i in unmatched:
            cx, cy = centroids[i]
            t = Track(track_id=self.next_id, cx=cx, cy=cy)
            t.history.append((cx, cy))
            self.tracks[self.next_id] = t
            self.next_id += 1

        lost = [tid for tid, t in self.tracks.items()
                if t.frames_missing > self.max_missing]
        for tid in lost:
            del self.tracks[tid]

        return list(self.tracks.values())


# ══════════════════════════════════════════════════════════
#  Interactive ROI selector
# ══════════════════════════════════════════════════════════

def select_roi(first_frame: np.ndarray) -> tuple[int, int, int, int]:
    """Draw a rectangle on the first frame.  Returns (x1,y1,x2,y2)."""
    H, W  = first_frame.shape[:2]
    roi   = [0, 0, W, H]
    drawing  = [False]
    start_pt = [None]

    def draw_ui(img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, H - 68), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        for i, txt in enumerate([
            "DRAW ROI: click & drag to set the detection zone.",
            "Press ENTER or SPACE to confirm.  ESC = use full frame.",
        ]):
            cv2.putText(img, txt, (12, H - 44 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 1)

    def on_mouse(event, x, y, flags, param):
        img = first_frame.copy()
        draw_ui(img)

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            start_pt[0] = (x, y)
            roi[:] = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            roi[2], roi[3] = x, y
            x1, y1 = start_pt[0]
            cv2.rectangle(img, (x1, y1), (x, y), (0, 255, 100), 2)
            cv2.putText(img, f"({x1},{y1}) → ({x},{y})",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
            cv2.imshow(WIN_ROI, img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            roi[2], roi[3] = x, y
            x1, y1 = start_pt[0]
            img2 = first_frame.copy()
            draw_ui(img2)
            cv2.rectangle(img2, (x1, y1), (x, y), (0, 255, 100), 2)
            cv2.putText(img2, "Press ENTER to confirm", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
            cv2.imshow(WIN_ROI, img2)

    WIN_ROI = "Step 1/2 — Draw ROI, then press ENTER"
    cv2.namedWindow(WIN_ROI, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_ROI, on_mouse)

    start = first_frame.copy()
    draw_ui(start)
    cv2.imshow(WIN_ROI, start)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):   # ENTER / SPACE
            break
        if key == 27:         # ESC → full frame
            roi[:] = [0, 0, W, H]
            break

    cv2.destroyWindow(WIN_ROI)

    x1, y1, x2, y2 = roi
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        x1, y1, x2, y2 = 0, 0, W, H

    print(f"  ROI : ({x1},{y1}) → ({x2},{y2})  "
          f"[reuse: --roi {x1},{y1},{x2},{y2}]")
    return x1, y1, x2, y2


# ══════════════════════════════════════════════════════════
#  Interactive tripwire placement
# ══════════════════════════════════════════════════════════

def select_tripwire(first_frame: np.ndarray,
                    roi: tuple[int, int, int, int],
                    default_y: int) -> int:
    """Click on the frame to position the tripwire.  ENTER to confirm."""
    rx1, ry1, rx2, ry2 = roi
    H, W = first_frame.shape[:2]
    ty   = [default_y]

    def on_mouse(event, x, y, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            ty[0] = y

    WIN_TW = "Step 2/2 — Click to set tripwire, then press ENTER"
    cv2.namedWindow(WIN_TW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_TW, on_mouse)

    while True:
        dim = (first_frame * 0.35).astype(np.uint8)
        dim[ry1:ry2, rx1:rx2] = first_frame[ry1:ry2, rx1:rx2]
        cv2.rectangle(dim, (rx1, ry1), (rx2, ry2), (100, 255, 100), 2)
        cv2.line(dim, (0, ty[0]), (W, ty[0]), (0, 255, 255), 2)
        label_y = ty[0] - 10 if ty[0] > 30 else ty[0] + 22
        cv2.putText(dim, f"Tripwire Y={ty[0]}  (click to move, ENTER to confirm)",
                    (12, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(WIN_TW, dim)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32, 27):
            break

    cv2.destroyWindow(WIN_TW)
    print(f"  Tripwire Y={ty[0]}  [reuse: --line-y {ty[0]}]")
    return ty[0]


# ══════════════════════════════════════════════════════════
#  Trackbar window
# ══════════════════════════════════════════════════════════

TB_WIN = "Live Tuning  (changes apply in real time)"

def create_trackbars(var_threshold, min_area, blur_size, history):
    cv2.namedWindow(TB_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TB_WIN, 440, 200)
    cv2.createTrackbar("varThreshold",  TB_WIN, var_threshold,      254, lambda _: None)
    cv2.createTrackbar("Min Area /10",  TB_WIN, min_area // 10,     500, lambda _: None)
    cv2.createTrackbar("Blur Size /2",  TB_WIN, max(1, blur_size // 2), 10, lambda _: None)
    cv2.createTrackbar("History /10",   TB_WIN, history // 10,      100, lambda _: None)

def read_trackbars():
    vt   = max(1,  cv2.getTrackbarPos("varThreshold", TB_WIN))
    area = max(50, cv2.getTrackbarPos("Min Area /10", TB_WIN) * 10)
    blur = max(1,  cv2.getTrackbarPos("Blur Size /2", TB_WIN)) * 2 + 1
    hist = max(10, cv2.getTrackbarPos("History /10",  TB_WIN) * 10)
    return vt, area, blur, hist


# ══════════════════════════════════════════════════════════
#  Main processing loop
# ══════════════════════════════════════════════════════════

def run(
    video_path:    str,
    line_y:        Optional[int],
    min_area:      int,
    roi_fixed:     Optional[tuple],
    var_threshold: int,
    history:       int,
    blur_size:     int,
    skip_frames:   int,
    output_path:   Optional[str],
    display:       bool,
    debug_area:    bool,
    use_trackbars: bool,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\nVideo: {W}×{H}  @  {FPS:.1f} fps")

    # Read first frame for interactive setup
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── ROI ──────────────────────────────────────────────
    if roi_fixed:
        roi = roi_fixed
        print(f"  ROI (fixed): {roi}")
    elif display:
        roi = select_roi(first_frame)
    else:
        roi = (0, 0, W, H)

    rx1, ry1, rx2, ry2 = roi

    # ── Tripwire ─────────────────────────────────────────
    default_y = line_y if line_y is not None else int((ry1 + ry2) * 0.60)
    if line_y is None and display:
        tripwire_y = select_tripwire(first_frame, roi, default_y)
    else:
        tripwire_y = default_y

    print(f"  varThreshold={var_threshold}  min-area={min_area}  "
          f"blur={blur_size}  history={history}  skip={skip_frames}\n")

    # ── Background subtractor ─────────────────────────────
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=False)
    last_history = history

    # ── Trackbars ─────────────────────────────────────────
    if display and use_trackbars:
        create_trackbars(var_threshold, min_area, blur_size, history)

    tracker    = CentroidTracker(max_distance=80, max_missing=20)
    jump_count = 0
    all_areas  = []

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, FPS / skip_frames, (W, H))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % skip_frames != 0:
            continue

        # ── Live trackbar sync ────────────────────────────
        if display and use_trackbars:
            var_threshold, min_area, blur_size, history = read_trackbars()
            if history != last_history:
                bg_sub = cv2.createBackgroundSubtractorMOG2(
                    history=history, varThreshold=var_threshold, detectShadows=False)
                last_history = history
            else:
                bg_sub.setVarThreshold(var_threshold)

        # ── Crop ROI ──────────────────────────────────────
        roi_crop = frame[ry1:ry2, rx1:rx2]

        # ── Gaussian blur (kills water texture) ───────────
        ksize   = blur_size if blur_size % 2 == 1 else blur_size + 1
        blurred = cv2.GaussianBlur(roi_crop, (ksize, ksize), 0)

        # ── Background subtraction ────────────────────────
        fg = bg_sub.apply(blurred)

        # ── Dual-kernel morphology ────────────────────────
        #   Large OPEN  → destroys tiny water-noise speckles
        #   Large CLOSE → fills holes inside fish blobs
        #   Small DILATE → slightly expands fish for better contour
        k_big  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_sml  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_big)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_big)
        fg = cv2.dilate(fg, k_sml, iterations=2)

        # ── Full-frame mask for display ───────────────────
        fg_full = np.zeros((H, W), dtype=np.uint8)
        fg_full[ry1:ry2, rx1:rx2] = fg

        # ── Contour detection ─────────────────────────────
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids, bboxes = [], []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            all_areas.append(area)
            if debug_area:
                print(f"  [area] {area:.0f}")
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = rx1 + x + bw // 2
            cy = ry1 + y + bh // 2
            centroids.append((cx, cy))
            bboxes.append((rx1 + x, ry1 + y, bw, bh))

        # ── Tracking ──────────────────────────────────────
        active_tracks = tracker.update(centroids)

        # ── Jump detection (upward tripwire crossing) ─────
        for track in active_tracks:
            if track.counted or len(track.history) < 3:
                continue
            prev_y = track.history[-2][1]
            curr_y = track.history[-1][1]
            # Upward motion = Y decreasing (origin is top-left)
            if prev_y >= tripwire_y > curr_y:
                jump_count += 1
                track.counted = True
                print(f"  🐟  Jump #{jump_count}  "
                      f"(track {track.track_id}, frame {frame_idx})")

        # ── Draw visualisation ────────────────────────────
        vis = frame.copy()

        # Dim area outside ROI
        dim = (vis * 0.35).astype(np.uint8)
        dim[ry1:ry2, rx1:rx2] = vis[ry1:ry2, rx1:rx2]
        vis = dim

        # ROI border
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (80, 255, 80), 2)

        # Tripwire
        cv2.line(vis, (rx1, tripwire_y), (rx2, tripwire_y), (0, 255, 255), 2)
        label_y = tripwire_y - 8 if tripwire_y > ry1 + 20 else tripwire_y + 20
        cv2.putText(vis, f"Tripwire  Y={tripwire_y}",
                    (rx1 + 6, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Bounding boxes for accepted blobs
        for (x, y, bw, bh) in bboxes:
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 210, 100), 1)

        # Track trails + IDs
        for track in active_tracks:
            pts = track.history[-25:]
            for i in range(1, len(pts)):
                cv2.line(vis, pts[i - 1], pts[i], (255, 130, 0), 2)
            color = (0, 70, 255) if track.counted else (255, 210, 0)
            cv2.putText(vis, f"#{track.track_id}",
                        (track.cx + 5, track.cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Jump counter (top-left)
        cv2.rectangle(vis, (0, 0), (250, 56), (0, 0, 0), -1)
        cv2.putText(vis, f"Salmon Jumps: {jump_count}",
                    (10, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 180), 2)

        # Live params (bottom bar)
        params_str = (f"thr={var_threshold}  area={min_area}  "
                      f"blur={ksize}  hist={history}  skip={skip_frames}")
        cv2.rectangle(vis, (0, H - 26), (W, H), (0, 0, 0), -1)
        cv2.putText(vis, params_str, (6, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (190, 190, 190), 1)
        cv2.putText(vis, f"Frame {frame_idx}", (W - 130, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (150, 150, 150), 1)

        if writer:
            writer.write(vis)

        if display:
            cv2.imshow("Salmon Counter  |  Q = quit", vis)
            cv2.imshow("Foreground Mask", fg_full)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Interrupted by user.")
                break

    # ── Cleanup ───────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────
    print("\n" + "═" * 48)
    print(f"  Total salmon jumps counted : {jump_count}")
    print("═" * 48)

    if all_areas:
        arr = np.array(all_areas)
        p25, p50, p75 = (np.percentile(arr, p) for p in (25, 50, 75))
        print(f"\n  Contour area stats (all blobs including water noise):")
        print(f"    min={arr.min():.0f}  p25={p25:.0f}  "
              f"median={p50:.0f}  p75={p75:.0f}  max={arr.max():.0f}")
        print(f"\n  💡 Tip: set --min-area to ~{int(p75 * 1.2)} "
              f"(20 %% above p75) to suppress water noise blobs.")
    print()

    return jump_count


# ══════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Salmon jump counter — background subtraction + contour tracking.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--video",         required=True)
    p.add_argument("--line-y",        type=int,   default=None,
                   help="Tripwire Y pixel (default: interactive click)")
    p.add_argument("--min-area",      type=int,   default=800,
                   help="Min contour area in pixels (default: 800)")
    p.add_argument("--roi",           default=None,
                   help="Fixed ROI as x1,y1,x2,y2  e.g. 100,80,1820,900")
    p.add_argument("--var-threshold", type=int,   default=80,
                   help="MOG2 varThreshold — higher = less sensitive (default: 80)")
    p.add_argument("--history",       type=int,   default=300,
                   help="MOG2 history frames (default: 300)")
    p.add_argument("--blur-size",     type=int,   default=7,
                   help="Gaussian blur kernel size, odd (default: 7)")
    p.add_argument("--skip-frames",   type=int,   default=2,
                   help="Process every Nth frame (default: 2)")
    p.add_argument("--output",        default=None,
                   help="Save annotated output video here")
    p.add_argument("--no-display",    action="store_true",
                   help="Headless mode — no GUI windows")
    p.add_argument("--debug-area",    action="store_true",
                   help="Print every contour area to the terminal")
    p.add_argument("--no-trackbars",  action="store_true",
                   help="Disable the live-tuning trackbar window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    roi_fixed = None
    if args.roi:
        parts = [int(v) for v in args.roi.split(",")]
        if len(parts) != 4:
            print("ERROR: --roi must be four comma-separated integers: x1,y1,x2,y2")
            sys.exit(1)
        roi_fixed = tuple(parts)

    run(
        video_path    = args.video,
        line_y        = args.line_y,
        min_area      = args.min_area,
        roi_fixed     = roi_fixed,
        var_threshold = args.var_threshold,
        history       = args.history,
        blur_size     = args.blur_size,
        skip_frames   = args.skip_frames,
        output_path   = args.output,
        display       = not args.no_display,
        debug_area    = args.debug_area,
        use_trackbars = not args.no_trackbars,
    )