"""
Dump per-frame track IDs from ByteTrack without generating a video.
Useful for debugging merge logic: you can see exactly which raw IDs appear in
which frames, and what the merge step would do with them.

Usage:
    python -m src.detection.track_dump --video bears/xxx.mp4 \
        --model models/trained/bear_detector3/weights/best.pt \
        --classes 0 --conf 0.7

Output: prints to stdout, and optionally saves JSON via --json-out.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import BearDetector
from src.config import RAW_DATA_DIR, TRAINED_MODELS_DIR

DEFAULT_MODEL = str(TRAINED_MODELS_DIR / "bear_detector2" / "weights" / "best.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Dump per-frame track IDs (no video output) for merge debugging"
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--classes", type=int, nargs="+", default=[0])
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--tracker", type=str, default="bytetrack")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to save full dump as JSON")
    parser.add_argument("--show-every", type=int, default=1,
                        help="Only print every Nth frame to stdout (default 1)")
    parser.add_argument("--max-gap-frames", type=int, default=3600,
                        help="Merge: max frame gap (default 3600 = 2min @ 30fps)")
    parser.add_argument("--max-dist-px", type=int, default=150,
                        help="Merge: max pixel distance at transition (default 150)")
    parser.add_argument("--cooccur-tol-frames", type=int, default=60,
                        help="Merge: tolerate up to N co-occurrence frames as detection "
                             "artifact if mean dist < max_dist_px (default 60)")
    parser.add_argument("--cooccur-artifact-iou", type=float, default=0.3,
                        help="Merge: if two co-occurring bboxes overlap with mean IoU >= "
                             "this, treat as same animal regardless of duration (default 0.3)")
    parser.add_argument("--min-duration", type=int, default=150,
                        help="Filter: drop groups shorter than this many frames (default 150 = 5s @ 30fps)")
    parser.add_argument("--min-mean-conf", type=float, default=0.80,
                        help="Filter: drop groups whose mean conf < this (default 0.80)")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        candidates = [
            PROJECT_ROOT / video_path,
            RAW_DATA_DIR / video_path,
            PROJECT_ROOT / "data" / "raw" / video_path,
        ]
        found = next((c for c in candidates if c.exists()), None)
        video_path = found if found is not None else (RAW_DATA_DIR / video_path)
    if not video_path.exists():
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    detector = BearDetector(model_path=args.model)
    tracker = args.tracker if args.tracker.endswith('.yaml') else f'{args.tracker}.yaml'

    print(f"\n📹 Running tracker on: {video_path.name}")
    print(f"   conf={args.conf}  frame_skip={args.frame_skip}  tracker={tracker}\n")

    results_stream = detector.model.track(
        source=str(video_path),
        conf=args.conf,
        classes=args.classes,
        tracker=tracker,
        save=False,
        stream=True,
        verbose=False,
        vid_stride=args.frame_skip,
        persist=True,
    )

    frame_data = []
    per_frame_dump = []  # [{frame, ids, positions, confs}]

    for result in results_stream:
        boxes = result.boxes
        track_ids = []
        positions = {}
        track_boxes = {}
        confs = {}
        if boxes.id is not None:
            ids_arr = boxes.id.cpu().numpy().astype(int).tolist()
            xywh = boxes.xywh.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            for i, tid in enumerate(ids_arr):
                track_ids.append(tid)
                positions[tid] = (round(float(xywh[i][0]), 1), round(float(xywh[i][1]), 1))
                track_boxes[tid] = (
                    float(xyxy[i][0]), float(xyxy[i][1]),
                    float(xyxy[i][2]), float(xyxy[i][3]),
                )
                confs[tid] = round(float(conf_arr[i]), 3)

        frame_idx = len(frame_data)
        frame_data.append({
            'frame': frame_idx,
            'track_ids': track_ids,
            'track_positions': positions,
            'track_boxes': track_boxes,
            'track_confs': confs,
        })
        per_frame_dump.append({
            'frame': frame_idx,
            'ids': track_ids,
            'positions': positions,
            'confs': confs,
        })

    # ---- Per-frame print ----
    print(f"{'='*70}")
    print(f"PER-FRAME RAW IDs (every {args.show_every} frame)")
    print(f"{'='*70}")
    for entry in per_frame_dump:
        if entry['frame'] % args.show_every != 0:
            continue
        ids_str = ', '.join(f"{tid}@({entry['positions'][tid][0]:.0f},{entry['positions'][tid][1]:.0f})c{entry['confs'][tid]:.2f}"
                            for tid in entry['ids'])
        print(f"Frame {entry['frame']:5d}: [{ids_str}]")

    # ---- Raw track summary ----
    all_ids = sorted({tid for e in per_frame_dump for tid in e['ids']})
    print(f"\n{'='*70}")
    print(f"RAW TRACK SUMMARY")
    print(f"{'='*70}")
    print(f"Total unique raw IDs: {len(all_ids)}")
    print(f"Raw IDs: {all_ids}")

    id_first_last = {}
    for e in per_frame_dump:
        for tid in e['ids']:
            if tid not in id_first_last:
                id_first_last[tid] = [e['frame'], e['frame']]
            id_first_last[tid][1] = e['frame']

    print(f"\n{'ID':>5} {'first':>7} {'last':>7} {'duration':>9}")
    for tid in all_ids:
        first, last = id_first_last[tid]
        print(f"{tid:>5} {first:>7} {last:>7} {last - first + 1:>9}")

    # ---- Merge analysis ----
    merged_count, id_map = BearDetector._merge_fragmented_tracks(
        frame_data,
        max_gap_frames=args.max_gap_frames,
        max_dist_px=args.max_dist_px,
        cooccurrence_tolerance_frames=args.cooccur_tol_frames,
        cooccurrence_artifact_iou=args.cooccur_artifact_iou,
    )

    # ---- Filter spurious groups ----
    id_map_filtered, dropped_roots = BearDetector._filter_spurious_groups(
        frame_data, id_map,
        min_duration=args.min_duration,
        min_mean_conf=args.min_mean_conf,
    )

    def summarize(id_map_to_show, header):
        print(f"\n{'='*70}")
        print(header)
        print(f"{'='*70}")

        groups = {}
        for raw, root in id_map_to_show.items():
            groups.setdefault(root, []).append(raw)

        if not groups:
            print("(no groups)")
            return

        group_first_frame = {
            root: min(id_first_last[m][0] for m in mems)
            for root, mems in groups.items()
        }
        ordered_roots = sorted(groups.keys(), key=lambda r: group_first_frame[r])
        print(f"Unique bears: {len(ordered_roots)}")

        for i, root in enumerate(ordered_roots, start=1):
            members_sorted = sorted(groups[root])
            first_seen = group_first_frame[root]
            # Compute group duration and mean conf for diagnostic
            durations = [id_first_last[m][1] - id_first_last[m][0] + 1 for m in members_sorted]
            total_span = max(id_first_last[m][1] for m in members_sorted) - min(id_first_last[m][0] for m in members_sorted) + 1
            confs_all = []
            for fd in per_frame_dump:
                for m in members_sorted:
                    if m in fd['confs']:
                        confs_all.append(fd['confs'][m])
            mean_conf = sum(confs_all) / len(confs_all) if confs_all else 0
            tag = f"  ← MERGED {len(members_sorted)} fragments" if len(members_sorted) > 1 else ""
            print(f"  Bear {i} (first @ {first_seen}, span {total_span}f, mean_conf {mean_conf:.2f}): {members_sorted}{tag}")

    summarize(id_map,
              f"MERGE RESULT (max_gap={args.max_gap_frames}, max_dist={args.max_dist_px}px) — BEFORE FILTER")
    summarize(id_map_filtered,
              f"AFTER FILTER (min_duration={args.min_duration}, min_mean_conf={args.min_mean_conf})")
    if dropped_roots:
        print(f"\nDropped {len(dropped_roots)} group(s) as spurious.")
    # Use filtered for output below
    id_map = id_map_filtered
    groups = {}
    for raw, root in id_map.items():
        groups.setdefault(root, []).append(raw)

    if args.json_out:
        out = {
            'video': str(video_path),
            'conf': args.conf,
            'frame_skip': args.frame_skip,
            'per_frame': per_frame_dump,
            'merge_params': {
                'max_gap_frames': args.max_gap_frames,
                'max_dist_px': args.max_dist_px,
                'min_duration': args.min_duration,
                'min_mean_conf': args.min_mean_conf,
            },
            'merge_result': {
                'unique_bears': len(groups),
                'id_map': {str(k): int(v) for k, v in id_map.items()},
                'groups': {str(root): sorted(members) for root, members in groups.items()},
            },
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n✓ JSON saved: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
