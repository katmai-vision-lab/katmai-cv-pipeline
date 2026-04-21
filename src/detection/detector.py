from ultralytics import YOLO
from pathlib import Path
import sys
import json
import time
import subprocess
from datetime import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    PRETRAINED_MODELS_DIR,
    TRAINED_MODELS_DIR,
    PREDICTIONS_DIR,
    TRAINED_BEAR_DETECTOR_PATH
)

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):  # np.float32, np.int64, etc.
        return obj.item()
    else:
        return obj

class BearDetector:
    """YOLO-based bear detector with training, prediction, and evaluation"""

    def __init__(self, model_path=None):
        """
        Initialize detector
        
        Args:
            model_path: Path to model file. If None, uses yolov8n pretrained
        """
        if model_path is None:
            model_path = TRAINED_BEAR_DETECTOR_PATH
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.model = YOLO(str(model_path))
        print(f"✓ Loaded model: {model_path.name}")


    def train(self, data_yaml, epochs=50, imgsz=640, batch=8, 
              project=None, name='bear_detector', resume=False, **kwargs):
        """
        Train/fine-tune the model
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            project: Output directory for training results
            name: Experiment name
            resume: Resume from last checkpoint
            **kwargs: Additional YOLO training parameters
        
        Returns:
            Training results
        """
        if project is None:
            project = TRAINED_MODELS_DIR

        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

        print(f"\n{'='*60}")
        print(f"Training Bear Detector")
        print(f"{'='*60}")
        print(f"Data: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch}")
        print(f"Output: {project}/{name}")
        print(f"{'='*60}\n")

        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(project),
            name=name,
            resume=resume,
            **kwargs
        )

        # Use actual save_dir: when name already exists, ultralytics saves to name2, name3, etc.
        save_dir = Path(self.model.trainer.save_dir)
        self.model_path = save_dir / 'weights' / 'best.pt'
        self.model = YOLO(str(self.model_path))

        print(f"\n✓ Training complete!")
        print(f"Best weights: {self.model_path}")

        return results

    def predict_video(self, video_path, output_name=None, conf=0.25, 
                      classes=None, save=True, **kwargs):
        """
        Run detection on video
        
        Args:
            video_path: Path to video file or filename in RAW_DATA_DIR
            output_name: Output directory name
            conf: Confidence threshold
            classes: List of class IDs to detect (None = all classes)
            save: Save annotated video
            **kwargs: Additional YOLO prediction parameters
        
        Returns:
            results: YOLO results object
            output_dir: Path to output directory
        """
        # Handle video path
        video_path = Path(video_path)
        if not video_path.is_absolute():
            video_path = RAW_DATA_DIR / video_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Generate output name
        if output_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_name = f"{timestamp}_{video_path.stem}"

        print(f"\n{'='*60}")
        print(f"Bear Detection")
        print(f"{'='*60}")
        print(f"Video: {video_path.name}")
        print(f"Model: {self.model_path.name}")
        print(f"Confidence: {conf}")
        print(f"Output: {PREDICTIONS_DIR / output_name}")
        print(f"{'='*60}\n")

        # Run prediction with streaming to avoid memory overload
        results = self.model.predict(
            source=str(video_path),
            conf=conf,
            classes=classes,
            save=save,
            show_labels=True,
            show_conf=True,
            project=str(PREDICTIONS_DIR),
            name=output_name,
            exist_ok=True,
            stream=True,  
            **kwargs
        )

        output_dir = PREDICTIONS_DIR / output_name

        # Save metadata 
        self._save_prediction_metadata(video_path, output_dir, conf, results)

        print(f"\n✓ Detection complete!")
        print(f"Results: {output_dir}")

        return None, output_dir

    def count_bears_in_video(self, video_path, conf=0.25, frame_skip=30, 
                            classes=21, verbose=True):
        """
        Count bears in a video without saving annotated output (fast)
        
        Args:
            video_path: Path to video file or filename in RAW_DATA_DIR
            conf: Confidence threshold
            frame_skip: Process every Nth frame (default: 30, ~1fps at 30fps video)
            classes: Class ID(s) to detect (default: 21 for bear)
            verbose: Print progress
            
        Returns:
            dict with counting statistics
        """
        video_path = Path(video_path)
        if not video_path.is_absolute():
            video_path = RAW_DATA_DIR / video_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if verbose:
            print(f"\n📹 Counting bears in: {video_path.name}")
        
        start_time = time.time()
        # Run prediction (streaming mode, no saving)
        results = self.model.predict(
            source=str(video_path),
            conf=conf,
            classes=classes,
            save=False,  # Don't save output
            stream=True,  # Stream for memory efficiency
            verbose=False,
            vid_stride=frame_skip
        )

        # Collect frame-by-frame statistics
        frame_data = []
        bear_counts_per_frame = []

        for frame_id, result in enumerate(results):
            boxes = result.boxes
            num_bears = len(boxes)
            confidences = boxes.conf.cpu().numpy() if len(boxes) > 0 else []

            bear_counts_per_frame.append(num_bears)
            frame_data.append({
                'frame': frame_id * frame_skip,
                'num_bears': num_bears,
                'avg_confidence': confidences.mean() if len(confidences) > 0 else 0,
                'max_confidence': confidences.max() if len(confidences) > 0 else 0
            })

        duration = time.time() - start_time

        # Calculate statistics
        bear_counts_array = np.array(bear_counts_per_frame)

        stats = {
            'video_name': video_path.name,
            'video_path': str(video_path),
            'processing_time': duration,
            'frames_analyzed': len(frame_data),
            'frames_with_bears': int((bear_counts_array > 0).sum()),
            'max_bears_in_frame': int(bear_counts_array.max()) if len(bear_counts_array) > 0 else 0,
            'min_bears_in_frame': int(bear_counts_array.min()) if len(bear_counts_array) > 0 else 0,
            'avg_bears_per_frame': float(bear_counts_array.mean()) if len(bear_counts_array) > 0 else 0,
            'median_bears_per_frame': float(np.median(bear_counts_array)) if len(bear_counts_array) > 0 else 0,
            'total_detections': int(bear_counts_array.sum()),
            'unique_bear_estimate': int(bear_counts_array.max()) if len(bear_counts_array) > 0 else 0,
            'frame_data': frame_data
        }

        if verbose:
            print(f"  ✓ Processed {len(frame_data)} frames in {duration:.1f}s")
            print(f"  🐻 Max bears: {stats['max_bears_in_frame']}")
            print(f"  📊 Avg bears: {stats['avg_bears_per_frame']:.2f}")

        return stats

    def track_bears_in_video(self, video_path, conf=0.25, frame_skip=30,
                            classes=21, tracker='bytetrack', verbose=True):
        """
        Track bears in a video using ByteTrack (returns unique bear count)

        Args:
            video_path: Path to video file
            conf: Confidence threshold
            frame_skip: Process every Nth frame
            classes: Class ID(s) to detect
            tracker: Tracker config ('bytetrack', 'botsort', or path to yaml)
            verbose: Print progress

        Returns:
            dict with tracking statistics
        """
        video_path = Path(video_path)
        if not video_path.is_absolute():
            video_path = RAW_DATA_DIR / video_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if verbose:
            print(f"\n📹 Tracking bears in: {video_path.name}")
        
        start_time = time.time()
        
        # Use model.track() instead of model.predict()
        results = self.model.track(
            source=str(video_path),
            conf=conf,
            classes=classes,
            tracker=tracker if tracker.endswith('.yaml') else f'{tracker}.yaml',
            save=False,
            stream=True,
            verbose=False,
            vid_stride=frame_skip,
            persist=True
        )

        # Collect tracking data
        frame_data = []
        bear_counts_per_frame = []
        unique_track_ids = set()

        for frame_id, result in enumerate(results):
            boxes = result.boxes
            track_ids = []
            track_positions = {}
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int).tolist()
                unique_track_ids.update(track_ids)
                if boxes.xywh is not None:
                    centers = boxes.xywh[:, :2].cpu().numpy()
                    for tid, (cx, cy) in zip(track_ids, centers):
                        track_positions[tid] = (float(cx), float(cy))

            num_bears = len(track_ids)
            confidences = boxes.conf.cpu().numpy() if len(boxes) > 0 else []

            bear_counts_per_frame.append(num_bears)
            frame_data.append({
                'frame': frame_id * frame_skip,
                'num_bears': num_bears,
                'track_ids': track_ids,
                'track_positions': track_positions,
                'avg_confidence': confidences.mean() if len(confidences) > 0 else 0
            })

        duration = time.time() - start_time
        bear_counts_array = np.array(bear_counts_per_frame)

        merged_count, _ = self._merge_fragmented_tracks(frame_data)

        stats = {
            'video_name': video_path.name,
            'video_path': str(video_path),
            'processing_time': duration,
            'tracker': tracker,
            'frames_analyzed': len(frame_data),
            'frames_with_bears': int((bear_counts_array > 0).sum()),
            'max_bears_in_frame': int(bear_counts_array.max()) if len(bear_counts_array) > 0 else 0,
            'avg_bears_per_frame': float(bear_counts_array.mean()) if len(bear_counts_array) > 0 else 0,
            'total_detections': int(bear_counts_array.sum()),
            'unique_bears_tracked': merged_count,
            'raw_track_ids': sorted(list(unique_track_ids)),
            'frame_data': frame_data
        }

        if verbose:
            print(f"  ✓ Processed {len(frame_data)} frames in {duration:.1f}s")
            print(f"  🐻 Unique bears: {stats['unique_bears_tracked']}")
            print(f"  📊 Max in frame: {stats['max_bears_in_frame']}")

        return stats

    @staticmethod
    def _merge_fragmented_tracks(frame_data, max_gap_frames=3600, max_dist_px=150):
        """
        Post-process track IDs to merge fragments from the same bear.

        A bear that briefly disappears (occlusion, enters water) may lose its track
        and reappear with a new ID. Two tracks are merged when:
          1. They never appear in the same frame (can't be different bears simultaneously)
          2. One ends before the other starts (temporally sequential)
          3. The gap between them is within max_gap_frames
          4. The last known position of the earlier track is within max_dist_px of
             the first known position of the later track

        Returns (merged_unique_count, {track_id: group_id})
        """
        from collections import defaultdict

        track_first_frame = {}
        track_last_frame = {}
        track_first_pos = {}
        track_last_pos = {}
        co_occurring = set()

        for fd in frame_data:
            frame = fd['frame']
            ids = fd['track_ids']
            positions = fd.get('track_positions', {})

            for tid in ids:
                if tid not in track_first_frame:
                    track_first_frame[tid] = frame
                    if tid in positions:
                        track_first_pos[tid] = positions[tid]
                track_last_frame[tid] = frame
                if tid in positions:
                    track_last_pos[tid] = positions[tid]

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = min(ids[i], ids[j]), max(ids[i], ids[j])
                    co_occurring.add((a, b))

        track_ids = sorted(track_first_frame.keys())
        parent = {tid: tid for tid in track_ids}
        # Members of each group root, so we can check transitive co-occurrence
        members = {tid: {tid} for tid in track_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def groups_can_merge(root_a, root_b):
            """Reject merge if any member of one group co-occurs with any member of the other."""
            for ma in members[root_a]:
                for mb in members[root_b]:
                    pair = (min(ma, mb), max(ma, mb))
                    if pair in co_occurring:
                        return False
            return True

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            parent[rx] = ry
            members[ry] |= members[rx]
            del members[rx]

        # Evaluate candidate pairs in order of (temporal gap, spatial dist) so the
        # most confident merges happen first — this reduces bad transitive chains.
        candidates = []
        for i, a in enumerate(track_ids):
            for b in track_ids[i + 1:]:
                pair = (min(a, b), max(a, b))
                if pair in co_occurring:
                    continue

                if track_last_frame[a] < track_first_frame[b]:
                    early, late = a, b
                elif track_last_frame[b] < track_first_frame[a]:
                    early, late = b, a
                else:
                    continue

                gap = track_first_frame[late] - track_last_frame[early]
                if gap > max_gap_frames:
                    continue

                if early in track_last_pos and late in track_first_pos:
                    lx, ly = track_last_pos[early]
                    fx, fy = track_first_pos[late]
                    dist = ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5
                    if dist > max_dist_px:
                        continue
                else:
                    dist = max_dist_px

                candidates.append((gap, dist, a, b))

        candidates.sort()  # prefer smallest gap, then smallest dist

        for _gap, _dist, a, b in candidates:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if not groups_can_merge(ra, rb):
                continue
            union(a, b)

        groups = set(find(tid) for tid in track_ids)
        return len(groups), {tid: find(tid) for tid in track_ids}

    @staticmethod
    def _filter_spurious_groups(frame_data, id_map, min_duration=30, min_mean_conf=0.80,
                                long_duration=500):
        """
        Drop groups that look like noise. Compound rule:
          - duration < min_duration → always drop (too short to be real).
          - duration >= long_duration → always keep (long-lived tracks are trustworthy
            even if mean_conf is slightly below threshold due to occlusion).
          - otherwise → require mean_conf >= min_mean_conf.

        Returns: (kept_id_map, dropped_groups)
        """
        from collections import defaultdict
        group_first = {}
        group_last = {}
        group_confs = defaultdict(list)

        for fd in frame_data:
            frame = fd['frame']
            confs = fd.get('track_confs', {})
            for tid in fd['track_ids']:
                root = id_map.get(tid)
                if root is None:
                    continue
                if root not in group_first:
                    group_first[root] = frame
                group_last[root] = frame
                if tid in confs:
                    group_confs[root].append(confs[tid])

        dropped = set()
        for root in group_first:
            duration = group_last[root] - group_first[root] + 1
            confs_list = group_confs[root]
            mean_conf = (sum(confs_list) / len(confs_list)) if confs_list else 0.0
            if duration < min_duration:
                dropped.add(root)
            elif duration < long_duration and mean_conf < min_mean_conf:
                dropped.add(root)

        kept = {tid: root for tid, root in id_map.items() if root not in dropped}
        return kept, dropped

    def track_and_save_video(self, video_path, output_name=None, conf=0.25,
                             frame_skip=1, classes=None, tracker='bytetrack',
                             max_gap_frames=3600, max_dist_px=150,
                             min_duration=30, min_mean_conf=0.80, **kwargs):
        """
        Track bears in a video and save an output video with bounding boxes and track IDs
        overlaid. Track IDs are post-processed with _merge_fragmented_tracks so that bears
        which briefly disappear keep the same display ID when they reappear.

        Two-pass approach:
          Pass 1 — stream tracking results to collect bbox data and positions.
          Merge  — compute id_map remapping fragmented track IDs.
          Pass 2 — read original video with cv2 and render with remapped IDs.

        Args:
            video_path: Path to video file or filename in RAW_DATA_DIR
            output_name: Output folder name under predictions/ (default: timestamp_videostem_track)
            conf: Confidence threshold
            frame_skip: Process every Nth frame (1 = every frame; 30 = ~1 fps)
            classes: Class ID(s) to detect (None = all)
            tracker: Tracker name, e.g. 'bytetrack', 'botsort'
            **kwargs: Extra arguments for model.track()

        Returns:
            (None, output_dir). Output video is in output_dir (MP4 if ffmpeg available, else AVI).
        """
        import cv2

        video_path = Path(video_path)
        if not video_path.is_absolute():
            video_path = RAW_DATA_DIR / video_path
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if output_name is None:
            output_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_path.stem}_track"

        print(f"\n📹 Tracking & saving video: {video_path.name}")
        print(f"   Output: {PREDICTIONS_DIR / output_name}\n")

        # --- Pass 1: stream to collect tracking data (no video writing) ---
        results_stream = self.model.track(
            source=str(video_path),
            conf=conf,
            classes=classes,
            tracker=tracker if tracker.endswith('.yaml') else f'{tracker}.yaml',
            save=False,
            stream=True,
            verbose=False,
            vid_stride=frame_skip,
            persist=True,
            **kwargs
        )

        frame_data = []   # for _merge_fragmented_tracks
        frame_boxes = []  # [{raw_track_id: (x1, y1, x2, y2, conf)}, ...]

        for result in results_stream:
            boxes = result.boxes
            track_ids = []
            track_positions = {}
            track_confs = {}
            box_data = {}

            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int).tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                xywh = boxes.xywh.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for i, tid in enumerate(track_ids):
                    track_positions[tid] = (float(xywh[i][0]), float(xywh[i][1]))
                    track_confs[tid] = float(confs[i])
                    box_data[tid] = (
                        int(xyxy[i][0]), int(xyxy[i][1]),
                        int(xyxy[i][2]), int(xyxy[i][3]),
                        float(confs[i]),
                    )

            frame_data.append({
                'frame': len(frame_data),
                'track_ids': track_ids,
                'track_positions': track_positions,
                'track_confs': track_confs,
            })
            frame_boxes.append(box_data)

        # --- Merge fragmented track IDs ---
        _, id_map = self._merge_fragmented_tracks(
            frame_data, max_gap_frames=max_gap_frames, max_dist_px=max_dist_px
        )
        # --- Filter spurious (short or low-confidence) groups ---
        id_map, dropped = self._filter_spurious_groups(
            frame_data, id_map, min_duration=min_duration, min_mean_conf=min_mean_conf
        )
        if dropped:
            print(f"   Dropped {len(dropped)} spurious track group(s) "
                  f"(duration<{min_duration} or mean_conf<{min_mean_conf})")
        # Display IDs ordered by first-appearance frame, so Bear 1 = first bear seen
        track_first_frame = {}
        for fd in frame_data:
            for tid in fd['track_ids']:
                if tid not in track_first_frame:
                    track_first_frame[tid] = fd['frame']
        group_first_frame = {}
        for raw, root in id_map.items():
            if root not in group_first_frame or track_first_frame[raw] < group_first_frame[root]:
                group_first_frame[root] = track_first_frame[raw]
        ordered_roots = sorted(group_first_frame.keys(), key=lambda r: group_first_frame[r])
        group_to_display = {root: i + 1 for i, root in enumerate(ordered_roots)}

        # BGR color palette — one consistent color per display ID
        palette = [
            (233, 180, 86), (0, 159, 230), (115, 158, 0),
            (66, 228, 240), (178, 114, 0), (0, 94, 213),
            (167, 121, 204), (255, 255, 0), (255, 0, 255), (0, 255, 0),
        ]

        # --- Pass 2: read original video, render with remapped IDs ---
        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = src_fps / frame_skip if frame_skip > 1 else src_fps

        output_dir = Path(PREDICTIONS_DIR) / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        avi_path = output_dir / f"{video_path.stem}.avi"

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(str(avi_path), fourcc, out_fps, (width, height))

        processed = 0
        src_frame = 0
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            if src_frame % frame_skip == 0:
                if processed < len(frame_boxes):
                    for raw_id, (x1, y1, x2, y2, conf_score) in frame_boxes[processed].items():
                        if raw_id not in id_map:
                            continue  # filtered as spurious (noise/low-conf)
                        group_id = id_map[raw_id]
                        display_id = group_to_display.get(group_id, raw_id)
                        color = palette[(display_id - 1) % len(palette)]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = f"Bear {display_id}  {conf_score:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(img, label, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                writer.write(img)
                processed += 1
            src_frame += 1

        cap.release()
        writer.release()

        # --- Export per-bear trajectories as JSON ---
        from collections import defaultdict
        raw_to_display = {
            raw: group_to_display[root] for raw, root in id_map.items() if root in group_to_display
        }
        display_to_raws = defaultdict(list)
        for raw, disp in raw_to_display.items():
            display_to_raws[disp].append(raw)

        bear_trajectories = defaultdict(list)
        for frame_idx, box_data in enumerate(frame_boxes):
            for raw_id, (x1, y1, x2, y2, conf_score) in box_data.items():
                disp = raw_to_display.get(raw_id)
                if disp is None:
                    continue
                w = x2 - x1
                h = y2 - y1
                bear_trajectories[disp].append({
                    'frame': frame_idx,
                    'cx': round(x1 + w / 2, 1),
                    'cy': round(y1 + h / 2, 1),
                    'w': w,
                    'h': h,
                    'conf': round(conf_score, 3),
                    'raw_id': raw_id,
                })

        trajectory_payload = {
            'video': video_path.name,
            'total_frames': len(frame_boxes),
            'fps': src_fps,
            'bears': {
                f'bear_{disp}': {
                    'raw_track_ids': sorted(display_to_raws[disp]),
                    'num_detections': len(bear_trajectories[disp]),
                    'first_frame': bear_trajectories[disp][0]['frame'] if bear_trajectories[disp] else None,
                    'last_frame': bear_trajectories[disp][-1]['frame'] if bear_trajectories[disp] else None,
                    'trajectory': bear_trajectories[disp],
                }
                for disp in sorted(bear_trajectories.keys())
            },
        }
        trajectory_path = output_dir / 'trajectories.json'
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory_payload, f, indent=2)
        print(f"   Saved trajectories: {trajectory_path}")

        # Convert AVI to MP4 for better browser streaming
        out_video = avi_path
        mp4_path = avi_path.with_suffix(".mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(avi_path), "-c:v", "libx264", "-c:a", "aac", str(mp4_path)],
                check=True, capture_output=True
            )
            avi_path.unlink()
            out_video = mp4_path
            print(f"\n✓ Tracked video saved (MP4): {out_video}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"\n✓ Tracked video saved (AVI): {out_video}")
            if not mp4_path.exists():
                print("  (Install ffmpeg to get MP4 for browser playback)")

        return None, output_dir

    def batch_track_bears(self, video_paths=None, video_dir=None, pattern='*.mkv',
                         conf=0.25, frame_skip=30, classes=21, tracker='bytetrack',
                         verbose=False, save_results=True):
        """
        Track bears in multiple videos (batch processing with ByteTrack)
        
        Args:
            video_paths: List of video file paths
            video_dir: Directory containing videos (alternative to video_paths)
            pattern: Glob pattern for videos in video_dir (default: *.mkv)
            conf: Confidence threshold
            frame_skip: Process every Nth frame
            classes: Class ID(s) to detect
            tracker: Tracker name (default: bytetrack)
            verbose: Print detailed info for each video
            
        Returns:
            dict with batch results
        """
        # Get video list
        if video_paths is None:
            if video_dir is None:
                raise ValueError("Either video_paths or video_dir must be provided")
            
            video_dir = Path(video_dir)
            if not video_dir.is_absolute():
                video_dir = RAW_DATA_DIR / video_dir
            
            # Support multiple video formats
            if pattern == '*':
                video_paths = []
                for ext in ['*.mkv', '*.mp4', '*.avi', '*.mov']:
                    video_paths.extend(video_dir.glob(ext))
            else:
                video_paths = list(video_dir.glob(pattern))
            
            if not video_paths:
                raise ValueError(f"No videos found in {video_dir} matching '{pattern}'")
        
        print(f"\n{'='*70}")
        print(f"BATCH BEAR TRACKING: {len(video_paths)} videos with {tracker}")
        print(f"{'='*70}\n")

        # Process each video with tracking
        results = []
        for video_path in video_paths:
            try:
                stats = self.track_bears_in_video(
                    video_path=video_path,
                    conf=conf,
                    frame_skip=frame_skip,
                    classes=classes,
                    tracker=tracker,
                    verbose=verbose
                )
                results.append(stats)
                print(f"  ✓ {stats['video_name']}: {stats['unique_bears_tracked']} bears (tracked)")
            except Exception as e:
                print(f"  ✗ {Path(video_path).name}: Error - {e}")
                results.append({
                    'video_name': Path(video_path).name,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Print summary
        successful = [r for r in results if 'error' not in r]
        print(f"\n{'='*70}")
        print(f"SUMMARY: Processed {len(successful)}/{len(video_paths)} videos successfully")
        print(f"{'='*70}")

        batch_results = {
            'videos': results,
            'total': len(video_paths),
            'successful': len(successful),
            'failed': len(video_paths) - len(successful)
        }

        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = PREDICTIONS_DIR / 'batch_counting' / f'batch_{timestamp}'
            output_dir.mkdir(parents=True, exist_ok=True)

            json_path = output_dir / 'batch_results.json'
            with open(json_path, 'w') as f:
                json.dump(make_json_safe(batch_results), f, indent=2)

            if successful:
                df = pd.DataFrame([{
                    'video_name': v['video_name'],
                    'unique_bears': v['unique_bears_tracked'],
                    'max_bears_in_frame': v['max_bears_in_frame'],
                    'avg_bears_per_frame': v['avg_bears_per_frame'],
                    'total_detections': v['total_detections'],
                    'frames_analyzed': v['frames_analyzed'],
                    'processing_time_sec': v['processing_time'],
                } for v in successful])
                csv_path = output_dir / 'batch_summary.csv'
                df.to_csv(csv_path, index=False)

            print(f"\n📁 Results saved to: {output_dir}")
            print(f"  - {json_path.name}")
            if successful:
                print(f"  - {csv_path.name}")

        return batch_results

    def batch_count_bears(self, video_paths=None, video_dir=None, pattern='*.mkv',
                         conf=0.25, frame_skip=30, classes=21, ground_truth=None,
                         save_results=True):
        """
        Count bears in multiple videos (batch processing)
        
        Args:
            video_paths: List of video file paths
            video_dir: Directory containing videos (alternative to video_paths)
            pattern: Glob pattern for videos in video_dir (default: *.mkv)
            conf: Confidence threshold
            frame_skip: Process every Nth frame
            classes: Class ID(s) to detect
            ground_truth: Dict mapping video names to actual bear counts
            save_results: Save results to JSON and CSV
            
        Returns:
            dict with batch results and aggregate statistics
        """

        # Get video list
        if video_paths is None:
            if video_dir is None:
                raise ValueError("Either video_paths or video_dir must be provided")
            
            video_dir = Path(video_dir)
            if not video_dir.is_absolute():
                video_dir = RAW_DATA_DIR / video_dir
            
            video_paths = list(video_dir.glob(pattern))
            
            if not video_paths:
                raise ValueError(f"No videos found in {video_dir} matching '{pattern}'")
        
        print(f"\n{'='*70}")
        print(f"BATCH BEAR COUNTING: {len(video_paths)} videos")
        print(f"{'='*70}")

        # Process each video
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': str(self.model_path),
            'config': {
                'confidence': conf,
                'frame_skip': frame_skip,
                'classes': classes
            },
            'videos': []
        }

        for video_path in tqdm(video_paths, desc="Processing videos"):
            try:
                # Count bears in this video
                stats = self.count_bears_in_video(
                    video_path=video_path,
                    conf=conf,
                    frame_skip=frame_skip,
                    classes=classes,
                    verbose=False
                )
                
                # Add ground truth if available
                if ground_truth and stats['video_name'] in ground_truth:
                    stats['ground_truth'] = ground_truth[stats['video_name']]
                    stats['accuracy'] = (
                        stats['unique_bear_estimate'] == stats['ground_truth']
                    )
                
                results['videos'].append(stats)
                
                # Print progress
                print(f"  ✓ {stats['video_name']}: {stats['unique_bear_estimate']} bears")
                
            except Exception as e:
                print(f"  ❌ Error processing {Path(video_path).name}: {e}")
                results['videos'].append({
                    'video_name': Path(video_path).name,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Calculate aggregate statistics
        results['aggregate'] = self._calculate_aggregate_stats(results['videos'])
        
        # Save results if requested
        if save_results:
            self._save_batch_results(results)
        
        # Print summary
        self._print_batch_summary(results)
        
        return results

    def _calculate_aggregate_stats(self, video_results):
        """Calculate statistics across all videos"""
        successful = [v for v in video_results if 'error' not in v]
        
        if not successful:
            return {
                'total_videos': len(video_results),
                'successful_videos': 0,
                'failed_videos': len(video_results)
            }
        
        aggregate = {
            'total_videos': len(video_results),
            'successful_videos': len(successful),
            'failed_videos': len(video_results) - len(successful),
            'total_frames_analyzed': sum(v['frames_analyzed'] for v in successful),
            'total_detections': sum(v['total_detections'] for v in successful),
            'total_unique_bears': sum(v['unique_bear_estimate'] for v in successful),
            'avg_bears_per_video': np.mean([v['unique_bear_estimate'] for v in successful]),
            'max_bears_single_video': max(v['max_bears_in_frame'] for v in successful),
            'total_processing_time': sum(v['processing_time'] for v in successful)
        }
        
        # Calculate accuracy if ground truth available
        videos_with_gt = [v for v in successful if 'ground_truth' in v]
        if videos_with_gt:
            correct = sum(1 for v in videos_with_gt if v.get('accuracy', False))
            aggregate['accuracy'] = correct / len(videos_with_gt)
            aggregate['correct_predictions'] = correct
            aggregate['total_ground_truth_videos'] = len(videos_with_gt)
        
        return aggregate

    def _save_batch_results(self, results):
        """Save batch results to files"""
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PREDICTIONS_DIR / 'batch_counting' / f'batch_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
        

        # Save full JSON
        json_path = output_dir / 'batch_results.json'
        json_safe_results = make_json_safe(results)
        with open(json_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        # Save summary CSV
        successful = [v for v in results['videos'] if 'error' not in v]
        if successful:
            df = pd.DataFrame([{
                'video_name': v['video_name'],
                'unique_bears': v['unique_bear_estimate'],
                'max_bears_in_frame': v['max_bears_in_frame'],
                'avg_bears_per_frame': v['avg_bears_per_frame'],
                'total_detections': v['total_detections'],
                'frames_analyzed': v['frames_analyzed'],
                'processing_time_sec': v['processing_time'],
                'ground_truth': v.get('ground_truth', None),
                'correct': '✓' if v.get('accuracy', False) else ('✗' if 'ground_truth' in v else 'N/A')
            } for v in successful])
            
            csv_path = output_dir / 'batch_summary.csv'
            df.to_csv(csv_path, index=False)
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"  - {json_path.name}")
        if successful:
            print(f"  - {csv_path.name}")

    def _print_batch_summary(self, results):
        """Print batch processing summary"""
        agg = results.get('aggregate', {})
        
        print(f"\n{'='*70}")
        print("BATCH COUNTING SUMMARY")
        print(f"{'='*70}")
        print(f"\n📊 Videos Processed: {agg.get('total_videos', 0)}")
        print(f"  ✓ Successful: {agg.get('successful_videos', 0)}")
        print(f"  ✗ Failed: {agg.get('failed_videos', 0)}")
        
        print(f"\n🐻 Total Unique Bears: {agg.get('total_unique_bears', 0)}")
        print(f"📈 Average Bears/Video: {agg.get('avg_bears_per_video', 0):.2f}")
        print(f"📍 Max Bears (Single Video): {agg.get('max_bears_single_video', 0)}")
        print(f"📊 Total Detections: {agg.get('total_detections', 0)}")
        
        if 'accuracy' in agg:
            print(f"\n🎯 Accuracy: {agg['accuracy']*100:.1f}%")
            print(f"   ({agg['correct_predictions']}/{agg['total_ground_truth_videos']} correct)")
        
        print(f"\n⏱️  Total Processing Time: {agg.get('total_processing_time', 0):.1f}s")
        print(f"{'='*70}\n")

    def _save_prediction_metadata(self, video_path, output_dir, conf, results):
        """Save prediction metadata"""
        # 只遍历一次流式生成器，同时收集帧数和检测数
        total_frames = 0
        total_detections = 0
        
        for result in results:
            total_frames += 1
            boxes = result.boxes
            if boxes is not None:
                total_detections += len(boxes)

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'video': str(video_path),
            'model': str(self.model_path),
            'confidence_threshold': conf,
            'total_frames': total_frames,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)