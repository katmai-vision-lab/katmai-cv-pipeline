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

    @staticmethod
    def canonical_detection_cache(video_path):
        """Stable cache path for a video: predictions/<stem>/detection_cache.json."""
        return Path(PREDICTIONS_DIR) / Path(video_path).stem / "detection_cache.json"

    def predict_video(self, video_path, output_name=None, conf=0.25,
                      classes=None, save=True, use_cache=True, **kwargs):
        """
        Run detection on video

        Args:
            video_path: Path to video file or filename in RAW_DATA_DIR
            output_name: Output directory name
            conf: Confidence threshold
            classes: List of class IDs to detect (None = all classes)
            save: Save annotated video
            use_cache: Return cached output if model/conf unchanged (default True)
            **kwargs: Additional YOLO prediction parameters

        Returns:
            results: YOLO results object
            output_dir: Path to output directory
        """
        video_path = Path(video_path)
        if not video_path.is_absolute():
            video_path = RAW_DATA_DIR / video_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Stable output dir used for caching (avoids timestamp-churn on repeat runs)
        stable_name = f"{video_path.stem}_detect"
        cache_path  = self.canonical_detection_cache(video_path)

        if use_cache and cache_path.exists():
            with open(cache_path) as _f:
                _c = json.load(_f)
            cached_dir = Path(_c.get("output_dir", ""))
            cached_video = next(
                iter(list(cached_dir.glob("*.mp4")) + list(cached_dir.glob("*.avi"))), None
            ) if cached_dir.exists() else None
            if (
                _c.get("model") == self.model_path.name
                and _c.get("conf")  == conf
                and cached_video is not None
            ):
                print(f"   Detection cache hit → {cache_path.parent.name}", flush=True)
                return None, cached_dir

        if output_name is None:
            output_name = stable_name

        print(f"   Detecting bears: {video_path.name}", flush=True)

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
        self._save_prediction_metadata(video_path, output_dir, conf, results)

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as _f:
                json.dump({"model": self.model_path.name, "conf": conf,
                           "output_dir": str(output_dir)}, _f)
            print(f"   Detection cache saved → {cache_path.parent.name}", flush=True)

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
    def _merge_fragmented_tracks(frame_data, max_gap_frames=3600, max_dist_px=150,
                                  cooccurrence_tolerance_frames=60,
                                  cooccurrence_artifact_iou=0.3,
                                  decisions=None):
        """
        Post-process track IDs to merge fragments from the same bear.

        A bear that briefly disappears (occlusion, enters water) may lose its track
        and reappear with a new ID. Two tracks are merged when:
          1. They are not "really" co-occurring (see below)
          2. One ends before the other starts (temporally sequential, possibly with
             a brief overlap caused by detection-artifact double-bbox)
          3. The gap between them is within max_gap_frames
          4. The last known position of the earlier track is within max_dist_px of
             the first known position of the later track

        Co-occurrence tolerance: a pair (a, b) is considered "really" co-occurring
        (i.e. different individuals) only if it appeared in the same frame for at
        least `cooccurrence_tolerance_frames` frames OR was apart by more than
        max_dist_px on average during the overlap. Brief & spatially-close
        co-occurrence is treated as a detection-artifact double bbox of the same
        animal and tolerated for merge.

        Returns (merged_unique_count, {track_id: group_id})
        """
        from collections import defaultdict

        track_first_frame = {}
        track_last_frame = {}
        track_first_pos = {}
        track_last_pos = {}
        co_occur_count = defaultdict(int)
        co_occur_dist_sum = defaultdict(float)
        co_occur_dist_n = defaultdict(int)
        co_occur_iou_sum = defaultdict(float)
        co_occur_iou_n = defaultdict(int)

        def _iou(box_a, box_b):
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
            area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
            union = area_a + area_b - inter
            return (inter / union) if union > 0 else 0.0

        for fd in frame_data:
            frame = fd['frame']
            ids = fd['track_ids']
            positions = fd.get('track_positions', {})
            boxes = fd.get('track_boxes', {})

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
                    pair = (a, b)
                    co_occur_count[pair] += 1
                    if a in positions and b in positions:
                        ax, ay = positions[a]
                        bx, by = positions[b]
                        co_occur_dist_sum[pair] += ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                        co_occur_dist_n[pair] += 1
                    if a in boxes and b in boxes:
                        co_occur_iou_sum[pair] += _iou(boxes[a], boxes[b])
                        co_occur_iou_n[pair] += 1

        def cooccur_info(a, b):
            pair = (min(a, b), max(a, b))
            n = co_occur_count.get(pair, 0)
            mean_dist = None
            mean_iou = None
            if co_occur_dist_n.get(pair, 0) > 0:
                mean_dist = co_occur_dist_sum[pair] / co_occur_dist_n[pair]
            if co_occur_iou_n.get(pair, 0) > 0:
                mean_iou = co_occur_iou_sum[pair] / co_occur_iou_n[pair]
            return n, mean_dist, mean_iou

        def is_real_cooccurrence(a, b):
            """True if (a,b) is genuinely two different individuals. False if it
            looks like a detection-artifact double-bbox of one animal.

            Tolerated as artifact when ANY of:
              - bboxes substantially overlap (mean IoU >= cooccurrence_artifact_iou) —
                strongest signal, regardless of duration
              - brief co-occurrence (< tolerance) AND spatially close (< max_dist_px)
            """
            n, mean_dist, mean_iou = cooccur_info(a, b)
            if n == 0:
                return False
            # IoU signal: bboxes overlap a lot → same animal double-detected
            if mean_iou is not None and mean_iou >= cooccurrence_artifact_iou:
                return False
            # Count + center-distance signal: brief & spatially close → tolerate
            if n < cooccurrence_tolerance_frames:
                if mean_dist is not None and mean_dist <= max_dist_px:
                    return False
            return True

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
            """Reject merge if any member of one group really co-occurs with any
            member of the other (artifact overlaps are tolerated)."""
            for ma in members[root_a]:
                for mb in members[root_b]:
                    if ma == mb:
                        continue
                    if is_real_cooccurrence(ma, mb):
                        return False
            return True

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            parent[rx] = ry
            members[ry] |= members[rx]
            del members[rx]

        # Pre-compute the set of pairs confirmed to be different individuals so
        # the candidate loop below can skip them in O(1).
        co_occurring = {
            (min(a, b), max(a, b))
            for i, a in enumerate(track_ids)
            for b in track_ids[i + 1:]
            if is_real_cooccurrence(a, b)
        }

        # Evaluate candidate pairs in order of (temporal gap, spatial dist) so the
        # most confident merges happen first — this reduces bad transitive chains.
        candidates = []
        for i, a in enumerate(track_ids):
            for b in track_ids[i + 1:]:
                pair = (min(a, b), max(a, b))
                if pair in co_occurring:
                    if decisions is not None:
                        decisions.append({
                            'a': a, 'b': b, 'result': 'rejected',
                            'reason': 'co-occurred in at least one frame (different bears)',
                        })
                    continue

                # Determine temporal ordering. With co-occurrence tolerance, the
                # ranges may briefly overlap (artifact), so don't require strict
                # sequentiality — fall back to whichever started first.
                if track_last_frame[a] < track_first_frame[b]:
                    early, late = a, b
                elif track_last_frame[b] < track_first_frame[a]:
                    early, late = b, a
                elif track_first_frame[a] <= track_first_frame[b]:
                    early, late = a, b
                else:
                    if decisions is not None:
                        decisions.append({
                            'a': a, 'b': b, 'result': 'rejected',
                            'reason': 'time ranges overlap but never co-occurred (ambiguous)',
                        })
                    continue

                gap = max(track_first_frame[late] - track_last_frame[early], 0)
                if gap > max_gap_frames:
                    if decisions is not None:
                        decisions.append({
                            'a': early, 'b': late, 'gap': gap, 'result': 'rejected',
                            'reason': f'gap ({gap}) > max_gap_frames ({max_gap_frames})',
                        })
                    continue

                # If they had any tolerated-artifact co-occurrence, the "transition"
                # distance is between two positions in the same frame, not at a real
                # gap — skip the dist check (the IoU/proximity check during overlap
                # already validated they're the same animal).
                pair_key = (min(a, b), max(a, b))
                n_overlap = co_occur_count.get(pair_key, 0)
                if n_overlap > 0:
                    dist = 0.0
                elif early in track_last_pos and late in track_first_pos:
                    lx, ly = track_last_pos[early]
                    fx, fy = track_first_pos[late]
                    dist = ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5
                    if dist > max_dist_px:
                        if decisions is not None:
                            decisions.append({
                                'a': early, 'b': late, 'gap': gap,
                                'dist_px': round(dist, 1), 'result': 'rejected',
                                'reason': f'dist_px ({dist:.1f}) > max_dist_px ({max_dist_px})',
                            })
                        continue
                else:
                    dist = max_dist_px

                candidates.append((gap, dist, early, late))

        candidates.sort()  # prefer smallest gap, then smallest dist

        for _gap, _dist, a, b in candidates:
            ra, rb = find(a), find(b)
            if ra == rb:
                if decisions is not None:
                    decisions.append({
                        'a': a, 'b': b, 'gap': _gap, 'dist_px': round(_dist, 1),
                        'result': 'redundant',
                        'reason': 'already in same group via earlier merge',
                    })
                continue
            if not groups_can_merge(ra, rb):
                if decisions is not None:
                    decisions.append({
                        'a': a, 'b': b, 'gap': _gap, 'dist_px': round(_dist, 1),
                        'result': 'rejected',
                        'reason': 'transitive co-occurrence with another raw ID in same group',
                    })
                continue
            if decisions is not None:
                decisions.append({
                    'a': a, 'b': b, 'gap': _gap, 'dist_px': round(_dist, 1),
                    'result': 'merged',
                    'reason': f'gap={_gap}f, dist={_dist:.1f}px (both within thresholds)',
                })
            union(a, b)

        groups = set(find(tid) for tid in track_ids)
        return len(groups), {tid: find(tid) for tid in track_ids}

    @staticmethod
    def _filter_spurious_groups(frame_data, id_map, min_duration=30, min_mean_conf=0.80,
                                long_duration=500, drop_log=None):
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
                if drop_log is not None:
                    drop_log[root] = {
                        'duration': duration, 'mean_conf': round(mean_conf, 3),
                        'reason': f'duration ({duration}) < min_duration ({min_duration})',
                    }
            elif duration < long_duration and mean_conf < min_mean_conf:
                dropped.add(root)
                if drop_log is not None:
                    drop_log[root] = {
                        'duration': duration, 'mean_conf': round(mean_conf, 3),
                        'reason': (f'duration ({duration}) < long_duration ({long_duration}) '
                                   f'and mean_conf ({mean_conf:.3f}) < min_mean_conf ({min_mean_conf})'),
                    }

        kept = {tid: root for tid, root in id_map.items() if root not in dropped}
        return kept, dropped


    @staticmethod
    def canonical_tracking_cache(video_path):
        """Stable cache path for a video: predictions/<stem>/tracking_cache.json."""
        return Path(PREDICTIONS_DIR) / Path(video_path).stem / "tracking_cache.json"

    def compute_display_boxes(self, video_path, conf=0.25, iou=0.7,
                              classes=None, tracker=None, show_progress=False,
                              cache_path=None):
        """
        Run ByteTrack + merge + filter on a video.

        Returns list[dict[int, tuple]]: one dict per frame mapping
        display_id -> (x1, y1, x2, y2, conf).  Display IDs start at 1,
        ordered by first appearance, with spurious tracks removed.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                with open(cache_path) as _f:
                    _c = json.load(_f)
                print(f"   Loaded tracking cache: {cache_path}")
                return [
                    {int(k): tuple(v) for k, v in frame.items()}
                    for frame in _c["display_frame_boxes"]
                ]

        if classes is None:
            classes = [0]
        if tracker is None:
            tracker = str(TRACKERS_CONFIG_DIR / "bytetrack.yaml")

        total = None
        if show_progress:
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        results_stream = self.model.track(
            source=str(video_path),
            conf=conf,
            iou=iou,
            classes=classes,
            tracker=tracker,
            save=False,
            stream=True,
            verbose=False,
            persist=True,
        )

        if show_progress:
            from tqdm import tqdm
            results_stream = tqdm(results_stream, total=total, desc="  Tracking")

        frame_data, frame_boxes = [], []
        for result in results_stream:
            track_ids, positions, boxes = [], {}, {}
            if result.boxes.id is not None:
                tids  = result.boxes.id.cpu().numpy().astype(int).tolist()
                xyxy  = result.boxes.xyxy.cpu().numpy()
                xywh  = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for i, tid in enumerate(tids):
                    positions[tid] = (float(xywh[i][0]), float(xywh[i][1]))
                    boxes[tid] = (
                        int(xyxy[i][0]), int(xyxy[i][1]),
                        int(xyxy[i][2]), int(xyxy[i][3]),
                        float(confs[i]),
                    )
                track_ids = tids
            frame_data.append({"frame": len(frame_data), "track_ids": track_ids,
                               "track_positions": positions})
            frame_boxes.append(boxes)

        _, id_map_pre = self._merge_fragmented_tracks(frame_data)
        id_map, dropped = self._filter_spurious_groups(frame_data, id_map_pre)
        if dropped:
            print(f"   Dropped {len(dropped)} spurious track group(s)")

        track_first_frame = {}
        for fd in frame_data:
            for tid in fd["track_ids"]:
                if tid not in track_first_frame:
                    track_first_frame[tid] = fd["frame"]
        group_first_frame = {}
        for raw, root in id_map.items():
            ff = track_first_frame.get(raw, 0)
            if root not in group_first_frame or ff < group_first_frame[root]:
                group_first_frame[root] = ff
        ordered_roots = sorted(group_first_frame, key=lambda r: group_first_frame[r])
        group_to_display = {root: i + 1 for i, root in enumerate(ordered_roots)}

        display_boxes = []
        for rb in frame_boxes:
            disp = {}
            for raw_id, bbox in rb.items():
                gid = id_map.get(raw_id)
                if gid is None:
                    continue
                disp[group_to_display[gid]] = bbox
            display_boxes.append(disp)

        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cap_meta = cv2.VideoCapture(str(video_path))
            _cache = {
                "src_fps":      cap_meta.get(cv2.CAP_PROP_FPS) or 30.0,
                "total_frames": int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT)),
                "frame_w":      int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "frame_h":      int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "display_frame_boxes": [
                    {str(k): list(v) for k, v in frame.items()}
                    for frame in display_boxes
                ],
            }
            cap_meta.release()
            with open(cache_path, "w") as _f:
                json.dump(_cache, _f)
            print(f"   Tracking cache saved → {cache_path}")

        return display_boxes

    def track_and_save_video(self, video_path, output_name=None, conf=0.25,
                             frame_skip=1, classes=None, tracker='bytetrack',
                             imgsz=1280,
                             max_gap_frames=3600, max_dist_px=150,
                             cooccurrence_tolerance_frames=60,
                             cooccurrence_artifact_iou=0.3,
                             min_duration=150, min_mean_conf=0.80,
                             min_raw_duration=None, **kwargs):
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

        # Stable cache for the saved output video (conf + frame_skip keyed)
        _save_cache_path = (
            Path(PREDICTIONS_DIR) / video_path.stem / "track_save_cache.json"
        )
        if output_name is None:
            if _save_cache_path.exists():
                with open(_save_cache_path) as _f:
                    _sc = json.load(_f)
                _cached_dir = Path(_sc.get("output_dir", ""))
                _cached_video = next(
                    iter(list(_cached_dir.glob("*.mp4")) + list(_cached_dir.glob("*.avi"))), None
                ) if _cached_dir.exists() else None
                if (
                    _sc.get("model")      == self.model_path.name
                    and _sc.get("conf")       == conf
                    and _sc.get("frame_skip") == frame_skip
                    and _sc.get("imgsz")      == imgsz
                    and _cached_video is not None
                ):
                    print(f"   Tracking save cache hit → {_save_cache_path.parent.name}", flush=True)
                    return None, _cached_dir

            output_name = f"{video_path.stem}_track"

        print(f"\n📹 Tracking & saving video: {video_path.name}")
        print(f"   Output: {PREDICTIONS_DIR / output_name}\n")

        _cap = cv2.VideoCapture(str(video_path))
        _total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _cap.release()
        _processed = _total_frames // frame_skip if frame_skip > 1 else _total_frames

        # --- Pass 1: stream to collect tracking data (no video writing) ---
        results_stream = self.model.track(
            source=str(video_path),
            conf=conf,
            classes=classes,
            tracker=tracker if tracker.endswith('.yaml') else f'{tracker}.yaml',
            imgsz=imgsz,
            save=False,
            stream=True,
            verbose=False,
            vid_stride=frame_skip,
            persist=True,
            **kwargs
        )

        frame_data = []   # for _merge_fragmented_tracks
        frame_boxes = []  # [{raw_track_id: (x1, y1, x2, y2, conf)}, ...]

        print(f"   Pass 1/2: tracking {_processed} frames …", flush=True)
        for result in results_stream:
            _n = len(frame_data)
            if _n > 0 and _n % 100 == 0:
                _pct = min(100, int(_n / max(_processed, 1) * 100))
                print(f"   Pass 1/2: {_pct}%  (frame {_n}/{_processed})", flush=True)
            boxes = result.boxes
            track_ids = []
            track_positions = {}
            track_confs = {}
            box_data = {}

            track_boxes = {}
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int).tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                xywh = boxes.xywh.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for i, tid in enumerate(track_ids):
                    track_positions[tid] = (float(xywh[i][0]), float(xywh[i][1]))
                    track_confs[tid] = float(confs[i])
                    track_boxes[tid] = (
                        float(xyxy[i][0]), float(xyxy[i][1]),
                        float(xyxy[i][2]), float(xyxy[i][3]),
                    )
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
                'track_boxes': track_boxes,
            })
            frame_boxes.append(box_data)
        print(f"   Pass 1/2: done ({len(frame_data)} frames)", flush=True)

        # --- Pre-merge filter: drop raw tracks that, on their own, are shorter
        # than min_raw_duration. This prevents brief detections from being used
        # as merge "bridges" that pull short fragments into long-lived bears.
        if min_raw_duration is None:
            min_raw_duration = min_duration
        pre_merge_dropped = {}
        if min_raw_duration > 0:
            raw_first = {}
            raw_last = {}
            for fd in frame_data:
                for tid in fd['track_ids']:
                    if tid not in raw_first:
                        raw_first[tid] = fd['frame']
                    raw_last[tid] = fd['frame']
            short_raws = {tid for tid in raw_first
                          if raw_last[tid] - raw_first[tid] + 1 < min_raw_duration}
            for tid in short_raws:
                pre_merge_dropped[tid] = {
                    'first_frame': raw_first[tid],
                    'last_frame': raw_last[tid],
                    'duration': raw_last[tid] - raw_first[tid] + 1,
                    'reason': f'duration < min_raw_duration ({min_raw_duration})',
                }
            for fd in frame_data:
                fd['track_ids'] = [t for t in fd['track_ids'] if t not in short_raws]
                for key in ('track_positions', 'track_confs', 'track_boxes'):
                    if key in fd:
                        fd[key] = {t: v for t, v in fd[key].items() if t not in short_raws}
            for box_data in frame_boxes:
                for tid in list(box_data.keys()):
                    if tid in short_raws:
                        del box_data[tid]
            if pre_merge_dropped:
                print(f"   Pre-merge filter dropped {len(pre_merge_dropped)} raw track(s) "
                      f"shorter than {min_raw_duration} frames")

        # --- Merge fragmented track IDs ---
        merge_decisions = []
        _, id_map_pre_filter = self._merge_fragmented_tracks(
            frame_data, max_gap_frames=max_gap_frames, max_dist_px=max_dist_px,
            cooccurrence_tolerance_frames=cooccurrence_tolerance_frames,
            cooccurrence_artifact_iou=cooccurrence_artifact_iou,
            decisions=merge_decisions,
        )
        # --- Filter spurious (short or low-confidence) groups ---
        filter_drop_log = {}
        id_map, dropped = self._filter_spurious_groups(
            frame_data, id_map_pre_filter,
            min_duration=min_duration, min_mean_conf=min_mean_conf,
            drop_log=filter_drop_log,
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

        # Save canonical tracking cache so feeding analysis can reuse without re-tracking
        _cache_path = self.canonical_tracking_cache(video_path)
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        cap_meta = cv2.VideoCapture(str(video_path))
        _cache = {
            "src_fps":      cap_meta.get(cv2.CAP_PROP_FPS) or 30.0,
            "total_frames": int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT)),
            "frame_w":      int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "frame_h":      int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "display_frame_boxes": [
                {str(group_to_display[id_map[raw]]): list(bbox)
                 for raw, bbox in fb.items() if raw in id_map and id_map[raw] in group_to_display}
                for fb in frame_boxes
            ],
        }
        cap_meta.release()
        with open(_cache_path, "w") as _f:
            json.dump(_cache, _f)
        print(f"   Tracking cache saved → {_cache_path}")

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

        _total_pass2 = len(frame_boxes)
        print(f"   Pass 2/2: rendering {_total_pass2} frames …", flush=True)
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
                if processed > 0 and processed % 100 == 0:
                    _pct = min(100, int(processed / max(_total_pass2, 1) * 100))
                    print(f"   Pass 2/2: {_pct}%  (frame {processed}/{_total_pass2})", flush=True)
            src_frame += 1

        print(f"   Pass 2/2: done ({processed} frames)", flush=True)
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

        _max_per_frame = max(
            (sum(1 for raw_id in fb if raw_id in id_map) for fb in frame_boxes),
            default=0
        )
        trajectory_payload = {
            'video': video_path.name,
            'total_frames': len(frame_boxes),
            'unique_bears': len(bear_trajectories),
            'max_per_frame': _max_per_frame,
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

        # --- Export merge decision report ---
        track_first = {}
        track_last = {}
        track_confs_acc = defaultdict(list)
        for fd in frame_data:
            for tid in fd['track_ids']:
                if tid not in track_first:
                    track_first[tid] = fd['frame']
                track_last[tid] = fd['frame']
                if tid in fd.get('track_confs', {}):
                    track_confs_acc[tid].append(fd['track_confs'][tid])

        def _bear_label_for(raw_id):
            disp = raw_to_display.get(raw_id)
            if disp is not None:
                return f'Bear {disp}'
            root = id_map_pre_filter.get(raw_id)
            if root in dropped:
                return 'dropped (filtered)'
            return None

        raw_tracks_report = {}
        for tid in sorted(track_first.keys()):
            confs = track_confs_acc.get(tid, [])
            raw_tracks_report[str(tid)] = {
                'first_frame': track_first[tid],
                'last_frame': track_last[tid],
                'duration': track_last[tid] - track_first[tid] + 1,
                'mean_conf': round(sum(confs) / len(confs), 3) if confs else 0.0,
                'final_bear': _bear_label_for(tid),
            }

        # Per-bear merge chain: only the accepted merges that built each final group.
        groups_pre_filter = defaultdict(list)
        for raw, root in id_map_pre_filter.items():
            groups_pre_filter[root].append(raw)
        bear_merge_chain = {}
        for disp in sorted(display_to_raws.keys()):
            members = set(display_to_raws[disp])
            chain = [
                {'from': d['a'], 'to': d['b'],
                 'gap': d.get('gap'), 'dist_px': d.get('dist_px')}
                for d in merge_decisions
                if d['result'] == 'merged' and d['a'] in members and d['b'] in members
            ]
            bear_merge_chain[f'Bear {disp}'] = {
                'raw_ids': sorted(display_to_raws[disp]),
                'merge_chain': chain,
            }

        filtered_groups_report = {}
        for root, info in filter_drop_log.items():
            filtered_groups_report[str(root)] = {
                'raw_ids': sorted(groups_pre_filter.get(root, [])),
                **info,
            }

        merge_report = {
            'params': {
                'max_gap_frames': max_gap_frames,
                'max_dist_px': max_dist_px,
                'min_duration': min_duration,
                'min_mean_conf': min_mean_conf,
            },
            'summary': {
                'raw_track_count': len(track_first),
                'final_bear_count': len(display_to_raws),
                'filtered_group_count': len(filter_drop_log),
                'merged_pairs': sum(1 for d in merge_decisions if d['result'] == 'merged'),
                'rejected_pairs': sum(1 for d in merge_decisions if d['result'] == 'rejected'),
            },
            'raw_tracks': raw_tracks_report,
            'final_bears': bear_merge_chain,
            'filtered_groups': filtered_groups_report,
            'decisions': merge_decisions,
        }
        report_path = output_dir / 'merge_report.json'
        with open(report_path, 'w') as f:
            json.dump(merge_report, f, indent=2, default=make_json_safe)
        print(f"   Saved merge report: {report_path}")

        # Convert AVI to MP4 for better browser streaming
        out_video = avi_path
        mp4_path = avi_path.with_suffix(".mp4")
        try:
            print("   Converting AVI → MP4 …", flush=True)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(avi_path), "-c:v", "libx264", "-c:a", "aac", str(mp4_path)],
                check=True, capture_output=True
            )
            avi_path.unlink()
            out_video = mp4_path
            print(f"   Conversion done → {mp4_path.name}", flush=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        _save_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_save_cache_path, "w") as _f:
            json.dump({"model": self.model_path.name, "conf": conf,
                       "frame_skip": frame_skip, "imgsz": imgsz,
                       "output_dir": str(output_dir)}, _f)
        print(f"   Track save cache saved → {_save_cache_path.parent.name}", flush=True)

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
        
        print(f"   Processing {len(video_paths)} video(s) …", flush=True)

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
        
        successful = [r for r in results if 'error' not in r]

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
        
        print(f"   Processing {len(video_paths)} video(s) …", flush=True)

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
        # Iterate the streaming generator once, accumulating frame and detection counts
        total_frames = 0
        total_detections = 0
        
        max_per_frame = 0
        for result in results:
            total_frames += 1
            boxes = result.boxes
            if boxes is not None:
                n = len(boxes)
                total_detections += n
                if n > max_per_frame:
                    max_per_frame = n

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'video': str(video_path),
            'model': str(self.model_path),
            'confidence_threshold': conf,
            'total_frames': total_frames,
            'total_detections': total_detections,
            'max_per_frame': max_per_frame,
            'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)