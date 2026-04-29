"""
src/cli.py  —  Katmai CV Pipeline interactive TUI
Run: python -m src.cli
"""

import glob
import json
import readline
import sys
import termios
import tty
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — prevents popup windows in TUI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import PREDICTIONS_DIR, TRAINED_BEAR_DETECTOR_PATH

MODEL    = str(TRAINED_BEAR_DETECTOR_PATH)
TEST_DIR = PROJECT_ROOT.parent / "katmai-cv-data" / "test"
EXTS     = ("*.mp4", "*.mkv", "*.avi", "*.mov")

# Tab-complete file paths in all c.input() calls
def _path_complete(text, state):
    return (glob.glob(text + "*") + [None])[state]
readline.set_completer(_path_complete)
readline.set_completer_delims(" \t\n;")
readline.parse_and_bind("tab: complete")

c = Console()


# ── Raw single-keypress ───────────────────────────────────────────────────────

def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch

def _key(prompt: str, valid: str, default: str = "") -> str:
    """Print prompt then wait for a single valid keypress (no Enter needed)."""
    opts = "/".join(
        ch.upper() if ch == default else ch
        for ch in valid
    )
    c.print(f"  [cyan]{prompt}[/cyan]  [dim]({opts})[/dim] : ", end="")
    while True:
        ch = _getch().lower()
        if ch in valid.lower() or ch in ("\r", "\n"):
            result = default if ch in ("\r", "\n") else ch
            c.print(result)
            return result

def _confirm(question: str, default: bool = True) -> bool:
    result = _key(question, "yn", "y" if default else "n")
    return result == "y"


# ── Free-text input ───────────────────────────────────────────────────────────

def _ask(label: str, default: str = "", desc: str = "") -> str:
    """Inline prompt — label (short_default)  ·  desc : _"""
    disp = Path(default).name if default and ("/" in default or "\\" in default) else default
    hint = f" [dim]({disp})[/dim]" if default else ""
    note = f"  [dim]· {desc}[/dim]" if desc else ""
    val  = c.input(f"  [cyan]{label}[/cyan]{hint}{note} : ").strip()
    return val or default


# ── Display helpers ───────────────────────────────────────────────────────────

def _header(title: str, sub: str = ""):
    body = Text(title, style="bold white")
    if sub:
        body.append(f"\n{sub}", style="dim")
    c.print(Panel(body, border_style="cyan", padding=(0, 2)))

def _ok(msg):   c.print(f"\n[bold green]✓[/bold green]  {msg}")
def _err(msg):  c.print(f"\n[bold red]✗[/bold red]  {msg}")
def _step(msg): c.print(f"\n[bold cyan]→[/bold cyan]  {msg}")

def _config(rows: dict):
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column(style="dim cyan", min_width=18)
    t.add_column(style="white")
    for k, v in rows.items():
        t.add_row(k, str(v))
    c.print(t)


# ── Video picker ──────────────────────────────────────────────────────────────

def _pick_video(directory: Path = TEST_DIR) -> Optional[str]:
    videos: list[Path] = []
    if directory.exists():
        for ext in EXTS:
            videos.extend(sorted(directory.glob(ext)))
    videos = videos[:9]  # single-keypress supports 1–9

    if videos:
        c.print("\n  [dim]Available videos:[/dim]")
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column(style="bold cyan", width=4, no_wrap=True)
        t.add_column(style="white", max_width=60, no_wrap=True, overflow="ellipsis")
        for i, v in enumerate(videos, 1):
            t.add_row(str(i), v.name)
        t.add_row("p", "Enter a custom path")
        c.print(t)
        c.print()

        valid = "".join(str(i) for i in range(1, len(videos) + 1)) + "p"
        c.print(f"  [cyan]Pick[/cyan]  [dim]({'/'.join(valid)})[/dim] : ", end="")
        while True:
            ch = _getch().lower()
            if ch in valid:
                c.print(ch)
                if ch != "p":
                    return str(videos[int(ch) - 1])
                break

    return _ask("Video path") or None


# ── Welcome & main menu ───────────────────────────────────────────────────────

def _welcome():
    c.clear()
    banner = Text("🐻  Katmai CV Pipeline\n", style="bold white")
    banner.append("Computer Vision · Katmai National Park & Preserve", style="dim")
    c.print(Panel(banner, border_style="bold cyan", padding=(1, 4)))

def _menu() -> str:
    c.print()
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="bold cyan", width=4)
    t.add_column(style="white", min_width=16)
    t.add_column(style="dim")
    t.add_row("1", "Track bears",    "ByteTrack · annotated output video with bounding boxes")
    t.add_row("2", "Count bears",    "per-video bear counts · optional unique-bear tracking")
    t.add_row("3", "Evaluate model", "mAP · precision · recall · counting accuracy vs ground truth")
    t.add_row("4", "Train model",    "fine-tune YOLOv8n on a new labeled dataset")
    t.add_row("q", "Quit", "")
    c.print(Panel(t, title="[bold]Main Menu[/bold]", border_style="cyan", padding=(0, 1)))
    c.print(f"\n  [cyan]Select[/cyan]  [dim](1/2/3/4/q)[/dim] : ", end="")
    while True:
        ch = _getch()
        if ch in "1234q":
            c.print(ch)
            return ch


# ── 1. Track ──────────────────────────────────────────────────────────────────

def track_bears():
    c.print(); _header("Track Bears", "ByteTrack  ·  saves annotated output video")

    video   = _pick_video()
    if not video: return _err("No video selected.")

    model   = _ask("Model weights",  MODEL,    "path to .pt weights file")
    conf    = float(_ask("Confidence", "0.25", "min detection score to accept a box  (0–1)"))
    skip    = int(_ask("Frame skip",   "1",    "process every Nth frame  (1 = every frame, smoothest video)"))
    tracker = _ask("Tracker",        "bytetrack", "bytetrack or botsort")

    c.print()
    _config({"video": Path(video).name, "model": Path(model).name,
             "confidence": conf, "frame_skip": skip, "tracker": tracker})

    if not _confirm("Run?"):
        return

    _step("Loading model and starting tracker…")
    try:
        from src.detection.detector import BearDetector
        det = BearDetector(model_path=model)
        _, out_dir = det.track_and_save_video(
            video_path=video, conf=conf, frame_skip=skip, tracker=tracker)
        out = next(iter(list(out_dir.glob("*.mp4")) + list(out_dir.glob("*.avi"))), out_dir)
        _ok(f"Saved → {out}")
    except KeyboardInterrupt:
        c.print("\n  [yellow]Cancelled.[/yellow]")
    except Exception as e:
        _err(str(e))


# ── 2. Count ──────────────────────────────────────────────────────────────────

def count_bears():
    c.print(); _header("Count Bears", "Batch counting  ·  optional ByteTrack unique-bear estimates")

    mode = _key("Input mode", "vd", "v")   # v = video, d = directory
    if mode == "v":
        video  = _pick_video()
        vpaths = [video] if video else None
        vdir, pat = None, "*.mp4"
    else:
        vdir   = _ask("Video directory", str(TEST_DIR), "folder containing video files")
        pat    = _ask("File pattern", "*.mp4", "glob to match video files  e.g. *.mkv")
        vpaths = None

    model    = _ask("Model weights",  MODEL,   "path to .pt weights file")
    conf     = float(_ask("Confidence", "0.25", "min detection score  (0–1)"))
    skip     = int(_ask("Frame skip",   "30",   "sample every Nth frame  (30 ≈ 1 fps at 30fps video)"))
    tracking = _confirm("Use ByteTrack for unique-bear estimates?", default=False)
    gt_path  = _ask("Ground-truth JSON", "", "optional  {\"video.mp4\": N}  for accuracy scoring")

    ground_truth = None
    if gt_path:
        with open(gt_path) as f:
            ground_truth = json.load(f)

    c.print()
    _config({"mode": "tracking" if tracking else "counting",
             "model": Path(model).name, "confidence": conf, "frame_skip": skip})

    if not _confirm("Run?"):
        return

    _step("Initializing detector…")
    try:
        from src.detection.detector import BearDetector
        det = BearDetector(model_path=model)
        if tracking:
            res = det.batch_track_bears(
                video_paths=vpaths, video_dir=vdir, pattern=pat,
                conf=conf, frame_skip=skip)
            _show_track_results(res)
        else:
            res = det.batch_count_bears(
                video_paths=vpaths, video_dir=vdir, pattern=pat,
                conf=conf, frame_skip=skip, ground_truth=ground_truth)
            _show_count_results(res)
    except KeyboardInterrupt:
        c.print("\n  [yellow]Cancelled.[/yellow]")
    except Exception as e:
        _err(str(e))


def _show_track_results(res: dict):
    t = Table(title="Tracking Results", box=box.ROUNDED, border_style="cyan")
    t.add_column("Video",        style="white", max_width=44, no_wrap=True, overflow="ellipsis")
    t.add_column("Unique Bears", style="bold green", justify="right")
    t.add_column("Max/Frame",    justify="right")
    t.add_column("Avg/Frame",    justify="right")
    t.add_column("",             justify="center")
    for v in res.get("videos", []):
        ok = "[green]✓[/green]" if "error" not in v else "[red]✗[/red]"
        t.add_row(Path(v.get("video_name","?")).name,
                  str(v.get("unique_bears_tracked","?")),
                  str(v.get("max_bears_in_frame","?")),
                  f"{v.get('avg_bears_per_frame',0):.2f}", ok)
    c.print(); c.print(t)
    c.print(f"  [dim]total {res['total']}  ·  ok {res['successful']}  ·  failed {res['failed']}[/dim]")


def _show_count_results(res: dict):
    t = Table(title="Count Results", box=box.ROUNDED, border_style="cyan")
    t.add_column("Video",       style="white", max_width=44, no_wrap=True, overflow="ellipsis")
    t.add_column("Max/Frame",   style="bold green", justify="right")
    t.add_column("Unique Est.", justify="right")
    t.add_column("Avg/Frame",   justify="right")
    t.add_column("Total Det.",  justify="right")
    t.add_column("",            justify="center")
    for v in res.get("videos", []):
        ok = "[green]✓[/green]" if v.get("status") == "success" else "[red]✗[/red]"
        t.add_row(Path(v.get("video_name","?")).name,
                  str(v.get("max_bears_in_frame","?")),
                  str(v.get("unique_bear_estimate","?")),
                  f"{v.get('avg_bears_per_frame',0):.2f}",
                  str(v.get("total_detections","?")), ok)
    c.print(); c.print(t)
    agg = res.get("aggregate", {})
    if agg:
        c.print(f"  [dim]total {agg.get('total_videos','?')}  ·  "
                f"failed {agg.get('failed_videos',0)}  ·  "
                f"time {agg.get('total_processing_time',0):.1f}s[/dim]")


# ── 3. Evaluate ───────────────────────────────────────────────────────────────

def evaluate():
    c.print(); _header("Evaluate Model", "mAP · precision · recall · counting accuracy")

    c.print("\n  [dim]Modes:[/dim]")
    c.print("  [cyan]d[/cyan]  dataset   [dim]· YOLO native val — full mAP / precision / recall on labeled data[/dim]")
    c.print("  [cyan]c[/cyan]  counting  [dim]· counting accuracy vs a known ground-truth bear count[/dim]")
    c.print("  [cyan]s[/cyan]  simple    [dim]· frame-by-frame detection stats + optional confidence plot[/dim]")
    mode  = _key("Mode", "dcs", "s")
    model = _ask("Model weights", MODEL, "path to .pt weights file")
    conf  = float(_ask("Confidence", "0.25", "min detection score  (0–1)"))

    try:
        from src.detection.detector import BearDetector
        from src.detection.metrics import VideoEvaluator
        out_dir = PREDICTIONS_DIR / "evaluations"
        out_dir.mkdir(parents=True, exist_ok=True)
        det = BearDetector(model_path=model)
        ev  = VideoEvaluator(det, conf_threshold=conf)

        if mode == "d":
            data = _ask("Dataset YAML", "", "bear.yaml with train/val image paths and class names")
            if not data: return _err("Dataset YAML required.")
            _config({"mode": "dataset", "data": data, "model": Path(model).name, "conf": conf})
            if not _confirm("Run?"): return
            _step("Running YOLO validation…")
            m = ev.evaluate_dataset_with_yolo(data_yaml=data, save_dir=out_dir)
            _show_metrics(m)

        elif mode == "c":
            video = _pick_video()
            if not video: return _err("No video selected.")
            gt   = int(_ask("Ground-truth count", "", "the actual number of bears in this video"))
            skip = int(_ask("Frame skip", "1", "process every Nth frame for evaluation"))
            _config({"video": Path(video).name, "ground_truth": gt,
                     "frame_skip": skip, "conf": conf})
            if not _confirm("Run?"): return
            _step("Evaluating counting accuracy…")
            ev.evaluate_counting_accuracy(
                video_path=video, ground_truth_counts=gt,
                frame_skip=skip, save_dir=out_dir)
            _ok(f"Results saved → {out_dir}")

        else:  # simple
            video = _pick_video()
            if not video: return _err("No video selected.")
            gt_raw = _ask("Ground-truth count", "", "optional — leave blank to skip accuracy check")
            plot   = _confirm("Generate confidence plot?", default=True)
            _config({"video": Path(video).name, "conf": conf, "plot": plot})
            if not _confirm("Run?"): return
            _step("Running frame-by-frame analysis…")
            from src.detection.evaluate import evaluate_video, plot_evaluation
            df   = evaluate_video(det, video,
                                  ground_truth_count=int(gt_raw) if gt_raw else None,
                                  conf=conf)
            stem = Path(video).stem
            df.to_csv(out_dir / f"simple_eval_{stem}.csv", index=False)
            if plot:
                plot_evaluation(df, out_dir / f"simple_eval_{stem}.png")
            _ok(f"Saved → {out_dir}")

    except KeyboardInterrupt:
        c.print("\n  [yellow]Cancelled.[/yellow]")
    except Exception as e:
        _err(str(e))


def _show_metrics(m: dict):
    t = Table(title="Evaluation Metrics", box=box.ROUNDED, border_style="cyan")
    t.add_column("Metric",  style="cyan")
    t.add_column("Value",   style="bold green", justify="right")
    t.add_row("Precision",    f"{m.get('precision',0):.4f}")
    t.add_row("Recall",       f"{m.get('recall',0):.4f}")
    t.add_row("F1 Score",     f"{m.get('f1',0):.4f}")
    t.add_row("mAP@0.5",      f"{m.get('map50',0):.4f}")
    t.add_row("mAP@0.5:0.95", f"{m.get('map50_95',0):.4f}")
    c.print(); c.print(t)


# ── 4. Train ──────────────────────────────────────────────────────────────────

def train():
    c.print(); _header("Train Model", "Fine-tune YOLOv8n on new labeled data")

    data   = _ask("Dataset YAML", "",    "bear.yaml with train/val paths — required")
    if not data: return _err("Dataset YAML is required.")
    base   = _ask("Base model",   MODEL, "start from these weights  (pretrained or fine-tuned)")
    epochs = int(_ask("Epochs",   "50",  "passes over the full dataset  (3–5 for a quick demo)"))
    batch  = int(_ask("Batch size","8",  "images per gradient step  (lower if GPU runs out of memory)"))
    imgsz  = int(_ask("Image size","640","resize all frames to NxN pixels before training"))
    name   = _ask("Experiment name", "bear_detector_demo", "output folder under models/trained/")
    resume = _confirm("Resume from last checkpoint?", default=False)

    c.print()
    _config({"data": data, "base_model": Path(base).name, "epochs": epochs,
             "batch": batch, "imgsz": imgsz, "name": name, "resume": resume})

    if not _confirm("Start training?"):
        return

    _step("Initializing training run…")
    try:
        from src.detection.detector import BearDetector
        det = BearDetector(model_path=base)
        res = det.train(data_yaml=data, name=name, resume=resume,
                        epochs=epochs, batch=batch, imgsz=imgsz)
        map50 = res.results_dict.get("metrics/mAP50(B)", "N/A")
        _ok(f"Training complete  ·  mAP@0.5 = {map50}")
        c.print(f"  [dim]Best weights → {det.model_path}[/dim]")
    except KeyboardInterrupt:
        c.print("\n  [yellow]Cancelled.[/yellow]")
    except Exception as e:
        _err(str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _welcome()
    dispatch = {"1": track_bears, "2": count_bears, "3": evaluate, "4": train}
    while True:
        try:
            ch = _menu()
        except (KeyboardInterrupt, EOFError):
            break
        if ch == "q":
            break
        dispatch[ch]()
    c.print("\n[dim]Goodbye.[/dim]\n")


if __name__ == "__main__":
    main()
