"""
src/cli.py  —  Katmai CV Pipeline interactive TUI
Run: python -m src.cli
"""

import glob
import json
import readline
import sys
import os
import platform
if platform.system() != "Windows":
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

class GoBack(Exception):
    pass


# ── Raw single-keypress ───────────────────────────────────────────────────────

def _getch() -> str:
    if os.name == "nt":
        import msvcrt
        while True:
            ch = msvcrt.getch()
            # Swallow the second byte of special keys (arrows, F-keys, etc.)
            if ch in (b'\x00', b'\xe0'):
                msvcrt.getch()
                continue
            ch = ch.decode("utf-8", errors="ignore")
            if ch == "\x03":
                raise KeyboardInterrupt
            return ch
    else:
        import termios
        import tty
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
    c.print(f"  [cyan]{prompt}[/cyan]  [dim]({opts} · b=back)[/dim] : ", end="")
    while True:
        ch = _getch().lower()
        if ch == "b":
            c.print("b")
            raise GoBack
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
    if val.lower() == "b":
        raise GoBack
    return val or default


# ── Display helpers ───────────────────────────────────────────────────────────

def _header(title: str, sub: str = ""):
    body = Text(title, style="bold white")
    if sub:
        body.append(f"\n{sub}", style="dim")
    c.print(Panel(body, border_style="cyan", padding=(0, 2)))
    c.print("  [dim]b · back to main menu[/dim]")

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

        valid = "".join(str(i) for i in range(1, len(videos) + 1)) + "pb"
        c.print(f"  [cyan]Pick[/cyan]  [dim]({'/'.join(valid[:-1])} · b=back)[/dim] : ", end="")
        while True:
            ch = _getch().lower()
            if ch in valid:
                c.print(ch)
                if ch == "b":
                    raise GoBack
                if ch != "p":
                    return str(videos[int(ch) - 1])
                break

    return _ask("Video path") or None


def _pick_videos(directory: Path = TEST_DIR) -> Optional[list]:
    """Multi-select video picker — returns list of path strings or None."""
    videos: list[Path] = []
    if directory.exists():
        for ext in EXTS:
            videos.extend(sorted(directory.glob(ext)))
    videos = videos[:9]

    if videos:
        c.print("\n  [dim]Available videos:[/dim]")
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column(style="bold cyan", width=4, no_wrap=True)
        t.add_column(style="white", max_width=60, no_wrap=True, overflow="ellipsis")
        for i, v in enumerate(videos, 1):
            t.add_row(str(i), v.name)
        c.print(t)
        c.print()
        valid_nums = set(str(i) for i in range(1, len(videos) + 1))
        raw = c.input(
            f"  [cyan]Pick[/cyan]  [dim](e.g. 1 3  or  all  ·  b=back)[/dim] : "
        ).strip()
        if raw.lower() == "b":
            raise GoBack
        if raw.lower() == "all":
            return [str(v) for v in videos]
        tokens  = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
        chosen  = [t for t in tokens if t in valid_nums]
        ignored = [t for t in tokens if t not in valid_nums]
        if ignored:
            c.print(f"  [yellow]⚠[/yellow]  Ignored unrecognised entries: {', '.join(ignored)}")
        if chosen:
            return [str(videos[int(n) - 1]) for n in chosen]
        return None

    raw = c.input("  [cyan]Video paths[/cyan]  [dim](space-separated  ·  b=back)[/dim] : ").strip()
    if raw.lower() == "b":
        raise GoBack
    paths = [p for p in raw.split() if p]
    return paths if paths else None


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
    t.add_column(style="white", min_width=20)
    t.add_column(style="dim")
    t.add_row("1", "Detect bears",         "raw YOLO inference · annotated output video with bear bounding boxes")
    t.add_row("2", "Track bears",          "ByteTrack · annotated output video with bounding boxes")
    t.add_row("3", "Batch count bears",     "fast frame-sampled counts across many videos · no output video")
    t.add_row("4", "Detect feeding events","VLM frame analysis · timestamped behavior JSON")
    t.add_row("5", "Count salmon jumps",   "CV-based jump detection · jump count + timestamps")
    t.add_row("6", "Fetch environmental data", "RAWS · NADP · USGS · weather, precipitation, hydrology")
    t.add_row("7", "Evaluate model",       "mAP · precision · recall · counting accuracy vs ground truth")
    t.add_row("8", "Train model",          "fine-tune YOLOv8n on a new labeled dataset")
    t.add_row("q", "Quit", "")
    c.print(Panel(t, title="[bold]Main Menu[/bold]", border_style="cyan", padding=(0, 1)))
    c.print(f"\n  [cyan]Select[/cyan]  [dim](1/2/3/4/5/6/7/8/q)[/dim] : ", end="")
    while True:
        ch = _getch().lower()
        if ch in "12345678q":
            c.print(ch)
            return ch


# ── 1. Detect ─────────────────────────────────────────────────────────────────

def detect_bears():
    c.print(); _header("Detect Bears", "Raw YOLO inference  ·  saves annotated output video with bear bounding boxes")

    video = _pick_video()
    if not video: return _err("No video selected.")

    model = _ask("Model weights", MODEL, "path to .pt weights file")
    conf  = float(_ask("Confidence", "0.25", "min detection score to accept a box  (0–1)"))

    c.print()
    _config({"video": Path(video).name, "model": Path(model).name, "confidence": conf})

    if not _confirm("Run?"):
        return

    _step("Loading model and running detection…")
    try:
        from src.detection.detector import BearDetector
        det = BearDetector(model_path=model)
        _, out_dir = det.predict_video(video_path=video, conf=conf)
        out = next(iter(list(out_dir.glob("*.mp4")) + list(out_dir.glob("*.avi"))), out_dir)
        meta_path = out_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        c.print()
        t = Table(title="Detection Results", box=box.ROUNDED, border_style="cyan",
                  show_header=False)
        t.add_column(style="dim cyan", min_width=14)
        t.add_column(style="white")
        t.add_row("video",        Path(video).name)
        t.add_row("model",        Path(model).name)
        t.add_row("confidence",   str(conf))
        if meta:
            t.add_row("frames",       str(meta.get("total_frames", "?")))
            t.add_row("detections",   str(meta.get("total_detections", "?")))
            t.add_row("max/frame",    str(meta.get("max_per_frame", "?")))
            t.add_row("avg/frame",    f"{meta.get('avg_detections_per_frame', 0):.2f}")
        t.add_row("status",       "[green]✓ complete[/green]")
        c.print(t)
        _ok(f"Saved → predictions/{out_dir.name}")
    except KeyboardInterrupt:
        c.print("\n  [yellow]Cancelled.[/yellow]")
    except Exception as e:
        _err(str(e))


# ── 2. Track ──────────────────────────────────────────────────────────────────

def track_bears():
    c.print(); _header("Track Bears", "ByteTrack  ·  saves annotated output video")

    video   = _pick_video()
    if not video: return _err("No video selected.")

    model  = _ask("Model weights",  MODEL,    "path to .pt weights file")
    conf   = float(_ask("Confidence", "0.25", "min detection score to accept a box  (0–1)"))
    skip   = int(_ask("Frame skip",   "1",    "process every Nth frame  (1 = every frame, smoothest video)"))
    imgsz  = int(_ask("Image size",   "1280", "resize frames before inference  (640 = faster, 1280 = more accurate)"))

    c.print()
    _config({"video": Path(video).name, "model": Path(model).name,
             "confidence": conf, "frame_skip": skip, "imgsz": imgsz})

    if not _confirm("Run?"):
        return

    _step("Loading model and starting tracker…")
    try:
        from src.detection.detector import BearDetector
        det = BearDetector(model_path=model)
        _, out_dir = det.track_and_save_video(
            video_path=video, conf=conf, frame_skip=skip, imgsz=imgsz, tracker="bytetrack")
        out = next(iter(list(out_dir.glob("*.mp4")) + list(out_dir.glob("*.avi"))), out_dir)
        is_mp4 = str(out).endswith(".mp4")
        traj_path = out_dir / "trajectories.json"
        traj = json.loads(traj_path.read_text()) if traj_path.exists() else {}
        c.print()
        t = Table(title="Tracking Results", box=box.ROUNDED, border_style="cyan",
                  show_header=False)
        t.add_column(style="dim cyan", min_width=14)
        t.add_column(style="white")
        t.add_row("video",         Path(video).name)
        t.add_row("model",         Path(model).name)
        t.add_row("confidence",    str(conf))
        t.add_row("frame skip",    str(skip))
        t.add_row("imgsz",         str(imgsz))
        if traj:
            t.add_row("frames",        str(traj.get("total_frames", "?")))
            t.add_row("unique bears",  str(traj.get("unique_bears", "?")))
            t.add_row("max/frame",     str(traj.get("max_per_frame", "?")))
        t.add_row("format",        "MP4" if is_mp4 else "AVI (install ffmpeg for MP4)")
        t.add_row("status",        "[green]✓ complete[/green]")
        c.print(t)
        _ok(f"Saved → predictions/{out_dir.name}")
    except KeyboardInterrupt:
        c.print("\n  [yellow]Cancelled.[/yellow]")
    except Exception as e:
        _err(str(e))


# ── 3. Count ──────────────────────────────────────────────────────────────────

def count_bears():
    c.print(); _header("Batch Count Bears", "Fast frame-sampled counts across many videos  ·  no output video")

    c.print()
    c.print("  [cyan]v[/cyan]  video files  [dim]· pick one or more specific videos[/dim]")
    c.print("  [cyan]d[/cyan]  directory    [dim]· process all videos in a folder[/dim]")
    mode = _key("Input mode", "vd", "v")
    if mode == "v":
        vpaths = _pick_videos()
        if not vpaths:
            return _err("No videos selected.")
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
        ok = res.get("successful", res.get("aggregate", {}).get("successful_videos", 0))
        total = res.get("total", res.get("aggregate", {}).get("total_videos", 0))
        _ok(f"{ok}/{total} video(s) processed")
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


# ── 4. Detect feeding events ─────────────────────────────────────────────────

def detect_feeding():
    c.print(); _header("Detect Feeding Events",
                       "YOLO + ByteTrack  ·  VLM frame analysis  ·  timestamped behavior JSON")

    video = _pick_video()
    if not video: return _err("No video selected.")

    backend = _key("VLM backend", "maog", "m")
    # m=molmo2, a=anthropic, o=openai, g=gemini
    backend_map = {"m": "molmo2", "a": "anthropic", "o": "openai", "g": "gemini"}
    backend_name = backend_map[backend]

    model    = _ask("YOLO weights",  MODEL,  "path to .pt bear detector weights")
    interval = float(_ask("Sample interval", "0.5", "analyze one frame every N seconds"))
    conf     = float(_ask("Confidence",      "0.25","min detection score  (0–1)"))

    c.print()
    _config({"video": Path(video).name, "backend": backend_name,
             "interval": f"{interval}s", "confidence": conf})

    if not _confirm("Run?"):
        return

    _step(f"Starting analysis with {backend_name} backend…")
    c.print("  [dim]Progress will appear below. This may take several minutes.[/dim]\n")
    try:
        from src.behavior.analyze_feeding import run as run_feeding
        json_path = run_feeding(
            video_path=video,
            interval=interval,
            model=model,
            conf=conf,
            backend=backend_name,
        )
        c.print()
        t = Table(title="Feeding Event Results", box=box.ROUNDED, border_style="cyan",
                  show_header=False)
        t.add_column(style="dim cyan", min_width=14)
        t.add_column(style="white")
        t.add_row("video",    Path(video).name)
        t.add_row("backend",  backend_name)
        t.add_row("interval", f"{interval}s")
        t.add_row("status",   "[green]✓ complete[/green]")
        c.print(t)
        _ok(f"Saved → predictions/{Path(str(json_path)).parent.name}")
        c.print(f"\n  [dim]View with:[/dim]  python -m src.behavior.feeding_viewer "
                f"--video \"{Path(video).name}\" --analysis \"{json_path}\"")
    except KeyboardInterrupt:
        c.print("\n  [yellow]Cancelled.[/yellow]")
    except Exception as e:
        _err(str(e))


# ── 5. Count salmon jumps ─────────────────────────────────────────────────────

def count_salmon_jumps():
    c.print(); _header("Count Salmon Jumps",
                       "MOG2 background subtraction · centroid tracking · tripwire counting")

    video = _pick_video()
    if not video: return _err("No video selected.")

    import os as _os
    has_display = bool(_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"))

    c.print("\n  [dim]Interactive mode opens OpenCV GUI windows for ROI draw and live tuning.[/dim]")
    c.print("  [dim]Fixed params mode runs headless — useful after you know your values.[/dim]\n")

    if has_display:
        interactive = _confirm("Run in interactive mode (ROI selector + trackbars)?", default=True)
    else:
        interactive = False
        c.print("  [yellow]⚠  No display detected — interactive mode unavailable.[/yellow]")
        c.print("  [dim]Using fixed params mode.[/dim]\n")

    roi      = None
    line_y   = None
    var_thr  = 80
    min_area = 800
    blur     = 7
    history  = 300
    skip     = 2
    output   = None

    if not interactive:
        roi_str  = _ask("ROI", "", "x1,y1,x2,y2  e.g. 434,720,710,1062  (blank = full frame)")
        roi      = roi_str.strip() or None
        line_y   = _ask("Tripwire Y", "", "pixel row of the counting line")
        var_thr  = int(_ask("varThreshold", "40",  "MOG2 sensitivity — raise to suppress water noise"))
        min_area = int(_ask("Min area",     "300", "min contour size in pixels"))
        blur     = int(_ask("Blur size",    "7",   "Gaussian kernel size, odd number"))
        history  = int(_ask("History",      "300", "frames used to build the background model"))
        skip     = int(_ask("Frame skip",   "2",   "process every Nth frame"))

    save_output = _confirm("Save annotated output video?", default=False)
    if save_output:
        stem   = Path(video).stem
        output = str(PREDICTIONS_DIR / "salmon_jumps" / f"{stem}_result.mp4")

    c.print()
    _config({
        "video":        Path(video).name,
        "mode":         "interactive (GUI)" if interactive else "fixed params (headless)",
        "roi":          roi or ("(draw with mouse)" if interactive else "(full frame)"),
        "line_y":       line_y or ("(click to place)" if interactive else "(auto 60%)"),
        "varThreshold": var_thr,
        "min_area":     min_area,
        "blur_size":    blur,
        "history":      history,
        "frame_skip":   skip,
        "output":       output or "(none)",
    })
    if not _confirm("Run?"): return

    (PREDICTIONS_DIR / "salmon_jumps").mkdir(parents=True, exist_ok=True)

    # ── Build the CLI command ─────────────────────────────
    script = str(PROJECT_ROOT / "src" / "behavior" / "count_salmon_jumps.py")
    cmd = [sys.executable, script, "--video", video]

    if not interactive:
        # Fixed params — pass everything, suppress GUI
        if roi:      cmd += ["--roi",           roi]
        if line_y:   cmd += ["--line-y",        line_y]
        cmd += ["--var-threshold", str(var_thr)]
        cmd += ["--min-area",      str(min_area)]
        cmd += ["--blur-size",     str(blur)]
        cmd += ["--history",       str(history)]
        cmd += ["--skip-frames",   str(skip)]
        cmd += ["--no-display"]          # headless — no cv2 windows at all
        cmd += ["--no-trackbars"]
    # Interactive: no extra flags — script opens ROI selector + trackbars itself

    if output:
        cmd += ["--output", output]

    _step("Running salmon jump counter…")
    if interactive:
        c.print("  [dim]OpenCV windows will open. Draw ROI → press ENTER, "
                "click tripwire → press ENTER, then press Q to finish.[/dim]\n")
    else:
        c.print("  [dim]Running headless — progress printed below.[/dim]\n")

    # Print the exact command so user can copy-paste it later
    c.print("  [dim]Command:[/dim]")
    c.print(f"  [dim]{' '.join(cmd)}[/dim]\n")

    try:
        import subprocess as _sp
        result = _sp.run(cmd, check=False)

        if result.returncode != 0:
            _err(f"Script exited with code {result.returncode}")
        else:
            if output:
                _ok(f"Annotated video saved → {output}")
            else:
                _ok("Done.")

        # Re-run hint with the values used (or discovered interactively)
        c.print(f"\n  [dim]Re-run with fixed params:[/dim]")
        hint_parts = [
            f'python {script}',
            f'  --video "{video}"',
        ]
        if roi:    hint_parts.append(f"  --roi {roi}")
        if line_y: hint_parts.append(f"  --line-y {line_y}")
        hint_parts += [
            f"  --var-threshold {var_thr}",
            f"  --min-area {min_area}",
        ]
        if output: hint_parts.append(f'  --output "{output}"')
        c.print("  [dim]" + " \\\n".join(hint_parts) + "[/dim]")

    except KeyboardInterrupt:
        c.print("\n  [yellow]Cancelled.[/yellow]")
    except Exception as e:
        _err(str(e))
# ── 6. Fetch environmental data ───────────────────────────────────────────────

def fetch_environmental_data():
    c.print(); _header("Fetch Environmental Data",
                       "RAWS weather  ·  NADP precipitation  ·  USGS hydrology  ·  all fetched together")

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    t.add_column("Source", style="bold cyan",  width=6)
    t.add_column("Station / Site",  style="white",    min_width=34)
    t.add_column("Variables",       style="dim")
    t.add_row("RAWS", "ATHF Three Forks · ACOV Coville · APFA Pfaff Mine",
              "hourly wind, temp, humidity, precip")
    t.add_row("NADP", "AK97 Southwest Alaska",
              "daily precipitation (inches)")
    t.add_row("USGS", "Kvichak River at Igiugig (~106 km from Brooks Falls)",
              "15-min water level, stream flow, water temp")
    c.print(); c.print(t)

    from datetime import date as _date
    date_str = _ask("Date", _date.today().isoformat(), "YYYY-MM-DD  (bear season: Jul–Sep)")
    c.print("  [cyan]Output format[/cyan]  [dim](json/csv)[/dim] : ", end="")
    while True:
        ch = _getch().lower()
        if ch in "jc\r\n":
            fmt_name = "csv" if ch == "c" else "json"
            c.print(fmt_name)
            break

    c.print()
    _config({"date": date_str, "format": fmt_name, "sources": "RAWS + NADP + USGS"})
    if not _confirm("Fetch all three sources?"):
        return

    try:
        from datetime import datetime as _dt
        d = _dt.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return _err(f"Invalid date '{date_str}'. Use YYYY-MM-DD.")

    rows = []  # (source, station_label, n_records, out_path_str, error_str)

    # ── RAWS (all three stations) ─────────────────────────────────────────────
    _step("RAWS — fetching wind / temperature / humidity…")
    try:
        from src.environment.raws_weather import (
            fetch_day as _raws_fetch, save_json as _raws_json, save_csv as _raws_csv,
        )
        raws = _raws_fetch(d.year, d.month, d.day, station_ids=None)
        out  = PREDICTIONS_DIR / "raws_weather" / f"{date_str}_all-stations.{fmt_name}"
        (_raws_csv if fmt_name == "csv" else _raws_json)(raws, out)
        n     = sum(len(v) for v in raws["stations"].values())
        errs  = raws.get("errors", {})
        note  = f"{len(errs)} station(s) offline" if errs else ""
        rows.append(("RAWS", "ATHF / ACOV / APFA", n, str(out), note))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        rows.append(("RAWS", "ATHF / ACOV / APFA", 0, "", str(e)))

    # ── NADP precipitation ────────────────────────────────────────────────────
    _step("NADP — fetching daily precipitation…")
    try:
        from src.environment.nadp_precip import (
            fetch_day as _nadp_fetch, save_json as _nadp_json, save_csv as _nadp_csv,
        )
        nadp = _nadp_fetch(d.year, d.month, d.day, site_ids=["AK97"])
        out  = PREDICTIONS_DIR / "nadp_precip" / f"{date_str}_AK97.{fmt_name}"
        (_nadp_csv if fmt_name == "csv" else _nadp_json)(nadp, out)
        n    = sum(len(v) for v in nadp["sites"].values())
        errs = nadp.get("errors", {})
        note = "; ".join(errs.values()) if errs else ""
        rows.append(("NADP", "AK97", n, str(out), note))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        rows.append(("NADP", "AK97", 0, "", str(e)))

    # ── USGS hydrology ────────────────────────────────────────────────────────
    _step("USGS — fetching water level / stream flow / water temp…")
    try:
        from src.environment.usgs_hydro import (
            fetch_day as _usgs_fetch, save_json as _usgs_json, save_csv as _usgs_csv,
        )
        usgs = _usgs_fetch(date_str)
        out  = PREDICTIONS_DIR / "usgs_hydro" / f"{date_str}.{fmt_name}"
        (_usgs_csv if fmt_name == "csv" else _usgs_json)(usgs, out)
        n    = len(usgs.get("readings", []))
        errs = usgs.get("errors", [])
        note = "; ".join(errs) if errs else ""
        rows.append(("USGS", "Kvichak River (15300500)", n, str(out), note))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        rows.append(("USGS", "Kvichak River (15300500)", 0, "", str(e)))

    # ── Results summary ───────────────────────────────────────────────────────
    c.print()
    t = Table(title="Environmental Data Results", box=box.ROUNDED, border_style="cyan")
    t.add_column("Source",         style="bold cyan", width=6)
    t.add_column("Station / Site", style="white",     min_width=26)
    t.add_column("Records",        style="bold green", justify="right", width=8)
    t.add_column("Saved to",       style="dim",        max_width=46, no_wrap=True, overflow="ellipsis")
    t.add_column("",               width=3,            justify="center")

    all_ok = True
    for source, label, n, path, err in rows:
        if err and n == 0:
            status  = "[red]✗[/red]"
            rec_str = "[dim]—[/dim]"
            path_str = f"[red]{err[:44]}[/red]"
            all_ok = False
        elif err:
            status   = "[yellow]⚠[/yellow]"
            rec_str  = str(n)
            path_str = Path(path).name
        else:
            status   = "[green]✓[/green]"
            rec_str  = str(n)
            path_str = Path(path).name
        t.add_row(source, label, rec_str, path_str, status)

    c.print(t)

    for source, _, n, _, err in rows:
        if err and n > 0:
            c.print(f"  [yellow]⚠[/yellow]  {source}: {err}")

    if all_ok:
        _ok(f"All sources fetched → {PREDICTIONS_DIR}")
    else:
        c.print(f"\n  [yellow]Some sources failed — check errors above.[/yellow]")


# ── 7. Evaluate ───────────────────────────────────────────────────────────────

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
            df = ev.evaluate_counting_accuracy(
                video_path=video, ground_truth_counts=gt,
                frame_skip=skip, save_dir=out_dir)
            accuracy = df['is_correct'].sum() / max(len(df), 1) * 100
            mae      = df['absolute_error'].mean()
            rmse     = (df['absolute_error'] ** 2).mean() ** 0.5
            c.print()
            t = Table(title="Counting Accuracy", box=box.ROUNDED, border_style="cyan")
            t.add_column("Metric", style="cyan")
            t.add_column("Value",  style="bold green", justify="right")
            t.add_row("Frames evaluated", str(len(df)))
            t.add_row("Exact match",      f"{accuracy:.2f}%")
            t.add_row("MAE",              f"{mae:.3f} bears")
            t.add_row("RMSE",             f"{rmse:.3f} bears")
            c.print(t)
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
            c.print()
            t = Table(title="Evaluation Results", box=box.ROUNDED, border_style="cyan")
            t.add_column("Metric", style="cyan")
            t.add_column("Value",  style="bold green", justify="right")
            t.add_row("Total frames",      str(len(df)))
            t.add_row("Frames with bears", str((df['num_bears'] > 0).sum()))
            t.add_row("Avg bears/frame",   f"{df['num_bears'].mean():.2f}")
            t.add_row("Max bears/frame",   str(int(df['num_bears'].max())))
            t.add_row("Avg confidence",    f"{df['avg_confidence'].mean():.2f}")
            if gt_raw:
                match = df['num_bears'].max() == int(gt_raw)
                t.add_row("Ground truth match", "[green]✓[/green]" if match else "[red]✗[/red]")
            c.print(t)
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


# ── 8. Train ──────────────────────────────────────────────────────────────────

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
    dispatch = {
        "1": detect_bears,
        "2": track_bears,
        "3": count_bears,
        "4": detect_feeding,
        "5": count_salmon_jumps,
        "6": fetch_environmental_data,
        "7": evaluate,
        "8": train,
    }
    while True:
        try:
            ch = _menu()
        except (KeyboardInterrupt, EOFError):
            break
        if ch == "q":
            break
        try:
            dispatch[ch]()
        except GoBack:
            pass
        except (KeyboardInterrupt, EOFError):
            break
    c.print("\n[dim]Goodbye.[/dim]\n")


if __name__ == "__main__":
    main()
