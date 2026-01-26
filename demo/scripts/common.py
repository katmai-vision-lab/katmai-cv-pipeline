from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ToolMissing(RuntimeError):
    pass


def which_or_raise(tool: str) -> str:
    found = shutil.which(tool)
    if not found:
        raise ToolMissing(
            f"找不到外部工具 `{tool}`。请先安装并确保在 PATH 中可用。\n"
            f"- macOS: `brew install ffmpeg`（ffmpeg 会包含 ffprobe）\n"
            f"- Ubuntu/Debian: `sudo apt-get install -y ffmpeg`"
        )
    return found


def run(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_capture(cmd: list[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True
    )


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Scene:
    video_id: str
    scene_id: str
    start_time: str  # HH:MM:SS.mmm
    end_time: str    # HH:MM:SS.mmm
    notes: str = ""


def load_scenes_csv(path: Path) -> list[Scene]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    scenes: list[Scene] = []
    for r in rows:
        scenes.append(
            Scene(
                video_id=(r.get("video_id") or "").strip(),
                scene_id=(r.get("scene_id") or "").strip(),
                start_time=(r.get("start_time") or "").strip(),
                end_time=(r.get("end_time") or "").strip(),
                notes=(r.get("notes") or "").strip(),
            )
        )
    return scenes


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ffprobe_duration_seconds(video: Path) -> float:
    which_or_raise("ffprobe")
    p = run_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ]
    )
    s = (p.stdout or "").strip()
    if not s:
        raise RuntimeError("ffprobe 未返回 duration")
    return float(s)


