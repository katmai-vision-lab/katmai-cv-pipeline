from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

# Allow running as a script (python3 scripts/extract_frames.py) by ensuring repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.common import (  # noqa: E402
    ToolMissing,
    ensure_dir,
    ffprobe_duration_seconds,
    run,
    which_or_raise,
)


def _parse_timecode_to_seconds(tc: str) -> float:
    """
    Parse HH:MM:SS(.mmm) to seconds, or accept plain seconds like "12.345".
    """
    tc = tc.strip()
    if not tc:
        raise ValueError("empty timecode")
    if ":" not in tc:
        return float(tc)
    parts = tc.split(":")
    if len(parts) != 3:
        raise ValueError(f"bad timecode: {tc}")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = float(parts[2])
    return hh * 3600.0 + mm * 60.0 + ss


def extract_segment_frames(
    *,
    video: Path,
    start_time: str,
    end_time: str,
    out_dir: Path,
    fps: float,
    scale_width: int | None,
    jpg_quality: int,
    image_format: str,
) -> None:
    which_or_raise("ffmpeg")
    ensure_dir(out_dir)

    start_s = _parse_timecode_to_seconds(start_time)
    end_s = _parse_timecode_to_seconds(end_time)
    dur = end_s - start_s
    if dur <= 0:
        raise ValueError(f"片段时长 <= 0（start={start_time}, end={end_time}）")

    vf_parts: list[str] = []
    if fps > 0:
        vf_parts.append(f"fps={fps}")
    if scale_width and scale_width > 0:
        vf_parts.append(f"scale={scale_width}:-2")
    # JPEG on some ffmpeg builds is picky; force a safe pixel format.
    if image_format.lower() in {"jpg", "jpeg"}:
        vf_parts.append("format=yuvj420p")
    vf = ",".join(vf_parts) if vf_parts else "null"

    if image_format.lower() in {"jpg", "jpeg"}:
        out_pattern = out_dir / "%06d.jpg"
    elif image_format.lower() == "png":
        out_pattern = out_dir / "%06d.png"
    else:
        raise ValueError("--image-format 仅支持 jpg 或 png")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        start_time,
        "-i",
        str(video),
        "-t",
        f"{dur:.3f}",
        "-vf",
        vf,
        "-an",
    ]
    if image_format.lower() in {"jpg", "jpeg"}:
        cmd += ["-q:v", str(jpg_quality), "-pix_fmt", "yuvj420p", "-strict", "-1"]
    cmd += [str(out_pattern)]
    run(cmd)


def _iter_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    imgs.sort()
    return imgs


def _safe_stem(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in s).strip("_")


def _write_empty_label(path: Path) -> None:
    ensure_dir(path.parent)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="从视频抽帧并直接输出 train/val（依赖 ffmpeg/ffprobe）")
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument(
        "--video-id",
        default="",
        help="视频短 id（用于输出命名/前缀）。默认自动使用视频文件名（不含扩展名）。",
    )
    ap.add_argument(
        "--start-time",
        default="0",
        help="从视频的哪个时间开始抽帧（支持 HH:MM:SS.mmm 或秒数，如 12.5）。默认 0。",
    )
    ap.add_argument(
        "--end-time",
        default="",
        help="抽帧到视频的哪个时间结束（支持 HH:MM:SS.mmm 或秒数）。默认空=到视频结尾。",
    )
    ap.add_argument(
        "--yolo-out",
        required=True,
        help="输出为 YOLO 目录结构（只保留 train/val 两套）：<yolo-out>/images/train|val 和 labels/train|val。",
    )
    ap.add_argument("--train-ratio", type=float, default=0.9, help="yolo-out 模式下训练集比例")
    ap.add_argument("--seed", type=int, default=42, help="yolo-out 模式下随机种子")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖 yolo-out 下已有文件")
    ap.add_argument(
        "--every-seconds",
        type=float,
        default=1.0,
        help="每隔多少秒抽一帧（例如 1.0 表示 1fps；0.5 表示 2fps）",
    )
    ap.add_argument("--scale-width", type=int, default=1280, help="缩放宽度（保持比例），0 表示不缩放")
    ap.add_argument("--jpg-quality", type=int, default=2, help="ffmpeg -q:v（2=高质量，31=低质量）")
    ap.add_argument("--image-format", default="jpg", help="输出图片格式：jpg 或 png（png 更稳但更大）")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    video_id = _safe_stem(args.video_id) if args.video_id else _safe_stem(video.stem)
    yolo_out = Path(args.yolo_out).expanduser().resolve()

    if not video.exists():
        raise SystemExit(f"找不到视频：{video}")

    every = float(args.every_seconds)
    if every <= 0:
        raise SystemExit("--every-seconds 必须 > 0")
    fps = 1.0 / every

    scale_width = int(args.scale_width)
    if scale_width == 0:
        scale_width = None

    try:
        if not (0.0 < float(args.train_ratio) < 1.0):
            raise SystemExit("--train-ratio 必须在 (0, 1) 之间")

        total = ffprobe_duration_seconds(video)
        start_s = _parse_timecode_to_seconds(str(args.start_time))
        end_s = _parse_timecode_to_seconds(str(args.end_time)) if str(args.end_time).strip() else total

        if start_s < 0:
            raise SystemExit("--start-time 必须 >= 0")
        if end_s > total + 1e-6:
            raise SystemExit(f"--end-time 超出视频时长（video duration ≈ {total:.3f}s）")
        if end_s <= start_s:
            raise SystemExit("--end-time 必须 > --start-time")

        tmp_dir = yolo_out / "_tmp_extract"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        ensure_dir(tmp_dir)

        print(f"[extract] {start_s:.3f}s -> {end_s:.3f}s -> {tmp_dir}")
        extract_segment_frames(
            video=video,
            start_time=f"{start_s:.3f}",
            end_time=f"{end_s:.3f}",
            out_dir=tmp_dir,
            fps=fps,
            scale_width=scale_width,
            jpg_quality=int(args.jpg_quality),
            image_format=str(args.image_format),
        )

        imgs = _iter_images(tmp_dir)
        if not imgs:
            raise SystemExit("没有抽到任何图片（检查 start/end 是否有效，或 every-seconds 是否过大）")

        random.seed(int(args.seed))
        random.shuffle(imgs)
        n_train = int(len(imgs) * float(args.train_ratio))
        splits = [("train", imgs[:n_train]), ("val", imgs[n_train:])]

        for split_name, split_imgs in splits:
            for img in split_imgs:
                # Put each video's outputs under its own folder to avoid mixing datasets.
                # Layout:
                #   <yolo-out>/<video_id>/images/train|val/<frame>.jpg
                #   <yolo-out>/<video_id>/labels/train|val/<frame>.txt
                out_name = img.name
                out_img = yolo_out / video_id / "images" / split_name / out_name
                out_label = yolo_out / video_id / "labels" / split_name / Path(out_name).with_suffix(".txt").name

                ensure_dir(out_img.parent)
                ensure_dir(out_label.parent)

                if (out_img.exists() or out_label.exists()) and not args.overwrite:
                    continue

                shutil.copy2(img, out_img)
                _write_empty_label(out_label)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(
            f"[yolo] 已写出：{yolo_out / video_id}（train={n_train}, val={len(imgs)-n_train}）"
        )
    except ToolMissing as e:
        raise SystemExit(str(e))
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()


