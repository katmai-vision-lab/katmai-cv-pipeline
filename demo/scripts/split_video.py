from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import ToolMissing, ensure_dir, ffprobe_duration_seconds, run, which_or_raise


def main() -> None:
    ap = argparse.ArgumentParser(description="把整段视频按固定时长切成多个视频片段（依赖 ffmpeg）")
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--out-dir", default="data/video_segments", help="输出目录")
    ap.add_argument("--segment-seconds", type=float, required=True, help="每段时长（秒），例如 60")
    ap.add_argument("--format", default="mp4", help="输出容器格式：mp4 或 mkv")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    seg = float(args.segment_seconds)

    if not video.exists():
        raise SystemExit(f"找不到视频：{video}")
    if seg <= 0:
        raise SystemExit("--segment-seconds 必须 > 0")

    try:
        which_or_raise("ffmpeg")
        total = ffprobe_duration_seconds(video)
        ensure_dir(out_dir)

        # Use segment muxer; keyframe alignment depends on source. For ML data management it's fine.
        # If you need exact boundaries, you can re-encode, but it's slower.
        ext = args.format.lower()
        if ext not in {"mp4", "mkv"}:
            raise SystemExit("--format 仅支持 mp4 或 mkv")

        out_pattern = out_dir / f"{video.stem}_%04d.{ext}"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-map",
            "0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            f"{seg:.3f}",
            "-reset_timestamps",
            "1",
            str(out_pattern),
        ]
        run(cmd)
        print(f"已切割：{out_dir}（总时长约 {total:.1f}s，每段 {seg:.1f}s）")
    except ToolMissing as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()


