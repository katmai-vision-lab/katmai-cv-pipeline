#!/bin/bash
set -euo pipefail

# macOS double-click runnable script.
# It will prompt you to choose a video file, then run the extraction pipeline.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

choose_file() {
  /usr/bin/osascript <<'APPLESCRIPT'
set theFile to choose file with prompt "选择一个视频文件（mp4/mkv 等）"
POSIX path of theFile
APPLESCRIPT
}

prompt_value() {
  local prompt="$1"
  local default="$2"
  /usr/bin/osascript <<APPLESCRIPT
text returned of (display dialog "$prompt" default answer "$default")
APPLESCRIPT
}

VIDEO_PATH="$(choose_file)"
EVERY_SECONDS="$(prompt_value "每隔多少秒抽一张？（例如 1.0=每秒一张，2.0=每两秒一张）" "1.0")"
SCALE_WIDTH="$(prompt_value "输出图片缩放宽度？（例如 1280；填 0 表示不缩放）" "1280")"
START_TIME="$(prompt_value "从视频什么时间开始抽帧？（支持 HH:MM:SS.mmm 或秒数；默认 0）" "0")"
END_TIME="$(prompt_value "抽到什么时间结束？（支持 HH:MM:SS.mmm 或秒数；留空=到视频结尾）" "")"

YOLO_OUT="$ROOT_DIR/data/datasets/yolo"

echo ""
echo "== Running extract_frames =="
echo "video: $VIDEO_PATH"
echo "yolo_out: $YOLO_OUT"
echo "every_seconds: $EVERY_SECONDS"
echo "scale_width: $SCALE_WIDTH"
echo "start_time: $START_TIME"
echo "end_time: $END_TIME"
echo ""

cd "$ROOT_DIR"
python3 -m scripts.extract_frames \
  --video "$VIDEO_PATH" \
  --yolo-out "$YOLO_OUT" \
  --every-seconds "$EVERY_SECONDS" \
  --scale-width "$SCALE_WIDTH" \
  --start-time "$START_TIME" \
  --end-time "$END_TIME"

echo ""
echo "== Done =="
echo "Output: $YOLO_OUT/<video_name>/images/train|val and labels/train|val"
echo ""
read -r -p "按回车关闭窗口..."


