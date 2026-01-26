## 一键运行（不想每次手敲命令行）

### 方式 A（最简单）：双击 `run_extract_frames.command`

1) 先在终端执行一次（只需要一次）让它可运行：

```bash
chmod +x "/Users/rachel/CourseWork/WIN1/Capstone/demo/run_extract_frames.command"
```

2) 之后在 Finder 里双击 `run_extract_frames.command`

它会弹窗让你：

- 选择视频文件
- 输入抽帧间隔（秒）
- 输入缩放宽度

输出会自动写到：`data/datasets/yolo/images/train|val` 和 `data/datasets/yolo/labels/train|val`

### 方式 B（在 Cursor 里点 Run）

打开 `.vscode/launch.json`，把 `--video` 后面的路径改成你的新视频绝对路径，然后在 Cursor 的 Run/Debug 里选择：

- `Extract frames -> YOLO (promptless, edit args)`


