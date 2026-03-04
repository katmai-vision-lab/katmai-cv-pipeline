# Salmon Auto-Annotation System

这是从熊自动标注系统改编的**三文鱼检测系统**。

## 主要差异

### vs. 熊系统 (annotation-bear)

| 特性 | 熊系统 | 三文鱼系统 |
|------|--------|------------|
| 模型数量 | 3个 (GDINO + DETR + MegaDetector) | 2个 (GDINO + DETR) |
| MegaDetector | ✅ 启用 | ❌ 禁用（为陆地动物训练） |
| 默认prompt | "bear" | "salmon" |
| 模型权重 | gdino:0.406, detr:0.259, megadet:0.335 | gdino:0.61, detr:0.39 |
| min_agreement | 2 (3个模型中2个) | 1 (2个模型中1个) |

### 为什么禁用MegaDetector？

MegaDetector v5 专门为**陆地野生动物**（熊、鹿、狼等）训练，在以下场景表现很好：
- 森林、草原等陆地环境
- 红外相机陷阱图像
- 四足动物形态

但对**水生生物**表现差：
- 鱼类形态完全不同
- 水面反光、水下环境干扰
- 没有相关训练数据

因此在三文鱼系统中默认禁用，仅使用Grounding DINO和DETR。

## 使用方法

### 基础用法

```bash
# 1. 提取视频帧
python -m src.preprocessing.annotation_salmon.frame_extractor \
  --input salmon_video.mp4 \
  --output data/frames/salmon/ \
  --fps 0.2

# 2. 自动标注
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon/ \
  --output data/auto_labels_salmon/ \
  --review-queue data/review_queue_salmon/ \
  --prompt "salmon" \
  --min-agreement 1 \
  --auto-approve

# 3. 可视化验证
python -m src.preprocessing.annotation_salmon.visualize_labels \
  --images data/frames/salmon/subfolder/ \
  --labels data/auto_labels_salmon/ \
  --output data/visualized_salmon/ \
  --limit 10
```

### 高级：概率校准

如果你有标注好的三文鱼验证集：

```bash
# 训练校准器
python -m src.preprocessing.annotation_salmon.train_calibration \
  --images data/annotation/salmon/images/ \
  --labels data/annotation/salmon/labels/ \
  --output models/calibrators_salmon.pkl \
  --prompt "salmon" \
  --use-megadet False

# 使用校准器标注
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon/ \
  --output data/auto_labels_salmon/ \
  --prompt "salmon" \
  --auto-approve \
  --calibrator models/calibrators_salmon.pkl
```

## 提示词优化

除了 "salmon"，你可以尝试更具体的提示：

```bash
# 通用三文鱼
--prompt "salmon"

# 特定种类
--prompt "chinook salmon"
--prompt "sockeye salmon"  
--prompt "coho salmon"

# 上下文线索
--prompt "salmon fish in water"
--prompt "jumping salmon"

# 多目标检测
--prompt "salmon. bear."  # 同时检测三文鱼和熊
```

## 性能预期

由于只有2个模型，预期性能：
- **召回率**: 可能略低于熊系统（少一个模型）
- **精度**: 取决于GDINO和DETR对"salmon"的泛化能力
- **速度**: 更快（少加载一个模型）

**建议**：
1. 先在小样本上测试效果
2. 如果误报多，提高 `--min-agreement` 为 2
3. 如果漏检多，降低置信度阈值或使用单模型

## 进一步优化方向

1. **收集三文鱼验证集**（100-500张）
   - 运行模型竞技场评估
   - 计算针对三文鱼的最优权重
   - 训练专用校准器

2. **尝试其他模型**
   - OWL-ViT: 另一个强大的zero-shot检测器
   - 微调YOLOv8: 用自动标注数据训练专用模型

3. **prompt工程**
   - 测试不同的文本描述
   - 使用种类名称提高精度

4. **后处理优化**
   - 时序平滑（连续帧的检测一致性）
   - 尺寸过滤（排除过小/过大的框）

## 文件结构

```
annotation-salmon/
├── multi_model_annotator.py       # 核心标注系统（适配salmon）
├── train_calibration.py           # 校准器训练（默认禁用MegaDetector）
├── auto_annotator_gdino.py        # Grounding DINO包装器（通用）
├── auto_annotator_detr.py         # DETR包装器（通用）
├── auto_annotator_megadet.py      # MegaDetector（默认不使用）
├── probability_calibrator.py      # 概率校准模块（通用）
├── frame_extractor.py             # 视频帧提取（通用）
├── visualize_labels.py            # 标注可视化（通用）
└── README_SALMON.md              # 本文档
```

## 反馈与改进

这是从熊系统快速改编的版本。如果你发现：
- 检测效果不理想
- 特定场景问题
- 需要新功能

请提交issue或联系开发团队。我们可以根据实际数据进一步优化。
