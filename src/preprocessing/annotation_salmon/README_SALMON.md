# Salmon Auto-Annotation System

这是专为**跃出水面的三文鱼检测**优化的系统。

## 系统架构

### 3模型配置（最优推荐）

| 模型 | 版本 | 权重 | 专长 |
|------|------|------|------|
| **Grounding DINO** | base | 0.40 | 文本理解，通用目标检测 |
| **OWL-ViT v2** | base-ensemble | 0.35 | 动作理解（"跳跃"、"跃出"） |
| **Florence-2** | base | 0.25 | 复杂场景（水花、反光） |

**min_agreement**: 2 (3个模型中至少2个同意)

### vs. 熊系统 (annotation_bear)

| 差异 | 熊系统 | 三文鱼系统 |
|------|--------|------------|
| 模型组合 | GDINO + DETR + MegaDetector | GDINO + OWL-ViT v2 + Florence-2 |
| MegaDetector | ✅ 启用（陆地动物专用） | ❌ 禁用（不适合水生生物） |
| DETR | ✅ 启用 | ❌ 禁用（精度低35.4%） |
| 默认prompt | "bear" | "salmon" / "salmon jumping out of water" |
| 模型权重 | 0.406/0.335/0.259 | 0.40/0.35/0.25 |
| 场景优化 | 陆地森林环境 | 水面跳跃场景 |

### 模型选择理由

#### 🚀 OWL-ViT v2 的优势
- **CLIP架构**：擅长理解动作概念（"jumping"、"leaping"）
- **Zero-shot能力**：无需预训练即可理解"salmon jumping"
- **场景适应**：对提示词 "salmon jumping out of water" 响应好

#### 🧠 Florence-2 的优势
- **最新VLM** (2024发布)：视觉-语言多模态模型
- **鲁棒性强**：对水花飞溅、水面反光等复杂场景更稳定
- **泛化能力**：在各种光照、角度下表现一致

#### ❌ 为什么禁用MegaDetector和DETR？
- **MegaDetector v5**: 专门为**陆地野生动物**（熊、鹿、狼）训练，对鱼类形态识别能力差
- **DETR**: 在熊系统评估中精度仅35.4%，假阳性率高

## 使用方法

### 基础用法

```bash
# 1. 提取视频帧
python -m src.preprocessing.annotation_salmon.frame_extractor \
  --input salmon_jumping_video.mp4 \
  --output data/frames/salmon/ \
  --fps 0.5

# 2. 自动标注（3模型ensemble）
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon/ \
  --output data/auto_labels_salmon/ \
  --review-queue data/review_queue_salmon/ \
  --prompt "salmon jumping out of water" \
  --min-agreement 2 \
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
# 训练校准器（使用3个模型）
python -m src.preprocessing.annotation_salmon.train_calibration \
  --images data/annotation/salmon/images/train/ \
  --labels data/annotation/salmon/labels/train/ \
  --output models/calibrators_salmon.pkl \
  --prompt "salmon jumping out of water" \
  --use-gdino \
  --use-owlvit \
  --use-florence2

# 使用校准器标注
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon/ \
  --output data/auto_labels_salmon/ \
  --prompt "salmon jumping out of water" \
  --auto-approve \
  --calibrator models/calibrators_salmon.pkl
```

## 提示词优化

**推荐提示词（针对跳跃场景）**：

```bash
# 最优推荐（默认）
--prompt "jumping salmon"

# 备选动作词
--prompt "leaping salmon"
--prompt "salmon in mid-air"

# 种类特定
--prompt "jumping chinook salmon"
--prompt "leaping sockeye salmon"

# 简洁通用
--prompt "salmon"
```

**不推荐**：
- ❌ "salmon jumping out of water" (包含"water"会被Florence-2误检为大框)
- ❌ "salmon swimming" (水下场景，模型优化不匹配)
- ❌ "dead salmon" (非动态场景)
- ❌ "salmon in bear mouth" (复合场景，用熊系统更好)

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
