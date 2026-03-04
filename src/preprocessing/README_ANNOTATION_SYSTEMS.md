# 自动标注系统 - 熊 vs. 三文鱼

本目录包含两个独立的自动标注系统，分别针对不同的目标物种优化。

## 系统概览

| 系统 | 目录 | 目标 | 模型数量 | 状态 |
|------|------|------|----------|------|
| 🐻 **熊系统** | `annotation_bear/` | 棕熊检测 | 3个 | ✅ 生产级 |
| 🐟 **三文鱼系统** | `annotation_salmon/` | 三文鱼检测 | 2个 | 🧪 实验性 |

## 熊系统 (`annotation_bear/`)

### 特点
- **3个模型共识**: Grounding DINO + DETR + MegaDetector v5
- **科学验证**: 基于341张测试图片的模型竞技场评估
- **概率校准**: Isotonic regression校准置信度
- **高性能**: 89.3%精度, 99.8%召回率

### 使用场景
- Katmai国家公园熊监测
- 其他棕熊栖息地
- 陆地野生动物检测（通过修改prompt）

### 快速使用
```bash
python -m src.preprocessing.annotation_bear.multi_model_annotator \
  --input data/frames/bear_video/ \
  --output data/auto_labels_bear/ \
  --prompt "bear" \
  --min-agreement 2 \
  --auto-approve
```

### 文档
- 完整技术报告: [docs/bear_auto_annotation_system_report.md](../../docs/bear_auto_annotation_system_report.md)
- 使用说明: [README.md](../../README.md)

---

## 三文鱼系统 (`annotation_salmon/`)

### 特点
- **2个模型共识**: Grounding DINO + DETR（禁用MegaDetector）
- **快速改编**: 从熊系统修改而来
- **优化提示词**: 默认prompt为"salmon"
- **轻量级**: 更快的处理速度

### 为什么不用MegaDetector？
MegaDetector v5专为**陆地野生动物**训练（熊、鹿、狼等），对鱼类：
- ❌ 形态差异大（四足动物 vs 鱼类）
- ❌ 环境不同（森林 vs 水域）
- ❌ 无相关训练数据

### 使用场景
- 河流/溪流中的三文鱼
- 鱼类洄游监测
- 水生生物检测（需调整prompt）

### 快速使用
```bash
python -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon_video/ \
  --output data/auto_labels_salmon/ \
  --prompt "salmon" \
  --min-agreement 1 \
  --auto-approve
```

### 文档
- 三文魚系统说明: [annotation_salmon/README_SALMON.md](annotation_salmon/README_SALMON.md)

---

## 技术对比

### 模型配置

| 特性 | 熊系统 | 三文鱼系统 |
|------|--------|------------|
| **Grounding DINO** | ✅ base模型，0.25阈值 | ✅ base模型，0.25阈值 |
| **DETR ResNet-101** | ✅ 0.5阈值 | ✅ 0.5阈值 |
| **MegaDetector v5** | ✅ 0.3阈值 | ❌ 禁用 |
| **模型权重** | gdino:0.406, detr:0.259, megadet:0.335 | gdino:0.61, detr:0.39 |
| **默认min_agreement** | 2/3 | 1/2 |

### 性能特征

| 维度 | 熊系统 | 三文鱼系统 |
|------|--------|------------|
| **召回率** | 高（99.8%验证） | 中等（预期略低） |
| **精度** | 高（89.3%验证） | 待测试 |
| **速度** | 中等（~4-5秒/图） | 快（~3秒/图） |
| **GPU显存** | 6-8GB | 4-6GB |
| **生产就绪** | ✅ 是 | 🧪 实验阶段 |

---

## 选择指南

### 使用熊系统，如果你需要：
- ✅ 检测陆地野生动物（熊、鹿、狼等）
- ✅ 最高的检测质量和可靠性
- ✅ 经过验证的生产系统
- ✅ 完整的概率校准支持

### 使用三文鱼系统，如果你需要：
- ✅ 检测鱼类或水生生物
- ✅ 更快的处理速度
- ✅ 较低的GPU显存需求
- ⚠️  可接受实验性质的系统

---

## 扩展到新物种

想为其他物种创建自动标注系统？遵循以下步骤：

### 1. 复制现有系统
```bash
cd src/preprocessing
cp -r annotation_bear annotation_newspecies
```

### 2. 修改关键参数

**a. 确定是否使用MegaDetector**
- 陆地动物（鹿、狐狸、浣熊等）：保留MegaDetector ✅
- 鸟类、鱼类、昆虫等：禁用MegaDetector ❌

**b. 更新默认prompt**
```python
# multi_model_annotator.py 和 train_calibration.py
default="newspecies"  # 或更具体的描述
```

**c. 调整模型权重**
- 如果禁用MegaDetector，重新分配权重（如三文鱼系统）
- 如果有验证数据，运行模型竞技场评估计算最优权重

**d. 更新min_agreement**
- 3个模型：default=2
- 2个模型：default=1

### 3. 测试与优化
```bash
# 小样本测试
python -m src.preprocessing.annotation_newspecies.multi_model_annotator \
  --input test_frames/ \
  --output test_labels/ \
  --prompt "newspecies" \
  --limit 20

# 可视化验证
python -m src.preprocessing.annotation_newspecies.visualize_labels \
  --images test_frames/ \
  --labels test_labels/ \
  --output test_visualized/ \
  --limit 10
```

### 4. 可选：训练专用校准器
如果你有标注数据（推荐100+样本）：
```bash
python -m src.preprocessing.annotation_newspecies.train_calibration \
  --images data/annotation/newspecies/images/ \
  --labels data/annotation/newspecies/labels/ \
  --output models/calibrators_newspecies.pkl \
  --prompt "newspecies"
```

---

## 维护说明

### 文件组织
每个系统应保持独立：
- ✅ 独立的README文档
- ✅ 独立的校准器文件
- ✅ 独立的输出目录
- ✅ 独立的git提交

### 共享组件
以下模块在系统间共享（无需复制）：
- `frame_extractor.py` - 视频帧提取
- `visualize_labels.py` - 标注可视化
- `probability_calibrator.py` - 校准算法（数据分离）

### 版本控制
```bash
# 熊系统更新
git add src/preprocessing/annotation_bear/
git commit -m "feat(bear): ..."

# 三文鱼系统更新
git add src/preprocessing/annotation_salmon/
git commit -m "feat(salmon): ..."
```

---

## 常见问题

**Q: 能否用熊系统检测三文鱼？**
A: 可以，但效果不如专用系统。只需修改prompt为"salmon"，但MegaDetector可能产生误报。

**Q: 需要为每个物种重新训练模型吗？**
A: 不需要！我们使用zero-shot模型（Grounding DINO），只需修改文本提示即可。

**Q: 如何知道哪个系统适合我的物种？**
A: 看物种类别：
- 陆地哺乳动物 → 用熊系统
- 水生/飞行/小型动物 → 改编三文鱼系统（禁用MegaDetector）

**Q: 能否同时检测多个物种？**
A: 可以！Grounding DINO支持多prompt：
```bash
--prompt "bear. salmon. eagle."
```

---

## 参考资源

- 熊系统技术报告: [docs/bear_auto_annotation_system_report.md](../../docs/bear_auto_annotation_system_report.md)
- 主README: [README.md](../../README.md)
- Grounding DINO论文: https://arxiv.org/abs/2303.05499
- DETR论文: https://arxiv.org/abs/2005.12872
- MegaDetector: https://github.com/microsoft/CameraTraps

---

**最后更新**: 2026年3月4日  
**维护团队**: Katmai Vision Lab
