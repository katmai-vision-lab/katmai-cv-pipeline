# Stacking元学习器 - 多模型智能融合

## 📖 什么是Stacking？

**Stacking (堆叠泛化)** 是一种先进的集成学习方法，通过**元学习器 (Meta-Learner)** 学习如何最佳组合多个基础模型的预测。

### vs. 传统投票方法

| 方法 | 决策方式 | 优点 | 缺点 |
|------|---------|------|------|
| **简单投票** | 固定规则（如2/3同意） | 简单直接 | 无法适应数据，可能过松或过严 |
| **加权投票** | 手工设置权重 | 灵活性好 | 需要专家知识，难以优化 |
| **Stacking** ⭐ | 从数据学习最优策略 | 自适应，精度高 | 需要标注验证集 |

### Stacking工作原理

```
[图像] 
   ↓
[GDINO] [OWL-ViT] [Florence-2]  ← 基础模型
   ↓        ↓          ↓
 box1     box2       box3
 conf1    conf2      conf3
   ↓        ↓          ↓
   [特征提取: 11维特征向量]
   ↓
[随机森林元学习器]  ← 从验证数据训练
   ↓
[TP或FP判断] + [置信度分数]
   ↓
[最终检测结果]
```

## 🎯 特征设计 (11维)

Stacking提取以下特征来学习最优融合策略：

1. **模型特征 (3维)**
   - GDINO, OWL-ViT, Florence-2 的one-hot编码
   - 学习每个模型的可靠性模式

2. **置信度 (1维)**
   - 原始模型输出的confidence score
   - 高置信度≠一定正确（需要上下文）

3. **框大小 (2维)**
   - 归一化的宽度和高度
   - 学习：过大/过小的框可能是误检

4. **框位置 (2维)**
   - 归一化的中心坐标 (center_x, center_y)
   - 学习：边缘位置vs中心位置的可靠性

5. **模型一致性 (3维)**
   - `max_iou`: 与其他模型的最大重叠度
   - `num_overlaps`: 有多少其他模型也检测到这里
   - `avg_overlap_conf`: 重叠检测的平均置信度
   - **关键特征**：多模型一致性是最强信号

## 🚀 使用步骤

### 1. 准备验证数据集

你需要一个**人工标注的验证集**来训练Stacking模型。

```bash
data/annotation/salmon/
├── images/
│   └── train/
│       ├── salmon_001.jpg
│       ├── salmon_002.jpg
│       └── ...
└── labels/
    └── train/
        ├── salmon_001.txt  # YOLO格式
        ├── salmon_002.txt
        └── ...
```

**最佳实践**：
- 至少100-200张图像
- 覆盖不同场景（光照、角度、跳跃高度）
- 确保标注质量高

### 2. 训练Stacking元学习器

```bash
cd /home/katmai/katmai-cv-pipeline

python3 -m src.preprocessing.annotation_salmon.train_stacking \
  --images data/annotation/salmon/images/train/ \
  --labels data/annotation/salmon/labels/train/ \
  --output models/stacker_salmon.pkl \
  --prompt "jumping salmon" \
  --meta-learner rf
```

**参数说明**：
- `--images`: 验证图像目录
- `--labels`: YOLO格式标签目录
- `--output`: 输出stacker模型路径
- `--prompt`: 检测提示词（与推理时保持一致）
- `--meta-learner`: 元学习器类型
  - `rf` (Random Forest) ⭐ 推荐 - 鲁棒，不易过拟合
  - `gb` (Gradient Boosting) - 精度可能更高，但易过拟合
  - `lr` (Logistic Regression) - 最简单，适合小数据集

**输出示例**：
```
[1/4] Loading base models...
[2/4] Extracting features from validation set...
Processing images: 100%|████████| 150/150

Dataset collected:
  Total detections: 1247
  True Positives: 1089 (87.3%)
  False Positives: 158 (12.7%)
  Feature dimension: 11

[3/4] Training meta-learner...
[4/4] Evaluating meta-learner...

Validation Performance:
  Precision: 0.923
  Recall:    0.956
  F1 Score:  0.939
  AUC-ROC:   0.982

Feature Importances:
  num_overlaps: 0.287       ← 最重要！
  max_iou: 0.195
  avg_overlap_conf: 0.143
  confidence: 0.112
  model_gdino: 0.089

Saving stacking model to: models/stacker_salmon.pkl
Done!
```

### 3. 使用Stacking进行推理

```bash
python3 -m src.preprocessing.annotation_salmon.multi_model_annotator \
  --input data/frames/salmon/ \
  --output data/auto_labels_stacking/ \
  --review-queue data/review_queue/ \
  --stacker models/stacker_salmon.pkl
```

**关键变化**：
- 添加 `--stacker` 参数指定训练好的模型
- `--min-agreement` 参数会被忽略（Stacking自己决策）
- 不再需要人工审核队列（Stacking已过滤）

## 📊 性能对比

基于我们的测试数据（85帧三文鱼视频）：

| 方法 | 覆盖率 | 检测数 | 精度 (预估) | 召回率 (预估) |
|------|--------|--------|-------------|---------------|
| **min-agreement=2** | 36.5% | 92 | ⭐⭐⭐⭐⭐ 高 | ⭐⭐ 低 |
| **min-agreement=1** | 100% | 394 | ⭐⭐ 低 | ⭐⭐⭐⭐⭐ 高 |
| **Stacking** ⭐ | ~85-95% | ~250 | ⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐ 高 |

**Stacking优势**：
- ✅ 平衡精度和召回率
- ✅ 自动学习最优决策边界
- ✅ 适应不同场景和模型组合
- ✅ 无需手工调参

## 🔧 高级用法

### 调整IoU阈值

```bash
python3 -m src.preprocessing.annotation_salmon.train_stacking \
  --images ... \
  --labels ... \
  --output ... \
  --iou-threshold 0.7  # 更严格的TP判定
```

### 尝试不同元学习器

```bash
# Random Forest (推荐)
--meta-learner rf

# Gradient Boosting (更高精度)
--meta-learner gb

# Logistic Regression (最快，小数据集)
--meta-learner lr
```

### 查看特征重要性

训练时会自动输出特征重要性，帮助理解模型决策：

```
Feature Importances:
  num_overlaps: 0.287       # 最重要：其他模型是否也检测到
  max_iou: 0.195            # 重要：与其他检测的重叠度
  avg_overlap_conf: 0.143   # 重要：重叠检测的平均置信度
  confidence: 0.112         # 中等：原始置信度
  model_gdino: 0.089        # 较低：具体是哪个模型
```

## ⚠️ 注意事项

1. **需要标注数据**
   - 至少100张高质量标注的图像
   - 标注错误会直接影响Stacking性能

2. **训练集和测试集分布要一致**
   - 如果推理场景与训练场景差异大，Stacking可能表现不佳
   - 建议：从实际应用场景采样验证集

3. **过拟合风险**
   - 小数据集推荐使用Random Forest
   - 避免使用过深的Gradient Boosting

4. **计算时间**
   - Stacking推理比简单投票慢约10-20%
   - 特征提取需要额外计算

## 💡 最佳实践建议

### 场景1：精度优先（论文、生产系统）
```bash
# 1. 收集200+张高质量标注数据
# 2. 训练Random Forest stacker
--meta-learner rf

# 3. 验证精度>90%后部署
```

### 场景2：快速原型（探索阶段）
```bash
# 使用min-agreement=2的简单投票
--min-agreement 2

# 不需要额外标注，快速迭代
```

### 场景3：召回率优先（不能漏检）
```bash
# 训练Stacking，但调整决策阈值
# (需要修改代码，使用predict_proba并设置低阈值)
```

## 📚 技术参考

- **论文**: "Stacked Generalization" (Wolpert, 1992)
- **相关技术**:
  - Weighted Boxes Fusion (WBF)
  - Non-Maximum Suppression (NMS)
  - Soft-NMS

**为什么选择Stacking而不是WBF？**
- WBF: 融合**重叠的框**，生成平均框
- Stacking: 判断**每个框的真伪**，过滤假阳性
- 我们的问题是"太多误检"而不是"框不准"，所以Stacking更适合

## 🎓 扩展阅读

想深入了解多模型集成？查看这些资源：
- Kaggle目标检测竞赛方案
- COCO Detection Challenge技术报告
- Ensemble Methods in Machine Learning (书籍)
