# 模型、工具与方法报告 — Darian Ding

**项目：** Katmai CV Pipeline — 棕熊检测、跟踪与三文鱼捕食行为分析
**日期：** 2026 年 4 月 26 日
**作者：** Darian Ding（标注流水线、三文鱼检测 MoE、行为分析 VLM 集成）

---

## 0. 评估方法论

整个项目对候选模型进行比较时，采用了两种互补的方法：

### 0.1 公共 Benchmark 排名（用于候选模型初筛）

候选模型首先通过公开维护的 leaderboard 进行初筛。这让我们能够把数量庞大的开源模型缩减到少数几个值得在 Katmai 数据上实测的候选。

| 任务 | 参考的 Benchmark |
|---|---|
| 开放词汇检测（Open-vocabulary Detection） | **ODinW (Object Detection in the Wild)**、**COCO 零样本**、**LVIS** — 用于比较 Grounding DINO、OWL-ViT、Florence-2、MegaDetector。 |
| 闭集检测（Closed-vocabulary Detection） | **COCO mAP@0.5**、**COCO mAP@0.5:0.95**、**Roboflow 100** — 用于比较 YOLOv8 各版本、RetinaNet、DETR。 |
| 视觉-语言模型（图像） | **OpenCompass / OpenVLM Leaderboard**、**MMBench**、**MMMU** — 用于比较 Molmo2、LLaVA-OneVision、Qwen2.5-VL、InternVL2。 |
| 视觉-语言模型（视频） | **Video-MME**、**MVBench**、**TempCompass**、**LongVideoBench** — 用于比较 VideoLLaMA2、LLaVA-OneVision (video)、Qwen2.5-VL (video)、Molmo2 (video)。 |
| 多目标跟踪 | **MOT17 / MOT20 (HOTA, MOTA, IDF1)** — 用于比较 ByteTrack、BoT-SORT、DeepSORT。 |

Benchmark 排名是 **必要但不充分** 的：在 Video-MME（多为电影片段、镜头切换干净）上得分高的模型，并不一定能在 Katmai 这种静态摄像头、水花遮挡、纯动物的画面下表现良好。因此 benchmark 排名只用作过滤器，决定每个任务下哪 2–3 个模型值得拿到我们自己的数据上实测。

### 0.2 人工制造的 Ground Truth（用于最终模型选型）

每个任务我们都基于 Katmai 视频人工构建了一个小但具有代表性的 ground-truth 数据集，并据此衡量候选模型的准确率。这是流水线中每一项模型选型的最终依据。

| Ground Truth 数据集 | 构建方式 | 评估目标 |
|---|---|---|
| **848 帧 / 745 个熊 bbox** | 在 24,238 帧自动标注结果中人工抽取并核验。每帧检查；删除假阳性、补充漏检；以 YOLO 格式导出。 | YOLOv8 baseline 与 fine-tuned 之间的对比（Precision / Recall / F1 / mAP@0.5 / mAP@0.5:0.95）。 |
| **按摄像头视角分层的子集**（Brooks Falls Low、Multiview、Riffles、River Watch） | 对上述标注集按摄像头来源进行分层抽样。 | 验证 fine-tuned 检测器是否能跨摄像头视角泛化，而非过拟合到某一个视角。 |
| **完整视频片段的人工熊数计数**（如 `2025-09-19 23-30-11_Brooks_Falls_Low_5_bears.mp4`，人工标注计数 = 5） | 完整观看视频，统计真正出现过的不同熊只数量。 | ByteTrack vs. BoT-SORT 的比较 — 以 `unique_bears_tracked` 与人工计数对比，并统计 ID switch 次数。 |
| **~50 帧人工标注的熊捕食序列** | 人工打标，对每帧每只可见熊的捕食阶段打标签：`WAITING / LUNGING / CATCHING / EATING / MISSED`。 | 各 VLM 的逐帧准确率：Molmo2-8B vs. LLaVA-OneVision-7B vs. Qwen2.5-VL-7B vs. InternVL2-8B。指标包括阶段匹配准确率与熊 ID 正确性。 |
| **短视频片段三文鱼跳跃计数**（`salmon_jump_0.mp4` … `salmon_jump_9.mov`，5–22 秒） | 逐帧观看，记录每次可见跳跃的时间戳。 | Molmo2 视频模式的计数与人工 ground truth 对比；跳跃时间戳误差。 |

(a) Benchmark 驱动的初筛 + (b) Katmai 专属的人工 ground truth — 这两层方法让我们对流水线中每个模型选择都有可辩护的理由。

---

## 1. 实验过的模型

### 检测与自动标注模型

| 模型 | URL | 能力 |
|---|---|---|
| **Grounding DINO** | https://github.com/IDEA-Research/GroundingDINO | 开放词汇目标检测器；接受自由文本提示（如 "bear"、"fish jumping"）输出 bbox，无需任务特定训练。SwinT/SwinB 主干 + BERT 文本编码器融合。选型时在 **ODinW** 排行榜领先。 |
| **OWL-ViT (v2)** | https://huggingface.co/google/owlv2-base-patch16-ensemble | 基于 CLIP 的开放词汇检测器。通过 patch 级注意力，对小目标 / 密集目标表现较好。 |
| **Florence-2** | https://huggingface.co/microsoft/Florence-2-large | 微软统一视觉基础模型。多任务：检测、分割、描述、区域描述。在水域 / 反射场景表现好。 |
| **MegaDetector v6** | https://github.com/agentmorris/MegaDetector | 微软 AI-for-Earth 出品的检测器，基于数百万张相机陷阱图像训练。三类：动物、人、车辆。 |
| **DETR** | https://huggingface.co/facebook/detr-resnet-50 | 基于 Transformer 的端到端检测器。擅长建模图像中的全局空间关系。 |
| **YOLOv8 (n / s / m)** | https://github.com/ultralytics/ultralytics | 单阶段实时检测器。基于 COCO 预训练（包含 "bear" 类，索引 21）。被选为我们 fine-tune 的主干。 |
| **RetinaNet** | https://pytorch.org/vision/stable/models/retinanet.html | 单阶段检测器，使用 focal loss 处理类别不平衡；作为 YOLO 的替代方案被考虑。 |
| **CLIP ViT-L/14** | https://huggingface.co/openai/clip-vit-large-patch14 | 图文对比模型。用于三文鱼 Mixture-of-Experts 中的场景特征提取（2048 维特征 → 3 层 MLP → 专家权重）。 |

### 视觉-语言模型（行为分析与三文鱼计数）

| 模型 | URL | 能力 |
|---|---|---|
| **Molmo2-8B** *（已选用）* | https://huggingface.co/allenai/Molmo2-8B | Allen Institute 开源权重 VLM。通过 chat template 原生支持视频输入。输出结构化文本 + point 坐标。8B 参数；bf16 下可在 22 GB 显存运行。**OpenVLM Leaderboard** 开源权重模型中表现强劲。 |
| **LLaVA-OneVision-7B** | https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf | 单图、多图、视频理解三合一。基于 Qwen2-7B + SigLIP。**Video-MME** 7B 类排名靠前。 |
| **Qwen2.5-VL-7B-Instruct** | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | 阿里巴巴 VLM，原生支持视频、动态分辨率、grounding（bbox 输出）。指令跟随能力强。发布时位居 **MMBench** 榜首。 |
| **InternVL2-8B** | https://huggingface.co/OpenGVLab/InternVL2-8B | 开源 VLM，OCR 与多图推理强；视频通过交错帧输入支持。 |
| **VideoLLaMA2-7B** | https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B | 专为视频设计的 VLM，带时空卷积连接器。**MVBench** 表现强。 |
| **GPT-4o** | https://platform.openai.com/docs/models/gpt-4o | OpenAI 多模态闭源模型。通过 API 进行图像理解；不原生支持视频，必须先采样帧。 |
| **Gemini 1.5 Pro** | https://ai.google.dev/gemini-api | Google 闭源模型，1M token 上下文，原生接受长视频（约 1 小时）。 |

### 跟踪

| 模型 | URL | 能力 |
|---|---|---|
| **ByteTrack** *（已选用）* | https://github.com/ifzhang/ByteTrack | 两阶段、纯运动学的关联，配合 Kalman 滤波。通过低置信度检测恢复被遮挡目标。发布时 **MOT17 / MOT20** 排行榜第一。 |
| **BoT-SORT** | https://github.com/NirAharon/BoT-SORT | 运动 + 外观关联，带相机运动补偿。 |
| **DeepSORT** | https://github.com/nwojke/deep_sort | 运动 + 外观 Re-ID。需要单独的 Re-ID 网络。 |

---

## 2. 我们创建 / 修改的模型

| 模型 | 功能 |
|---|---|
| **Fine-tuned YOLOv8n 熊检测器（`bear_detector3/best.pt`）** | YOLOv8-Nano 在 24,238 帧 Katmai 图像上 fine-tune（40,000+ 自动生成的 bbox，由 Grounding DINO 产生）。单类：`bear`。验证集 mAP@0.5 = 95.1%，F1 = 91.4%。 |
| **三文鱼 Mixture-of-Experts 标注流水线** | CLIP-ViT-L gating 网络（3 层 MLP，ReLU + Dropout 0.2 + softmax）将每帧路由到 Grounding DINO、OWL-ViT、Florence-2 三个专家的加权组合。Platt scaling 校准每个专家的分数；加权融合 + IoU NMS 得到最终框。 |
| **Molmo2-8B 熊捕食行为分类器（prompt 工程）** | 用领域特定 prompt 包装 Molmo2，并在帧上叠加 ByteTrack 的 ID。逐帧推理，对每只跟踪熊产出 5 阶段分类（`[WAITING] / [LUNGING] / [CATCHING] / [EATING] / [MISSED]`），再用阶段感知去重合并。 |
| **Molmo2-8B 全片总结器** | 构造行为时间线 + 一帧降采样后的中段参考帧，发给 Molmo2 生成 2–4 句叙述性总结，附在 demo 视频结尾。 |
| **ByteTrack ID 碎片合并器** | 在 ByteTrack 之上的自定义后处理：将同一只熊的碎片化 ID 合并（`_merge_fragmented_tracks`），重映射为显示 ID 1..N。 |

---

## 3. 各模型的优缺点

下面表格中，"优点" 列引用 **公共 benchmark 分数**（解释为何进入候选短名单），"缺点" 列报告我们在 **Katmai 人工 ground truth** 上的实测表现。

### 检测 / 标注

| 模型 | 优点（benchmark + 实测） | 缺点（在人工 ground truth 上的实测） |
|---|---|---|
| **Grounding DINO** | 选型时在开源检测器中 ODinW 零样本得分最高；4,848 帧人工 ground truth 上抽检准确率 ~95%+；文本提示灵活；能处理部分遮挡 | 推理慢（2080 Ti 上 ~1.5 秒/帧）；显存占用大；水域纹理背景下假阳性率 ~3–5%；远距离小三文鱼召回率较差 |
| **OWL-ViT** | 在三文鱼抽检上对小密目标的召回优于 Grounding DINO；CLIP grounding 拓展性好 | 对纹理背景（水花）误报多；逐帧速度更慢 |
| **Florence-2** | 水 / 反射场景表现强；多任务（描述 + 检测）；速度快 | 小目标定位精度较低；prompt 格式较怪，需要自定义解析 |
| **MegaDetector** | 用野生动物相机陷阱数据训练；速度快；离线可用；在保护生物学 ML 中广泛引用 | 仅 3 个粗类（animal/person/vehicle）— 无法区分熊与其他动物；无三文鱼类别 |
| **DETR** | 全局上下文建模强；COCO mAP 高 | 计算开销大；缺少大规模任务特定数据时效果较差；工具链不够成熟 |
| **YOLOv8（COCO 预训练，bear 类）** | 实时；生态成熟；fine-tune 简单 | 在 Katmai 人工 ground truth 上即开即用召回率 = **1.72%**（不 fine-tune 基本不可用） |
| **YOLOv8（fine-tuned）** | 生产级：人工验证集上 precision 92.2%，recall 90.6%，mAP@0.5 95.1%；笔记本可跑 | 仅单类；不具备行为理解；新物种需重新训练 |
| **RetinaNet** | COCO 上准确率 / 速度平衡好 | 比 YOLO 需要更细致的超参数调节；计算开销更高；生态不如 YOLO 成熟 |

### VLM（用于行为分析 + 三文鱼计数）

每个候选 VLM 都在 **~50 帧人工标注的熊捕食 ground-truth 集**（三位组员共识标注）以及 **几段短三文鱼跳跃片段（人工逐帧计数）** 上进行了评分。

| 模型 | 优点（benchmark + 实测） | 缺点（在人工 ground truth 上的实测） |
|---|---|---|
| **Molmo2-8B** *（已选用）* | 开源权重 + 商用许可；OpenVLM Leaderboard 开源 8B 类得分高；领域推理强；通过 chat template 原生支持视频；支持 point 坐标和 bbox；bf16 下可在 22 GB 显存运行（双 2080 Ti）；可靠遵循结构化输出格式；**在我们 50 帧捕食集上的阶段匹配准确率最高** | 单帧推理 ~5 秒；视频模式在 22 GB 显存下只能容纳 ~8–10 帧（瓶颈是每帧 token 数 × 注意力开销）；不显式禁止时偶尔输出 `<points>` 而非纯文本 |
| **LLaVA-OneVision-7B** | Video-MME 7B 类排名靠前；强力开源视频 VLM；HuggingFace 集成好；支持多图输入 | 在我们 50 帧捕食 ground truth 上行为标签与人工标注不一致率 ~20%；不支持原生 bbox 输出，无法通过画框引用熊 ID |
| **Qwen2.5-VL-7B** | 原生 grounding（bbox 输出）；动态分辨率；指令跟随强；发布时 MMBench 第一 | 输出冗长，不做大量 prompt 工程会破坏我们的解析器；带轻微中文调风偏好；7B 模型在长视频上仍 OOM |
| **InternVL2-8B** | 通用 benchmark 强；OCR 优秀 | 视频支持本质上是 "交错帧"，无原生时序建模；输出描述发散，没有清晰的阶段标签；不满足我们的结构化输出要求 |
| **VideoLLaMA2-7B** | 专为视频设计；MVBench 强；时空连接器 | 我们的实测中对野生动物的零样本领域知识较弱；需要 fine-tune 但实验室硬件无法承受；生态不够成熟 |
| **GPT-4o** | 在 50 帧 ground truth 上原始质量最佳；API 速度快 | 闭源 / 收费（按帧 API 费用；24K 帧 × N 次运行成本难以承受）；无法本地推理；赞助方所有视频存在数据驻留顾虑；不原生支持视频，必须先采样帧 |
| **Gemini 1.5 Pro** | 原生支持长视频输入（1M token 上下文）；可一次性吞下完整 15 分钟片段 | 闭源 / 收费 API；需要把视频上传到 Google；有速率限制；内部采样策略不透明，对同一片段重复运行结果不稳定 |

### 跟踪

在一段 5 头熊的人工计数测试视频上对比。

| 跟踪器 | 优点 | 缺点 |
|---|---|---|
| **ByteTrack** *（已选用）* | 不需要外观网络（熊外观高度相似，Re-ID 不可靠）；通过第二阶段低置信度关联恢复遮挡目标；MOT17/MOT20 第一；在 5 头熊测试视频上（加上我们的 ID 合并后处理）能在水花遮挡下保持 5 个稳定 ID | 熊完全离开画面再返回时会分配新 ID；对置信度阈值敏感 |
| **BoT-SORT** | 相机运动补偿对镜头变焦有帮助 | 更重；外观分支对外观相同的熊不可靠（同一段 5 头熊视频上观察到 8–10 个碎片化 ID） |
| **DeepSORT** | 成熟；文档广 | 需要 Re-ID 提取器；外观特征在水花下对外观相同的熊失效 |

---

## 4. 各模型的帧 / 视频分析准则

| 模型 | 决策准则 |
|---|---|
| **Grounding DINO** | BERT 编码的文本提示与 Swin-Transformer 图像特征做跨模态对齐；每个 query 输出 `(bbox, confidence, matched-token)` |
| **OWL-ViT** | 每个 ViT 图像 patch 上做 CLIP 图文相似度 → patch 级 objectness + 类别得分 |
| **Florence-2** | 序列到序列：图像 patch + 任务提示 → 解析为结构化输出（boxes、labels、captions） |
| **YOLOv8** | 单次 anchor-free 回归：CSPDarknet 主干 → PAN neck → decoupled head，每个网格输出 `(x, y, w, h, objectness, class-prob)`；conf 阈值 0.25 |
| **CLIP gating 网络** | ViT-L 提取 2048 维语义向量 → 3 层 MLP → softmax 输出 3 个专家检测器的权重，按场景内容（瀑布 vs 水下 vs 水面）路由 |
| **ByteTrack** | 两阶段匈牙利匹配：(1) Kalman 预测的 track box 与高置信度检测按 IoU（阈值 0.7）匹配；(2) 未匹配的 track 与低置信度检测（阈值 0.15）二次匹配；`fuse_score=True` 把置信度与 IoU 融合 |
| **Molmo2-8B（逐帧模式）** | Chat-template 输入：图像（带 ByteTrack bbox + ID 叠加）+ 结构化 prompt 要求 `Bear N: [STAGE] description`；`do_sample=False` 贪心解码 |
| **Molmo2-8B（视频模式）** | 内部视频处理器按 `max_fps × duration` 均匀采样最多若干帧（默认 `max_fps=2.0`、`num_frames=384`）；每帧 tokenize 后拼成单个序列；对所有 token 做交叉注意力 |

---

## 5. 所做的修改与结果

### Grounding DINO
- **修改：** 用自定义批量推理脚本包装，含置信度过滤、YOLO 格式转换、可视化调试模块。
- **结果：** 在双 2080 Ti 上约 14 小时为 24,238 帧生成 40,000+ 个 bbox — 这是人工标注需要数周的工作量。抽检准确率 ~95%；剩下 ~3–5% 噪声被 YOLO fine-tune 容忍。

### 三文鱼 Mixture-of-Experts
- **修改：** 在 Grounding DINO + OWL-ViT + Florence-2 之上构建 CLIP 路由门控；引入 Platt scaling 做跨模型分数校准；用 IoU-NMS 融合。
- **结果：** 在视觉复杂的水域场景下，召回率高于任何单个检测器。春季学期仍在进行中；低置信度帧已启用人工复核队列。

### YOLOv8n Fine-Tuning
- **修改：** 从 COCO 权重做迁移学习；cosine 学习率（0.01 → 0.0001）；640×640 输入；混合精度；mosaic + 水平翻转 + 缩放增强；禁用旋转 / 透视（摄像头固定）；早停 patience=50。
- **结果：** 在人工 ground truth 上：Precision 34.37% → 92.2%（+168%）；Recall 1.72% → 90.6%（+5167%）；mAP@0.5 17.87% → 95.1%（+432%）。所有六个摄像头视角均收敛到 >90%。

### ByteTrack
- **修改：** 针对水花遮挡场景调参：`track_high_thresh=0.4`、`track_low_thresh=0.15`、`new_track_thresh=0.85`、`track_buffer=300–450`、`match_thresh=0.7`、`fuse_score=True`。新增自定义后处理 `_merge_fragmented_tracks` 把同一只熊的碎片 ID 合并、重映射为显示 ID 1..N。
- **结果：** ID switch 显著减少；在 5 头熊的人工计数测试视频上，跟踪器在水花遮挡全程保持 5 个稳定 ID，而不是碎成 8–10 个 ID。

### Molmo2-8B 用于熊捕食行为
- **修改：**
  1. **帧上 ID 标注：** 在发给 Molmo2 *之前*，把 ByteTrack 的 bbox + 熊 ID 画到帧上，让模型用与用户在 demo 视频中看到的相同编号引用熊。
  2. **阶段标签 prompt：** 强制每只熊输出 `[WAITING] / [LUNGING] / [CATCHING] / [EATING] / [MISSED]` 结构化格式，让去重逻辑能可靠检测阶段切换。
  3. **阶段感知去重：** 用阶段标签提取替代朴素的 `SequenceMatcher` 相似度去重 — 任何阶段切换始终视为新事件，即使周围文本相似。这修复了视频结尾漏报 `CATCHING` 的问题。
  4. **全片总结调用：** 构造行为时间线字符串 + 一帧降采样的参考帧（最大 512 px），作为最终的总结调用发给模型。在 0.25 秒采样间隔时，调用前先 `torch.cuda.empty_cache()` 避免 OOM。
- **结果：** 视频与分析文本中的熊 ID 现在一致。视频结尾的捕获事件（"鱼在嘴里"）不再被去重吞掉。Side-by-side 播放器渲染正常。

### Molmo2-8B 用于三文鱼跳跃计数
- **修改：** 测试原生视频输入模式；调整 `max_fps`、帧数、输入分辨率（1966×1102 → 720p → 480p → 360p）以适配显存预算。
- **结果：** **负面结果。** 双 2080 Ti 的 22 GB 显存在 Molmo2 视频模式下无论分辨率如何只能容纳 ~8–10 帧 — 瓶颈在于每帧 token 数 × 注意力开销。在稀疏采样下，模型会在不可能的时间点幻觉跳跃事件（例如在 4 秒视频里说 0:15 有跳跃）。计数准确率不可靠；目前正在探索 YOLO 三文鱼检测器 + ByteTrack 轨迹分析作为替代。

---

## 6. 表现最佳的模型（与原因）

选型准则按优先级：(1) 在 **Katmai 人工 ground truth** 上的准确率；(2) 能在消费级硬件上运行（PR-2）；(3) 开源 / 开放权重，避免持续 API 成本与数据驻留问题；(4) 工具链成熟度，HuggingFace / Ultralytics 集成度，便于快速迭代。

| 任务 | 最佳模型 | 原因 |
|---|---|---|
| **熊检测** | Fine-tuned YOLOv8n（`bear_detector3/best.pt`） | 人工验证集 mAP@0.5 = 95.1%，F1 = 91.4%；CPU 笔记本约 30 FPS，2080 Ti 约 200 FPS；满足消费级硬件约束（PR-2）。 |
| **熊自动标注** | Grounding DINO | 选型时开源检测器中 ODinW 得分最高，在我们的人工抽检集上零样本准确率最佳；文本提示灵活，让我们能在不组建人工标注团队的情况下完成 24K 帧标注。 |
| **多目标跟踪** | ByteTrack（加上我们的碎片合并器） | MOT17/MOT20 排行榜第一；在 5 头熊的人工计数测试视频上，纯运动学关联在外观相同的目标上是正确选择；第二阶段低置信度恢复能处理水花遮挡。 |
| **熊捕食行为分类** | Molmo2-8B（逐帧、带帧上 ID 叠加） | 在我们 50 帧人工捕食 ground truth 上，开源 7–8B 类 VLM 中阶段匹配准确率最高；开放权重可本地运行；原生支持 bbox / point；输出结构化稳定。 |
| **三文鱼跳跃计数** | *未解决。* | Molmo2 视频模式目前是已演示的最佳方案，但受硬件限制。 |

---

## 7. 缺失的模型能力

在我们尝试过的所有模型中，下列能力缺失实质性地拖慢了熊检测、跟踪与三文鱼捕食分析的进度：

1. **显存友好的长视频 VLM 推理。** 我们试过的所有开源权重 VLM（Molmo2、LLaVA-OneVision、Qwen2.5-VL、VideoLLaMA2）在 22 GB 显存下都被卡在 8–16 帧附近，因为每帧视觉 token 数 × O(n²) 注意力主导显存。目前没有一个开源 VLM 在消费级 GPU 上支持流式视频理解（KV-cache 淘汰、滑窗注意力、层级 token 压缩）。Gemini 1.5 Pro 这类闭源模型有这个能力但要付 API 费 + 数据驻留约束。**如果有这个能力，三文鱼跳跃计数与长片段熊行为分析就能一次过解决，而不需要逐帧。**

2. **外观相同动物的跨实例 Re-ID。** 预训练 Re-ID 网络都是用人 / 车数据训练的，对外观近乎相同的熊失效。我们试过的跟踪器没有一个能可靠地把熊完全离开画面再返回时重新关联起来。跨视频的熊个体识别需求实际上因为这个原因被从项目范围中移除。开源生态中缺少一个野生动物专用的 Re-ID 基础模型（类比检测领域的 MegaDetector）。

3. **原生时序动作 / 事件定位。** 检测模型逐帧输出 bbox，VLM 逐帧输出文本，但都没有原生输出 `(start_time, end_time, event_label, agent_id)` 元组。我们必须用自定义去重和阶段跟踪逻辑从逐帧输出反推事件。如果有真正的视频动作检测基础模型（类似在动物行为上 fine-tune 的 ActionFormer 或 VideoMAE-V2），我们就能把 "salmon catch" 当成一等公民的检测事件来处理。

4. **重纹理背景下的小目标检测。** 湍流水中的三文鱼（瀑布水花、水面反射）击败了我们试过的全部三个开放词汇检测器。召回低、对水花伪影的假阳性高。一个显式利用运动线索或光流条件的模型很可能会有帮助。

5. **大型 VLM 的廉价本地 fine-tune。** 我们无法在 Katmai 行为标签上 fine-tune Molmo2-8B，因为 LoRA + 8B + bf16 视频训练在双 2080 Ti 上没法轻松装下，除非做大量工程努力（梯度检查点、ZeRO-3 等）。更小的 VLM（~2B）能装下但推理质量损失太大不堪用。一个 2–3B 参数、Molmo2 级推理质量的 VLM 会改变实验室硬件能做的事情。

---

## 8. 使用 / 修改的 AI 工具

| 工具 | URL | 优点 | 缺点 / 需要监督的地方 |
|---|---|---|---|
| **Ultralytics YOLOv8 框架** | https://github.com/ultralytics/ultralytics | 一行 train/predict/track API；内置 ByteTrack/BoT-SORT 集成；HuggingFace + ONNX 导出；社区庞大。 | 跟踪器配置 YAML 格式文档欠佳 — `fuse_score: True` 是 ultralytics 跟踪器必需的，但官方 ByteTrack README 没提，调试花了好几个小时。默认 `vid_stride=1` 静默处理每一帧，长视频上灾难性慢。 |
| **HuggingFace `transformers` (4.57.6)** | https://github.com/huggingface/transformers | 统一的 `AutoModelForImageTextToText` + `AutoProcessor` 适用于 Molmo2；chat-template 处理视频加载；`device_map="auto"` 多 GPU 分布。 | 版本变更是大坑：transformers 5.5.4 让 Molmo2 报 `Unexpected keyword argument image_use_col_tokens`；transformers 4.57.6 又要求 `torch>=2.6`（`Using or_mask_function arguments require torch>=2.6`）。版本固定不容易。 |
| **Grounding DINO (HuggingFace 移植)** | https://huggingface.co/IDEA-Research/grounding-dino-base | 文本提示 API 简单；通过 HF Hub 跨机器复现性好。 | 慢（~1.5 秒/帧）；不显式降采样时 1080p 帧偶发 CUDA OOM。 |
| **CLIP ViT-L (HuggingFace)** | https://huggingface.co/openai/clip-vit-large-patch14 | 特征提取稳定；GPU 上快。 | 不同 CLIP 版本间 embedding 稳定性有差异；我们固定到了某一个 revision。 |
| **OpenCV + ffmpeg** | https://opencv.org / https://ffmpeg.org | 视频解码、帧提取、剪辑、格式转换。 | OpenCV `VideoCapture` 在 Linux 上对 `.mov` 文件不带 ffmpeg backend 时不可靠；`XVID` writer 输出 `.avi`，我们再 re-encode 成 `.mp4` 分享。 |
| **decord** | https://github.com/dmlc/decord | Molmo2 视频处理器要求；随机访问帧加载比 OpenCV 快。 | 安静地依赖 — Molmo2 报 `ImportError: requires decord, torchcodec, or av`，`pip install decord` 后才好。 |
| **CVAT** | https://github.com/opencv/cvat | 工业标准标注 UI；YOLO 格式导出；用于构建人工 ground-truth 集。 | 自部署较重（Docker、多服务）；时间紧因此只用于抽检验证而非完整标注。 |
| **Label Studio** | https://github.com/HumanSignal/label-studio | 比 CVAT 轻；标签 schema 灵活；适合三文鱼 MoE 流水线的 human-in-the-loop 复核队列以及 50 帧捕食 ground truth 集。 | 长视频帧上 Web UI 偶有卡顿；导出格式需要后处理才能转为 YOLO 格式。 |
| **bitsandbytes** | https://github.com/bitsandbytes-foundation/bitsandbytes | 4-bit / 8-bit 量化把更大 VLM 塞进显存。 | 需要可用的 C 编译器和较新的 CUDA；ENGINE Lab 机器上我们碰到 Triton 编译时 `RuntimeError: Failed to find C compiler`，回退到 bf16。不是即插即用的替代品。 |
| **PyTorch (2.6.0+cu124)** | https://pytorch.org | 最新 transformers 必需；`bf16` + `device_map="auto"` 把 Molmo2 干净地分布到双 2080 Ti 上。 | 与 transformers/CUDA 紧耦合；升级时要小心规划，避免破坏 Ultralytics。 |
| **OpenCompass / OpenVLM Leaderboard** | https://rank.opencompass.org.cn/leaderboard-multimodal | 集中提供开源 VLM 在 MMBench、MMMU、Video-MME、MVBench 上的排名；用于 VLM 候选短名单。 | Benchmark 分数未必能迁移到野生动物视频 — 仍需用人工 ground truth 验证。 |
| **HuggingFace Open VLM Leaderboard** | https://huggingface.co/spaces/opencompass/open_vlm_leaderboard | 同上目的；交叉核对排名。 | 同样的 benchmark 到领域迁移性问题。 |
| **Cursor / Claude Code (Anthropic)** | https://claude.com/claude-code | 作为 AI pair-programmer，用于 Molmo2 prompt 迭代、调试 ByteTrack ID 碎片合并器、编写 side-by-side 播放器渲染代码。极大加速了春季学期的原型迭代。 | 在 VLM 行为上需要密切监督 — LLM 生成的代码偶尔会编造我们固定版本 transformers 中不存在的 HuggingFace API 参数（如 `image_use_col_tokens`），需要逐个 API 调用核对源码。在不实际运行模型的情况下推理 Molmo2 视频输出时，也会幻觉三文鱼跳跃时间戳。 |
| **GitHub + GitHub Actions** | https://github.com | 分支保护（合并到 main 需要 2 个 reviewer 通过）、共享 SSH deploy keys、ENGINE Lab 机器上每天三次自动 pull。 | 标准用法，无意外。 |
| **Slack** | https://slack.com | 与赞助方和团队的异步沟通；方便分享文件 / 截图做可视化调试。 | – |

---

## 选型总结

流水线最终收敛到 **YOLOv8（fine-tuned）+ Grounding DINO（自动标注）+ ByteTrack（跟踪）+ Molmo2-8B（行为 + 总结）**，因为这是唯一同时满足以下条件的组合：

1. 能在现有硬件上运行（双 2080 Ti 训练，消费级笔记本推理）。
2. 全开放权重模型，开源交付物没有 API 依赖或许可证障碍。
3. 在 **人工核验的 Katmai 验证集** 上熊检测准确率 >90%。
4. 在 **人工标注的捕食 ground truth** 上行为分类阶段匹配准确率最高。
5. 输出结构化、可解析，下游捕食事件检测与环境数据关联可以消费。

每个模型选择都基于 **(a) 公共 benchmark 排名做候选短名单** 和 **(b) Katmai 专属人工 ground truth 做最终选型** 这两个依据。

剩下的开放问题是 **独立三文鱼检测 / 跳跃计数**，目前的开源 VLM 受 VRAM 帧数预算约束被卡住，下一步最可能的路径是 fine-tune 一个 YOLO 三文鱼检测器。
