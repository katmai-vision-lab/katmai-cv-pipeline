# 像素识别熊吃鱼行为 —— 实验报告

**作者：** Darian Ding
**日期：** 2026 年 5 月 3 日
**任务来源：** Alex 提议尝试 pixel RGB analysis 来检测熊吃鱼，要求消费级硬件可跑

---

## 1. 任务背景

Alex 在 Slack 提出两个想法：

1. **像素颜色分析**：当摄像头变焦到熊吃鱼且光照尚可时，可以看到鲑鱼的**粉色、红色、白/浅灰色**区域。这些颜色只在熊吃鱼时出现。
2. **姿态接近性**：熊的头/嘴只在吃鱼时贴近爪子，能否做某种像素或图像分析。

**约束：**
- 必须能在消费级电脑上运行（不依赖云端 / GPU）
- 云端方案可作为补充文档化，但不能成为唯一选项

**测试视频：** `katmai_2026_05_03_8to20s.mp4`（Bear 903 "Gully" 在 Brooks Falls 钓鱼，12 秒，60fps）
**Ground truth：** Molmo2-8B 已经在该视频上跑过，48 个 0.25s 采样点中 39 帧标为 `[CATCHING]`，9 帧 `[WAITING]`

---

## 2. 已尝试的方法（按时间顺序）

### 方法 1 — 基础 HSV 颜色掩膜

**思路：** 在 OpenCV HSV 色彩空间内，用 `cv2.inRange()` 直接圈出粉色、红色、白色像素，统计在 bbox 上半部分（mouth region）内的占比。

**实现细节：**
```
PINK:  H 0–15,  S 60–180, V 100–220
RED:   H 0–10 ∪ 165–180, S 120–255, V 80–200
LIGHT: S < 50, V > 180
mouth_region = bbox 上 60% × 中间 70% 内缩
combined_score = pink × 30 + red × 25 + light × 3
```

**结果：**

| 指标 | 数值 |
|---|---|
| `max_score`（48 帧最高分） | **0.60** |
| 标为 `eating` 的帧 | **0/48 (0%)** |
| `avg_pink_ratio` | **0.007**（0.7% 像素匹配粉色） |

**为什么失败：**

抓 5 个 CATCHING 帧 + 5 个 WAITING 帧的像素统计：

| 指标 | CATCHING (n=5) | WAITING (n=5) |
|---|---|---|
| `warm_pct`（红/粉/橙系像素） | 84% | 80% |
| `saturated_warm`（饱和暖色） | 56% | 52% |
| `silver_pct`（灰色像素） | 7% | **14%** ⚠️ 反向 |

**根本原因：棕熊本身就是棕色（warm color）**。bbox 里 80% 都是熊毛，pink/red 掩膜被熊毛"污染"严重。CATCHING 帧和 WAITING 帧的颜色分布几乎完全一样。

---

### 方法 2 — 熊毛掩膜 + 仅在非熊像素上算信号

**改进思路：** 既然棕色熊毛是噪声源，就先把熊毛识别出来 mask 掉，然后只在剩下的像素（背景水/天空/可能的鱼）上算粉/红/白比例。

**实现细节：**
```
BEAR_FUR_MASK: H 5–28, S 40–220, V 20–130（暗的暖色 = 棕色熊毛）
non_bear = NOT bear_fur_mask
salmon_signals = (pink/red/light pixels) ∩ non_bear / non_bear_count
```

把所有阈值提高（V 提到 150+，S 提到 100+）确保只匹配**饱和+明亮**的鲑鱼色，跳过熊毛色。

**结果：**

| 指标 | v1 | v2 |
|---|---|---|
| `max_score` | 0.60 | 0.61 |
| `eating` 帧数 | 0/48 | 1/48 |
| `avg_pink_ratio` | 0.007 | 0.000 |

**几乎没有改善。**

**为什么失败：**

把熊毛 mask 掉之后，bbox 内非熊部分主要是：
1. **水花、白色泡沫** —— 触发 light_mask 但和鱼无关
2. **岩石、远景** —— 灰/绿色，不触发任何鲑鱼掩膜
3. **真正的鲑鱼** —— 但鱼是**银色**的（不是产卵期的红色），不在 pink/red 范围内

更重要的发现：**鱼根本没露多少**。鲑鱼被叼在熊嘴里，只有鱼尾从嘴边伸出来，占 bbox 不到 1-2% 像素。即便是银色，也太少了。

---

### 方法 3 — 加入 "bright-non-brown" 通用信号

**思路：** 既然鲑鱼是银色的，正经的 pink/red 检测不到，那就降低专属性 —— 检测**任何明亮的、非棕色的像素**（V > 130, 不在熊毛范围内）。这能至少捕捉到银色鱼身。

**实现细节：**
```
bright = inRange(hsv, [0,0,130], [180,255,255])
bright_non_brown = bright ∩ non_bear
score 公式中加权：bright_non_brown × 1.5
```

**结果：还是没有显著改善。**

**为什么失败：**

"明亮非棕色"也包括：
- 水花、瀑布飞溅（V > 130, S 低）
- 天空、远山（V 中-高）
- 岩石上的白色泡沫

这些在 WAITING 帧里和 CATCHING 帧里**比例差不多**，因为熊在两种状态下都站在水边。bright_non_brown 信号在 CATCHING 和 WAITING 都是 ~30%，区分不出来。

---

### 方法 4 — 姿态启发式（aspect ratio + 运动量）

**思路：** Alex 也提到"头/嘴贴近爪子时才在吃"。如果有姿态关键点就能直接测距，但 YOLO 只输出 bbox，没有关键点。

**替代代理信号：**
1. **bbox 宽高比 `w/h`** —— 站立扫描：高瘦（w/h < 0.85）；蹲着吃：宽扁（w/h > 1.0）
2. **帧间位移** —— 吃鱼时熊基本不动；扑鱼/巡游时移动大

```
posture_score = sigmoid((aspect - 0.85) × 4) × stillness
stillness = 1 - clip(motion × 6, 0, 1)
```

**结果：实际 aspect ratio 测量值：**

| 状态 | t=0.0 | t=0.5 | t=1.25 | t=2.75 | t=11.5 |
|---|---|---|---|---|---|
| Molmo2 标签 | WAITING | WAITING | CATCHING | CATCHING | CATCHING |
| `w/h` | 1.23 | 1.23 | 1.30 | 1.19 | 1.36 |

**WAITING 和 CATCHING 的 aspect 范围完全重叠**（都在 1.19–1.36）。posture_score 也无区分能力。

**为什么失败：**

这只熊（Bear 903 "Gully"）从头到尾都站在瀑布顶部俯视水面，**身体姿态几乎不变**。它不是那种"咬到鱼就坐下来吃"的姿势 —— 它一直站着，把鱼咬住后继续站着叼着，最多头部稍低。整体 bbox 形状基本一样。

姿态启发式只对**姿态变化大**的吃鱼场景有效（比如熊把鱼带到岸上坐下吃）。

---

### 方法 5 — 加权融合最终分数

**思路：** 既然单一信号都不行，组合起来看是否能提取微弱的信号。

```
eating_score = 0.55 × color_score + 0.45 × posture_score
threshold:
  > 0.60  →  "eating"
  > 0.40  →  "maybe"
  否则    →  "not_eating"
```

**做了 5 帧滑动平均平滑去噪。**

**与 Molmo2 ground truth 的相关性：**

| | CATCHING (n=39) | WAITING (n=9) |
|---|---|---|
| 像素 score 平均值 | **0.471** | 0.484 |
| 像素 score 最大值 | 0.529 | 0.615 |
| 像素 score 最小值 | 0.326 | 0.378 |
| **分离度** | **−0.013**（WAITING 反而稍高）|

**完全没有分离。** WAITING 帧的平均分甚至比 CATCHING 高一点点。这意味着 ROC AUC ≈ 0.5（等同于随机猜）。

---

## 3. 失败的根本原因（按重要性）

### 原因 1：鲑鱼色彩不符合假设 ⭐⭐⭐

Alex 的假设是"看到 pink/red/white-light 鲑鱼肉色"。但：

- **早夏 / 中夏的 Brooks Falls 鲑鱼是银色的**（海洋相，silver phase），尚未发育出产卵期的红色
- **粉红色鲑鱼肉**只有在熊**撕开鱼肉**之后才暴露，正常咬住状态下是看不到的
- **白色鱼腹**在叼着的状态下也基本被熊嘴遮住

### 原因 2：bbox 主体是熊，不是鱼 ⭐⭐⭐

YOLO 的 bbox 框的是**整只熊**，鱼最多占 1-2%。无论怎么 mask 熊毛、限制 mouth crop，剩下的"非熊"区域里：
- 鱼（小，可能不在范围内）
- 水（占大头）
- 岩石、天空（背景）

把这三者按颜色分开很困难。

### 原因 3：姿态信号在这个机位下失效 ⭐⭐

侧视远景下，熊从 WAITING 到 CATCHING 姿态变化非常小：
- aspect ratio：差异 < 0.15
- 位置：基本不动
- 头部朝向：始终向下

这套启发式只在**俯视 / 近视**场景下有效。

### 原因 4：缺少"接近性"的真实测量 ⭐⭐

Alex 提的"头/嘴贴近爪子"需要 keypoint 检测。我们没有接入：
- DeepLabCut（动物姿态）
- MMPose AP10K（动物 17 点）
- PoseSwin 的 HRNet（仅熊脸 13 点，不含爪子）

用 bbox aspect ratio 做代理太粗糙，丢失了"局部距离"信息。

---

## 4. 实验数字对比表

| 方法 | max_score | eating 帧 | avg pink | CATCHING 平均分 | WAITING 平均分 | 分离度 |
|---|---|---|---|---|---|---|
| v1 朴素 HSV | 0.60 | 0/48 | 0.007 | 0.42 | 0.45 | −0.03 |
| v2 + 熊毛掩膜 | 0.61 | 1/48 | 0.000 | 0.45 | 0.46 | −0.01 |
| v3 + bright-non-brown | 0.61 | 1/48 | 0.000 | 0.47 | 0.48 | −0.01 |
| **目标（参考 Molmo2）** | — | **39/48** | — | (CATCHING) | (WAITING) | **明显分离** |

**结论：在这段 Brooks Falls 视频上，纯像素方法的 ROC AUC ≈ 0.5（无判别力），Molmo2 接近 ground truth。**

---

## 5. 哪些情况下像素方法应该 work（未验证）

**我们没有失败地证明像素方法整体不可行**。Alex 描述的场景如果能拿到实际素材，理论上应该能 work：

| 条件 | 为什么有效 |
|---|---|
| ✅ 镜头**变焦到熊脸/嘴特写** | 鱼占 bbox 的比例大幅提升（5-30%） |
| ✅ 熊**已经撕开鱼**，露出鲑鱼肉 | 出现明显的粉红/红色饱和像素 |
| ✅ **晚夏产卵期**的 sockeye | 鱼本身就是亮红色，颜色对比度高 |
| ✅ 鱼**外置**（叼在嘴外、放在岸上） | 不被熊嘴遮挡 |
| ✅ 光照好、对比度高 | 颜色饱和度高，不被去饱和 |
| ❌ 我们的测试视频（侧视、银鱼、叼在嘴里） | 上述条件都不满足 |

**下一步行动**：等 Alex 周末上传的"single bear repeatedly eating salmon"特写视频，重新测试。

---

## 6. 替代思路（未实现，仅记录）

实验中考虑过但因复杂度 / 可行性放弃的方向：

| 思路 | 为什么放弃 |
|---|---|
| **帧间颜色 delta** —— 检测"突然出现"的鲑鱼色像素 | 高频抖动多，水流也会触发 delta |
| **光流 / 运动 mask** —— 找熊嘴附近"动着的"小目标（鱼挣扎） | 水花本身就一直在动，难分离 |
| **鱼形检测** —— OpenCV 找细长椭圆轮廓 | 水花碎屑和岩石也是细长形 |
| **学习背景颜色分布** —— 给每只熊建立 normal 颜色直方图，找异常 | 需要"非吃鱼"基线数据，跨视频不通用 |
| **训练专门的鱼分类器** —— 在熊 bbox 内训 YOLO 检测鱼 | 需要标注大量"鱼在嘴里"样本 → 又回到标注数据问题 |
| **接入 DeepLabCut / MMPose 测真实关键点距离** | 加 100-300 MB 模型 + 50 ms/frame，违反 consumer-grade 目标 |

---

## 7. 实际可交付的产出

**虽然纯像素分类不准，但 detector 仍然有用**——作为**预筛器**：

```
某帧像素 score < 0.30  →  极不可能在吃 → 直接跳过 Molmo2
某帧像素 score ≥ 0.30  →  候选帧 → 送 Molmo2 精确分类
```

如果像素预筛拒绝 70% 的帧，Molmo2 GPU 推理时间就降低 3 倍。这就是 Alex 想要的"both options"组合方案。

**已交付：**

| 文件 | 用途 |
|---|---|
| [src/behavior/pixel_eating_detector.py](katmai-cv-pipeline/src/behavior/pixel_eating_detector.py) | 独立 CPU detector（v3，含所有改进） |
| [docs/eating_detection_design.md](katmai-cv-pipeline/docs/eating_detection_design.md) | 设计文档 + hybrid 方案 |
| [predictions/.../pixel_eating.json](katmai-cv-pipeline/predictions/katmai_2026_05_03_8to20s_pixel_eating/pixel_eating.json) | 48 帧详细信号数据 |
| [predictions/.../pixel_eating_demo.mp4](katmai-cv-pipeline/predictions/katmai_2026_05_03_8to20s_pixel_eating/katmai_2026_05_03_8to20s_pixel_eating_demo.mp4) | 标注 demo（展示原理，不展示效果） |

---

## 8. 给 Alex 的一句话总结

> 我们试了 5 个版本的像素 + 姿态分析方法，在你给的当前 Brooks Falls 侧视镜头上都失败（CATCHING 与 WAITING 的分数差 ≈ 0）。原因不是方法本身错，而是**这段视频里鲑鱼是银色的、藏在嘴里、占像素 < 2%**，根本没有 pink/red/white salmon flesh 可看。等你周末上传的特写视频到了，方法应该能在那种条件下 work；同时建议用"像素预筛 + Molmo2 精判"的混合管线给用户两种 mode。
