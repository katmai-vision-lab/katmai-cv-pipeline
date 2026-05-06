# Models, Tools, and Methods Report — Darian Ding

**Project:** Katmai CV Pipeline — Bear Detection, Tracking, and Salmon Feeding Behavior Analysis
**Date:** April 26, 2026
**Author:** Darian Ding (Annotation Pipeline, Salmon Detection MoE, Behavior-Analysis VLM Integration)

---

## 0. Evaluation Methodology

Two complementary methods were used to compare candidate models throughout the project:

### 0.1 Public Benchmark Rankings (model shortlisting)

Candidate models were first shortlisted by consulting publicly maintained leaderboards relevant to each task. This let us narrow a large field of open-source releases down to a small set worth bench-testing on Katmai footage.

| Task | Benchmarks consulted |
|---|---|
| Open-vocabulary detection | **ODinW (Object Detection in the Wild)**, **COCO zero-shot**, **LVIS** — used to compare Grounding DINO, OWL-ViT, Florence-2, MegaDetector. |
| Closed-vocabulary detection | **COCO mAP@0.5**, **COCO mAP@0.5:0.95**, **Roboflow 100** — used to compare YOLOv8 variants, RetinaNet, DETR. |
| Vision-language models (image) | **OpenCompass / OpenVLM Leaderboard**, **MMBench**, **MMMU** — used to compare Molmo2, LLaVA-OneVision, Qwen2.5-VL, InternVL2. |
| Vision-language models (video) | **Video-MME**, **MVBench**, **TempCompass**, **LongVideoBench** — used to compare VideoLLaMA2, LLaVA-OneVision (video), Qwen2.5-VL (video), Molmo2 (video). |
| Multi-object tracking | **MOT17 / MOT20 (HOTA, MOTA, IDF1)** — used to compare ByteTrack, BoT-SORT, DeepSORT. |

Benchmark rankings were **necessary but not sufficient**: a model that scores well on Video-MME (e.g., on movie clips with clean cuts) does not necessarily perform well on Katmai's static-camera, splash-occluded, animal-only footage. We therefore used the rankings only as a filter to decide which 2–3 models per task were worth running on our own data.

### 0.2 Manually-Created Ground Truth (final model selection)

For each task we manually constructed a small but representative ground-truth set from Katmai footage and measured candidate models against it. This was the deciding test for every modeling choice in the pipeline.

| Ground-truth set | How it was built | What it evaluated |
|---|---|---|
| **848 frames / 745 bear bboxes** | Manually verified subset of the 24,238-frame auto-annotation output. Each frame inspected; false positives removed; missed bears added. Labels exported in YOLO format. | YOLOv8 baseline vs. fine-tuned (Precision / Recall / F1 / mAP@0.5 / mAP@0.5:0.95). |
| **Per-camera-view subsets (Brooks Falls Low, Multiview, Riffles, River Watch)** | Stratified sampling from the same labeled set, grouped by source camera. | Whether the fine-tuned detector generalizes across camera angles or overfits to one view. |
| **Bear-count ground truth on full clips (e.g. `2025-09-19 23-30-11_Brooks_Falls_Low_5_bears.mp4`, manually labeled count = 5)** | Watched each clip end-to-end and counted distinct bears that appeared. | ByteTrack vs. BoT-SORT — measured by `unique_bears_tracked` against manual count, and by ID-switch count. |
| **~50-frame manually-labeled bear-feeding sequence** | Manual labeling of feeding stage per visible bear per frame: `WAITING / LUNGING / CATCHING / EATING / MISSED`. | Per-frame VLM accuracy: Molmo2-8B vs. LLaVA-OneVision-7B vs. Qwen2.5-VL-7B vs. InternVL2-8B. Measured stage-match accuracy and bear-ID correctness. |
| **Salmon-jump count on short clips (`salmon_jump_0.mp4` … `salmon_jump_9.mov`, 5–22 s)** | Watched each clip frame-by-frame and recorded the timestamp of each visible jump. | Molmo2 video-mode count vs. ground-truth count; jump-timestamp error. |

The combination of (a) benchmark-driven shortlisting and (b) Katmai-specific manual ground truth gave us defensible reasons for every model choice in the pipeline.

---

## 1. Models Experimented With

### Detection & Auto-Annotation Models

| Model | URL | Capability |
|---|---|---|
| **Grounding DINO** | https://github.com/IDEA-Research/GroundingDINO | Open-vocabulary object detector; takes free-text prompts (e.g. "bear", "fish jumping") and returns bounding boxes without task-specific training. SwinT/SwinB backbone fused with BERT text encoder. State of the art on **ODinW** at the time of selection. |
| **OWL-ViT (v2)** | https://huggingface.co/google/owlv2-base-patch16-ensemble | CLIP-based open-vocabulary detector. Strong on small / dense objects via patch-level attention. |
| **Florence-2** | https://huggingface.co/microsoft/Florence-2-large | Microsoft's unified vision foundation model. Multi-task: detection, segmentation, captioning, region description. Performs well on water / reflection scenes. |
| **MegaDetector v6** | https://github.com/agentmorris/MegaDetector | Microsoft AI-for-Earth detector trained on millions of camera-trap images. Three classes: animal, person, vehicle. |
| **DETR** | https://huggingface.co/facebook/detr-resnet-50 | Transformer-based end-to-end detector. Strong at modeling global spatial relationships. |
| **YOLOv8 (n / s / m)** | https://github.com/ultralytics/ultralytics | Single-stage real-time detector. Pretrained on COCO (which includes "bear" class 21). Selected as the backbone we fine-tuned. |
| **RetinaNet** | https://pytorch.org/vision/stable/models/retinanet.html | Single-stage detector with focal loss for class imbalance; considered as alternative to YOLO. |
| **CLIP ViT-L/14** | https://huggingface.co/openai/clip-vit-large-patch14 | Image-text contrastive model. Used as the scene-feature extractor for the salmon Mixture-of-Experts gating network (2048-d feature → 3-layer MLP → expert weights). |

### Vision-Language Models (Behavior Analysis & Salmon Counting)

| Model | URL | Capability |
|---|---|---|
| **Molmo2-8B** *(selected)* | https://huggingface.co/allenai/Molmo2-8B | Allen Institute open-weights VLM. Native video input via chat template. Outputs structured text + point coordinates. 8B parameters; runs on 22 GB VRAM in bf16. Strong on **OpenVLM Leaderboard** for open-weights models. |
| **LLaVA-OneVision-7B** | https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf | Single-image, multi-image, and video understanding in one model. Built on Qwen2-7B + SigLIP. Top-tier on **Video-MME** for 7B class. |
| **Qwen2.5-VL-7B-Instruct** | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | Alibaba's VLM with native video support, dynamic resolution, and grounding (bbox output). Strong instruction following. Top of **MMBench** at release. |
| **InternVL2-8B** | https://huggingface.co/OpenGVLab/InternVL2-8B | Open-source VLM with strong OCR and multi-image reasoning, video via interleaved frame input. |
| **VideoLLaMA2-7B** | https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B | Purpose-built video VLM with spatio-temporal convolutional connector. Strong on **MVBench**. |
| **GPT-4o** | https://platform.openai.com/docs/models/gpt-4o | OpenAI multimodal closed model. Image understanding via API; no native video, must pre-sample frames. |
| **Gemini 1.5 Pro** | https://ai.google.dev/gemini-api | Google closed model, 1M-token context, accepts long video natively (~1 hour). |

### Tracking

| Model | URL | Capability |
|---|---|---|
| **ByteTrack** *(selected)* | https://github.com/ifzhang/ByteTrack | Two-stage motion-only association with Kalman filter. Recovers occluded objects via low-confidence detections. Top of **MOT17 / MOT20** at release. |
| **BoT-SORT** | https://github.com/NirAharon/BoT-SORT | Motion + appearance association with camera motion compensation. |
| **DeepSORT** | https://github.com/nwojke/deep_sort | Motion + appearance Re-ID. Requires a separate Re-ID network. |

---

## 2. Models We Created / Modified

| Model | What it does |
|---|---|
| **Fine-tuned YOLOv8n bear detector (`bear_detector3/best.pt`)** | YOLOv8-Nano fine-tuned on 24,238 Katmai frames (40,000+ auto-generated bounding boxes from Grounding DINO). Single class: `bear`. mAP@0.5 = 95.1%, F1 = 91.4% on validation. |
| **Salmon Mixture-of-Experts annotation pipeline** | CLIP-ViT-L gating network (3-layer MLP, ReLU + Dropout 0.2 + softmax) routes each frame to a weighted combination of Grounding DINO, OWL-ViT, and Florence-2. Platt scaling calibrates per-expert scores; weighted fusion + IoU NMS produces the final boxes. |
| **Molmo2-8B Bear Feeding Behavior Classifier (prompt-engineered)** | Wraps Molmo2 with a domain-specific prompt and ByteTrack ID overlays. Per-frame inference produces a 5-stage classification (`[WAITING] / [LUNGING] / [CATCHING] / [EATING] / [MISSED]`) per tracked bear, then merged with stage-aware deduplication. |
| **Molmo2-8B Whole-Video Summarizer** | Builds a behavior timeline + a downscaled mid-video reference frame, sends both to Molmo2 for a 2–4-sentence narrative summary appended to the demo video. |
| **ByteTrack ID-fragmentation merger** | Custom post-processing on top of ByteTrack to merge fragmented IDs of the same bear (`_merge_fragmented_tracks`) and remap to display IDs 1..N. |

---

## 3. Pros / Cons of Each Model

Where applicable, the "Pros" column references **public benchmark scores** that drove shortlisting, and the "Cons" column reports what we observed when running each model against our **manually-created Katmai ground truth**.

### Detection / Annotation

| Model | Pros (benchmark + observed) | Cons (observed against manual ground truth) |
|---|---|---|
| **Grounding DINO** | Best zero-shot ODinW score among open detectors at selection time; ~95%+ accuracy on our 4,848-frame manual bear ground truth in spot-checks; flexible text prompts; handles partial occlusion | Slow inference (~1.5 s/frame on 2080 Ti); memory-hungry; ~3–5% false-positive rate on water-textured backgrounds; mediocre recall on small distant salmon |
| **OWL-ViT** | Better recall than Grounding DINO on small dense objects in our salmon spot-check; CLIP-based grounding scales well | More false positives on textured backgrounds (waterfall splash); slower per-frame |
| **Florence-2** | Strong on water / reflection scenes; multi-task (caption + detect); fast | Less precise localization on small objects; prompt format is quirky and required custom parsing |
| **MegaDetector** | Trained on wildlife camera-trap data; very fast; works offline; widely cited in conservation ML | Only 3 generic classes (animal/person/vehicle) — cannot distinguish bear from other animals; no salmon class |
| **DETR** | Models global context well; strong COCO mAP | Heavy compute; weaker without large task-specific data; mature tooling lacking |
| **YOLOv8 (pretrained, COCO bear)** | Real-time; mature ecosystem; trivial fine-tuning | Out-of-the-box recall on Katmai manual ground truth = **1.72%** (essentially unusable without fine-tuning) |
| **YOLOv8 (fine-tuned)** | Production-ready: precision 92.2%, recall 90.6%, mAP@0.5 95.1% on manual validation set; runs on laptop | Single-class only; no behavior understanding; needs re-training for new species |
| **RetinaNet** | Good accuracy / speed balance on COCO | More careful hyperparameter tuning required than YOLO; higher compute cost; ecosystem less mature |

### VLMs (for feeding behavior + salmon counting)

Each candidate VLM was scored against a **~50-frame manually-labeled bear-feeding ground-truth set** (consensus labels from three team members) and a **set of short salmon-jump clips with manually-counted ground-truth jump counts**.

| Model | Pros (benchmark + observed) | Cons (observed against manual ground truth) |
|---|---|---|
| **Molmo2-8B** *(selected)* | Open weights + commercial license; high score on OpenVLM Leaderboard for open 8B class; strong domain reasoning; native video input via chat template; supports point coordinates and bounding boxes; runs in bf16 on 22 GB VRAM (dual 2080 Ti); follows structured output format reliably; **highest stage-match accuracy** on our 50-frame feeding set among 7–8B open VLMs we tested | ~5 s per frame inference; video mode capped at ~8–10 frames on 22 GB VRAM (frame budget × tokens-per-frame is the bottleneck); occasionally outputs `<points>` instead of text without explicit prompt suppression |
| **LLaVA-OneVision-7B** | Top-tier on Video-MME for 7B class; strong open-source video VLM; HuggingFace integration; handles multi-image input | On our 50-frame feeding ground truth, behavior labels were ~20% inconsistent vs. manual labels; no native bbox output, so cannot reference bear IDs by drawing |
| **Qwen2.5-VL-7B** | Excellent grounding (bbox output natively); dynamic resolution; strong instruction following; top of MMBench at release | Verbose outputs that broke our parser without heavy prompt engineering; slight Chinese-tuned bias in style; 7B model still hit OOM on long videos |
| **InternVL2-8B** | Strong general benchmarks; OCR is excellent | Video support is essentially "interleaved frames" with no native temporal modeling; produced wandering descriptions rather than a clear stage tag; failed our structured-output requirement |
| **VideoLLaMA2-7B** | Purpose-built for video; strong on MVBench; spatio-temporal connector | Weaker zero-shot domain knowledge of wildlife in our pilot; required fine-tuning we couldn't afford on lab hardware; less mature ecosystem |
| **GPT-4o** | Best raw quality on our 50-frame ground truth; fast API | Closed/paid (per-frame API cost; 24K frames × N runs is prohibitive); no on-prem inference; data-residency concerns for sponsor-owned video; no native video — must pre-sample frames |
| **Gemini 1.5 Pro** | Native long-video input (1M-token context); can ingest entire 15-min clip in one call | Closed/paid API; requires upload of footage to Google; rate-limited; opaque internal sampling strategy gave inconsistent answers across repeated runs on the same clip |

### Tracking

Compared on a held-out 5-bear test clip (manually-counted ground truth = 5).

| Tracker | Pros | Cons |
|---|---|---|
| **ByteTrack** *(selected)* | No appearance net needed (bears look very similar so Re-ID is unreliable); recovers occluded tracks via second-stage low-conf association; top of MOT17/MOT20; on the 5-bear test clip held 5 stable IDs through splash occlusion (after our ID-merger post-processing) | Reassigns new ID when bear fully exits and re-enters frame; sensitive to confidence-threshold choice |
| **BoT-SORT** | Camera-motion compensation helps with zoom transitions | Heavier; appearance branch unreliable on visually-identical bears (we observed 8–10 fragmented IDs on the same 5-bear clip) |
| **DeepSORT** | Mature; widely documented | Re-ID extractor needed; appearance features fail for visually identical bears under water splash |

---

## 4. Criteria Each Model Uses to Analyze Frames / Videos

| Model | Decision criteria |
|---|---|
| **Grounding DINO** | Cross-modal alignment between BERT-encoded text prompt and Swin-Transformer image features; outputs `(bbox, confidence, matched-token)` per query |
| **OWL-ViT** | CLIP image-text similarity per ViT image patch → patch-level objectness + class score |
| **Florence-2** | Sequence-to-sequence: image patches + task prompt → token sequence parsed into structured output (boxes, labels, captions) |
| **YOLOv8** | Single-pass anchor-free regression: CSPDarknet backbone → PAN neck → decoupled head producing `(x, y, w, h, objectness, class-prob)` per grid cell; conf threshold 0.25 |
| **CLIP gating network** | 2048-d ViT-L semantic vector → 3-layer MLP → softmax weights over 3 expert detectors based on scene content (waterfall vs. underwater vs. surface) |
| **ByteTrack** | Two-stage Hungarian matching: (1) Kalman-predicted track box vs. high-conf detection by IoU (threshold 0.7); (2) unmatched tracks vs. low-conf detections (threshold 0.15); `fuse_score=True` blends conf with IoU |
| **Molmo2-8B (per-frame mode)** | Chat-template input: image (with ByteTrack bbox + ID overlay) + structured prompt requesting `Bear N: [STAGE] description`; greedy decode with `do_sample=False` |
| **Molmo2-8B (video mode)** | Internal video processor uniformly samples up to `max_fps × duration` frames (default `max_fps=2.0`, `num_frames=384`); each frame tokenized and concatenated into a single sequence; cross-attention over all tokens |

---

## 5. Modifications Made and Outcomes

### Grounding DINO
- **Modification:** Wrapped in a custom batch-inference script with confidence filtering, YOLO-format conversion, and a visualization debug module.
- **Outcome:** Generated 40,000+ bounding boxes across 24,238 frames in ~14 hours on dual 2080 Ti — work that would have taken weeks of manual annotation. Spot-check accuracy ~95%; the ~3–5% noise was tolerated by YOLO fine-tuning.

### Salmon Mixture-of-Experts
- **Modification:** Built CLIP-gated routing over Grounding DINO + OWL-ViT + Florence-2; added Platt scaling for cross-model score calibration and IoU-NMS for fusion.
- **Outcome:** Higher recall on visually-complex water scenes than any single detector. Ongoing in Spring Quarter; preliminary review queue is enabled for low-confidence frames.

### YOLOv8n Fine-Tuning
- **Modification:** Transfer learning from COCO weights; cosine LR schedule (0.01 → 0.0001); 640×640 input; mixed-precision; mosaic + horizontal-flip + scale augmentation; rotation/perspective disabled (fixed camera); early stopping with patience=50.
- **Outcome:** Measured against manual ground truth: Precision 34.37% → 92.2% (+168%); Recall 1.72% → 90.6% (+5167%); mAP@0.5 17.87% → 95.1% (+432%). All six camera views converged to >90%.

### ByteTrack
- **Modification:** Tuned thresholds for the splash-occlusion scenario: `track_high_thresh=0.4`, `track_low_thresh=0.15`, `new_track_thresh=0.85`, `track_buffer=300–450`, `match_thresh=0.7`, `fuse_score=True`. Added a custom post-processing step `_merge_fragmented_tracks` to collapse fragmented IDs of the same bear and remap to display IDs 1..N.
- **Outcome:** Reduced ID switches dramatically; on the 5-bear manually-counted test clip the tracker holds 5 stable IDs throughout splash occlusion instead of fragmenting into 8–10 IDs.

### Molmo2-8B for Bear Feeding Behavior
- **Modifications:**
  1. **On-frame ID annotation:** Draw ByteTrack bounding box + bear ID on each frame *before* sending to Molmo2, so the model can reference bears by the same numbering the user sees in the demo video.
  2. **Stage-tag prompting:** Force structured output `[WAITING] / [LUNGING] / [CATCHING] / [EATING] / [MISSED]` per bear, so dedup logic can detect stage transitions reliably.
  3. **Stage-aware deduplication:** Replaced naïve `SequenceMatcher` ratio-based dedup with stage-tag extraction — any stage change is always treated as a new event, even if the surrounding text is similar. This fixed missed `CATCHING` events at video end.
  4. **Whole-video summary pass:** Built a behavior timeline string + a downscaled reference frame (max 512 px) and sent it as a final summary call. `torch.cuda.empty_cache()` before this call avoided OOM at 0.25 s sampling intervals.
- **Outcome:** Bear IDs now match between video and analysis text. Catch events at end-of-clip ("fish in mouth") are no longer swallowed by dedup. Side-by-side viewer renders correctly.

### Molmo2-8B for Salmon Jump Counting
- **Modifications:** Tested native video input mode; varied `max_fps`, frame count, and input resolution (1966×1102 → 720p → 480p → 360p) to fit VRAM budget.
- **Outcome:** **Negative result.** 22 GB of VRAM (dual 2080 Ti) only fits ~8–10 video frames in Molmo2's video mode regardless of resolution — the bottleneck is the per-frame token count multiplied by attention cost. With sparse sampling, the model hallucinates jump events at impossible timestamps (e.g. claiming a jump at 0:15 in a 4-second clip). Counting accuracy is unreliable; we are now exploring a YOLO-based salmon detector + ByteTrack trajectory analysis as a fallback.

---

## 6. Best-Performing Models (and Why)

Selection criteria, in priority order: (1) accuracy on **Katmai-specific manual ground truth**; (2) ability to run on consumer hardware (PR-2); (3) open-source / open-weights to avoid recurring API cost and data-residency issues; (4) tooling maturity and HuggingFace / Ultralytics integration for fast iteration.

| Task | Best model | Why |
|---|---|---|
| **Bear detection** | Fine-tuned YOLOv8n (`bear_detector3/best.pt`) | mAP@0.5 = 95.1%, F1 = 91.4% on the manually-verified validation set; runs at ~30 FPS on a CPU-only laptop and ~200 FPS on the 2080 Ti; satisfies the consumer-grade hardware constraint (PR-2). |
| **Bear auto-annotation** | Grounding DINO | Best ODinW benchmark score among open detectors at selection time, and best zero-shot accuracy on our manual bear spot-check set; text-prompt flexibility allowed us to label 24K frames without building a manual labeling team. |
| **Multi-object tracking** | ByteTrack (with our fragmentation merger) | Top of MOT17/MOT20 leaderboard; on the 5-bear manually-counted test clip, motion-only association is the right choice when subjects look identical; second-stage low-conf recovery handles splash occlusion. |
| **Bear feeding behavior classification** | Molmo2-8B (per-frame, with on-image ID overlay) | Highest stage-match accuracy on our 50-frame manual feeding ground truth among open 7–8B VLMs tested; open weights so we can run on-prem; native bbox/point support; consistent structured output. |
| **Salmon jump counting** | *Unresolved.* | Molmo2 video mode is currently the best demonstrated approach but is hardware-limited. |

---

## 7. Missing Model Capabilities

Across every model we tried, the following capability gaps materially slowed progress on bear detection, tracking, and salmon-feeding analysis:

1. **Memory-efficient long-video VLM inference.** Every open-weights VLM (Molmo2, LLaVA-OneVision, Qwen2.5-VL, VideoLLaMA2) hits a hard wall around 8–16 video frames on 22 GB of VRAM because per-frame visual tokens × O(n²) attention dominate. There is currently no open VLM that supports streaming video understanding (KV-cache eviction, sliding-window attention, or hierarchical token compression) on consumer GPUs. Closed models like Gemini 1.5 Pro have this but at API cost and with data-residency constraints. **If this existed, salmon jump counting and long-form bear behavior analysis would be solvable in a single pass instead of frame-by-frame.**

2. **Cross-instance Re-ID for visually identical animals.** Pretrained Re-ID networks are trained on humans / vehicles and fail on bears that look near-identical. None of the trackers we tested can reliably re-associate a bear that fully exits and re-enters the frame. Solving cross-video bear identification was actually descoped from the project for this reason. A wildlife-specific Re-ID foundation model (analogous to MegaDetector for detection) is missing from the open-source ecosystem.

3. **Native temporal action / event localization.** Detection models output bounding boxes per frame, and VLMs output text per frame, but neither natively outputs `(start_time, end_time, event_label, agent_id)` tuples. We have to reconstruct events from per-frame outputs with custom dedup and stage-tracking logic. A true video-action-detection foundation model (something like a wildlife-domain ActionFormer or VideoMAE-V2 fine-tuned on animal behavior) would let us treat "salmon catch" as a first-class detected event.

4. **Small-object detection in heavily textured backgrounds.** Salmon in turbulent water (waterfall splash, surface reflections) defeat all three open-vocabulary detectors we tried. Recall is poor and false positives on splash artifacts are high. A model with explicit motion-cue or optical-flow conditioning would likely help.

5. **Cheap on-prem fine-tuning of large VLMs.** We could not fine-tune Molmo2-8B on Katmai-specific behavior labels because LoRA + 8B + bf16 video training does not fit comfortably on dual 2080 Ti without a heroic engineering effort (gradient checkpointing, ZeRO-3, etc.). Smaller VLMs (~2B) do fit but lose enough reasoning quality to be useless. A 2–3B parameter VLM with Molmo2-level reasoning would change what is achievable on lab hardware.

---

## 8. AI Tools Used or Modified

| Tool | URL | Pros | Cons / Where it needed supervision |
|---|---|---|---|
| **Ultralytics YOLOv8 framework** | https://github.com/ultralytics/ultralytics | Single-line train/predict/track API; built-in ByteTrack/BoT-SORT integration; HuggingFace + ONNX export; large community. | Tracker config YAML format was poorly documented — `fuse_score: True` is required for ultralytics' tracker but not mentioned in the official ByteTrack README, costing hours of debugging. Default `vid_stride=1` silently processes every frame, catastrophically slow on long video. |
| **HuggingFace `transformers` (4.57.6)** | https://github.com/huggingface/transformers | Unified `AutoModelForImageTextToText` + `AutoProcessor` for Molmo2; chat-template handles video loading; `device_map="auto"` for multi-GPU. | Version churn is a major hazard: transformers 5.5.4 broke Molmo2 with `Unexpected keyword argument image_use_col_tokens`; transformers 4.57.6 requires `torch>=2.6` (`Using or_mask_function arguments require torch>=2.6`). Pinning was non-trivial. |
| **Grounding DINO (HuggingFace port)** | https://huggingface.co/IDEA-Research/grounding-dino-base | Easy text-prompt API; reproducible across machines via HF Hub. | Slow (~1.5 s/frame); occasional CUDA OOM on 1080p frames without explicit downscaling. |
| **CLIP ViT-L (HuggingFace)** | https://huggingface.co/openai/clip-vit-large-patch14 | Reliable feature extractor; fast on GPU. | Embedding stability across CLIP versions varied; we pinned a specific revision. |
| **OpenCV + ffmpeg** | https://opencv.org / https://ffmpeg.org | Video decode, frame extraction, trimming, format conversion. | OpenCV `VideoCapture` on `.mov` files was unreliable on Linux without ffmpeg backend; `XVID` writer outputs `.avi` which we re-encode to `.mp4` for sharing. |
| **decord** | https://github.com/dmlc/decord | Required by Molmo2's video processor; faster than OpenCV for random-access frame loading. | Quietly required — Molmo2 raised `ImportError: requires decord, torchcodec, or av` and `pip install decord` fixed it. |
| **CVAT** | https://github.com/opencv/cvat | Industry-standard labeling UI; YOLO format export; used to construct manual ground-truth sets. | Heavy to self-host (Docker, multiple services); used sparingly for spot-check verification rather than full annotation due to timeline. |
| **Label Studio** | https://github.com/HumanSignal/label-studio | Lighter than CVAT; flexible label schemas; good for the human-in-the-loop review queue in the salmon MoE pipeline and for the 50-frame feeding ground-truth set. | Web UI sometimes lags on long video frames; export format requires post-processing to reach YOLO format. |
| **bitsandbytes** | https://github.com/bitsandbytes-foundation/bitsandbytes | 4-bit / 8-bit quantization to fit larger VLMs in VRAM. | Requires a working C compiler and recent CUDA; on our ENGINE Lab machine we hit `RuntimeError: Failed to find C compiler` on Triton compilation and reverted to bf16. Not a drop-in replacement. |
| **PyTorch (2.6.0+cu124)** | https://pytorch.org | Required for the latest transformers; `bf16` + `device_map="auto"` distributes Molmo2 across both 2080 Ti GPUs cleanly. | Tight version coupling to transformers/CUDA; required a careful upgrade plan to avoid breaking Ultralytics. |
| **OpenCompass / OpenVLM Leaderboard** | https://rank.opencompass.org.cn/leaderboard-multimodal | Centralized ranking of open VLMs across MMBench, MMMU, Video-MME, MVBench; used to shortlist candidate VLMs. | Benchmark scores do not always transfer to wildlife footage — we still had to verify with manual ground truth. |
| **HuggingFace Open VLM Leaderboard** | https://huggingface.co/spaces/opencompass/open_vlm_leaderboard | Same purpose as OpenCompass; cross-checked rankings. | Same benchmark-to-domain transfer caveat. |
| **Cursor / Claude Code (Anthropic)** | https://claude.com/claude-code | Used as an AI pair-programmer for prompt iteration on Molmo2, debugging the ByteTrack ID-fragmentation merger, and writing the side-by-side viewer rendering code. Hugely accelerated the Spring Quarter prototyping loop. | Needed close supervision on VLM behavior — LLM-generated code occasionally invented HuggingFace API arguments (e.g., `image_use_col_tokens`) that didn't exist in our pinned transformers version; needed to verify every API call against actual source. Also hallucinated salmon jump timestamps when reasoning about Molmo2's video output instead of running the model. |
| **GitHub + GitHub Actions** | https://github.com | Branch protection (2 reviewer approvals to merge to main), shared SSH deploy keys, thrice-daily auto-pulls on the ENGINE Lab machine. | Standard, no surprises. |
| **Slack** | https://slack.com | Async coordination with sponsor and team; easy file/screenshot sharing for visual debugging. | – |

---

## Summary of Selection Rationale

The pipeline converged on **YOLOv8 (fine-tuned) + Grounding DINO (auto-annotation) + ByteTrack (tracking) + Molmo2-8B (behavior + summary)** because that combination is the only one that simultaneously:

1. Runs on the available hardware (dual 2080 Ti for training, consumer laptop for inference).
2. Uses fully open-weights models so the open-source deliverable has no API dependency or licensing barrier.
3. Achieves >90% accuracy on the **manually-verified Katmai validation set** for bear detection.
4. Achieves the highest stage-match accuracy on our **manually-labeled feeding ground truth** for behavior classification.
5. Produces structured, parseable output that downstream feeding-event detection and environmental-data correlation can consume.

Every model choice was justified by **(a) public benchmark rankings to shortlist candidates** and **(b) Katmai-specific manual ground truth to make the final selection.**

The remaining open problem is **independent salmon detection / jump counting**, where current open VLMs are blocked by the VRAM-bound video-frame budget, and where a fine-tuned YOLO salmon detector is the most likely path forward.
