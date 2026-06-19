# 使用 diffusion 优化 CLIP 表征

## Direction Card

| 字段 | 内容 |
| --- | --- |
| 研究方向 | 使用 diffusion 生成/监督/蒸馏来优化 CLIP / VLM 表征 |
| 当前阶段 | 第一轮系统调研完成，正在收敛到可实验的方法路线 |
| 上一次探索时间 | 2026-06-18 |
| 调研范围 | 2024 年以来 diffusion-to-CLIP representation、dense supervision、counterfactual generation、feature fusion、reconstruction feedback |
| 已调研主题数 | 26 |
| 原始结果 | [`results/*.json`](../results/) |
| 完整报告 | [`report.md`](../report.md) |
| 聚焦 memo | [`focused_diffusion_clip_distillation_memo.md`](../focused_diffusion_clip_distillation_memo.md) |

## Research Question

CLIP 的全局图文对比学习很强，但经常压缩掉局部视觉证据：物体状态、细粒度属性、空间关系、局部异常区域、动作线索和 counterfactual 中真正变化的区域。Diffusion model 为了生成或重建图像，必须保留这些视觉结构，因此可以作为 CLIP 表征优化的教师、特征库或伪监督来源。

核心问题是：

> 如何利用 diffusion 的重建反馈、内部 dense feature、attention/concept map 或 counterfactual editing 能力，让 CLIP 获得更强的局部与细粒度表征，同时不破坏原本的 text-aligned embedding space？

## Current Summary

目前调研下来的主线可以概括为五类。

### 1. Diffusion reconstruction feedback

代表工作包括 DIVA、GenHancer、DCR、un2CLIP。它们把 CLIP image feature 作为 diffusion 或 unCLIP reconstructor 的条件，用 denoising / reconstruction loss 反向优化 CLIP image encoder。

这类工作的共同结论是：diffusion 重建信号能让 CLIP 保留更多视觉细节，但单纯追求重建可能损伤判别性。因此最重要的设计不是“生成质量越高越好”，而是控制信息泄漏、选择合适 conditioning token，并加入 contrastive / discriminative balance。

### 2. Diffusion feature as auxiliary representation

代表工作包括 DIFT、diffusion activation selection、Diffusion Hyperfeatures、Mustafar 等。它们不一定修改 CLIP，而是抽取 diffusion U-Net / DiT 的中间层、attention、query、residual 或 output-space concept feature，再与 CLIP token 融合或蒸馏。

这条路线适合快速验证，因为可以冻结 diffusion 和 CLIP，只训练轻量 projector / adapter。调研中反复出现的经验是：layer、timestep、feature type 是隐藏关键变量，不能只随便取一个 diffusion feature。

### 3. Dense region and patch supervision

代表工作包括 OVAM、ConceptAttention、DIFT、DiCLIP 以及 patch-adapter 方向。它们利用 diffusion attention map、concept map、correspondence 或 dense feature 给 CLIP patch token 提供局部监督。

这条路线最贴近 video anomaly 和 FGCLIP，因为异常往往不是全局类别，而是局部状态、动作、物体关系或细粒度 cue。最自然的做法是保持 CLIP/FGCLIP 基本冻结，在 patch token 后接 adapter，用 diffusion dense signal 监督 adapter。

### 4. Counterfactual and compositional generation

这类工作用 diffusion 生成最小语义编辑：属性替换、关系变化、空间位置变化、异常 cue 注入、背景 shortcut 控制等。它适合构造对比组、hard negative、source-counterfactual consistency loss。

对当前项目的价值在于：可以把“源视频/图像中未改变的部分保持不变”和“被编辑的异常 cue 必须改变表征”同时写进训练目标。

### 5. Evaluation, risk, and protocol design

调研结果显示，很多 synthetic 或 diffusion-assisted 方法容易只在某些 benchmark 上变好，但未必真的改进了通用表征。必须同时看：

- zero-shot / retrieval 是否退化；
- fine-grained / compositional task 是否提升；
- local cue localization 是否更准；
- anomaly normal/abnormal separation 是否更好；
- generator bias、prompt leakage、benchmark contamination 是否可控。

## Paper Map

### Highest Priority For Current Work

| 论文/方向 | 核心作用 | 对当前项目的启发 | 原始结果 |
| --- | --- | --- | --- |
| DIVA: Diffusion Feedback Helps CLIP See Better | diffusion 作为 frozen reconstructor，给 CLIP image encoder 提供重建反馈 | 可作为 diffusion-to-CLIP post-training baseline | [JSON](../results/Diffusion_Feedback_Reconstruction_for_CLIP_Posttraining.json) |
| GenHancer | 强调 global-token conditioning、两阶段训练、避免 local-token leakage | 说明不是重建越强越好，token 选择和泄漏控制更关键 | [JSON](../results/Globaltoken_Conditioned_Generative_Enhancement.json) |
| DCR | 用 contrastive signal 平衡 diffusion reconstruction，避免判别性下降 | 当前视频异常方向很需要这种 reconstruction + discrimination balance | [JSON](../results/Contrastiveguided_Diffusion_Reconstruction_for_Balanced_CLIP_Features.json) |
| un2CLIP | 用 unCLIP inversion 改善 CLIP 细节保留，同时保持 text space 兼容 | 如果使用与 CLIP embedding 对齐的生成器，蒸馏会更干净 | [JSON](../results/unCLIP_Inversion_for_CLIP_Detail_Preservation.json) |
| DIFT / diffusion feature selection | diffusion 中间层可作为 dense correspondence / feature | 先做 layer-timestep grid，找到对异常 cue 最有用的 feature | [JSON](../results/Layer_and_Timestep_Selection_for_Diffusion_Feature_Transfer.json) |
| Mustafar-style fusion | frozen diffusion feature 与 CLIP/VLM token 融合 | 最快验证 diffusion internals 是否能增强 FGCLIP pipeline | [JSON](../results/Diffusion_Feature_Fusion_for_CLIP_and_VLM_Encoders.json) |
| CLIP patch adapters from diffusion dense signals | patch adapter 接受 diffusion dense supervision | 最贴近“局部异常 cue 表征优化”的可执行路线 | [JSON](../results/CLIP_Patch_Adapters_from_Diffusion_Dense_Signals.json) |

### Supporting Directions

| 方向 | 主要问题 | 原始结果 |
| --- | --- | --- |
| Representation distillation between diffusion and CLIP | diffusion 与 CLIP 表征空间如何互相蒸馏 | [JSON](../results/Representation_Distillation_Between_Diffusion_and_CLIP.json) |
| Diffusion features as auxiliary representation | diffusion feature 是否可以直接补充 CLIP | [JSON](../results/Diffusion_Features_as_Auxiliary_Representation.json) |
| DiT internal dense priors | DiT / Flux-style backbone 内部 signal 如何用于 dense supervision | [JSON](../results/DiTInternal_Dense_Priors_for_Local_CLIP_Supervision.json) |
| Region, mask, attention supervision | diffusion attention / mask 是否能监督 region-text alignment | [JSON](../results/Diffusionbased_Region_Mask_and_Attention_Supervision.json) |
| Temporal and video counterfactual diffusion | 视频 counterfactual 是否能用于 video-CLIP / anomaly | [JSON](../results/Temporal_and_Video_Counterfactual_Diffusion_for_VideoCLIP.json) |
| Training objectives beyond contrastive | 除了 InfoNCE 之外还需要哪些目标 | [JSON](../results/Training_Objectives_Beyond_Standard_Contrastive_Learning.json) |
| Evaluation protocols | 如何证明表征真的变强，而不是 benchmark-specific gain | [JSON](../results/Evaluation_Protocols_for_Diffusionoptimized_CLIP_Representations.json) |
| Failure modes, bias, leakage | diffusion-assisted CLIP 的泄漏、偏差、污染风险 | [JSON](../results/Failure_Modes_Bias_and_Leakage_in_Diffusionoptimized_CLIP.json) |

### Full Result Index

| 路线 | Research item | 入口 |
| --- | --- | --- |
| Diffusion teacher / distillation | Representation Distillation Between Diffusion and CLIP | [JSON](../results/Representation_Distillation_Between_Diffusion_and_CLIP.json) |
| Diffusion teacher / distillation | Diffusion Feedback Reconstruction for CLIP Post-training | [JSON](../results/Diffusion_Feedback_Reconstruction_for_CLIP_Posttraining.json) |
| Diffusion teacher / distillation | Global-token Conditioned Generative Enhancement | [JSON](../results/Globaltoken_Conditioned_Generative_Enhancement.json) |
| Diffusion teacher / distillation | Contrastive-guided Diffusion Reconstruction for Balanced CLIP Features | [JSON](../results/Contrastiveguided_Diffusion_Reconstruction_for_Balanced_CLIP_Features.json) |
| Diffusion teacher / distillation | unCLIP Inversion for CLIP Detail Preservation | [JSON](../results/unCLIP_Inversion_for_CLIP_Detail_Preservation.json) |
| Diffusion feature / dense signal | Diffusion Features as Auxiliary Representation | [JSON](../results/Diffusion_Features_as_Auxiliary_Representation.json) |
| Diffusion feature / dense signal | DiT-Internal Dense Priors for Local CLIP Supervision | [JSON](../results/DiTInternal_Dense_Priors_for_Local_CLIP_Supervision.json) |
| Diffusion feature / dense signal | Diffusion-based Region, Mask, and Attention Supervision | [JSON](../results/Diffusionbased_Region_Mask_and_Attention_Supervision.json) |
| Diffusion feature / dense signal | Layer and Timestep Selection for Diffusion Feature Transfer | [JSON](../results/Layer_and_Timestep_Selection_for_Diffusion_Feature_Transfer.json) |
| Diffusion feature / dense signal | Diffusion Feature Fusion for CLIP and VLM Encoders | [JSON](../results/Diffusion_Feature_Fusion_for_CLIP_and_VLM_Encoders.json) |
| Diffusion feature / dense signal | REPA-style Hidden-state Alignment Between Diffusion and Vision Encoders | [JSON](../results/REPAstyle_Hiddenstate_Alignment_Between_Diffusion_and_Vision_Encoders.json) |
| Diffusion feature / dense signal | CLIP Patch Adapters from Diffusion Dense Signals | [JSON](../results/CLIP_Patch_Adapters_from_Diffusion_Dense_Signals.json) |
| Counterfactual / compositional data | Diffusion-generated Counterfactuals for Representation Learning | [JSON](../results/Diffusiongenerated_Counterfactuals_for_Representation_Learning.json) |
| Counterfactual / compositional data | Set-Structured Counterfactual Training for CLIP | [JSON](../results/SetStructured_Counterfactual_Training_for_CLIP.json) |
| Counterfactual / compositional data | Compositional and Relational Diffusion Data | [JSON](../results/Compositional_and_Relational_Diffusion_Data.json) |
| Counterfactual / compositional data | Diffusion-generated Spurious and Shortcut Probes for CLIP Debiasing | [JSON](../results/Diffusiongenerated_Spurious_and_Shortcut_Probes_for_CLIP_Debiasing.json) |
| Domain / continual / data efficiency | Diffusion-based Causal Prompt and Adapter Learning | [JSON](../results/Diffusionbased_Causal_Prompt_and_Adapter_Learning.json) |
| Domain / continual / data efficiency | Domain-specific Diffusion Priors for CLIP Adaptation | [JSON](../results/Domainspecific_Diffusion_Priors_for_CLIP_Adaptation.json) |
| Domain / continual / data efficiency | Generator Personalization for Representation Data Synthesis | [JSON](../results/Generator_Personalization_for_Representation_Data_Synthesis.json) |
| Domain / continual / data efficiency | Fully Synthetic CLIP Pretraining and Corpus Design | [JSON](../results/Fully_Synthetic_CLIP_Pretraining_and_Corpus_Design.json) |
| Domain / continual / data efficiency | Synthetic Replay for Continual CLIP/VLM Adaptation | [JSON](../results/Synthetic_Replay_for_Continual_CLIPVLM_Adaptation.json) |
| Domain / continual / data efficiency | Diffusion-based Data Condensation and Budgeted Synthetic Curriculum | [JSON](../results/Diffusionbased_Data_Condensation_and_Budgeted_Synthetic_Curriculum.json) |
| Video / evaluation / risk | Temporal and Video Counterfactual Diffusion for Video-CLIP | [JSON](../results/Temporal_and_Video_Counterfactual_Diffusion_for_VideoCLIP.json) |
| Video / evaluation / risk | Training Objectives Beyond Standard Contrastive Learning | [JSON](../results/Training_Objectives_Beyond_Standard_Contrastive_Learning.json) |
| Video / evaluation / risk | Evaluation Protocols for Diffusion-optimized CLIP Representations | [JSON](../results/Evaluation_Protocols_for_Diffusionoptimized_CLIP_Representations.json) |
| Video / evaluation / risk | Failure Modes, Bias, and Leakage in Diffusion-optimized CLIP | [JSON](../results/Failure_Modes_Bias_and_Leakage_in_Diffusionoptimized_CLIP.json) |

## What The Field Is Doing

这个方向现在不是单一路线，而是在回答同一个问题：CLIP 的 global contrastive representation 缺少细节，diffusion 是否能把生成模型中的视觉结构补给 CLIP？

已经比较清楚的趋势有：

- 从 synthetic data scaling 转向 representation feedback：早期直觉是多生成图片，现在更强的工作倾向于用 diffusion 直接监督 CLIP feature。
- 从 global embedding 转向 patch / dense token：仅优化全局 image embedding 不够，局部区域、属性、关系和 abnormal cue 才是关键。
- 从 reconstruction-only 转向 balanced objective：重建信号有用，但要同时保持 CLIP 的图文对齐和判别边界。
- 从 heavy full fine-tuning 转向 frozen backbone + adapter：为了降低成本和避免破坏 CLIP，很多路线倾向于 projector、LoRA、patch adapter、fusion adapter。
- 从 image-only 逐渐指向 video / temporal setting：现有强证据主要来自图像，但异常检测需要把 frame-level dense cue 扩展到 clip-level temporal reasoning。

## Proposed Method Direction

当前最建议推进的方法名可以暂定为：

> Diffusion-supervised CLIP Patch Adapter for Abnormal Cue Representation

基本结构：

```text
video frame
  -> frozen or mostly frozen CLIP / FGCLIP image encoder
  -> patch tokens + global token

same frame + text prompt
  -> frozen diffusion model
  -> dense feature / attention / concept / correspondence signal

CLIP patch adapter
  -> align local CLIP tokens to diffusion dense cues
  -> preserve global CLIP text alignment
  -> feed temporal anomaly / retrieval / localization head
```

候选目标函数：

```text
L = L_video_anomaly
  + lambda_global * L_clip_preservation
  + lambda_dense  * L_diffusion_feature_distill
  + lambda_region * L_region_text_contrast
  + lambda_cf     * L_source_counterfactual_consistency
```

## Experiment Plan

### Experiment 1: Diffusion feature utility probe

先不训练大模型，只比较特征是否有用。

比较组：

- CLIP / FGCLIP only；
- diffusion feature only；
- CLIP + diffusion concat；
- CLIP + shallow fusion adapter。

建议评估：

- UCF-Crime AUC；
- XD-Violence AP；
- abnormal text retrieval；
- source/counterfactual pair separation；
- 如有标注，评估 local cue localization。

### Experiment 2: Layer and timestep grid

先从小网格开始：

```text
backbone: Stable Diffusion v2.1 or SDXL
stage: mid block, early up block, late up block
timestep: 10, 50, 100
feature: cross-attention query, attention output, residual output
```

目标是确定哪种 diffusion signal 对异常 cue 最敏感，而不是一开始就做完整大搜索。

### Experiment 3: Patch adapter distillation

对比：

- frozen CLIP；
- frozen CLIP + adapter；
- frozen CLIP + adapter + diffusion dense supervision；
- frozen CLIP + adapter + diffusion dense supervision + global CLIP preservation。

重点看 diffusion supervision 是否真的改善局部异常 cue，而不是只提高训练集指标。

### Experiment 4: Reconstruction-balanced post-training

借鉴 DCR 的思想，测试 reconstruction 与 discrimination 的平衡。

对比：

- reconstruction only；
- contrastive only；
- reconstruction + contrastive；
- reconstruction + contrastive + region supervision。

## Risks And Checks

| 风险 | 检查方式 |
| --- | --- |
| diffusion feature 带来 style/aesthetic bias | 加入真实视频 anomaly benchmark 和 OOD split |
| reconstruction 让表征过度关注像素细节 | 同时报告 zero-shot / retrieval / anomaly separation |
| local-token leakage 让任务变成重建捷径 | 控制 conditioning token，比较 global-only 与 patch-token density |
| counterfactual 不够局部或不够忠实 | 检查 edit locality、source preservation、changed cue alignment |
| 视频时序不一致 | 先做 frame-level probe，再加入 clip-level consistency |

## Files To Read First

1. [`report.md`](../report.md)：完整调研总结。
2. [`focused_diffusion_clip_distillation_memo.md`](../focused_diffusion_clip_distillation_memo.md)：最贴近当前方法设计的聚焦 memo。
3. [`results/Diffusion_Feedback_Reconstruction_for_CLIP_Posttraining.json`](../results/Diffusion_Feedback_Reconstruction_for_CLIP_Posttraining.json)：DIVA / GenHancer / DCR / un2CLIP 相关路线。
4. [`results/CLIP_Patch_Adapters_from_Diffusion_Dense_Signals.json`](../results/CLIP_Patch_Adapters_from_Diffusion_Dense_Signals.json)：当前最可执行的 patch adapter 路线。
5. [`results/Layer_and_Timestep_Selection_for_Diffusion_Feature_Transfer.json`](../results/Layer_and_Timestep_Selection_for_Diffusion_Feature_Transfer.json)：feature selection 的实验依据。

## Update Log

| 日期 | 更新内容 |
| --- | --- |
| 2026-06-18 | 完成 26 个 research items 的 deep-research，生成 `results/*.json`、`report.md` 和聚焦 memo。 |
| 2026-06-19 | 将仓库结构调整为研究方向探索站点，新增方向页、主页索引和维护模板。 |
