# GitHub Code Bases for Implementation

Generated on 2026-06-20.

## Short Answer

如果目标是“尽快在现有代码上改出一个能跑的 diffusion-supervised CLIP / video anomaly 原型”，我建议不要只 fork 一个仓库，而是用三层组合：

```text
VadCLIP
  as downstream video anomaly framework

DCR or DIVA
  as CLIP representation post-training framework

DIFT / OVAM / DiCLIP
  as diffusion dense signal extraction modules
```

最实际的第一版路线是：

```text
fork VadCLIP
  -> 替换或扩展 CLIP feature extraction
  -> 接入 DIFT/OVAM 生成的 diffusion dense features
  -> 在 VadCLIP 的 MIL / prompt / temporal head 上加 adapter supervision

parallel fork DCR
  -> 学习 reconstruction + contrastive post-training 代码
  -> 将 image-level CC3M training 改成 video frame / anomaly cue frame post-training
```

也就是说：

- 想快速对 UCF-Crime / XD-Violence 出结果：从 **VadCLIP** 改。
- 想做“优化 CLIP 表征”这篇方法本体：从 **DCR** 改。
- 想做 patch-level / region-level diffusion supervision：把 **DIFT / OVAM / DiCLIP** 当模块接进来。

## Recommended Ranking

| Rank | Repo | Use As | Why It Fits | Main Risk |
| ---: | --- | --- | --- | --- |
| 1 | [nwpu-zxr/VadCLIP](https://github.com/nwpu-zxr/VadCLIP) | Video anomaly downstream base | 已经是 CLIP + weakly supervised video anomaly；有 UCF-Crime / XD-Violence 训练测试入口 | 主要用预提取 CLIP feature，需要你重写 feature cache / adapter 接口 |
| 2 | [boyuh/DCR](https://github.com/boyuh/DCR) | CLIP representation post-training base | 有 diffusion reconstruction + contrastive balance，两阶段训练脚本，支持 OpenAI CLIP / MetaCLIP / SigLIP | 图像级 CC3M 代码，需要迁移到视频帧和异常 cue |
| 3 | [Tsingularity/dift](https://github.com/Tsingularity/dift) | Diffusion dense feature extractor | 直接提供 diffusion feature extraction，能输出 torch tensor，适合做 patch supervision | 不是完整训练框架，只是 feature module |
| 4 | [baaivision/DIVA](https://github.com/baaivision/DIVA) | Baseline CLIP enhancement base | 官方 ICLR 2025 代码，思路直接，支持多个 CLIP backbone | 需要改 installed CLIP/OpenCLIP/timm 源码，工程侵入性较强 |
| 5 | [zwyang6/DiCLIP](https://github.com/zwyang6/DiCLIP) | Dense CLIP + diffusion mechanism reference | 有 diffusion correlation map、CLIP dense refinement、KV cache 思路 | 面向 WSSS，不是 video anomaly |
| 6 | [vpulab/ovam](https://github.com/vpulab/ovam) | Open-vocabulary attention pseudo-label module | training-free，基于 diffusers hook attention map，适合生成 region/text cue | 主要是 attention attribution，不是表征训练框架 |
| 7 | [huggingface/diffusers](https://github.com/huggingface/diffusers) + [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | Clean engineering base | 最稳、最模块化，长期维护好 | 需要自己搭训练 loop 和实验协议 |
| 8 | [helblazer811/ConceptAttention](https://github.com/helblazer811/ConceptAttention) | DiT / Flux / video concept heatmap module | 支持 Flux、SD3、CogVideoX concept heatmap，对未来视频 diffusion 有价值 | 比 SD U-Net 路线重，第一版成本高 |
| 9 | [mashijie1028/GenHancer](https://github.com/mashijie1028/GenHancer) | Lightweight generative enhancer reference | 两阶段训练、global-token conditioning 的经验很重要 | DCR 已经基于它并补了 contrastive balance |
| 10 | [LiYinqi/un2CLIP](https://github.com/LiYinqi/un2CLIP) | unCLIP inversion reference | embedding-compatible generator idea 很干净 | Stable unCLIP 官方 checkpoint 获取存在现实阻碍 |
| 11 | [joos2010kj/CLIP-TSA](https://github.com/joos2010kj/CLIP-TSA) | Simple VAD baseline | 结构简单，适合快速 sanity check | 不如 VadCLIP 贴近语言-视觉异常对齐 |
| 12 | [Sumutan/GV-VAD](https://github.com/Sumutan/GV-VAD) | Synthetic video augmentation reference | 正好是 video generation + WSVAD | 更偏数据增强，不是 diffusion-to-CLIP representation |

## Best Candidate 1: VadCLIP

Repo: <https://github.com/nwpu-zxr/VadCLIP>

Why it should be your downstream base:

- 它直接解决 weakly supervised video anomaly detection。
- 它本身就是 CLIP-based video anomaly framework，而不是普通 VAD。
- 代码结构有 `data/`、`list/`、`src/`，并提供 `xd_train.py`、`xd_test.py`、`ucf_train.py`、`ucf_test.py` 这类训练测试入口。
- README 说明它释放了 UCF-Crime / XD-Violence 的 CLIP features 和 pretrained models。

How to modify it:

```text
VadCLIP
  src/model temporal branch
  + CLIP patch/global feature cache
  + diffusion dense feature cache
  + patch adapter / fusion adapter
  + region-text or abnormal-cue alignment loss
```

具体改法：

- 把输入 feature 从纯 CLIP feature 扩展为 `{clip_global, clip_patch, diffusion_dense}`。
- 在 temporal branch 前加一个 `DiffusionPatchAdapter`。
- 保留 VadCLIP 的 MIL / prompt / LGT-Adapter 作为 video anomaly head。
- 先离线缓存 diffusion features，避免训练时每个 batch 跑 Stable Diffusion。

Verdict:

> 如果你想最快在 UCF-Crime / XD-Violence 上跑出 baseline 和 ablation，就从 VadCLIP fork。

## Best Candidate 2: DCR

Repo: <https://github.com/boyuh/DCR>

Why it should be your representation-learning base:

- 它正好做 diffusion-based reconstruction + contrastive balanced visual representation。
- README 显示它有 Stage-1 / Stage-2：Stage-1 冻结 CLIP 训练 projector，Stage-2 fine-tune CLIP。
- 支持 OpenAI CLIP、MetaCLIP、SigLIP 多种 backbone。
- 使用 Stable Diffusion v2.1，工程路线和你当前 diffusion-supervised CLIP 假设一致。

How to modify it:

```text
DCR image-level training
  -> replace CC3M samples with sampled video frames / cue frames
  -> add abnormal/normal or source/counterfactual grouping
  -> add patch-level diffusion dense distillation
  -> export enhanced CLIP or adapter weights
  -> plug into VadCLIP
```

Possible objective:

```text
L = L_DCR_reconstruction_contrast
  + lambda_patch * L_diffusion_dense_distill
  + lambda_clip  * L_global_clip_preservation
  + lambda_vad   * L_video_anomaly_or_prompt_align
```

Verdict:

> 如果你想把“用 diffusion 优化 CLIP 表征”做成方法论文，DCR 是目前最像直接代码底座的仓库。

## Best Candidate 3: DIFT

Repo: <https://github.com/Tsingularity/dift>

Why it matters:

- 它能直接从 diffusion U-Net 抽 dense feature。
- README 提供 `extract_dift.py`，可以保存 torch tensor。
- 参数包括 timestep `t`、U-Net upsampling block index、prompt、ensemble size。
- 对你的 patch adapter / local abnormal cue supervision 非常直接。

How to use it:

```text
video frame
  -> DIFT extract_dift.py
  -> diffusion dense feature tensor
  -> cache by video_id / frame_id
  -> supervise CLIP patch adapter
```

First grid:

```text
t: 51, 101, 261
up_ft_index: 1, 2
prompt: "a surveillance video frame", anomaly-specific prompt
```

Verdict:

> DIFT 不适合作为完整项目主仓库，但非常适合作为 diffusion dense feature extraction 子模块。

## Best Candidate 4: DIVA

Repo: <https://github.com/baaivision/DIVA>

Why it is useful:

- 官方 ICLR 2025 code。
- 已经有 OpenAI CLIP / MetaCLIP / SigLIP / DFN 的训练入口。
- 提供 training & evaluation code 和 released CLIP model weights。
- 目标就是通过 diffusion feedback post-training 改善 CLIP visual shortcomings。

Why I would not make it the first base:

- README 明确需要替换 conda 环境里 installed `clip`、`open_clip`、`timm` 的部分源码。
- 这种做法复现论文可以，但作为你自己的长期项目底座会比较脆。
- DCR 继承了同一大方向，并加入 contrastive balance，工程结构也更像可扩展训练框架。

Verdict:

> DIVA 适合做 baseline 和复现，不是我最建议的主开发底座。

## Dense Supervision Modules

### DiCLIP

Repo: <https://github.com/zwyang6/DiCLIP>

Use it for:

- diffusion correlation map；
- CLIP dense knowledge refinement；
- diffusion-generated KV cache；
- attention clustering refinement ideas。

Why it is relevant:

- README 说明代码、数据、checkpoint 已释放。
- 代码目录包含 `clip/`、`diffusion_model/`、`maintain_kv_cache/`、`model/`、`scripts/`。
- 它的 Visual Correlation Enhancement 思路非常接近“用 diffusion dense prior 改 CLIP patch representation”。

Main adaptation:

```text
WSSS image CAM refinement
  -> video frame abnormal cue localization
  -> diffusion correlation as patch adapter target
```

### OVAM

Repo: <https://github.com/vpulab/ovam>

Use it for:

- open-vocabulary token attention maps；
- text-to-region attribution；
- abnormal prompt heatmap pseudo-label。

Why it is convenient:

- 它是 training-free extension。
- 可以 `pip install -e .`。
- README 给了 diffusers `StableDiffusionPipeline` + `StableDiffusionHooker` 的使用方式。

### ConceptAttention

Repo: <https://github.com/helblazer811/ConceptAttention>

Use it for:

- Flux / SD3 / CogVideoX concept heatmaps；
- 未来扩展到 video diffusion 的 concept-level pseudo-label。

Why not first:

- 当前第一版用 SD v1.5 / SD 2.1 U-Net feature 更容易。
- Flux / CogVideoX 资源成本和工程复杂度更高。

## Clean Library Base

### open_clip

Repo: <https://github.com/mlfoundations/open_clip>

Use it as:

- CLIP backbone loader；
- tokenizer / preprocess；
- training utilities if building from scratch。

Important note:

- README 说明 main branch training stack 已经大改，旧训练 API 建议 pin v3 或 latest 3.x release。
- 如果你只是加载 CLIP feature，直接用 `open_clip_torch` 即可。

### diffusers

Repo: <https://github.com/huggingface/diffusers>

Use it as:

- Stable Diffusion / SDXL / SD3 / video diffusion pipeline loader；
- hook attention / UNet blocks；
- scheduler / model components。

Verdict:

> 如果你想搭一个长期干净的研究代码库，底层应该用 `open_clip + diffusers`；但第一版不要从零搭，先在 VadCLIP / DCR 上做最小可行实验。

## Not First-choice But Useful

### GenHancer

Repo: <https://github.com/mashijie1028/GenHancer>

Useful lessons:

- global token condition often better than local token leakage；
- lightweight denoisers can be enough；
- two-stage training is important。

Why secondary:

- DCR 已经 built upon GenHancer，并进一步处理 reconstruction/discrimination balance。

### un2CLIP

Repo: <https://github.com/LiYinqi/un2CLIP>

Useful lesson:

- unCLIP generator 与 CLIP embedding space 更兼容，理论上蒸馏更干净。

Why risky:

- README 提到官方 Stable unCLIP checkpoints / repos 当前不再公开可用，需要替代来源或 Wayback。
- 因此不适合做第一版工程底座。

### CLIP-TSA

Repo: <https://github.com/joos2010kj/CLIP-TSA>

Use it for:

- simple CLIP-feature WSVAD sanity check；
- lightweight temporal self-attention baseline。

Why secondary:

- 它没有 VadCLIP 那种 language-visual alignment branch。

### GV-VAD

Repo: <https://github.com/Sumutan/GV-VAD>

Use it for:

- text-conditioned video generation augmentation；
- synthetic sample loss scaling；
- anomaly video generation pipeline reference。

Why secondary:

- 它更偏 data augmentation，而你的主问题是 diffusion-to-CLIP representation。

## Proposed Fork Strategy

### Option A: Fastest Paper Prototype

```text
Fork VadCLIP
Add DIFT feature cache
Add CLIP patch adapter
Train on UCF-Crime / XD-Violence
Compare:
  VadCLIP
  VadCLIP + DIFT concat
  VadCLIP + patch adapter
  VadCLIP + patch adapter + diffusion dense supervision
```

This is the fastest route to a measurable video anomaly result.

### Option B: Strongest Representation Method

```text
Fork DCR
Replace CC3M image samples with selected video frames
Add abnormal / normal grouping
Add diffusion dense feature target from DIFT or DiCLIP
Train enhanced CLIP / adapter
Plug enhanced encoder into VadCLIP
```

This is the best route if the paper's core claim is representation optimization.

### Option C: Clean Long-term Codebase

```text
New repo
  open_clip for CLIP
  diffusers for diffusion
  VadCLIP-style VAD head
  DIFT/OVAM-style dense signal modules
```

This is cleanest long term, but slower for the first result.

## My Recommendation

Start with Option A, but keep DCR as a parallel reference.

Concrete first milestone:

```text
1. Fork VadCLIP.
2. Make its dataloader accept extra frame-level feature tensors.
3. Use DIFT to cache diffusion dense features for a small subset.
4. Add a small adapter:
   CLIP_patch_tokens -> adapter -> local_anomaly_feature
5. Train with existing VadCLIP losses plus diffusion feature distillation.
6. Run ablation on UCF-Crime first.
```

Reason:

- It touches the fewest moving parts.
- It gives direct video anomaly metrics.
- It lets you show whether diffusion dense signals help before committing to expensive CLIP post-training.

After that:

```text
Use DCR to post-train / LoRA-tune the CLIP visual encoder or adapter,
then plug the enhanced encoder back into the VadCLIP fork.
```

## Source Links

- DIVA: <https://github.com/baaivision/DIVA>
- DCR: <https://github.com/boyuh/DCR>
- GenHancer: <https://github.com/mashijie1028/GenHancer>
- un2CLIP: <https://github.com/LiYinqi/un2CLIP>
- DIFT: <https://github.com/Tsingularity/dift>
- DiCLIP: <https://github.com/zwyang6/DiCLIP>
- OVAM: <https://github.com/vpulab/ovam>
- ConceptAttention: <https://github.com/helblazer811/ConceptAttention>
- VadCLIP: <https://github.com/nwpu-zxr/VadCLIP>
- CLIP-TSA: <https://github.com/joos2010kj/CLIP-TSA>
- GV-VAD: <https://github.com/Sumutan/GV-VAD>
- open_clip: <https://github.com/mlfoundations/open_clip>
- diffusers: <https://github.com/huggingface/diffusers>
