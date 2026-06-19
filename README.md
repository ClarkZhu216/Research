# Research Directions

这是一个用来持续记录研究方向探索过程的仓库。它不是单篇论文笔记，而是一个研究路线站点：每个方向都会保存调研时间、论文地图、阶段性结论、可复现实验想法，以及原始 deep-research 结构化结果。

当前仓库会优先服务我的 CLIP / diffusion / video anomaly 方向探索。后续每开启一个新方向，都在 `directions/` 下建立独立页面，主页只保留方向索引和最新状态。

## Direction Index

| 方向 | 当前状态 | 上一次探索时间 | 论文/主题数 | 入口 |
| --- | --- | --- | ---: | --- |
| 使用 diffusion 优化 CLIP 表征 | 已完成第一轮系统调研，已补充工程代码底座选择 | 2026-06-20 | 26 | [方向页](directions/diffusion-clip-representation/) |

## What This Site Records

每个研究方向建议固定记录这些内容：

- 研究问题：这个方向到底想解决什么表征、数据或评估问题。
- 更新时间：上一次系统调研、补充论文、修改结论的日期。
- 论文地图：按方法路线整理论文，而不是只按年份堆列表。
- 当前共识：这个方向目前主流工作都在做什么。
- 对我项目的关系：它如何连接 CueBench、CLIPDiffusion、FGCLIP、video anomaly 或其他当前实验。
- 可执行下一步：最小可跑实验、关键 ablation、风险检查。
- 原始证据：deep-research 生成的 JSON、outline、field schema 和报告文件。

## Current Main Direction

### 使用 diffusion 优化 CLIP 表征

这个方向关注如何把 diffusion model 的生成先验、重建反馈、内部 dense feature、attention/concept map 或 counterfactual editing 能力转移到 CLIP / VLM 表征中。

阶段性结论是：最值得做的不是简单生成更多 synthetic image 去扩充 CLIP 训练集，而是把 diffusion 当作：

- visual teacher；
- reconstruction feedback provider；
- dense feature reservoir；
- region / patch pseudo-label source；
- counterfactual cue generator。

目前最适合当前项目的路线是：

```text
frozen or mostly frozen CLIP / FGCLIP
+ diffusion dense feature / reconstruction supervision
+ lightweight patch adapter
+ global CLIP text alignment preservation
+ abnormal cue / counterfactual consistency objective
```

更完整的论文地图和方法总结见：

- [方向页：使用 diffusion 优化 CLIP 表征](directions/diffusion-clip-representation/)
- [完整调研报告](directions/diffusion-clip-representation/report.md)
- [聚焦 memo：diffusion-to-CLIP distillation](directions/diffusion-clip-representation/focused_diffusion_clip_distillation_memo.md)

## Repository Structure

```text
.
├── README.md
├── directions/
│   ├── TEMPLATE.md
│   └── diffusion-clip-representation/
│       ├── README.md
│       ├── outline.yaml
│       ├── fields.yaml
│       ├── report.md
│       ├── focused_diffusion_clip_distillation_memo.md
│       └── results/
│           └── *.json
```

## Maintenance Notes

当继续调研某个方向时，建议同步更新三处：

1. `directions/<direction>/README.md`：更新探索时间、论文地图、阶段性结论。
2. `README.md`：更新主页方向索引里的“上一次探索时间”和状态。
3. `directions/<direction>/results/`：保留每轮 deep-research 的结构化 JSON，方便之后让 Codex 或本地脚本继续解析。

如果新增研究方向，复制 [directions/TEMPLATE.md](directions/TEMPLATE.md) 到 `directions/<new-direction>/README.md` 后填写即可。
