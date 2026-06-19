# Focused Memo: Diffusion-to-CLIP Representation Distillation

## Scope

This memo focuses on the two directions currently closest to the project goal:

1. Representation distillation between diffusion and CLIP.
2. Diffusion features as auxiliary representations for CLIP or VLM encoders.

The key shift is to treat diffusion as a teacher, reconstructor, feature reservoir, or dense pseudo-label source, not merely as a synthetic image generator.

## Core Research Claim

CLIP's contrastive pretraining gives strong global image-text alignment, but it often compresses away local structure, subtle attributes, object parts, spatial layout, and abnormal cues. Diffusion models must preserve these details to generate images, so their reconstruction feedback and internal states can provide supervision that CLIP lacks.

The useful research question is therefore:

> How can diffusion-derived reconstruction signals or internal features improve CLIP's local and fine-grained visual representation while preserving its original text-aligned embedding space?

## Main Subdirections

### 1. Diffusion Feedback Reconstruction

Representative route: DIVA.

Mechanism:

- Feed CLIP image features or visual tokens into a diffusion reconstructor.
- Use image reconstruction or denoising feedback as self-supervised signal.
- Update CLIP image encoder so it preserves more visual detail.
- Keep text encoder fixed to retain original CLIP alignment.

Why it matters:

- Does not require paired captions.
- Avoids building a huge synthetic corpus.
- Directly targets CLIP representation quality.

Main risk:

- Reconstruction may force CLIP to encode irrelevant pixel detail.
- It can hurt discrimination if not balanced.

### 2. Global-token Conditioned Generative Enhancement

Representative route: GenHancer.

Mechanism:

- Use selected CLIP tokens, especially global visual tokens, as denoiser conditions.
- Avoid giving the denoiser too much local-token leakage.
- Use staged training: first train projector/denoiser path, then tune CLIP visual encoder.
- Lightweight denoisers may be enough; perfect generation is not required.

Key lesson:

Better reconstruction quality is not automatically better representation. The denoiser should create gradients that help CLIP encode useful visual factors, not merely reproduce pixels.

Project implication:

For FGCLIP/video anomaly work, start with a lightweight reconstruction assistant and control which frame/patch/global tokens enter it.

### 3. Contrastive-guided Diffusion Reconstruction

Representative route: DCR.

Mechanism:

- Combine diffusion reconstruction with contrastive or discriminative signals.
- Use reconstructed images or reconstruction-conditioned features to preserve class separability.
- Optimize detail perception and discrimination together.

Why this is important:

This is probably the most important loss-design lesson for your direction. A pure diffusion reconstruction loss may improve local detail but damage CLIP's global semantics. A contrastive balancing term is needed.

Potential project objective:

```text
L = L_clip_global
  + lambda_rec * L_diffusion_reconstruction
  + lambda_region * L_diffusion_dense_region
  + lambda_cons * L_source_counterfactual_consistency
```

For video anomaly:

```text
L = L_video_mil
  + lambda_clip * L_text_video_contrast
  + lambda_rec * L_frame_diffusion_feedback
  + lambda_patch * L_abnormal_region_alignment
```

### 4. unCLIP / Generator Inversion

Representative route: un2CLIP.

Mechanism:

- Use a generator already conditioned on CLIP image embeddings.
- Invert or use the generator's feedback to improve CLIP detail preservation.
- Preserve compatibility with the original text embedding space.

Why it is attractive:

The generator and CLIP space are already structurally connected, so it may be cleaner than generic Stable Diffusion reconstruction.

Open question:

Whether this route can be adapted to video-frame CLIP or FGCLIP-style encoders without losing temporal consistency.

### 5. Diffusion Feature Extraction and Selection

Representative routes: DIFT, diffusion activation selection, Diffusion Hyperfeatures.

Mechanism:

- Extract U-Net or DiT activations from selected layers and timesteps.
- Use them as dense descriptors, correspondences, or semantic maps.
- Fuse them with CLIP tokens or distill them into CLIP patch adapters.

Important signal choices:

- U-Net cross-attention queries.
- Attention output vectors.
- Residual block outputs.
- Early-to-mid denoising timesteps.
- Up-stage U-Net features.
- DiT output-space concept vectors.

Project implication:

For a first experiment, do not search all layers. Start with a small controlled grid:

```text
backbone: Stable Diffusion v2.1 or SDXL
features: cross-attention query, attention output, residual output
timesteps: 10, 50, 100
stages: mid block, up block early, up block late
```

Evaluate which features best separate normal/abnormal or source/counterfactual pairs.

### 6. Diffusion Feature Fusion

Representative route: Mustafar.

Mechanism:

- Keep diffusion backbone frozen.
- Extract task-aware diffusion features.
- Fuse with CLIP or VLM visual tokens through a projector, cross-attention module, or adapter.

Why it is practical:

- Fastest route to test value before retraining CLIP.
- Compatible with existing CLIP/FGCLIP pipelines.
- Can be ablated cleanly against CLIP-only features.

Suggested project prototype:

```text
frame -> CLIP image encoder -> patch/global tokens
frame + prompt -> frozen diffusion -> dense features / attention maps
concat or cross-attend -> lightweight adapter
adapter output -> anomaly score / text alignment / MIL head
```

### 7. REPA-style Hidden-state Alignment

Representative routes: REPA, U-REPA.

Mechanism:

- Align diffusion hidden states with strong vision encoder representations.
- Preserve relative manifold geometry using feature or manifold losses.

Why it matters:

These works mainly improve diffusion training, but they prove that diffusion hidden states and discriminative visual representations can be made compatible. The inverse direction is interesting: use diffusion hidden states as targets to improve CLIP patch or frame features.

Possible adaptation:

```text
CLIP_patch_adapter(frame) ~= Projector(diffusion_hidden_state(frame, t, layer))
```

Add a global CLIP contrastive loss to prevent the adapter from drifting away from text semantics.

## Best Near-term Method Idea

The most project-aligned method is:

> Diffusion-supervised CLIP patch adapter for fine-grained abnormal cue representation.

Architecture:

```text
Input frame or sampled video frame
  -> frozen CLIP image encoder
  -> CLIP patch tokens

Same frame + text prompt
  -> frozen diffusion model
  -> selected dense feature / attention / concept map

CLIP patch tokens
  -> lightweight adapter
  -> local representation aligned to diffusion dense cues

Video aggregation
  -> FGCLIP / MIL / temporal anomaly head
```

Training signals:

- Global image-text CLIP contrast to preserve semantics.
- Diffusion dense feature distillation for local cues.
- Region-text contrast for abnormal or counterfactual regions.
- Source-counterfactual consistency for unchanged content.
- MIL or temporal localization loss for video anomaly.

Why this is better than plain synthetic augmentation:

- It directly changes the representation.
- It can be trained with real frames.
- It keeps CLIP mostly frozen, lowering compute and overfitting risk.
- It gives a clean bridge from diffusion priors to video anomaly localization.

## Suggested Experiments

### Experiment A: Feature Utility Probe

Goal:

Find which diffusion features are useful before training a large model.

Setup:

- Extract CLIP frame embeddings and diffusion features from the same frames.
- Compare normal/abnormal separation with linear probes.
- Test feature fusion by concatenation or shallow adapter.

Metrics:

- UCF-Crime AUC.
- XD-Violence AP.
- Frame-level or segment-level anomaly localization if available.
- Retrieval accuracy for abnormal text prompts.

### Experiment B: Patch Adapter Distillation

Goal:

Train a CLIP patch adapter supervised by diffusion dense features.

Losses:

- Patch-feature distillation loss.
- Region-text contrastive loss.
- Global CLIP preservation loss.

Baseline:

- Frozen CLIP.
- CLIP + simple adapter.
- CLIP + synthetic augmentation only.

### Experiment C: Reconstruction-balanced Post-training

Goal:

Test whether diffusion reconstruction feedback improves CLIP frame features.

Losses:

- Diffusion reconstruction loss.
- Contrastive preservation loss.
- Optional anomaly classification or MIL loss.

Key ablation:

- Reconstruction only.
- Contrastive only.
- Reconstruction + contrastive.
- Reconstruction + contrastive + region supervision.

### Experiment D: Source-counterfactual Consistency

Goal:

Use your existing counterfactual generation pipeline to enforce local semantic changes.

Loss idea:

- Changed region should move toward changed text.
- Unchanged region should remain close to source representation.
- Global embedding should reflect normal/abnormal label change only when the anomaly cue changes.

## Most Important Risks

1. Reconstruction gradients may encode irrelevant visual detail.
2. Diffusion features may be expensive to extract at scale.
3. Dense pseudo-labels may fail on surveillance images with low resolution or clutter.
4. Gains may be limited to fine-grained probes and not improve broad zero-shot behavior.
5. Video temporal consistency is not solved by image diffusion features alone.
6. If the text encoder is also fine-tuned, CLIP compatibility may degrade.

## Recommended Next Deep-research Items

Prioritize these newly added outline items:

1. Diffusion Feedback Reconstruction for CLIP Post-training.
2. Global-token Conditioned Generative Enhancement.
3. Contrastive-guided Diffusion Reconstruction for Balanced CLIP Features.
4. Diffusion Feature Fusion for CLIP and VLM Encoders.
5. Layer and Timestep Selection for Diffusion Feature Transfer.
6. CLIP Patch Adapters from Diffusion Dense Signals.

Keep unCLIP inversion and REPA-style alignment as secondary but important theoretical support.
