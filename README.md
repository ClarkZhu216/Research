# Diffusion-to-CLIP Representation Research

This repository contains a structured deep-research study on using diffusion models to improve CLIP / VLM representations, with emphasis on:

- diffusion-to-CLIP representation distillation;
- diffusion internal features as auxiliary visual representations;
- dense region / patch supervision from diffusion attention or feature maps;
- counterfactual and domain-specific generation for CLIP adaptation;
- relevance to CueBench, CLIPDiffusion, FGCLIP, and video anomaly detection.

## Key Takeaway

The most promising direction is not simply generating more synthetic images for CLIP. The stronger research path is:

```text
frozen or mostly frozen CLIP / FGCLIP encoder
+ diffusion reconstruction feedback or dense feature supervision
+ lightweight patch adapters
+ contrastive preservation of CLIP text alignment
+ anomaly / counterfactual local cue objectives
```

In short: use diffusion as a teacher, feature reservoir, reconstructor, or dense pseudo-label source.

## Files

- `outline.yaml`: research outline and item list.
- `fields.yaml`: schema used by the deep-research JSON files.
- `focused_diffusion_clip_distillation_memo.md`: focused memo on the directions closest to the current project.
- `report.md`: human-readable research summary.
- `results/*.json`: structured deep-research outputs for 26 research items.

## Most Relevant Results

Start with these files:

- `results/Diffusion_Feedback_Reconstruction_for_CLIP_Posttraining.json`
- `results/Globaltoken_Conditioned_Generative_Enhancement.json`
- `results/Contrastiveguided_Diffusion_Reconstruction_for_Balanced_CLIP_Features.json`
- `results/unCLIP_Inversion_for_CLIP_Detail_Preservation.json`
- `results/Layer_and_Timestep_Selection_for_Diffusion_Feature_Transfer.json`
- `results/Diffusion_Feature_Fusion_for_CLIP_and_VLM_Encoders.json`
- `results/CLIP_Patch_Adapters_from_Diffusion_Dense_Signals.json`
- `results/Representation_Distillation_Between_Diffusion_and_CLIP.json`
- `results/Diffusion_Features_as_Auxiliary_Representation.json`

## Suggested Prototype

```text
video frame
  -> CLIP / FGCLIP image encoder
  -> patch tokens

same frame + prompt
  -> frozen diffusion model
  -> selected dense feature / attention / concept map

patch adapter
  -> align CLIP local features to diffusion dense cues
  -> preserve global CLIP image-text alignment
  -> feed into video anomaly / temporal localization head
```

Recommended first experiments:

1. Probe diffusion layer / timestep features.
2. Add a lightweight CLIP patch adapter.
3. Compare CLIP-only, CLIP+adapter, CLIP+diffusion-fusion, and CLIP+diffusion-distillation.
4. Evaluate on fine-grained cue retrieval and video anomaly metrics.

