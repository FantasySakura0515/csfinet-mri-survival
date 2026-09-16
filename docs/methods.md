# Implementation and version map

This repository supports **CSFINet-Based Multimodal MRI Image Segmentation and Survival Prediction**. The model is a paper-guided implementation; [NOTICE.md](../NOTICE.md) identifies the architecture source and the upstream commit consulted. It is not a recovery of the manuscript's earlier historical experiment.

| Component | Released implementation |
| --- | --- |
| Input | Four volume-normalized MRI channels, ordered FLAIR, T1, T1ce, T2; axial 240 x 240 slices |
| Segmentation encoder | Convolutional widths 64/128/256/512/1024 |
| CSFINet global branch | torchvision Swin blocks at the three deeper scales, depths 2/2/6 and heads 4/8/16 |
| PIU | Local cross-scale groups of 4 x 4, 2 x 2 and 1 x 1 positions, 21 tokens per group, common dimension 512, eight attention heads |
| FIU | Shared flow for two feature warps followed by convolution/BatchNorm/ReLU; bilinear sampling, zero padding, align_corners=True |
| Output | Four segmentation logits; native BraTS labels 0/1/2/4 |
| U-Net | Matching convolutional encoder widths, transposed-convolution upsampling, concatenated skips and four-class output |
| Segmentation training | Dice + cross-entropy; Adam; patient-level accumulated update; flip/intensity augmentation; warmup plus cosine schedule |
| Survival | WT mask plus four masked MRI channels, 5 x 155 x 128 x 128; 3D convolution widths 16/32/64; 128-dimensional image embedding |
| Clinical features | Training-fitted age standardization and GTR/STR/NA one-hot status; four image/clinical combinations |
| Survival training | Adam, Huber loss with delta 365 days; GT masks for training and internal validation |

Dimensions, architecture choices and survival heads are explicitly implemented in `csfinet_repro/models.py`; full numeric settings are in the configuration files. Standard framework layers and default initialization are used except for explicitly specified initialization, including zero-initialized flow convolution. Activation checkpointing preserves BatchNorm buffers during recomputation. CUDA grid-sampling backward does not guarantee bitwise reproducibility.

## Configuration scope

- `configs/reconstruction-v21.json`: actual released segmentation training recipe, output tag `v2`.
- `configs/survival-v2.json`: actual five-channel survival recipe, training tag `v2mri`. Its embedded older segmentation block is not used for the released segmentation models.
- `configs/analysis.json`: fixed-region Gradient SHAP and statistical settings.
- `configs/reconstruction-v2.json`: retained earlier configuration for protocol tests; not the released segmentation recipe.

Configuration bytes are retained because checkpoint provenance identifies them by SHA-256. The `declared_inference` entries describe evaluated candidates, not a requirement to combine all candidate procedures. The validation-selected procedure was flip TTA; neither LCC filtering nor BatchNorm recalibration is adopted in the reported TTA comparison.

## Fixed-weight evaluation

Raw and TTA segmentation use the same checkpoint for each model. TTA averages inverse-flipped class probabilities over identity, horizontal flip, vertical flip and both flips before argmax. The same four orientations are applied to CSFINet and U-Net. The release evaluator applies this already recorded decision; it does not make a new test-set selection.

The four survival models retain their original training weights and clinical transformations. `v2mask-fixed` substitutes raw v2 CSFINet test masks and rebuilds the corresponding masked MRI channels. Preprocessing rounds the five-channel volume through float16 cache representation, then uses FP32 neural inference with TF32 disabled. This is a post-hoc reevaluation on previously inspected test patients; it is neither retraining nor external validation.

SHAP explains mean WT probability within a fixed GT-WT or predicted-WT region. It uses 32 GradientShap samples and an all-zero normalized MRI baseline. Quantitative magnitude sums absolute attribution over modalities; the viewer sums signed modalities and scales each slice symmetrically by its maximum absolute value. These are different, documented uses of the same attribution array. The localization and randomization diagnostics do not establish clinical explanation fidelity.

## Commands retained for research workflows

The package also retains lower-level training, reporting, validation-variant and provenance-audit modules with their data-free tests. Historical delivery/report commands can require a full original run directory and can refer to archived `v2mri` analyses; they are not the public release evaluation entry point. Use `scripts/evaluate_release.py` for the current manuscript's released models and `results/reference/` for its aggregate results. See [reproduction.md](reproduction.md) for supported command sequences.
