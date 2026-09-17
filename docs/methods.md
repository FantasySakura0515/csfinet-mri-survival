# Implementation and evaluation protocol

This repository supports **CSFINet-Based Multimodal MRI Image Segmentation and Survival Prediction**. The model is a paper-guided implementation; [NOTICE.md](../NOTICE.md) identifies the architecture source and the upstream commit consulted.

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

## Configuration

- `configs/segmentation.json`: segmentation architecture, optimization and inference settings.
- `configs/survival.json`: five-channel survival model and optimization settings; its segmentation block is not used by the released segmentation models.
- `configs/analysis.json`: fixed-region Gradient SHAP and statistical settings.
- `configs/training-options.json`: configuration fixture for testing supported training options.

The manifest binds each checkpoint to its configuration using SHA-256. The validation-selected procedure is four-orientation flip TTA; LCC filtering and BatchNorm recalibration are not adopted in the reported TTA comparison.

## Fixed-weight evaluation

Raw and TTA segmentation use the same checkpoint for each model. TTA averages inverse-flipped class probabilities over identity, horizontal flip, vertical flip and both flips before argmax. The same four orientations are applied to CSFINet and U-Net. The release evaluator applies this already recorded decision; it does not make a new test-set selection.

The four survival models use fixed training weights and training-fitted clinical transformations. Training and internal validation use GT WT masks; held-out test inputs use raw CSFINet WT masks and four correspondingly masked MRI channels. Preprocessing rounds the five-channel volume through float16 cache representation, then uses FP32 neural inference with TF32 disabled. Model and method choices preceded test evaluation without test-score-driven selection. All test patients were excluded from model fitting and internal validation. This is internal held-out evaluation, without external validation.

SHAP explains mean WT probability within a fixed GT-WT or predicted-WT region. It uses 32 GradientShap samples and an all-zero normalized MRI baseline. Quantitative magnitude sums absolute attribution over modalities; the viewer sums signed modalities and scales each slice symmetrically by its maximum absolute value. These are different, documented uses of the same attribution array. The localization and randomization diagnostics do not establish clinical explanation fidelity.

## Statistical summaries

The public evaluator and `scripts/summarize_results.py` share the manuscript bootstrap schedule. See [statistical methods](statistics.md) and the per-interval seed manifest in `results/reference/segmentation-statistics.json`.
