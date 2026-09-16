# Reproduction guide

[English overview](../README.md) · [繁體中文總覽](../README.zh-TW.md)

Run commands from the repository root after installing `python -m pip install -e ".[all]"` in an activated environment. The commands below work in PowerShell and Bash because each command is on one line. Use Python 3.10 with PyTorch 2.7.1 and torchvision 0.22.1; select a compatible CPU or CUDA build from the [official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/#v271). Training requires CUDA, while the release evaluator also accepts CPU.

## 1. Reconstruct the local cohort

Source metadata comes from [awsaf49/brats2020-training-data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data), **version 3**. Native MRI comes from [awsaf49/brats20-dataset-training-validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation), **version 1**. The former is used for clinical labels, name mapping and source checks; the latter supplies the four original MRI modalities and segmentation labels.

```bash
python scripts/prepare_data.py --download-source
python scripts/prepare_data.py --verify-only
```

The first command downloads the small source files, verifies their pinned hashes, reconstructs `data/manifests/cohort/patients.csv` locally, and checks the supplied patient-ID split definition. The public repository does not contain that clinical CSV. If the source files are already available, place them in `data/raw/source-check/` and run `python scripts/prepare_data.py` without the download flag. Filenames and SHA-256 values are listed in `configs/source-files.json`; use the listed version rather than substituting a newer export.

The cohort contains 235 patients with numerical survival labels. The split uses seed `20260914` for 188 training/47 test patients and seed `20260915` for 150 development-training/38 internal-validation patients. All slices and modalities of each patient stay together. The split is reconstructed and verified locally; do not redraw it when evaluating the released checkpoints.

Download and validate all 235 patients:

```bash
python -m csfinet_repro download-nifti --workers 3
python -m csfinet_repro audit-nifti
```

The NIfTI downloader pins version 1, records file provenance, and resumes verified files. Each patient supplies five volumes, for **1,175 files** in total. The expected local layout is:

```text
data/raw/brats2020-nifti/
  BraTS20_Training_001/
    BraTS20_Training_001_flair.nii.gz
    BraTS20_Training_001_t1.nii.gz
    BraTS20_Training_001_t1ce.nii.gz
    BraTS20_Training_001_t2.nii.gz
    BraTS20_Training_001_seg.nii.gz
```

The downloader also writes per-file JSON records used by `audit-nifti`. Copying arbitrary NIfTI files into the directory without those records is not equivalent to completing this download-and-audit workflow. The downloader handles the nonstandard source segmentation filename for patient 355.

MRI has shape `240 × 240 × 155`, 1 mm spacing and native labels `0, 1, 2, 4`; the model maps label 4 to class index 3 internally. Intensities are standardized per patient and modality over the entire volume, including background, with population SD and epsilon `1e-8`. Dataset access remains subject to the source datasets' terms.

## 2. Download the release checkpoints

```bash
python scripts/download_weights.py --component all
python scripts/download_weights.py --verify-only
```

The downloader uses the tracked `release-manifest.json` and this repository's [v2.0.1 Release](https://github.com/FantasySakura0515/csfinet-mri-survival/releases/tag/v2.0.1), verifies checksums, and extracts into `runs/`. `--component segmentation` and `--component survival` select a subset. Checkpoints include the CSFINet/U-Net segmentation models and the four final survival configurations: `image`, `image_age`, `image_resection`, and `image_age_resection`.

The four survival checkpoints retain their `v2mri` training provenance. Training and internal validation used GT whole-tumor masks. Their earlier evaluation used archived v1 predicted test masks; the manuscript's current `v2mask-fixed` analysis holds weights and fitted clinical transformations fixed and uses raw v2 CSFINet masks instead. The mask substitution also changes the four masked MRI channels.

Use `evaluate_release.py` for this declared fixed-model reevaluation. The core `infer-survival` command intentionally checks the cache manifest associated with a training run; it must not be made to accept a different cache by editing hashes or run metadata. Core inference remains appropriate for newly trained models evaluated with their matching cache.

## 3. Evaluate the released models

```bash
python scripts/evaluate_release.py --stage segmentation --device auto
python scripts/evaluate_release.py --stage survival --device auto
python scripts/evaluate_release.py --stage shap --device auto
```

`--device auto` chooses an available supported device; `--device cpu` and `--device cuda` select it explicitly. `--stage all` runs the three stages in order. Full evaluation uses the frozen 47 test patients.

Add `--resume` to continue an interrupted stage only when its recorded inputs and settings still match. `--slice-batch 2` can reduce segmentation inference memory use; the recorded default is 20. An altered inference batch can cause small floating-point differences and is saved in the output manifest. This option does not change training settings or SHAP sampling. Downloaded weights are verified against the manifest before evaluation.

| Stage | Main local output |
| --- | --- |
| Segmentation | `results/segmentation/csfinet-test-v2/` and `unet-test-v2/`; prediction NIfTIs under `artifacts/segmentation/` |
| Survival | `results/survival/v2mask-fixed/`, including the six-model patient prediction CSV |
| SHAP | `results/explainability/v2/` and per-patient attribution files under `artifacts/explainability/v2/` |

The segmentation stage evaluates raw and matched in-plane flip TTA for both models. The reported TTA decision uses the internal validation set. Survival and SHAP use **raw CSFINet** output, not TTA masks. SHAP explains fixed GT-WT and predicted-WT segmentation targets, not the survival model.

A small check can be run before the full segmentation and survival stages:

```bash
python scripts/evaluate_release.py --stage segmentation --device auto --limit 1
python scripts/evaluate_release.py --stage survival --device auto --limit 1
```

Limited runs write to `results/smoke/` and `artifacts/smoke/`, separate from full-cohort outputs. They check that the pipeline runs; they do not reproduce the paper's 47-patient metrics. SHAP requires full-cohort evaluation.

The main reference values are WT Dice **0.7625/0.7373** for raw CSFINet/U-Net and **0.7664/0.7412** with matched TTA. Current fixed-model survival MAE is **256.50 days** for Image + Age + Resection Status versus **254.54 days** for age-only OLS. These are reported results of the recorded runs. Compare numerical differences with the evaluation settings and device recorded in the generated manifests before interpreting them as a model change.

The survival reevaluation reuses a previously inspected test partition. Neither it nor retraining the same split constitutes external validation. All eight neural-versus-reference comparisons have Holm-adjusted p = 1.0 in the reported analysis. SHAP localization alone does not establish explanation fidelity; the reported parameter-randomization diagnostics retain substantial similarity.

## 4. Open the local viewer

```bash
python -m streamlit run web.py
```

The viewer reads locally generated raw MRI, segmentation masks, survival predictions and SHAP artifacts. Downloading weights and MRI does not generate these outputs. Complete the evaluation stages first to make all views available; a segmentation-only evaluation provides no survival or SHAP outputs. The viewer is intended for research inspection.

## 5. Train new models

Training produces a new run of the reconstructed protocol, not the missing historical experiment. Use separate directories so the released checkpoints and evaluation outputs remain traceable. The examples below use `runs/retrained/`, `results/retrained/` and `artifacts/retrained/`.

Use `configs/reconstruction-v21.json` for current segmentation. Its v2 protocol uses foreground Dice plus cross-entropy, spatial/intensity augmentation, warmup and cosine decay. Select duration on the 150/38 internal partition, then refit on 188 patients. The released runs selected 28 CSFINet epochs and 22 U-Net epochs; a new run should follow its own recorded validation selection rather than force those counts.

```bash
python -m csfinet_repro train-segmentation --config configs/reconstruction-v21.json --model csfinet --phase development --output runs/retrained/csfinet-development
python -m csfinet_repro train-segmentation --config configs/reconstruction-v21.json --model csfinet --phase refit --selection runs/retrained/csfinet-development/run.json --output runs/retrained/csfinet-refit
python -m csfinet_repro infer-segmentation --config configs/reconstruction-v21.json --run runs/retrained/csfinet-refit --output results/retrained/csfinet-test --predictions artifacts/retrained/csfinet-test
python -m csfinet_repro train-segmentation --config configs/reconstruction-v21.json --model unet --phase development --output runs/retrained/unet-development
python -m csfinet_repro train-segmentation --config configs/reconstruction-v21.json --model unet --phase refit --selection runs/retrained/unet-development/run.json --output runs/retrained/unet-refit
python -m csfinet_repro infer-segmentation --config configs/reconstruction-v21.json --run runs/retrained/unet-refit --output results/retrained/unet-test --predictions artifacts/retrained/unet-test
python -m csfinet_repro summarize-segmentation --input results/retrained/csfinet-test/segmentation_patient_metrics.csv --input results/retrained/unet-test/segmentation_patient_metrics.csv --output results/retrained/segmentation-table
```

The inference commands above perform **raw** evaluation. They do not silently add TTA. A separate validation/test inference-variant workflow is available through `python -m csfinet_repro validate-inference-variants --help` and `evaluate-inference-variants --help`; it expects its documented run-directory layout.

Use `configs/survival-v2.json` for the five-channel survival models. Its embedded legacy segmentation block is not the configuration for current v2 segmentation. Build a fresh cache from the new CSFINet predictions and train all four configurations:

```bash
python -m csfinet_repro cache-survival-mri --predictions artifacts/retrained/csfinet-test --segmentation-manifest results/retrained/csfinet-test/inference-manifest.json --output data/cache/survival-retrained
python -m csfinet_repro.survival_suite --config configs/survival-v2.json --cache data/cache/survival-retrained --runs runs/retrained --tag localv2
python -m csfinet_repro infer-survival --config configs/survival-v2.json --cache data/cache/survival-retrained --image-run runs/retrained/survival-image-refit-localv2 --image-age-run runs/retrained/survival-image_age-refit-localv2 --image-resection-run runs/retrained/survival-image_resection-refit-localv2 --image-age-resection-run runs/retrained/survival-image_age_resection-refit-localv2 --output results/retrained/survival/survival_predictions.csv
python -m csfinet_repro summarize-survival-table --input results/retrained/survival/survival_predictions.csv --output results/retrained/survival-table
```

The cache uses GT masks for training/validation and predicted masks for test patients. Its five channels are WT and masked FLAIR, T1, T1ce and T2, with volume shape `5 × 155 × 128 × 128`. Clinical preprocessing is fitted only on the applicable training partition. Mean-survival and age-only OLS references use the 188 training patients.

The training suite records provenance and resumes verified completed runs. If you change the data, configuration or mask source, create a new cache and run directory rather than reusing an incompatible completed run. The core table command produces the comparisons documented in its output; results from newly trained models must not be labeled as the released `v2mask-fixed` analysis.

## Checks and troubleshooting

```bash
python -m pytest -q
python -m csfinet_repro --help
python scripts/prepare_data.py --help
python scripts/download_weights.py --help
python scripts/evaluate_release.py --help
```

- **Source hash mismatch:** check the Kaggle dataset and version, file name, and whether the response was a login page. Do not update the expected hash to bypass a mismatch.
- **Missing prediction masks:** run segmentation before survival or SHAP. Smoke outputs are separate from full-cohort artifacts.
- **Missing viewer panels:** generate the corresponding stage outputs locally; the release does not distribute patient predictions or attribution arrays.
- **CUDA unavailable:** verify the installed PyTorch build and driver. CPU evaluation is supported; training uses CUDA.
- **Out of memory:** preserve the original run and use a separately documented configuration for any altered training settings. A successful smoke evaluation is not a guarantee that full training or SHAP will fit.

The source repository and release weights provide the reconstructed implementation and fixed models. They do not establish exact recovery of the earlier manuscript's missing historical execution, external generalization, or clinical utility.

See [methods.md](methods.md) for the implementation choices, configuration scope and raw/TTA/mask-version distinctions, and [NOTICE.md](../NOTICE.md) for source attribution and license status.
