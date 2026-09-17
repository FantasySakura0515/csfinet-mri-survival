# Recorded aggregate results

These files contain the recorded summaries for the manuscript. They contain cohort/model aggregates; individual clinical records, patient-level predictions, MRI and attribution arrays are not distributed. The identifier-only partition is available in `../../data/splits/patients.csv`.

| Manuscript material | Public data |
| --- | --- |
| Tables 5 and 5b | `segmentation-raw.csv`, `segmentation-tta.csv`, `segmentation-paired-tests.csv` |
| Table 7 | `survival-v2mask-fixed.csv`, `survival-paired-tests.csv` |
| Survival dispersion supporting table | `survival-dispersion.csv` |
| Figures 8–11 | `figure8-9-bin-counts.csv`, `figure10-11-correlations.csv` |
| Table 9 | `shap.csv`, `shap-randomization.csv`, `shap-area-reference.csv` |

`manuscript-data-map.csv` describes the mapping. `segmentation-statistics.json` records the input hashes, software versions, per-interval seeds and output hashes. Segmentation summaries use 10,000 patient-level BCa resamples, a lexicographic patient order and base seed 20260914. Dice/precision/recall consume three consecutive seeds per region (WT, TC, ET), first CSFINet and then U-Net; TTA starts at base + 18. IoU retains its separate diagnostic schedule and cannot shift the manuscript measures. Paired comparisons start at 20260964; Holm adjustment is applied separately within families A and C. See [statistical reconciliation](../../docs/statistical-reconciliation.md).

Run `python scripts/summarize_results.py` after generating patient metrics, or use the segmentation evaluator, which calls the same summarizer. Authoritative segmentation outputs are written to `results/publication/`. The model inference is unchanged. Numerical environments can produce small differences in regenerated predictions; record device, batch size and precision.

The study uses patient-disjoint internal held-out evaluation: 188 training and 47 test patients, with development training/validation of 150/38 within the training group. Model and method choices were fixed before test evaluation, as confirmed by the authors. Training membership records support patient separation. This is not external validation.

Survival and SHAP use raw CSFINet masks. Survival training/internal validation use GT WT masks; test inputs use predicted WT masks and four correspondingly masked MRI volumes, with fixed weights and training-fitted clinical transformations. The eight neural/reference error comparisons did not reach significance after Holm correction. SHAP explains segmentation, not survival.

File and run names such as `v2mask-fixed` are stable artifact identifiers; they do not denote a comparison between study versions. Individual illustrative-case clinical values and image arrays are not supplied in these aggregate files.
