# Statistical methods

Patients are the statistical unit. Mean values and sample standard deviations use equal patient weighting. Pointwise 95% BCa intervals use 10,000 patient-level resamples with patient IDs sorted lexicographically. The bootstrap base seed is 20260914.

Dice, precision and recall use three consecutive seeds per region (WT, TC, ET), first for CSFINet and then U-Net. Raw inference uses offsets 0–17; TTA uses offsets 18–35. Diagnostic IoU uses a separate schedule. Paired comparisons start at seed 20260964; the individual seeds appear in `results/reference/segmentation-paired-tests.csv`. Holm correction is applied separately within the two segmentation comparison families and the eight survival neural/reference comparisons.

The evaluation command and `scripts/summarize_results.py` share the same segmentation summarizer. `results/reference/segmentation-statistics.json` records seeds, software versions and input/output hashes. The manuscript and distributed aggregate tables use this schedule. Patient-level metrics are generated locally using the separately obtained dataset.

Survival errors, prediction dispersion and correlations are supplied as aggregate CSVs. The reported neural models do not demonstrate a statistically significant improvement over the reference models after multiplicity correction. SHAP correlations and secondary segmentation-region summaries are exploratory. These intervals describe patient-level uncertainty within the evaluated cohort; they do not estimate variability across repeated model training or external cohorts.
