# Recorded aggregate results

These CSVs are copies of the recorded manuscript analysis, not outputs created by installing the repository. They contain cohort/model summaries only. Patient-level predictions and clinical values are not distributed.

- `segmentation-raw.csv`: Table 5, raw CSFINet and U-Net, 47 test patients.
- `segmentation-tta.csv`: matched four-orientation TTA, same models and patients; selected on internal validation.
- `segmentation-paired-tests.csv`: paired raw/TTA comparisons and multiplicity-adjusted tests.
- `survival-v2mask-fixed.csv`: four fixed neural models and two training-fitted reference models after raw v2 mask substitution.
- `survival-paired-tests.csv`: eight exploratory neural/reference comparisons; the previously inspected test set is reused.
- `shap.csv`, `shap-randomization.csv`: raw CSFINet fixed-region attribution summaries and randomization diagnostics.

The raw CSFINet masks, rather than TTA masks, feed both survival and SHAP. Neither age-only OLS nor the training-mean reference requires a neural checkpoint: the released evaluator fits these references from the 188 locally reconstructed training labels. Re-evaluation on another numerical environment can differ slightly; record device, batch size and precision before comparing values.
