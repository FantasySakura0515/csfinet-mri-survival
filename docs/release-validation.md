# Release validation

The v2.0.1 package was checked on 17 September 2026.

- A fresh Python 3.10 environment installed `pip install -e ".[all]"` successfully; `pip check` reported no broken requirements.
- All 114 data-free tests passed, including 10 subtests. The suite covers models, statistics, preprocessing, protocol guards, download integrity and source-only split validation.
- All six released checkpoints contain tensor-only state dictionaries. Each passed strict model loading and tensor equality checks; the ZIP members retain the original checkpoint bytes and SHA-256 values. Run metadata retain aggregate clinical transformations and anonymous partition IDs, without individual clinical records or machine paths.
- The clinical cohort was reconstructed locally from the pinned CSVs; its exact original SHA-256 and the published identifier-only partition both matched.
- A one-patient CUDA evaluation exercised raw and matched TTA inference for both segmentation models, then all four survival models and both clinical references. This used an inference slice batch of 2. Relative to the recorded patient, the largest raw region Dice difference was approximately 0.0000027; the largest survival prediction difference was approximately 0.00235 days.
- The independent Streamlit viewer passed missing-data checks and a local test with real MRI, predictions, survival rows and the exact saved SHAP slice.

These checks validate packaging and a small integration path. The release preparation did not retrain the models or rerun the full 47-patient segmentation, survival and SHAP experiments. The aggregate reference CSVs remain the recorded manuscript results. Hardware, precision and batch size can cause small numerical differences.

Dataset files, locally reconstructed clinical CSVs, prediction masks, attribution arrays, working environments and internal drafts are excluded from Git. Checkpoints are distributed separately in the same repository's Release.
