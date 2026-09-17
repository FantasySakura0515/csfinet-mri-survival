# Release validation

## v2.0.3 statistics and reporting checks

The consistency update was checked on 17 September 2026. All **121 data-free tests and 13 subtests passed**. New regression coverage compares the public summarizer against the manuscript reporting routine despite the additional IoU measure, reversed input files and reversed patient rows; it also checks aggregate-file hashes against the statistics manifest.

All 36 Dice/precision/recall mean intervals and all 20 paired summary rows were recomputed from the recorded patient metrics. Shared means, sample SDs, interval endpoints and available raw/Holm-adjusted p values agree with the manuscript source to 1e-12. The correction changes 11 public Dice intervals at four decimal places, plus the affected precision/recall intervals. The manuscript's scientific results remain unchanged. See [statistical reconciliation](statistical-reconciliation.md).

The current evaluation description is patient-disjoint internal held-out evaluation. The authors confirmed no test-score-driven model or method selection; 12 completed development/refit records support patient separation. [Evidence and limits](evaluation-design.json) distinguish record-based membership verification from author-confirmed decision chronology. No new external cohort is claimed.

Additional aggregate CSVs cover survival dispersion, the displayed bins/correlations and the SHAP target-area reference. No dataset files or individual clinical/prediction records were added. The six checkpoints and their run metadata remain byte-identical to the preceding release. This update performs statistical recomputation and packaging, without model training or inference.

## v2.0.2 license and packaging checks

The Apache 2.0 release was checked on 17 September 2026. All 118 data-free tests passed, including 13 subtests. The added checks cover license-document installation, tamper detection, invalid archive paths and agreement between the published manifest and repository notices.

Both repackaged weight archives were installed and verified with the release downloader. All six checkpoint files and sanitized run metadata retain their previous SHA-256 values; only archive packaging and licensing metadata changed. The Python wheel built successfully and includes the `Apache-2.0` SPDX expression, complete license and attribution documents.

The licensing update does not rerun training or cohort inference. The scientific validation below describes the unchanged model and evaluation artifacts.

## v2.0.1 scientific and integration checks

The v2.0.1 package was checked on 17 September 2026.

- A fresh Python 3.10 environment installed `pip install -e ".[all]"` successfully; `pip check` reported no broken requirements.
- All 114 data-free tests passed, including 10 subtests. The suite covers models, statistics, preprocessing, protocol guards, download integrity and source-only split validation.
- All six released checkpoints contain tensor-only state dictionaries. Each passed strict model loading and tensor equality checks; the ZIP members retain the original checkpoint bytes and SHA-256 values. Run metadata retain aggregate clinical transformations and anonymous partition IDs, without individual clinical records or machine paths.
- The clinical cohort was reconstructed locally from the pinned CSVs; its exact original SHA-256 and the published identifier-only partition both matched.
- A one-patient CUDA evaluation exercised raw and matched TTA inference for both segmentation models, then all four survival models and both clinical references. This used an inference slice batch of 2. Relative to the recorded patient, the largest raw region Dice difference was approximately 0.0000027; the largest survival prediction difference was approximately 0.00235 days.
- The independent Streamlit viewer passed missing-data checks and a local test with real MRI, predictions, survival rows and the exact saved SHAP slice.

These checks validate packaging and a small integration path. The release preparation did not retrain the models or rerun the full 47-patient segmentation, survival and SHAP experiments. The aggregate reference CSVs remain the recorded manuscript results. Hardware, precision and batch size can cause small numerical differences.

Dataset files, locally reconstructed clinical CSVs, prediction masks, attribution arrays, working environments and internal drafts are excluded from Git. Checkpoints are distributed separately in the same repository's Release.
