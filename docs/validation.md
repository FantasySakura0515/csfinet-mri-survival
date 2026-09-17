# Software validation

The source distribution includes automated tests for model construction, training options, preprocessing, partition guards, statistical summaries, checkpoint installation, integrity checks and the research viewer. Run `python -m pytest -q` from the repository root.

The segmentation summary tests verify agreement with the manuscript reporting routine despite additional diagnostic metrics and reordered input rows. The statistics manifest binds distributed aggregate files to their SHA-256 hashes. Weight installation verifies archive members, checkpoint hashes, configuration hashes and license notices.

The trained weights and numerical model settings correspond to the reported study. Dataset files, individual clinical records, complete patient-level predictions and attribution arrays are obtained or generated locally. Code and packaging checks support reproducibility; external generalization and clinical utility remain unassessed.

Publication-package verification on 17 September 2026 passed 120 automated tests and 13 subtests. All six distributed checkpoints passed installation and integrity verification. A CUDA execution check on one held-out case completed raw and matched-TTA inference for both segmentation models and inference for all four neural survival configurations. This limited execution check is separate from the full-cohort results distributed under `results/reference/`.
