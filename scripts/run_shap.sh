#!/usr/bin/env bash
# After the CSFINet refit and its single test inference complete: 47-patient
# Gradient SHAP, cumulative parameter randomization and Table 9 for the trained model.
# Usage: TAG=study CONFIG=configs/segmentation.json bash scripts/run_shap.sh <project_root> <python> <raw_root>
set -euo pipefail
root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${2:-$root/.venv/bin/python}"
raw="${3:-$root/data/raw/brats2020-nifti}"
tag="${TAG:-study}"; config="${CONFIG:-configs/segmentation.json}"
patients="data/manifests/cohort/patients.csv"; analysis="configs/analysis.json"
cd "$root"
manifest="results/segmentation/csfinet-test-${tag}/inference-manifest.json"
until [[ -f "$manifest" ]] && grep -q '"status": "completed"' "$manifest"; do sleep 120; done
echo "csfinet test inference completed $(date --iso-8601=seconds)"
output="results/explainability/${tag}"; artifacts="artifacts/explainability/${tag}"
resume=(); if [[ -e "$output" || -e "$artifacts" ]]; then resume=(--resume); fi
if [[ ! -f "$output/manifest.json" ]]; then
  "$python_bin" -u -m csfinet_repro explain-segmentation --config "$config" --analysis "$analysis" \
    --patients "$patients" --root "$raw" --run "runs/csfinet-refit-${tag}" \
    --predictions "artifacts/segmentation/csfinet-test-${tag}" --inference-manifest "$manifest" \
    --output "$output" --artifacts "$artifacts" "${resume[@]}"
fi
randomization="$output/parameter_randomization.csv"
resume=(); if [[ -e "$randomization" || -e "${randomization%.csv}.start.json" ]]; then resume=(--resume); fi
if [[ ! -f "${randomization%.csv}.manifest.json" ]]; then
  "$python_bin" -u -m csfinet_repro randomize-segmentation --config "$config" --analysis "$analysis" \
    --patients "$patients" --root "$raw" --run "runs/csfinet-refit-${tag}" --output "$randomization" "${resume[@]}"
fi
if [[ ! -f "results/tables/${tag}/shap/table9.json" ]]; then
  "$python_bin" -m csfinet_repro summarize-shap --shap "$output/shap_patient_metrics.csv" \
    --segmentation "results/segmentation/csfinet-test-${tag}/segmentation_patient_metrics.csv" \
    --randomization "$randomization" --output "results/tables/${tag}/shap"
fi
echo "STUDY_SHAP_DONE $(date --iso-8601=seconds)"
