#!/usr/bin/env bash
# After both pre-registered v2 refits and their single test inferences complete:
# Table 5 (v2), then the declared inference variants for v2 decided on the 38
# validation patients and evaluated once on the frozen 47 test patients.
# Usage: bash scripts/run_v2_finish.sh <project_root> <python> <raw_root>
set -euo pipefail
root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${2:-$root/.venv/bin/python}"
raw="${3:-$root/data/raw/brats2020-nifti}"
tag="${TAG:-v2}"; config="${CONFIG:-configs/reconstruction-v21.json}"; patients="data/manifests/cohort/patients.csv"
cd "$root"
for model in csfinet unet; do
  manifest="results/segmentation/${model}-test-${tag}/inference-manifest.json"
  until [[ -f "$manifest" ]] && grep -q '"status": "completed"' "$manifest"; do sleep 120; done
  echo "$model test inference completed $(date --iso-8601=seconds)"
done
if [[ ! -f "results/tables/${tag}/segmentation/table5.json" ]]; then
  "$python_bin" -m csfinet_repro summarize-segmentation \
    --input "results/segmentation/csfinet-test-${tag}/segmentation_patient_metrics.csv" \
    --input "results/segmentation/unet-test-${tag}/segmentation_patient_metrics.csv" \
    --output "results/tables/${tag}/segmentation"
fi
if [[ ! -f "results/inference-variants/${tag}/validation/decision.json" ]]; then
  "$python_bin" -u -m csfinet_repro validate-inference-variants --config "$config" --patients "$patients" \
    --root "$raw" --runs runs --output "results/inference-variants/${tag}/validation" --tag "$tag"
fi
if [[ ! -f "results/inference-variants/${tag}/test/manifest.json" ]]; then
  "$python_bin" -u -m csfinet_repro evaluate-inference-variants --config "$config" --patients "$patients" \
    --root "$raw" --runs runs --results results --artifacts artifacts \
    --decision "results/inference-variants/${tag}/validation/decision.json" \
    --output "results/inference-variants/${tag}/test" --tag "$tag"
fi
echo "V2_FINISH_DONE $(date --iso-8601=seconds)"
