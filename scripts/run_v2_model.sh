#!/usr/bin/env bash
# Pre-registered v2 segmentation protocol for ONE model: development (150/38 epoch
# selection) -> refit on all 188 -> single frozen 47-patient test inference.
# Usage: bash scripts/run_v2_model.sh <csfinet|unet> [project_root] [python] [raw_root]
# Re-running resumes from the last verified checkpoint; completed outputs remain immutable.
set -euo pipefail

model="${1:?model name required: csfinet or unet}"
root="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${3:-$root/.venv/bin/python}"
raw="${4:-$root/data/raw/brats2020-nifti}"
tag="${TAG:-v2}"
config="${CONFIG:-configs/reconstruction-v21.json}"
patients="data/manifests/cohort/patients.csv"
cd "$root"

if [[ "$model" != "csfinet" && "$model" != "unet" ]]; then
  echo "model must be csfinet or unet" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked files are modified; commit the v2 protocol before launching" >&2
  exit 3
fi
echo "v2 launch commit: $(git rev-parse HEAD)"

development="runs/${model}-development-${tag}"
refit="runs/${model}-refit-${tag}"

resume=()
if [[ -e "$development/last.pt" || -e "$development/pending.pt" ]]; then resume=(--resume); fi
if [[ ! -f "$development/run.json" ]] || ! grep -q '"status": "completed"' "$development/run.json"; then
  "$python_bin" -u -m csfinet_repro train-segmentation --model "$model" --phase development \
    --output "$development" --config "$config" --patients "$patients" --root "$raw" "${resume[@]}"
fi

resume=()
if [[ -e "$refit/last.pt" || -e "$refit/pending.pt" ]]; then resume=(--resume); fi
if [[ ! -f "$refit/run.json" ]] || ! grep -q '"status": "completed"' "$refit/run.json"; then
  "$python_bin" -u -m csfinet_repro train-segmentation --model "$model" --phase refit \
    --output "$refit" --config "$config" --patients "$patients" --root "$raw" \
    --selection "$development/run.json" "${resume[@]}"
fi

output="results/segmentation/${model}-test-${tag}"
predictions="artifacts/segmentation/${model}-test-${tag}"
resume=()
if [[ -e "$output" || -e "$predictions" ]]; then resume=(--resume); fi
if [[ ! -f "$output/inference-manifest.json" ]] || ! grep -q '"status": "completed"' "$output/inference-manifest.json"; then
  "$python_bin" -u -m csfinet_repro infer-segmentation --config "$config" --patients "$patients" --root "$raw" \
    --run "$refit" --output "$output" --predictions "$predictions" "${resume[@]}"
fi
echo "V2_${model}_DONE $(date --iso-8601=seconds)"
