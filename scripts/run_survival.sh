#!/usr/bin/env bash
# Declared study survival experiment: whole-tumor mask plus masked MRI intensities as input.
# Training uses ground-truth masks; test inputs use the selected frozen segmentation predictions.
# cache -> four variants development/refit -> six-model frozen-test inference -> Table 7 (survival).
# Usage: bash scripts/run_survival.sh [project_root] [python] [raw_root] [segmentation_tag]
set -euo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${2:-$root/.venv/bin/python}"
raw="${3:-$root/data/raw/brats2020-nifti}"
segmentation_tag="${4:-study}"
tag="survival"
config="configs/survival.json"
patients="data/manifests/cohort/patients.csv"
cd "$root"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked files are modified; commit the study protocol before launching" >&2
  exit 3
fi
echo "survival study launch commit: $(git rev-parse HEAD); test masks from CSFINet ${segmentation_tag}"

cache="data/cache/survival-${tag}"
predictions="artifacts/segmentation/csfinet-test-${segmentation_tag}"
manifest="results/segmentation/csfinet-test-${segmentation_tag}/inference-manifest.json"
if [[ ! -f "$cache/manifest.json" ]]; then
  "$python_bin" -u -m csfinet_repro cache-survival-mri --patients "$patients" --root "$raw" \
    --predictions "$predictions" --segmentation-manifest "$manifest" --output "$cache"
fi

# A different segmentation source requires new survival runs, not reuse of an old cache.
"$python_bin" - "$cache/manifest.json" "$manifest" "$patients" <<'PY'
import json
import sys
from pathlib import Path
from csfinet_repro.files import sha256

cache_path, segmentation_path, patients_path = sys.argv[1:]
cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
if (cache.get("status") != "completed"
        or cache.get("split_sha256") != sha256(patients_path)
        or cache.get("segmentation_manifest_sha256") != sha256(segmentation_path)):
    raise SystemExit("Survival cache does not match the selected segmentation source. Use new cache/run/output directories.")
PY

"$python_bin" -u -m csfinet_repro.survival_suite --config "$config" --patients "$patients" \
  --cache "$cache" --runs runs --tag "$tag"

output="results/survival/${tag}/survival_predictions.csv"
if [[ ! -f "results/survival/${tag}/survival_predictions.manifest.json" ]]; then
  "$python_bin" -u -m csfinet_repro infer-survival --config "$config" --patients "$patients" --cache "$cache" \
    --output "$output" \
    --image-run "runs/survival-image-refit-${tag}" \
    --image-age-run "runs/survival-image_age-refit-${tag}" \
    --image-resection-run "runs/survival-image_resection-refit-${tag}" \
    --image-age-resection-run "runs/survival-image_age_resection-refit-${tag}"
fi
if [[ ! -f "results/tables/${tag}/survival/table7.json" ]]; then
  "$python_bin" -m csfinet_repro summarize-survival-table --input "$output" --output "results/tables/${tag}/survival"
fi
echo "STUDY_SURVIVAL_DONE $(date --iso-8601=seconds)"
