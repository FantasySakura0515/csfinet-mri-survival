"""Validate study artifacts and assemble a results package."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .files import read_csv, sha256, write_csv, write_json
from .metrics import descriptive
from .segmentation_evaluation import MEASURES, REGIONS
from .survival import VARIANTS

SEGMENTATION_MODELS = ("csfinet", "unet")
SURVIVAL_MODELS = (*VARIANTS, "training_mean", "age_ols")
def _completed_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed":
        raise ValueError(f"Artifact is not completed: {path}")
    return value


def _inventory_file(root, path, inventory, expected_sha=None, role=None, logical_path=None):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"Delivery file missing or outside project: {path}")
    logical_path = Path(logical_path).resolve() if logical_path is not None else path
    if not logical_path.is_relative_to(root):
        raise ValueError(f"Delivery inventory path is outside project: {logical_path}")
    digest = sha256(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"Hash mismatch: {path}")
    inventory.append(dict(path=str(logical_path.relative_to(root)).replace("\\", "/"), role=role or "artifact",
                          bytes=path.stat().st_size, sha256=digest))
    return digest


def _validate_rows(root, tag, split_sha, test_ids, inventory):
    results, artifacts = root / "results", root / "artifacts"
    cache_dir = root / "data" / "cache" / f"survival-{tag}"
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = _completed_json(cache_manifest_path)
    cache_items = cache_manifest.get("artifacts", [])
    if (cache_manifest.get("split_sha256") != split_sha or len(cache_items) != 235
            or len({item["patient_id"] for item in cache_items}) != 235):
        raise ValueError("Survival cache manifest does not contain 235 unique patients")
    _inventory_file(root, cache_manifest_path, inventory, role="survival_cache_manifest")
    for item in cache_items:
        _inventory_file(root, cache_dir / item["cache_file"], inventory, item["cache_sha256"], "survival_mask_cache")
    all_segmentation = []
    for model in SEGMENTATION_MODELS:
        directory = results / "segmentation" / f"{model}-test-{tag}"
        manifest_path = directory / "inference-manifest.json"
        manifest = _completed_json(manifest_path)
        if manifest.get("model") != model or set(manifest.get("test_ids", [])) != test_ids:
            raise ValueError(f"Segmentation manifest patient/model mismatch: {model}")
        _inventory_file(root, manifest_path, inventory, role="segmentation_manifest")
        metrics_path = directory / "segmentation_patient_metrics.csv"
        metrics = read_csv(metrics_path, ["patient_id", "model", "region", *MEASURES])
        expected = {(patient, region) for patient in test_ids for region in REGIONS}
        if len(metrics) != 141 or {(row["patient_id"], row["region"]) for row in metrics} != expected \
                or {row["model"] for row in metrics} != {model}:
            raise ValueError(f"Segmentation metrics are not 47x3 for {model}")
        _inventory_file(root, metrics_path, inventory, manifest["metrics_sha256"], "segmentation_patient_metrics")
        for item in manifest["artifacts"]:
            prediction = artifacts / "segmentation" / item["relative_path"]
            _inventory_file(root, prediction, inventory, item["prediction_sha256"], "segmentation_prediction")
        if len(manifest["artifacts"]) != 47:
            raise ValueError(f"Expected 47 prediction artifacts for {model}")
        all_segmentation.extend(metrics)

    predictions_path = results / "survival" / tag / "survival_predictions.csv"
    predictions = read_csv(predictions_path, ["patient_id", "model", "y_true", "y_pred", "residual",
                                               "absolute_error", "age", "resection_status", "mask_source"])
    expected = {(patient, model) for patient in test_ids for model in SURVIVAL_MODELS}
    if len(predictions) != 282 or {(row["patient_id"], row["model"]) for row in predictions} != expected:
        raise ValueError("Survival predictions are not six models x 47 patients")
    survival_manifest_path = predictions_path.with_suffix(".manifest.json")
    survival_manifest = _completed_json(survival_manifest_path)
    _inventory_file(root, predictions_path, inventory, survival_manifest["predictions_sha256"], "survival_predictions")
    _inventory_file(root, survival_manifest_path, inventory, role="survival_manifest")

    explain_dir = results / "explainability" / tag
    explain_manifest_path = explain_dir / "manifest.json"
    explain_manifest = _completed_json(explain_manifest_path)
    shap_path = explain_dir / "shap_patient_metrics.csv"
    shap = read_csv(shap_path, ["patient_id", "target", "eligible_slices", "lesion_attribution_mass",
                                "pointing_accuracy", "deletion_auc", "insertion_auc"])
    targets = {"ground_truth_WT_region", "predicted_WT_region"}
    if len(shap) != 94 or {(row["patient_id"], row["target"]) for row in shap} != {
            (patient, target) for patient in test_ids for target in targets}:
        raise ValueError("SHAP patient metrics are not two targets x 47 patients")
    _inventory_file(root, shap_path, inventory, explain_manifest["patient_metrics_sha256"], "shap_patient_metrics")
    slice_path = explain_dir / "shap_slice_metrics.csv"
    _inventory_file(root, slice_path, inventory, explain_manifest["slice_metrics_sha256"], "shap_slice_metrics")
    _inventory_file(root, explain_manifest_path, inventory, role="shap_manifest")
    for item in explain_manifest["artifacts"]:
        attribution = artifacts / "explainability" / item["relative_path"]
        _inventory_file(root, attribution, inventory, item["sha256"], "shap_attribution")
    if len(explain_manifest["artifacts"]) != 94:
        raise ValueError("Expected 94 raw SHAP artifacts")

    randomization_path = explain_dir / "parameter_randomization.csv"
    randomization = read_csv(randomization_path, ["patient_id", "stage", "spearman_attribution_similarity"])
    stages = {"head", "decoder", "piu_swin", "encoder"}
    if len(randomization) != 188 or {(row["patient_id"], row["stage"]) for row in randomization} != {
            (patient, stage) for patient in test_ids for stage in stages}:
        raise ValueError("Parameter randomization is not four stages x 47 patients")
    random_manifest_path = randomization_path.with_suffix(".manifest.json")
    random_manifest = _completed_json(random_manifest_path)
    _inventory_file(root, randomization_path, inventory, random_manifest["output_sha256"], "parameter_randomization")
    _inventory_file(root, random_manifest_path, inventory, role="randomization_manifest")
    return all_segmentation, predictions, shap


def _validate_schematics(root, tag, split_sha, inventory):
    schematic_dir = root / "results" / "schematics" / tag
    manifest_path = schematic_dir / "manifest.json"
    manifest = _completed_json(manifest_path)
    items = manifest.get("artifacts", [])
    if (manifest.get("scope") != "implementation_matched_figures_1_to_7"
            or manifest.get("split_sha256") != split_sha
            or manifest.get("config_sha256") != sha256(root / "configs" / "segmentation.json")
            or len(items) != 14 or len({item["path"] for item in items}) != 14):
        raise ValueError("Fig.1-7 schematic manifest is incomplete or has the wrong lineage")
    for item in items:
        _inventory_file(root, schematic_dir / item["path"], inventory, item["sha256"],
                        "manuscript_schematic")
    _inventory_file(root, manifest_path, inventory, role="schematic_manifest")
    return manifest


def _validate_runs_and_reports(root, tag, split_sha, inventory):
    for model in SEGMENTATION_MODELS:
        for phase in ("development", "refit"):
            run_dir = root / "runs" / f"{model}-{phase}-{tag}"
            run_path = run_dir / "run.json"
            run = _completed_json(run_path)
            if run.get("model") != model or run.get("phase") != phase or run.get("split_sha256") != split_sha:
                raise ValueError(f"Segmentation run lineage mismatch: {model}/{phase}")
            _inventory_file(root, run_path, inventory, role="training_run")
            checkpoint_name = "final.pt" if phase == "refit" else "best.pt"
            checkpoint_key = "final_checkpoint_sha256" if phase == "refit" else "best_checkpoint_sha256"
            _inventory_file(root, run_dir / checkpoint_name, inventory, run[checkpoint_key], "model_checkpoint")
    for variant in VARIANTS:
        for phase in ("development", "refit"):
            run_dir = root / "runs" / f"survival-{variant}-{phase}-{tag}"
            run_path = run_dir / "run.json"
            run = _completed_json(run_path)
            if run.get("variant") != variant or run.get("phase") != phase or run.get("split_sha256") != split_sha:
                raise ValueError(f"Survival run lineage mismatch: {variant}/{phase}")
            _inventory_file(root, run_path, inventory, role="training_run")
            checkpoint_name = "final.pt" if phase == "refit" else "best.pt"
            checkpoint_key = "final_checkpoint_sha256" if phase == "refit" else "best_checkpoint_sha256"
            _inventory_file(root, run_dir / checkpoint_name, inventory, run[checkpoint_key], "model_checkpoint")

    report_specs = (("segmentation", "table5"), ("survival", "table7"), ("shap", "table9"))
    reports = {}
    for area, name in report_specs:
        directory = root / "results" / "tables" / tag / area
        report_path = directory / f"{name}.json"
        report = _completed_json(report_path)
        reports[name] = report
        _inventory_file(root, report_path, inventory, role="statistical_report")
        for suffix in ("csv", "md", "tex"):
            _inventory_file(root, directory / f"{name}.{suffix}", inventory, role="manuscript_table")
        if name == "table9":
            _inventory_file(root, directory / "table9_randomization.csv", inventory, role="manuscript_table")

    figure_dir = root / "results" / "figures" / tag
    figure_manifest_path = figure_dir / "manifest.json"
    figure_manifest = _completed_json(figure_manifest_path)
    for section in ("artifacts", "tables"):
        for item in figure_manifest[section]:
            _inventory_file(root, figure_dir / item["path"], inventory, item["sha256"], "manuscript_figure" if section == "artifacts" else "figure_source_table")
    _inventory_file(root, figure_manifest_path, inventory, role="figure_manifest")
    reports["schematics"] = _validate_schematics(root, tag, split_sha, inventory)
    return reports


def _cohort_table(patient_rows, output):
    groups = (("all", patient_rows), ("train", [row for row in patient_rows if row["split"] == "train"]),
              ("test", [row for row in patient_rows if row["split"] == "test"]))
    result = []
    for name, rows in groups:
        age = descriptive(float(row["age"]) for row in rows)
        survival = descriptive(float(row["survival_days"]) for row in rows)
        status = Counter(row["resection_status"] for row in rows)
        result.append(dict(group=name, n=len(rows), age_mean=age["mean"], age_sd=age["sd"],
                           age_median=age["median"], age_q1=age["q1"], age_q3=age["q3"],
                           survival_mean=survival["mean"], survival_sd=survival["sd"],
                           survival_median=survival["median"], survival_q1=survival["q1"],
                           survival_q3=survival["q3"], GTR=status["GTR"], STR=status["STR"], NA=status["NA"]))
    path = output / "table1_cohort.csv"
    write_csv(path, result, list(result[0]))
    markdown = ["# Table 1 — New reconstruction cohort", "",
                "All SD values are patient-level sample SD (ddof=1).", "",
                "| Group | n | Age, mean ± SD | Age, median [IQR] | Survival, mean ± SD | Survival, median [IQR] | GTR / STR / NA |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result:
        markdown.append(f"| {row['group']} | {row['n']} | {row['age_mean']:.2f} ± {row['age_sd']:.2f} | "
                        f"{row['age_median']:.2f} [{row['age_q1']:.2f}, {row['age_q3']:.2f}] | "
                        f"{row['survival_mean']:.2f} ± {row['survival_sd']:.2f} | "
                        f"{row['survival_median']:.2f} [{row['survival_q1']:.2f}, {row['survival_q3']:.2f}] | "
                        f"{row['GTR']} / {row['STR']} / {row['NA']} |")
    md_path = output / "table1_cohort.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    return result, (path, md_path)


def _mean_sd(row, prefix):
    if row[f"{prefix}_mean"] in (None, "") or row[f"{prefix}_sd"] in (None, ""):
        return "NA"
    mean, sd = float(row[f"{prefix}_mean"]), float(row[f"{prefix}_sd"])
    return f"{mean:.4f} ± {sd:.4f}"


def _manuscript_text(root, tag, cohort, table5_rows, table7_rows, table9_rows, output):
    wt = {row["model"]: row for row in table5_rows if row["region"] == "WT"}
    best_survival = min(table7_rows, key=lambda row: float(row["mae"]))
    lines = ["# Study results", "",
             "Results are generated from validated patient-level artifacts for the fixed study partition.", "",
             "## Methods — reproducibility and statistics", "",
             "We created a deterministic patient-level split of 235 eligible BraTS 2020 cases into 188 training and 47 held-out test cases. Model selection used a fixed 150/38 development split within the training cohort; the held-out test set was not used for epoch selection. After epoch selection, each model was refitted from initialization on all 188 training cases. Segmentation measures were calculated for each test patient and summarized as mean ± sample standard deviation (ddof=1), median and interquartile range, with 95% BCa bootstrap confidence intervals (10,000 resamples). Survival absolute errors were summarized analogously. Paired model comparisons used two-sided Wilcoxon signed-rank tests after patient-ID alignment; the prespecified survival comparison family used Holm adjustment. Pearson intervals used Fisher's z transformation, Spearman intervals used paired BCa bootstrap, and dependent age correlations were compared with Steiger's overlapping-correlation z test.", "",
             "## Results — cohort", "",
             f"The study cohort contained {cohort[0]['n']} patients: {cohort[1]['n']} training and {cohort[2]['n']} held-out test patients. The test cohort contained {cohort[2]['GTR']} GTR, {cohort[2]['STR']} STR and {cohort[2]['NA']} unknown-resection cases.", "",
             "## Results — segmentation", ""]
    for model in SEGMENTATION_MODELS:
        row = wt[model]
        lines.append(f"- {model.upper()} WT Dice: {_mean_sd(row, 'dice')} (n={row['dice_n']}); precision {_mean_sd(row, 'precision')}; recall {_mean_sd(row, 'recall')}.")
    lines += ["", "## Results — survival", "",
              f"The lowest held-out MAE among the six prespecified models was {float(best_survival['mae']):.2f} days for `{best_survival['model']}` with SD(AE) {float(best_survival['sd_ae']):.2f} days and RMSE {float(best_survival['rmse']):.2f} days.", "",
              "## Results — explainability", ""]
    for row in table9_rows:
        lines.append(f"- `{row['target']}`: lesion attribution mass {_mean_sd(row, 'lesion_attribution_mass')}; pointing accuracy {_mean_sd(row, 'pointing_accuracy')}; deletion AUC {_mean_sd(row, 'deletion_auc')}; insertion AUC {_mean_sd(row, 'insertion_auc')}.")
    lines += ["", "## Data sources", "",
              f"Tables: `results/tables/{tag}/`. Figures: `results/figures/{tag}/`. File hashes are recorded in `delivery-audit.json`."]
    path = output / "study-results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def audit_delivery(project_root, tag="study", output=None):
    if not tag.isalnum():
        raise ValueError("Tag must be alphanumeric")
    root = Path(project_root).resolve()
    final_output = Path(output).resolve() if output else root / "results" / "delivery" / tag
    if final_output.exists():
        raise FileExistsError("Delivery output exists; completed delivery packages are immutable")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    patient_csv = root / "data" / "manifests" / "cohort" / "patients.csv"
    patient_rows = read_csv(patient_csv, ["patient_id", "age", "survival_days", "resection_status", "split"])
    train_ids = {row["patient_id"] for row in patient_rows if row["split"] == "train"}
    test_ids = {row["patient_id"] for row in patient_rows if row["split"] == "test"}
    if len(patient_rows) != 235 or len(train_ids) != 188 or len(test_ids) != 47 or train_ids & test_ids:
        raise ValueError("Cohort is not the frozen 188/47 patient partition")
    split_sha = sha256(patient_csv)
    nifti_audit_path = root / "data" / "manifests" / "cohort" / "nifti-audit.json"
    nifti_audit = json.loads(nifti_audit_path.read_text(encoding="utf-8"))
    if (nifti_audit.get("status") != "passed" or nifti_audit.get("patients") != 235
            or nifti_audit.get("files") != 1175 or nifti_audit.get("split_sha256") != split_sha):
        raise ValueError("Full 235-patient / 1,175-file NIfTI audit evidence is invalid")
    post_path = root / "runs" / f"postprocess-suite-{tag}.json"
    post = _completed_json(post_path)
    if post.get("split_sha256") != split_sha:
        raise ValueError("Postprocess split does not match frozen patient CSV")
    inventory = []
    _inventory_file(root, patient_csv, inventory, role="frozen_split")
    _inventory_file(root, nifti_audit_path, inventory, role="raw_nifti_audit")
    _inventory_file(root, post_path, inventory, role="postprocess_suite")
    segmentation, survival, shap = _validate_rows(root, tag, split_sha, test_ids, inventory)
    reports = _validate_runs_and_reports(root, tag, split_sha, inventory)
    output = Path(tempfile.mkdtemp(prefix=f".{final_output.name}.tmp-", dir=final_output.parent)).resolve()
    cohort, cohort_files = _cohort_table(patient_rows, output)
    table5_path = root / "results" / "tables" / tag / "segmentation" / "table5.csv"
    table7_path = root / "results" / "tables" / tag / "survival" / "table7.csv"
    table9_path = root / "results" / "tables" / tag / "shap" / "table9.csv"
    table5 = read_csv(table5_path, ["model", "region", "dice_mean", "dice_sd", "precision_mean", "precision_sd",
                                           "recall_mean", "recall_sd"])
    table7 = read_csv(table7_path, ["model", "mae", "sd_ae", "rmse", "median_ae", "pearson", "spearman"])
    table9 = read_csv(table9_path, ["target", "lesion_attribution_mass_mean", "lesion_attribution_mass_sd",
                                           "pointing_accuracy_mean", "pointing_accuracy_sd", "deletion_auc_mean",
                                           "deletion_auc_sd", "insertion_auc_mean", "insertion_auc_sd"])
    manuscript_path = _manuscript_text(root, tag, cohort, table5, table7, table9, output)
    generated = [(cohort_files[0], "cohort_table"), (cohort_files[1], "cohort_table"),
                 (manuscript_path, "study_results")]
    for path, role in generated:
        _inventory_file(root, path, inventory, role=role, logical_path=final_output / path.name)
    manifest_path = output / "delivery-manifest.csv"
    write_csv(manifest_path, sorted(inventory, key=lambda row: row["path"]), ["path", "role", "bytes", "sha256"])
    audit = dict(status="completed", scope="validated_study_results", tag=tag,
                 git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=root, text=True).strip(),
                 split_sha256=split_sha, counts=dict(patients=235, train=188, test=47,
                 segmentation_patient_region_rows=len(segmentation), survival_prediction_rows=len(survival),
                 shap_patient_target_rows=len(shap), schematic_files=len(reports["schematics"]["artifacts"]),
                 inventory_files=len(inventory)),
                 upstream=dict(postprocess_sha256=sha256(post_path), table5_sha256=sha256(table5_path),
                               table7_sha256=sha256(table7_path), table9_sha256=sha256(table9_path)),
                 report_status={name: report["status"] for name, report in reports.items()},
                 manifest_sha256=sha256(manifest_path), study_results_sha256=sha256(manuscript_path))
    write_json(output / "delivery-audit.json", audit)
    os.replace(output, final_output)
    return audit
