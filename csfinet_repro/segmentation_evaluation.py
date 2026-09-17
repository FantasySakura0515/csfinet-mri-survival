"""Frozen test inference and Table 5 generation from per-patient rows."""

from collections import defaultdict
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .files import read_csv, sha256, write_csv, write_json
from .metrics import bca_mean_ci, descriptive, paired_difference, paired_wilcoxon
from .models import build_segmentation
from .nifti import load_patient
from .segmentation import original_labels, predict_patient
from .training import seed_everything

REGIONS = ("WT", "TC", "ET")
MEASURES = ("dice", "precision", "recall", "iou")


def save_prediction(reference_path, prediction_zyx, target):
    reference = nib.load(reference_path)
    labels = np.asarray(prediction_zyx, dtype=np.uint8).transpose(1, 2, 0)
    if labels.shape != reference.shape or not np.isin(labels, [0, 1, 2, 4]).all():
        raise ValueError("Prediction geometry or labels invalid")
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name.replace(".nii.gz", ".part.nii.gz"))
    nib.save(nib.Nifti1Image(labels, reference.affine, header), temporary)
    temporary.replace(target)
    return sha256(target)


def evaluate_segmentation(config_path, patient_csv, root, run_dir, output, predictions, resume=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_dir, output, predictions = Path(run_dir), Path(output), Path(predictions)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model_name = run.get("model")
    checkpoint = run_dir / "final.pt"
    if (run.get("phase") != "refit" or run.get("status") != "completed"
            or run.get("config_sha256") != sha256(config_path)
            or run.get("split_sha256") != sha256(patient_csv)
            or run.get("final_checkpoint_sha256") != sha256(checkpoint)):
        raise ValueError("A completed, provenance-matched refit run is required")
    rows = read_csv(patient_csv, ["patient_id", "split"])
    tests = [row for row in rows if row["split"] == "test"]
    if len(tests) != 47:
        raise ValueError("Expected exactly 47 frozen test patients")
    identity = dict(model=model_name, config_sha256=sha256(config_path), split_sha256=sha256(patient_csv),
                    refit_run_sha256=sha256(run_dir / "run.json"), checkpoint_sha256=sha256(checkpoint))
    start_path = output / "inference-start.json"
    if output.exists() or predictions.exists():
        if not resume or not start_path.exists() or json.loads(start_path.read_text(encoding="utf-8")) != identity:
            raise FileExistsError("Evaluation outputs exist without a matching explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    write_json(start_path, identity)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation(model_name, config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    metric_rows, artifacts = [], []
    for position, row in enumerate(tests, 1):
        patient = row["patient_id"]
        images, training_targets = load_patient(root, patient)
        target = predictions / patient / f"{patient}_{model_name}_prediction.nii.gz"
        if resume and target.exists():
            stored = nib.load(target)
            predicted = stored.get_fdata(dtype=np.float32).astype(np.uint8).transpose(2, 0, 1)
            if stored.shape != (240, 240, 155) or not np.allclose(stored.affine, nib.load(Path(root) / patient / f"{patient}_seg.nii.gz").affine):
                raise ValueError(f"Resumed prediction geometry mismatch: {patient}")
        else:
            predicted = original_labels(predict_patient(model, images, config["segmentation"]["slice_batch"], device))
        truth = original_labels(training_targets)
        from .metrics import region_metrics
        regions = region_metrics(truth, predicted)
        prediction_sha = (sha256(target) if target.exists() else
                          save_prediction(Path(root) / patient / f"{patient}_seg.nii.gz", predicted, target))
        truth_sha = sha256(Path(root) / patient / f"{patient}_seg.nii.gz")
        artifacts.append(dict(patient_id=patient, relative_path=str(target.relative_to(predictions.parent)).replace("\\", "/"),
                              prediction_sha256=prediction_sha, truth_sha256=truth_sha))
        for region in REGIONS:
            metric_rows.append(dict(run_id=run_dir.name, patient_id=patient, model=model_name,
                                    region=region, **regions[region]))
        del images, training_targets, predicted, truth
        print(f"{model_name} test inference {position}/47: {patient}", flush=True)
    metrics_path = output / "segmentation_patient_metrics.csv"
    write_csv(metrics_path, metric_rows, list(metric_rows[0]))
    manifest = dict(status="completed", scope="frozen_47_patient_test", **identity,
                    metrics_sha256=sha256(metrics_path), prediction_root=str(predictions), artifacts=artifacts,
                    test_ids=[row["patient_id"] for row in tests], test_access="first model evaluation after refit; no selection")
    write_json(output / "inference-manifest.json", manifest)
    return manifest


def _format(value):
    return "NA" if value is None else f"{value:.4f}"


def summarize_segmentation(inputs, output, seed=20260914, n_resamples=10000, publication_variant=None):
    if publication_variant not in (None, 'raw', 'tta'):
        raise ValueError('Publication variant must be raw or tta')
    rows = []
    for path in inputs:
        rows.extend(read_csv(path, ["run_id", "patient_id", "model", "region", "tp", "fp", "fn", *MEASURES]))
    expected_ids = None
    values = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["region"], row["patient_id"])
        if key in values or row["region"] not in REGIONS:
            raise ValueError("Duplicate or invalid segmentation metric row")
        values[key] = {measure: None if row[measure] == "" else float(row[measure]) for measure in MEASURES}
    models = sorted({key[0] for key in values})
    if len(models) != 2:
        raise ValueError("Table 5 requires exactly two models")
    for model in models:
        ids = {key[2] for key in values if key[0] == model}
        if len(ids) != 47 or any((model, region, patient) not in values for region in REGIONS for patient in ids):
            raise ValueError("Each model must contain all three regions for the same 47 patients")
        expected_ids = ids if expected_ids is None else expected_ids
        if ids != expected_ids:
            raise ValueError("Models must use identical test patient IDs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table, detailed, index = [], {}, 0
    for model in models:
        detailed[model] = {}
        for region in REGIONS:
            detailed[model][region] = {}
            table_row = dict(model=model, region=region)
            for measure in MEASURES:
                observations = [values[(model, region, patient)][measure] for patient in sorted(expected_ids)]
                summary = descriptive(observations)
                # Preserve the manuscript's three-measure schedule. IoU must not
                # shift subsequent Dice/precision/recall seeds.
                offset = index
                if publication_variant is not None and measure != 'iou':
                    offset = (18 if publication_variant == 'tta' else 0) + models.index(model) * 9 + REGIONS.index(region) * 3 + MEASURES.index(measure)
                ci = bca_mean_ci(observations, seed=seed + offset, n_resamples=n_resamples)
                index += 1
                detailed[model][region][measure] = dict(summary=summary, mean_ci=ci)
                table_row[f"{measure}_mean"] = summary["mean"]
                table_row[f"{measure}_sd"] = summary["sd"]
                table_row[f"{measure}_n"] = summary["n_valid"]
                table_row[f"{measure}_ci_low"] = ci["low"]
                table_row[f"{measure}_ci_high"] = ci["high"]
            table.append(table_row)
    paired = {}
    left, right = models
    paired["WT"] = {}
    for measure in ("dice", "precision", "recall"):
        a = {patient: values[(left, "WT", patient)][measure] for patient in expected_ids}
        b = {patient: values[(right, "WT", patient)][measure] for patient in expected_ids}
        offset = index if publication_variant is None else 50 + (5 if publication_variant == 'tta' else 0) + {'dice': 0, 'precision': 3, 'recall': 4}[measure]
        paired["WT"][measure] = dict(difference=paired_difference(a, b, seed=seed + offset, n_resamples=n_resamples),
                                      wilcoxon=paired_wilcoxon(a, b))
        index += 1
    table_path = output / "table5.csv"
    write_csv(table_path, table, list(table[0]))
    markdown = ["# Table 5 — New frozen 47-patient experiment", "",
                "Mean ± sample SD (ddof=1); CI is 95% BCa over patients. NA metrics retain their reduced valid n.", "",
                "| Model | Region | Dice | Precision | Recall | IoU |", "| --- | --- | --- | --- | --- | --- |"]
    latex = [r"\begin{tabular}{llrrrr}", r"Model & Region & Dice & Precision & Recall & IoU \\", r"\hline"]
    for row in table:
        cells = [f"{_format(row[f'{m}_mean'])} ± {_format(row[f'{m}_sd'])} (n={row[f'{m}_n']})" for m in MEASURES]
        markdown.append(f"| {row['model']} | {row['region']} | " + " | ".join(cells) + " |")
        latex.append(f"{row['model']} & {row['region']} & " + " & ".join(cells) + r" \\")
    latex.append(r"\end{tabular}")
    (output / "table5.md").write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    (output / "table5.tex").write_text("\n".join(latex) + "\n", encoding="utf-8", newline="\n")
    report = dict(status="completed", scope="new_frozen_47_patient_experiment", direction=f"{left}_minus_{right}", seed=seed,
                  n_resamples=n_resamples, input_sha256={str(path): sha256(path) for path in inputs},
                  table_sha256=sha256(table_path), models=detailed, paired=paired)
    write_json(output / "table5.json", report)
    return report
