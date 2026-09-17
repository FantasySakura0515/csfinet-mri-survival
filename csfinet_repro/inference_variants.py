"""Inference variants selected on validation patients and evaluated on frozen test patients.

Raw and LCC reuse hash-verified NIfTI predictions; TTA and BatchNorm variants
are recomputed. All candidates are reported and the adopted candidate becomes
Table 5b. Test metrics never select a variant.
"""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
from torch import nn

from .files import read_csv, sha256, write_csv, write_json
from .metrics import descriptive, paired_difference, paired_wilcoxon, region_metrics
from .models import build_segmentation
from .nifti import load_patient
from .segmentation import original_labels
from .segmentation_evaluation import MEASURES, REGIONS, summarize_segmentation
from .training import seed_everything

VARIANTS = ("raw", "lcc", "tta", "tta_lcc", "bn", "bn_lcc", "bn_tta", "bn_tta_lcc")
FLIPS_RAW = ((),)
FLIPS_TTA = ((), (-1,), (-2,), (-2, -1))
CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)  # 26-connected 3D components
MODELS = ("csfinet", "unet")
COLUMNS = ["run_id", "patient_id", "model", "variant", "region", "tp", "fp", "fn", *MEASURES]
RULE = ("Adopt the non-raw variant with the highest mean validation WT Dice averaged over both models, "
        "provided it is non-inferior to raw for each model; ties prefer the simpler variant "
        "(earlier in the VARIANTS order); otherwise keep raw.")


def largest_component(labels):
    """Keep the largest 26-connected whole-tumor component; clear every other label."""
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1, 2, 4]).all():
        raise ValueError("Expected original BraTS labels 0, 1, 2, 4")
    foreground = labels > 0
    components, count = ndimage.label(foreground, structure=CONNECTIVITY)
    if count <= 1:
        return labels.copy()
    sizes = ndimage.sum_labels(foreground, components, index=np.arange(1, count + 1))
    keep = components == (int(np.argmax(sizes)) + 1)
    result = labels.copy()
    result[~keep] = 0
    return result


@torch.inference_mode()
def predict_labels(model, images, slice_batch, device, flips=FLIPS_RAW):
    """Argmax of the mean softmax over a flip set; each flip lists in-plane axes to flip."""
    model.eval()
    output = []
    for start in range(0, len(images), slice_batch):
        batch = torch.as_tensor(images[start:start + slice_batch], device=device, dtype=torch.float32)
        total = None
        for axes in flips:
            source = torch.flip(batch, dims=list(axes)) if axes else batch
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(source)
            probability = logits.float().softmax(dim=1)
            if axes:
                probability = torch.flip(probability, dims=list(axes))
            total = probability if total is None else total + probability
        output.append((total / len(flips)).argmax(dim=1).cpu().numpy().astype(np.uint8))
    return original_labels(np.concatenate(output))


def calibration_ids(patient_ids, seed, count):
    """Fixed, seeded subset of training-only patients used to re-estimate BatchNorm statistics."""
    ordered = sorted(patient_ids)
    if not 1 <= count <= len(ordered):
        raise ValueError("Calibration count must be between 1 and the number of training patients")
    return sorted(np.random.default_rng(seed).permutation(ordered)[:count].tolist())


def recalibrate_batchnorm(model, slice_arrays, slice_batch, device):
    """Re-estimate every BatchNorm running mean/variance from ``slice_arrays`` with frozen weights.

    ``slice_arrays`` yields (N, C, H, W) numpy arrays. Only BatchNorm layers enter
    training mode; dropout and stochastic depth stay disabled. ``momentum=None``
    accumulates an exact average over all calibration batches. Returns the number
    of recalibrated layers.
    """
    layers = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not layers:
        raise ValueError("Model has no BatchNorm layers to recalibrate")
    model.eval()
    momenta = []
    for layer in layers:
        momenta.append(layer.momentum)
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    batches = 0
    with torch.no_grad():
        for images in slice_arrays:
            for start in range(0, len(images), slice_batch):
                batch = torch.as_tensor(images[start:start + slice_batch], device=device, dtype=torch.float32)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    model(batch)
                batches += 1
    if batches == 0:
        raise ValueError("No calibration slices were provided")
    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum
        layer.eval()
    model.eval()
    return len(layers)


def _calibration_slices(root, patient_ids):
    for patient in patient_ids:
        images, _ = load_patient(root, patient)
        yield images


def calibrated_copy(model, root, patient_ids, slice_batch, device):
    calibrated = copy.deepcopy(model)
    layers = recalibrate_batchnorm(calibrated, _calibration_slices(root, patient_ids), slice_batch, device)
    return calibrated, layers


def variant_labels(raw, tta, bn_raw, bn_tta):
    return {"raw": raw, "lcc": largest_component(raw), "tta": tta, "tta_lcc": largest_component(tta),
            "bn": bn_raw, "bn_lcc": largest_component(bn_raw), "bn_tta": bn_tta,
            "bn_tta_lcc": largest_component(bn_tta)}


def decide(validation_mean_wt_dice):
    """Pre-specified rule over {model: {variant: mean validation WT Dice}}."""
    models = sorted(validation_mean_wt_dice)
    raw_mean = float(np.mean([validation_mean_wt_dice[m]["raw"] for m in models]))
    candidates = []
    for variant in VARIANTS[1:]:
        mean = float(np.mean([validation_mean_wt_dice[m][variant] for m in models]))
        non_inferior = all(validation_mean_wt_dice[m][variant] >= validation_mean_wt_dice[m]["raw"] for m in models)
        candidates.append(dict(variant=variant, cross_model_mean_WT_dice=mean,
                               non_inferior_for_every_model=non_inferior,
                               improves_cross_model_mean=mean > raw_mean))
    eligible = [c for c in candidates if c["non_inferior_for_every_model"] and c["improves_cross_model_mean"]]
    if not eligible:
        return dict(adopted="raw", rule=RULE, raw_cross_model_mean_WT_dice=raw_mean, candidates=candidates,
                    reason="no variant improved the cross-model mean validation WT Dice while staying "
                           "non-inferior for every model")
    best = max(eligible, key=lambda c: (c["cross_model_mean_WT_dice"], -VARIANTS.index(c["variant"])))
    return dict(adopted=best["variant"], rule=RULE, raw_cross_model_mean_WT_dice=raw_mean, candidates=candidates,
                reason="highest cross-model mean validation WT Dice among variants non-inferior for every model")


def _metric_rows(run_id, model, patient, truth, variants):
    rows = []
    for variant, labels in variants.items():
        metrics = region_metrics(truth, labels)
        for region in REGIONS:
            rows.append(dict(run_id=run_id, patient_id=patient, model=model, variant=variant, region=region,
                             **metrics[region]))
    return rows


def _environment():
    cuda = torch.cuda.is_available()
    return dict(torch=torch.__version__, cuda=torch.version.cuda if cuda else None,
                device=torch.cuda.get_device_name() if cuda else "cpu", python=platform.python_version(),
                amp="float16" if cuda else "float32")


def _summary_rows(long_rows, seed, n_resamples):
    keyed = {}
    for row in long_rows:
        keyed.setdefault((row["model"], row["variant"], row["patient_id"]), {})[row["region"]] = row
    table, index = [], 0
    for model in MODELS:
        patients = sorted({key[2] for key in keyed if key[0] == model})
        raw_wt = {patient: keyed[(model, "raw", patient)]["WT"]["dice"] for patient in patients}
        for variant in VARIANTS:
            row = dict(model=model, variant=variant, n=len(patients))
            for region in REGIONS:
                for measure in ("dice", "precision", "recall"):
                    summary = descriptive(keyed[(model, variant, patient)][region][measure] for patient in patients)
                    row[f"{region}_{measure}_mean"] = summary["mean"]
                    row[f"{region}_{measure}_sd"] = summary["sd"]
                    row[f"{region}_{measure}_n"] = summary["n_valid"]
            row.update(WT_dice_diff_vs_raw_mean=None, WT_dice_diff_ci_low=None, WT_dice_diff_ci_high=None,
                       WT_dice_wilcoxon_p=None)
            if variant != "raw":
                current = {patient: keyed[(model, variant, patient)]["WT"]["dice"] for patient in patients}
                difference = paired_difference(current, raw_wt, seed=seed + index, n_resamples=n_resamples)
                wilcoxon = paired_wilcoxon(current, raw_wt)
                index += 1
                row.update(WT_dice_diff_vs_raw_mean=difference["summary"]["mean"],
                           WT_dice_diff_ci_low=difference["mean_ci"]["low"],
                           WT_dice_diff_ci_high=difference["mean_ci"]["high"],
                           WT_dice_wilcoxon_p=wilcoxon["p_value"])
            table.append(row)
    return table


def _fmt(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def _write_summary(table, output, title, note):
    path = output / "variant_summary.csv"
    write_csv(path, table, list(table[0]))
    lines = [f"# {title}", "", note, "",
             "| Model | Variant | WT Dice | WT Precision | WT Recall | TC Dice | ET Dice | "
             "WT Dice diff vs raw [95% BCa] | Wilcoxon p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in table:
        if row["WT_dice_diff_vs_raw_mean"] is None:
            delta, p_value = "-", "-"
        else:
            delta = (f"{row['WT_dice_diff_vs_raw_mean']:+.4f} "
                     f"[{_fmt(row['WT_dice_diff_ci_low'])}, {_fmt(row['WT_dice_diff_ci_high'])}]")
            p_value = _fmt(row["WT_dice_wilcoxon_p"], 3)
        lines.append(f"| {row['model']} | {row['variant']} | "
                     f"{_fmt(row['WT_dice_mean'])} +/- {_fmt(row['WT_dice_sd'])} | "
                     f"{_fmt(row['WT_precision_mean'])} +/- {_fmt(row['WT_precision_sd'])} | "
                     f"{_fmt(row['WT_recall_mean'])} +/- {_fmt(row['WT_recall_sd'])} | "
                     f"{_fmt(row['TC_dice_mean'])} +/- {_fmt(row['TC_dice_sd'])} | "
                     f"{_fmt(row['ET_dice_mean'])} +/- {_fmt(row['ET_dice_sd'])} | {delta} | {p_value} |")
    markdown = output / "variant_summary.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path, markdown


def _check_run(run, phase, model_name, config_path, patient_csv, checkpoint, key):
    return (run.get("phase") == phase and run.get("status") == "completed" and run.get("model") == model_name
            and run.get("config_sha256") == sha256(config_path) and run.get("split_sha256") == sha256(patient_csv)
            and run.get(key) == sha256(checkpoint))


def validate_variants(config_path, patient_csv, root, runs_root, output, tag="study", seed=20260914,
                      n_resamples=10000, calibration_patients=30):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    slice_batch = config["segmentation"]["slice_batch"]
    rows = read_csv(patient_csv, ["patient_id", "split", "development_split"])
    validation = [row for row in rows if row["development_split"] == "validation"]
    development = [row["patient_id"] for row in rows if row["development_split"] == "train"]
    if len(rows) != 235 or len(validation) != 38 or len(development) != 150 \
            or any(row["split"] != "train" for row in validation):
        raise ValueError("Expected the frozen 150/38 development split inside the 188 training patients")
    calibration = calibration_ids(development, config["seed"], calibration_patients)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Validation-variant output exists; recorded decisions are immutable")
    output.mkdir(parents=True)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    long_rows, lineage, means = [], {}, {}
    for model_name in MODELS:
        run_dir = Path(runs_root) / f"{model_name}-development-{tag}"
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        checkpoint = run_dir / "best.pt"
        if (not _check_run(run, "development", model_name, config_path, patient_csv, checkpoint, "best_checkpoint_sha256")
                or set(run.get("validation_ids", [])) != {row["patient_id"] for row in validation}
                or not set(calibration).issubset(set(run.get("training_ids", [])))):
            raise ValueError(f"Completed, provenance-matched development run required: {model_name}")
        model = build_segmentation(model_name, config).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        calibrated, layers = calibrated_copy(model, root, calibration, slice_batch, device)
        print(f"{model_name}: recalibrated {layers} BatchNorm layers on {len(calibration)} training patients", flush=True)
        wt = {variant: [] for variant in VARIANTS}
        for position, row in enumerate(validation, 1):
            patient = row["patient_id"]
            images, targets = load_patient(root, patient)
            truth = original_labels(targets)
            variants = variant_labels(predict_labels(model, images, slice_batch, device, FLIPS_RAW),
                                      predict_labels(model, images, slice_batch, device, FLIPS_TTA),
                                      predict_labels(calibrated, images, slice_batch, device, FLIPS_RAW),
                                      predict_labels(calibrated, images, slice_batch, device, FLIPS_TTA))
            metric_rows = _metric_rows(f"{model_name}-development-{tag}-validation", model_name, patient, truth, variants)
            long_rows.extend(metric_rows)
            for item in metric_rows:
                if item["region"] == "WT":
                    wt[item["variant"]].append(item["dice"])
            del images, targets, truth, variants
            print(f"{model_name} validation variants {position}/38: {patient} "
                  + " ".join(f"{v}={wt[v][-1]:.4f}" for v in VARIANTS), flush=True)
        means[model_name] = {variant: float(np.mean(wt[variant])) for variant in VARIANTS}
        lineage[model_name] = dict(run_sha256=sha256(run_dir / "run.json"), checkpoint="best.pt",
                                   checkpoint_sha256=sha256(checkpoint), best_epoch=run.get("best_epoch"),
                                   batchnorm_layers=layers)
        del model, calibrated
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics_path = output / "validation_patient_metrics.csv"
    write_csv(metrics_path, long_rows, COLUMNS)
    summary = _summary_rows(long_rows, seed, n_resamples)
    summary_csv, summary_md = _write_summary(
        summary, output, "Inference variants on the 38 development-validation patients",
        "Decided before any test evaluation. Development best.pt of each model; mean +/- sample SD over 38 "
        "patients. BatchNorm statistics were re-estimated on a fixed seeded subset of development-training "
        "patients only.")
    decision = dict(status="completed", scope="inference_variant_decision_on_38_validation_patients_before_test",
                    tag=tag, **decide(means), validation_mean_WT_dice=means, config_sha256=sha256(config_path),
                    split_sha256=sha256(patient_csv), lineage=lineage,
                    batchnorm_calibration=dict(source="development_training_patients_only", seed=config["seed"],
                                               count=len(calibration), patient_ids=calibration),
                    environment=_environment(), metrics_sha256=sha256(metrics_path),
                    summary_sha256=sha256(summary_csv), bootstrap=dict(seed=seed, n_resamples=n_resamples),
                    elapsed_seconds=time.perf_counter() - started,
                    decided_at=datetime.now(timezone.utc).isoformat(),
                    disclosure="validation-selected inference variants; frozen test results do not select the variant")
    write_json(output / "decision.json", decision)
    print(f"Decision: adopt '{decision['adopted']}' ({decision['reason']})", flush=True)
    return decision


def _raw_dice(results_root, model_name, tag):
    path = Path(results_root) / "segmentation" / f"{model_name}-test-{tag}" / "segmentation_patient_metrics.csv"
    return {(row["patient_id"], row["region"]): float(row["dice"])
            for row in read_csv(path, ["patient_id", "region", "dice"])}


def evaluate_test_variants(config_path, patient_csv, root, runs_root, results_root, artifacts_root, decision_path,
                           output, tag="study", seed=20260914, n_resamples=10000):
    decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    if (decision.get("status") != "completed" or decision.get("config_sha256") != sha256(config_path)
            or decision.get("split_sha256") != sha256(patient_csv) or decision.get("adopted") not in VARIANTS):
        raise ValueError("A completed, provenance-matched validation decision is required before test evaluation")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    slice_batch = config["segmentation"]["slice_batch"]
    rows = read_csv(patient_csv, ["patient_id", "split"])
    tests = [row for row in rows if row["split"] == "test"]
    training = [row["patient_id"] for row in rows if row["split"] == "train"]
    if len(rows) != 235 or len(tests) != 47 or len(training) != 188:
        raise ValueError("Expected the frozen 188/47 split")
    calibration = calibration_ids(training, config["seed"], decision["batchnorm_calibration"]["count"])
    output = Path(output)
    if output.exists():
        raise FileExistsError("Test-variant output exists; test evaluations are immutable")
    output.mkdir(parents=True)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    long_rows, lineage = [], {}
    for model_name in MODELS:
        run_dir = Path(runs_root) / f"{model_name}-refit-{tag}"
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        checkpoint = run_dir / "final.pt"
        manifest_path = Path(results_root) / "segmentation" / f"{model_name}-test-{tag}" / "inference-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not _check_run(run, "refit", model_name, config_path, patient_csv, checkpoint, "final_checkpoint_sha256")
                or manifest.get("status") != "completed" or manifest.get("checkpoint_sha256") != sha256(checkpoint)
                or set(manifest.get("test_ids", [])) != {row["patient_id"] for row in tests}
                or set(run.get("training_ids", [])) != set(training)):
            raise ValueError(f"Completed, provenance-matched refit run and raw test inference required: {model_name}")
        saved = {item["patient_id"]: item for item in manifest["artifacts"]}
        raw_dice = _raw_dice(results_root, model_name, tag)
        model = build_segmentation(model_name, config).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        calibrated, layers = calibrated_copy(model, root, calibration, slice_batch, device)
        print(f"{model_name}: recalibrated {layers} BatchNorm layers on {len(calibration)} training patients", flush=True)
        for position, row in enumerate(tests, 1):
            patient = row["patient_id"]
            item = saved[patient]
            prediction_path = Path(artifacts_root) / "segmentation" / item["relative_path"]
            if sha256(prediction_path) != item["prediction_sha256"]:
                raise ValueError(f"Saved raw prediction hash mismatch: {prediction_path}")
            raw = nib.load(prediction_path).get_fdata(dtype=np.float32).astype(np.uint8).transpose(2, 0, 1)
            if raw.shape != (155, 240, 240) or not np.isin(raw, [0, 1, 2, 4]).all():
                raise ValueError(f"Invalid saved raw prediction: {prediction_path}")
            images, targets = load_patient(root, patient)
            truth = original_labels(targets)
            variants = variant_labels(raw, predict_labels(model, images, slice_batch, device, FLIPS_TTA),
                                      predict_labels(calibrated, images, slice_batch, device, FLIPS_RAW),
                                      predict_labels(calibrated, images, slice_batch, device, FLIPS_TTA))
            metric_rows = _metric_rows(f"{model_name}-refit-{tag}-test-variants", model_name, patient, truth, variants)
            for item_row in metric_rows:
                if item_row["variant"] == "raw" and abs(item_row["dice"] - raw_dice[(patient, item_row["region"])]) > 1e-9:
                    raise ValueError(f"Raw metrics do not reproduce the raw Table 5 inputs: {patient}/{item_row['region']}")
            long_rows.extend(metric_rows)
            wt = {item_row["variant"]: item_row["dice"] for item_row in metric_rows if item_row["region"] == "WT"}
            del images, targets, truth, variants, raw
            print(f"{model_name} test variants {position}/47: {patient} "
                  + " ".join(f"{v}={wt[v]:.4f}" for v in VARIANTS), flush=True)
        lineage[model_name] = dict(run_sha256=sha256(run_dir / "run.json"), checkpoint="final.pt",
                                   checkpoint_sha256=sha256(checkpoint),
                                   raw_inference_manifest_sha256=sha256(manifest_path), batchnorm_layers=layers)
        del model, calibrated
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics_path = output / "test_patient_metrics.csv"
    write_csv(metrics_path, long_rows, COLUMNS)
    per_variant = {}
    for model_name in MODELS:
        for variant in VARIANTS:
            path = output / "per-variant" / f"{model_name}_{variant}_patient_metrics.csv"
            write_csv(path, [r for r in long_rows if r["model"] == model_name and r["variant"] == variant], COLUMNS)
            per_variant[(model_name, variant)] = path
    summary = _summary_rows(long_rows, seed, n_resamples)
    adopted = decision["adopted"]
    summary_csv, summary_md = _write_summary(
        summary, output, "Inference variants on the frozen 47 test patients",
        f"Adopted before test evaluation from decision.json: **{adopted}**. raw and lcc reuse the hash-verified "
        f"{tag} test predictions; every other variant was recomputed on the machine recorded in manifest.json with "
        "BatchNorm statistics re-estimated on a fixed seeded subset of the 188 training patients only. Every "
        "variant is reported for transparency; only the adopted variant forms Table 5b.")
    table5b = None
    if adopted != "raw":
        table5b = summarize_segmentation([per_variant[("csfinet", adopted)], per_variant[("unet", adopted)]],
                                         output / "table5b", seed, n_resamples)
    manifest = dict(status="completed", scope="frozen_47_patient_test_inference_variants", tag=tag,
                    adopted_variant=adopted, decision_sha256=sha256(decision_path), rule=decision.get("rule"),
                    config_sha256=sha256(config_path), split_sha256=sha256(patient_csv), lineage=lineage,
                    batchnorm_calibration=dict(source="all_188_training_patients_only", seed=config["seed"],
                                               count=len(calibration), patient_ids=calibration),
                    environment=_environment(), metrics_sha256=sha256(metrics_path),
                    summary_sha256=sha256(summary_csv), summary_markdown_sha256=sha256(summary_md),
                    per_variant_sha256={f"{m}_{v}": sha256(p) for (m, v), p in per_variant.items()},
                    table5b=None if table5b is None else dict(
                        directory="table5b", table_sha256=table5b["table_sha256"],
                        report_sha256=sha256(output / "table5b" / "table5.json")),
                    bootstrap=dict(seed=seed, n_resamples=n_resamples), elapsed_seconds=time.perf_counter() - started,
                    test_access="single evaluation after a recorded validation decision; no selection on test",
                    disclosure="validation-selected inference variants; frozen test results do not select the variant")
    write_json(output / "manifest.json", manifest)
    return manifest
