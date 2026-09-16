"""Gradient SHAP and quantitative Table 9 analysis for the frozen CSFINet test run."""

import copy
from collections import defaultdict
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn

from .files import read_csv, sha256, write_csv, write_json
from .metrics import bca_mean_ci, bca_statistic_ci, correlation_ci, descriptive
from .models import build_segmentation
from .nifti import load_patient
from .training import seed_everything


class FixedRegionWTScore(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, region):
        probability = self.model(image).softmax(dim=1)[:, 1:].sum(dim=1)
        if region.shape[0] == 1 and len(probability) > 1:
            region = region.expand(len(probability), -1, -1)
        denominator = region.sum(dim=(1, 2)).clamp_min(1)
        return (probability * region).sum(dim=(1, 2)) / denominator


def gradient_shap(model, image, region, n_samples, stdev, seed):
    from captum.attr import GradientShap

    if image.shape != (1, 4, 240, 240) or region.shape != (1, 240, 240) or not region.any():
        raise ValueError("Gradient SHAP requires one slice and a nonempty fixed region")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    explainer = GradientShap(FixedRegionWTScore(model), multiply_by_inputs=True)
    return explainer.attribute(image, baselines=torch.zeros_like(image), n_samples=n_samples,
                               stdevs=float(stdev), additional_forward_args=region)


@torch.inference_mode()
def perturbation_curves(model, image, region, magnitude, steps):
    if steps < 2 or magnitude.shape != (240, 240):
        raise ValueError("Perturbation curve requires at least two steps and 240x240 attribution")
    order = torch.argsort(magnitude.reshape(-1), descending=True)
    fractions = torch.linspace(0, 1, steps + 1, device=image.device)
    deletion, insertion = [], []
    scorer = FixedRegionWTScore(model)
    flat_source = image.reshape(1, 4, -1)
    for fraction in fractions:
        count = int(round(float(fraction) * magnitude.numel()))
        deleted = flat_source.clone()
        inserted = torch.zeros_like(flat_source)
        if count:
            pixels = order[:count]
            deleted[:, :, pixels] = 0
            inserted[:, :, pixels] = flat_source[:, :, pixels]
        deletion.append(float(scorer(deleted.reshape_as(image), region)))
        insertion.append(float(scorer(inserted.reshape_as(image), region)))
    x = fractions.cpu().numpy()
    return dict(fractions=x.tolist(), deletion=deletion, insertion=insertion,
                deletion_auc=float(np.trapz(deletion, x)), insertion_auc=float(np.trapz(insertion, x)))


def attribution_metrics(attribution, lesion):
    magnitude = attribution.detach().abs().sum(dim=1)[0]
    total = float(magnitude.sum())
    inside = float(magnitude[lesion[0]].sum())
    maximum = int(torch.argmax(magnitude))
    pointing = int(bool(lesion.reshape(-1)[maximum]))
    return magnitude, dict(lesion_attribution_mass=inside / total if total else None,
                           pointing_accuracy=pointing if total else None, total_attribution_magnitude=total)


def _prediction_labels(path):
    image = nib.load(path)
    labels = image.get_fdata(dtype=np.float32).astype(np.uint8)
    if labels.shape != (240, 240, 155) or not np.isin(labels, [0, 1, 2, 4]).all():
        raise ValueError(f"Invalid saved segmentation prediction: {path}")
    return labels.transpose(2, 0, 1)


def explain_test(config_path, analysis_path, patient_csv, raw_root, refit_run, prediction_root,
                 inference_manifest, output, artifacts, resume=False, device_override=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    settings = analysis["gradient_shap"]
    run = json.loads((Path(refit_run) / "run.json").read_text(encoding="utf-8"))
    inference = json.loads(Path(inference_manifest).read_text(encoding="utf-8"))
    checkpoint = Path(refit_run) / "final.pt"
    if (run.get("status") != "completed" or run.get("model") != "csfinet" or run.get("phase") != "refit"
            or run.get("final_checkpoint_sha256") != sha256(checkpoint)
            or inference.get("status") != "completed" or inference.get("checkpoint_sha256") != sha256(checkpoint)
            or inference.get("split_sha256") != sha256(patient_csv)):
        raise ValueError("Completed matching CSFINet refit and test inference required")
    tests = [row for row in read_csv(patient_csv, ["patient_id", "split"]) if row["split"] == "test"]
    if len(tests) != 47:
        raise ValueError("Expected 47 frozen test patients")
    output, artifacts = Path(output), Path(artifacts)
    identity = dict(config_sha256=sha256(config_path), analysis_sha256=sha256(analysis_path),
                    split_sha256=sha256(patient_csv), checkpoint_sha256=sha256(checkpoint),
                    inference_manifest_sha256=sha256(inference_manifest))
    start_path = output / "explainability-start.json"
    if output.exists() or artifacts.exists():
        if not resume or not start_path.exists() or json.loads(start_path.read_text(encoding="utf-8")) != identity:
            raise FileExistsError("Explainability outputs exist without a matching explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    write_json(start_path, identity)
    seed_everything(settings["seed"])
    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_segmentation("csfinet", config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    patient_path, slice_path = output / "shap_patient_metrics.csv", output / "shap_slice_metrics.csv"
    progress_path = output / "progress.json"
    patient_rows = read_csv(patient_path, ["patient_id", "target"]) if resume and patient_path.exists() else []
    slice_rows = read_csv(slice_path, ["patient_id", "target", "slice_z"]) if resume and slice_path.exists() else []
    artifact_rows = json.loads(progress_path.read_text(encoding="utf-8")).get("artifacts", []) if resume and progress_path.exists() else []
    completed = {patient for patient in {row["patient_id"] for row in patient_rows}
                 if sum(row["patient_id"] == patient for row in patient_rows) == 2
                 and sum(row["patient_id"] == patient for row in artifact_rows) == 2}
    for patient_index, row in enumerate(tests):
        patient = row["patient_id"]
        if patient in completed:
            for artifact in [item for item in artifact_rows if item["patient_id"] == patient]:
                path = artifacts.parent / artifact["relative_path"]
                if not path.exists() or sha256(path) != artifact["sha256"]:
                    raise ValueError(f"Resumed SHAP artifact hash mismatch: {path}")
            print(f"Gradient SHAP patient {patient_index + 1}/47: {patient} (verified existing)", flush=True)
            continue
        patient_rows = [item for item in patient_rows if item["patient_id"] != patient]
        slice_rows = [item for item in slice_rows if item["patient_id"] != patient]
        artifact_rows = [item for item in artifact_rows if item["patient_id"] != patient]
        images, targets = load_patient(raw_root, patient)
        truth_wt = targets > 0
        predicted_wt = _prediction_labels(Path(prediction_root) / patient / f"{patient}_csfinet_prediction.nii.gz") > 0
        for target_name, fixed_region in (("ground_truth_WT_region", truth_wt), ("predicted_WT_region", predicted_wt)):
            per_slice, saved_indices, saved_attr = [], [], []
            eligible = np.flatnonzero(fixed_region.reshape(155, -1).any(axis=1))
            for slice_position, z in enumerate(eligible):
                image = torch.from_numpy(images[z:z + 1]).to(device)
                region = torch.from_numpy(fixed_region[z:z + 1]).to(device)
                attribution = gradient_shap(model, image, region, settings["n_samples"], settings["stdev"],
                                            settings["seed"] + patient_index * 1000 + slice_position)
                magnitude, measures = attribution_metrics(attribution, region)
                curves = perturbation_curves(model, image, region, magnitude, settings["perturbation_steps"])
                slice_rows.append(dict(patient_id=patient, target=target_name, slice_z=int(z), **measures,
                                       deletion_auc=curves["deletion_auc"], insertion_auc=curves["insertion_auc"]))
                per_slice.append((measures, curves))
                saved_indices.append(int(z))
                saved_attr.append(attribution.detach().cpu().numpy().astype(np.float16)[0])
            patient_result = dict(patient_id=patient, target=target_name, eligible_slices=len(eligible))
            for measure in ("lesion_attribution_mass", "pointing_accuracy", "deletion_auc", "insertion_auc"):
                values = [item[0][measure] if measure in item[0] else item[1][measure] for item in per_slice]
                valid = [value for value in values if value is not None]
                patient_result[measure] = float(np.mean(valid)) if valid else None
            patient_rows.append(patient_result)
            artifact = artifacts / patient / f"{target_name}.npz"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact.with_name(artifact.stem + ".part.npz")
            np.savez_compressed(temporary, slice_z=np.asarray(saved_indices, dtype=np.int16),
                                attribution=np.asarray(saved_attr, dtype=np.float16))
            temporary.replace(artifact)
            artifact_rows.append(dict(patient_id=patient, target=target_name,
                                      relative_path=str(artifact.relative_to(artifacts.parent)).replace("\\", "/"),
                                      sha256=sha256(artifact), slices=len(saved_indices)))
        write_csv(patient_path, patient_rows, list(patient_rows[0]))
        write_csv(slice_path, slice_rows, list(slice_rows[0]))
        complete_now = sum(sum(item["patient_id"] == pid for item in patient_rows) == 2
                           for pid in {item["patient_id"] for item in patient_rows})
        write_json(progress_path, dict(**identity, completed_patients=complete_now, artifacts=artifact_rows))
        print(f"Gradient SHAP patient {patient_index + 1}/47: {patient}", flush=True)
    write_csv(patient_path, patient_rows, list(patient_rows[0]))
    write_csv(slice_path, slice_rows, list(slice_rows[0]))
    manifest = dict(status="completed", scope="47_patient_segmentation_GradientShap", **identity,
                    patient_metrics_sha256=sha256(patient_path), slice_metrics_sha256=sha256(slice_path),
                    settings=settings, artifacts=artifact_rows)
    write_json(Path(output) / "manifest.json", manifest)
    return manifest


def summarize_table9(shap_metrics, segmentation_metrics, randomization_metrics, output, seed=20260914, n_resamples=10000):
    shap = read_csv(shap_metrics, ["patient_id", "target", "eligible_slices", "lesion_attribution_mass",
                                   "pointing_accuracy", "deletion_auc", "insertion_auc"])
    segmentation = read_csv(segmentation_metrics, ["patient_id", "model", "region", "dice"])
    if len(shap) != 94 or len({row["patient_id"] for row in shap}) != 47:
        raise ValueError("Table 9 requires two targets for all 47 patients")
    dice = {row["patient_id"]: float(row["dice"]) for row in segmentation
            if row["model"] == "csfinet" and row["region"] == "WT"}
    if len(dice) != 47:
        raise ValueError("Table 9 requires 47 CSFINet WT Dice values")
    randomization = read_csv(randomization_metrics, ["patient_id", "stage", "spearman_attribution_similarity"])
    expected_stages = ("head", "decoder", "piu_swin", "encoder")
    if len(randomization) != 47 * len(expected_stages) or {row["patient_id"] for row in randomization} != set(dice):
        raise ValueError("Table 9 requires four cumulative randomization stages for 47 matching patients")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table, details = [], {}
    for target in ("ground_truth_WT_region", "predicted_WT_region"):
        rows = [row for row in shap if row["target"] == target]
        if len(rows) != 47 or {row["patient_id"] for row in rows} != set(dice):
            raise ValueError("SHAP and Dice patient IDs must match")
        details[target] = {}
        item = dict(target=target)
        for index, measure in enumerate(("lesion_attribution_mass", "pointing_accuracy", "deletion_auc", "insertion_auc")):
            values = [None if row[measure] == "" else float(row[measure]) for row in rows]
            summary = descriptive(values)
            ci = bca_mean_ci(values, seed=seed + index, n_resamples=n_resamples)
            details[target][measure] = dict(summary=summary, mean_ci=ci)
            item[f"{measure}_mean"] = summary["mean"]
            item[f"{measure}_sd"] = summary["sd"]
            item[f"{measure}_n"] = summary["n_valid"]
        joint = [(dice[row["patient_id"]], float(row["lesion_attribution_mass"])) for row in rows
                 if row["lesion_attribution_mass"] != ""]
        if len(joint) >= 3:
            wt_dice, masses = map(list, zip(*joint))
            association = correlation_ci(wt_dice, masses, seed=seed + 10, n_resamples=n_resamples)["spearman"]
            association["p_value"] = float(spearmanr(wt_dice, masses).pvalue)
        else:
            association = dict(coefficient=None, low=None, high=None, p_value=None,
                               status="undefined_fewer_than_three_joint_patients", n=len(joint))
        details[target]["mass_vs_WT_Dice_spearman"] = association
        item.update(mass_dice_spearman=association["coefficient"], mass_dice_ci_low=association["low"],
                    mass_dice_ci_high=association["high"], mass_dice_p=association["p_value"])
        table.append(item)
    randomization_summary = {}
    randomization_table = []
    for index, stage in enumerate(expected_stages):
        values = [None if row["spearman_attribution_similarity"] == "" else float(row["spearman_attribution_similarity"])
                  for row in randomization if row["stage"] == stage]
        summary = descriptive(values)
        ci = bca_mean_ci(values, seed=seed + 20 + index, n_resamples=n_resamples)
        randomization_summary[stage] = dict(summary=summary, mean_ci=ci)
        randomization_table.append(dict(stage=stage, mean_spearman_similarity=summary["mean"],
                                        sd_spearman_similarity=summary["sd"], n=summary["n_valid"],
                                        ci_low=ci["low"], ci_high=ci["high"]))
    table_path = output / "table9.csv"
    write_csv(table_path, table, list(table[0]))
    randomization_path = output / "table9_randomization.csv"
    write_csv(randomization_path, randomization_table, list(randomization_table[0]))
    markdown = ["# Table 9 — Gradient SHAP quantitative analysis", "",
                "Metrics are patient means over all slices with a nonempty fixed target region; uncertainty is across 47 patients.", "",
                "| Target | Lesion mass | Pointing accuracy | Deletion AUC | Insertion AUC | Mass–WT Dice ρ [95% CI] |",
                "| --- | --- | --- | --- | --- | --- |"]
    for row in table:
        def mean_sd(name):
            mean, sd, count = row[f"{name}_mean"], row[f"{name}_sd"], row[f"{name}_n"]
            return f"NA (n={count})" if mean is None or sd is None else f"{mean:.4f} ± {sd:.4f} (n={count})"
        association = ("NA" if any(row[key] is None for key in
                                   ("mass_dice_spearman", "mass_dice_ci_low", "mass_dice_ci_high")) else
                       f"{row['mass_dice_spearman']:.3f} [{row['mass_dice_ci_low']:.3f}, {row['mass_dice_ci_high']:.3f}]")
        markdown.append(f"| {row['target']} | {mean_sd('lesion_attribution_mass')} | {mean_sd('pointing_accuracy')} | "
                        f"{mean_sd('deletion_auc')} | {mean_sd('insertion_auc')} | {association} |")
    markdown += ["", "## Cumulative parameter randomization", "",
                 "| Randomized through stage | Attribution Spearman similarity |", "| --- | --- |"]
    for row in randomization_table:
        value = ("NA" if row["mean_spearman_similarity"] is None or row["sd_spearman_similarity"] is None else
                 f"{row['mean_spearman_similarity']:.4f} ± {row['sd_spearman_similarity']:.4f}")
        markdown.append(f"| {row['stage']} | {value} (n={row['n']}) |")
    (output / "table9.md").write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    def tex_value(row, name):
        mean, sd, count = row[f"{name}_mean"], row[f"{name}_sd"], row[f"{name}_n"]
        return f"NA (n={count})" if mean is None or sd is None else f"{mean:.4f} \\pm {sd:.4f} (n={count})"
    latex = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Target & Lesion mass & Pointing accuracy & Deletion AUC & Insertion AUC & Mass--Dice $\rho$ \\",
             r"\midrule"]
    for row in table:
        association = "NA" if row["mass_dice_spearman"] is None else f"{row['mass_dice_spearman']:.3f}"
        target = row["target"].replace("_", r"\_")
        latex.append(f"{target} & {tex_value(row, 'lesion_attribution_mass')} & {tex_value(row, 'pointing_accuracy')} & "
                     f"{tex_value(row, 'deletion_auc')} & {tex_value(row, 'insertion_auc')} & {association} \\\\")
    latex += [r"\bottomrule", r"\end{tabular}"]
    latex_path = output / "table9.tex"
    latex_path.write_text("\n".join(latex) + "\n", encoding="utf-8", newline="\n")
    report = dict(status="completed", scope="47_patient_GradientShap_quantitative_analysis", shap_sha256=sha256(shap_metrics),
                  segmentation_sha256=sha256(segmentation_metrics), randomization_sha256=sha256(randomization_metrics),
                  seed=seed, n_resamples=n_resamples, table_sha256=sha256(table_path), latex_sha256=sha256(latex_path),
                  randomization_table_sha256=sha256(randomization_path), targets=details,
                  cumulative_parameter_randomization=randomization_summary)
    write_json(output / "table9.json", report)
    return report


def _reset_modules(modules, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    visited = set()
    for root in modules:
        for module in root.modules():
            if id(module) in visited:
                continue
            visited.add(id(module))
            if isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()
            elif hasattr(module, "reset_parameters") and not list(module.children()):
                module.reset_parameters()


def parameter_randomization(config_path, analysis_path, patient_csv, raw_root, refit_run, output, resume=False, device_override=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    settings = json.loads(Path(analysis_path).read_text(encoding="utf-8"))["gradient_shap"]
    run = json.loads((Path(refit_run) / "run.json").read_text(encoding="utf-8"))
    checkpoint = Path(refit_run) / "final.pt"
    if (run.get("status") != "completed" or run.get("model") != "csfinet" or run.get("phase") != "refit"
            or run.get("final_checkpoint_sha256") != sha256(checkpoint)):
        raise ValueError("Completed CSFINet refit required for randomization")
    tests = [row for row in read_csv(patient_csv, ["patient_id", "split"]) if row["split"] == "test"]
    if len(tests) != 47:
        raise ValueError("Expected 47 test patients")
    output = Path(output)
    identity = dict(config_sha256=sha256(config_path), analysis_sha256=sha256(analysis_path),
                    split_sha256=sha256(patient_csv), checkpoint_sha256=sha256(checkpoint))
    start_path = output.with_suffix(".start.json")
    if output.exists() or start_path.exists():
        if not resume or not start_path.exists() or json.loads(start_path.read_text(encoding="utf-8")) != identity:
            raise FileExistsError("Randomization output exists without a matching explicit resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(start_path, identity)
    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    original = build_segmentation("csfinet", config).to(device)
    original.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    original.eval()
    randomized = copy.deepcopy(original)
    groups = (("head", [randomized.head]),
              ("decoder", [randomized.fiu, randomized.fusion]),
              ("piu_swin", [randomized.piu, randomized.swin, randomized.merge]),
              ("encoder", [randomized.encoder]))
    rows = read_csv(output, ["patient_id", "stage", "spearman_attribution_similarity"]) if resume and output.exists() else []
    completed = {patient for patient in {row["patient_id"] for row in rows}
                 if sum(row["patient_id"] == patient for row in rows) == len(groups)}
    for patient_index, row in enumerate(tests):
        patient = row["patient_id"]
        if patient in completed:
            print(f"Parameter randomization patient {patient_index + 1}/47: {patient} (existing)", flush=True)
            continue
        rows = [item for item in rows if item["patient_id"] != patient]
        images, targets = load_patient(raw_root, patient)
        counts = (targets > 0).reshape(155, -1).sum(axis=1)
        z = int(np.argmax(counts))
        if counts[z] == 0:
            raise ValueError(f"Test patient has no ground-truth WT: {patient}")
        image = torch.from_numpy(images[z:z + 1]).to(device)
        region = torch.from_numpy((targets[z:z + 1] > 0)).to(device)
        baseline = gradient_shap(original, image, region, settings["randomization_samples"], settings["stdev"],
                                 settings["seed"] + patient_index).detach().abs().sum(dim=1).cpu().numpy().reshape(-1)
        randomized.load_state_dict(original.state_dict())
        for stage_index, (stage, modules) in enumerate(groups):
            _reset_modules(modules, settings["seed"] + 10000 + stage_index)
            randomized.eval()
            changed = gradient_shap(randomized, image, region, settings["randomization_samples"], settings["stdev"],
                                    settings["seed"] + patient_index).detach().abs().sum(dim=1).cpu().numpy().reshape(-1)
            with np.errstate(invalid="ignore"):
                correlation = float(spearmanr(baseline, changed).statistic)
            rows.append(dict(patient_id=patient, slice_z=z, target="ground_truth_WT_region", stage=stage,
                             cumulative_randomization=True, spearman_attribution_similarity=(correlation if np.isfinite(correlation) else None)))
        write_csv(output, rows, list(rows[0]))
        print(f"Parameter randomization patient {patient_index + 1}/47: {patient}", flush=True)
    write_csv(output, rows, list(rows[0]))
    manifest = dict(status="completed", scope="47_patient_cumulative_parameter_randomization",
                    **identity,
                    output_sha256=sha256(output), stages=[stage for stage, _ in groups],
                    slice_policy=settings["randomization_slice"], n_samples=settings["randomization_samples"])
    write_json(Path(output).with_suffix(".manifest.json"), manifest)
    return manifest
