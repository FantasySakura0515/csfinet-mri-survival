"""Standalone study report from saved patient-level segmentation and survival results.

Run: python -m csfinet_repro.analysis_report --project-root . --output results/delivery/study
Missing inputs are disclosed; no training or inference is performed.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .files import read_csv, sha256, write_csv, write_json
from .metrics import (bca_mean_ci, bca_statistic_ci, correlation_ci, descriptive, holm_adjust, paired_difference,
                      paired_wilcoxon, survival_metrics)

TEST_PATIENTS = 47
REGIONS = ("WT", "TC", "ET")
MEASURES = ("dice", "precision", "recall")
SEGMENTATION_MODELS = ("csfinet", "unet")
SEGMENTATION_TAGS = ("study",)
SEGMENTATION_COLUMNS = ["run_id", "patient_id", "model", "region", "tp", "fp", "fn", "dice", "precision", "recall", "iou"]
VARIANT_COLUMNS = ["run_id", "patient_id", "model", "variant", "region", "tp", "fp", "fn", "dice", "precision",
                   "recall", "iou"]
# Mirrors inference_variants.VARIANTS without importing torch; unknown variants are appended alphabetically.
INFERENCE_VARIANTS = ("raw", "lcc", "tta", "tta_lcc", "bn", "bn_lcc", "bn_tta", "bn_tta_lcc")
SURVIVAL_TAGS = ("survival",)
SURVIVAL_NEURAL = ("image", "image_age", "image_resection", "image_age_resection")
SURVIVAL_REFERENCES = ("training_mean", "age_ols")
SURVIVAL_MODELS = (*SURVIVAL_NEURAL, *SURVIVAL_REFERENCES)
SURVIVAL_COLUMNS = ["run_id", "patient_id", "model", "y_true", "y_pred", "residual", "absolute_error", "age",
                    "resection_status", "mask_source"]
ROLES = {
    "study": "study_segmentation",
    "study_variant": "validation_selected_inference_variant",
    "survival": "masked_mri_survival",
}
FAMILIES = {
    "A": "CSFINet vs U-Net within each experiment, WT Dice",
    "C": "adopted inference variant vs raw for each model, WT Dice",
    "E": "survival neural models vs age_ols and training_mean absolute error",
}
PRIMARY_SEGMENTATION = "prespecified_primary_WT_dice"
EXPLORATORY_DICE = "exploratory_TC_ET_dice_interval_only"
EXPLORATORY_WT = "exploratory_WT_precision_recall_interval_only"
PROVENANCE_FILES = (("configs/segmentation.json", "segmentation_config"),
                    ("configs/survival.json", "survival_config"),
                    ("docs/protocol.md", "protocol_record"))
ALPHA = .05
TOLERANCE = 1e-9


def _take(root, relative, role, experiment, inputs, effect="experiment_unavailable"):
    """Hash and record an input when it exists; otherwise record it as missing and return None."""
    path = Path(root) / relative
    if path.is_file():
        inputs["present"].append(dict(path=relative, role=role, experiment=experiment, bytes=path.stat().st_size,
                                      sha256=sha256(path)))
        return path
    inputs["missing"].append(dict(path=relative, role=role, experiment=experiment, effect=effect))
    return None


def _completed_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("status") != "completed":
        raise ValueError(f"{path}: report is not completed")
    return value


def _close(left, right):
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= TOLERANCE


def _patients(values):
    return sorted({patient for patient, _ in values})


def _check_regions(values, label):
    patients = _patients(values)
    if len(patients) != TEST_PATIENTS:
        raise ValueError(f"{label}: expected {TEST_PATIENTS} test patients, found {len(patients)}")
    absent = [(patient, region) for patient in patients for region in REGIONS if (patient, region) not in values]
    if absent:
        raise ValueError(f"{label}: every patient needs all regions {REGIONS}; first missing {absent[0]}")


def _parse_metric_rows(rows, label, by_variant=False):
    grouped = defaultdict(dict)
    for row in rows:
        group = (row["model"], row["variant"]) if by_variant else row["model"]
        key = (row["patient_id"], row["region"])
        if row["region"] not in REGIONS or key in grouped[group]:
            raise ValueError(f"{label}: duplicate or invalid segmentation row {group}/{key}")
        grouped[group][key] = {measure: None if row[measure] == "" else float(row[measure]) for measure in MEASURES}
    for group, values in grouped.items():
        _check_regions(values, f"{label} [{group}]")
    return dict(grouped)


def load_segmentation_metrics(path):
    """One model's 47 x 3 rows keyed by (patient_id, region); blank precision/recall become None."""
    grouped = _parse_metric_rows(read_csv(path, SEGMENTATION_COLUMNS), path)
    if len(grouped) != 1:
        raise ValueError(f"{path}: expected exactly one model, found {sorted(grouped)}")
    return next(iter(grouped.items()))


def load_variant_metrics(path):
    """{(model, variant): values} from a test_patient_metrics.csv that carries a variant column."""
    return _parse_metric_rows(read_csv(path, VARIANT_COLUMNS), path, by_variant=True)


def load_survival_predictions(path):
    """{model: {patient_id: record}} for the six models; truth and age must agree across models."""
    rows = read_csv(path, SURVIVAL_COLUMNS)
    groups, clinical = defaultdict(dict), {}
    for row in rows:
        model, patient = row["model"], row["patient_id"]
        if patient in groups[model]:
            raise ValueError(f"{path}: duplicate survival row {model}/{patient}")
        observed, predicted, age = float(row["y_true"]), float(row["y_pred"]), float(row["age"])
        if clinical.setdefault(patient, (observed, age)) != (observed, age):
            raise ValueError(f"{path}: y_true or age differs across models for {patient}")
        groups[model][patient] = dict(y_true=observed, y_pred=predicted, age=age,
                                      absolute_error=abs(predicted - observed), mask_source=row["mask_source"])
    if set(groups) != set(SURVIVAL_MODELS):
        raise ValueError(f"{path}: expected the six models {sorted(SURVIVAL_MODELS)}, found {sorted(groups)}")
    patients = set(clinical)
    if len(patients) != TEST_PATIENTS or any(set(group) != patients for group in groups.values()):
        raise ValueError(f"{path}: every model must cover the same {TEST_PATIENTS} test patients")
    return dict(groups)


def _same_metrics(left, right, label):
    if set(left) != set(right):
        raise ValueError(f"{label}: patient/region keys differ from the plain test metrics")
    for key in sorted(left):
        for measure in MEASURES:
            if not _close(left[key][measure], right[key][measure]):
                raise ValueError(f"{label}: {key}/{measure} raw variant value differs from the plain test metrics")


def _check_table5(path, models, label):
    """A saved Table 5 report must summarize exactly these per-patient rows."""
    report = _completed_json(path)
    for model, values in models.items():
        patients = _patients(values)
        for region in REGIONS:
            for measure in MEASURES:
                mean = descriptive(values[(patient, region)][measure] for patient in patients)["mean"]
                try:
                    reported = report["models"][model][region][measure]["summary"]["mean"]
                except (KeyError, TypeError) as error:
                    raise ValueError(f"{path}: no {model}/{region}/{measure} summary") from error
                if not _close(mean, reported):
                    raise ValueError(f"{path}: {model}/{region}/{measure} mean does not match {label}")
    if all(model in models for model in SEGMENTATION_MODELS):
        left, right = SEGMENTATION_MODELS
        differences = [models[left][(patient, "WT")]["dice"] - models[right][(patient, "WT")]["dice"]
                       for patient in _patients(models[left])]
        try:
            reported = report["paired"]["WT"]["dice"]["difference"]["summary"]["mean"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path}: no paired WT Dice summary") from error
        if report.get("direction") != f"{left}_minus_{right}" or not _close(float(np.mean(differences)), reported):
            raise ValueError(f"{path}: paired WT Dice difference does not match {label}")


def _check_table7(path, groups, label):
    """A saved Table 7 report must summarize exactly these per-patient predictions."""
    report = _completed_json(path)
    for model in SURVIVAL_MODELS:
        mae = float(np.mean([groups[model][patient]["absolute_error"] for patient in sorted(groups[model])]))
        try:
            reported = report["models"][model]["metrics"]["mae"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path}: no {model} MAE") from error
        if not _close(mae, reported):
            raise ValueError(f"{path}: {model} MAE does not match {label}")


def _collect_segmentation(root, inputs):
    """Load every available segmentation experiment: plain study and the adopted inference variants."""
    experiments, variants = [], {}
    for tag in SEGMENTATION_TAGS:
        models = {}
        for model in SEGMENTATION_MODELS:
            path = _take(root, f"results/segmentation/{model}-test-{tag}/segmentation_patient_metrics.csv",
                         "segmentation_patient_metrics", tag, inputs)
            if path is not None:
                loaded, values = load_segmentation_metrics(path)
                if loaded != model:
                    raise ValueError(f"{path}: file describes model {loaded!r}, expected {model!r}")
                models[model] = values
        table5 = _take(root, f"results/tables/{tag}/segmentation/table5.json", "table5_report", tag, inputs,
                       "cross_check_skipped")
        if models:
            if table5 is not None:
                _check_table5(table5, models, f"results/segmentation/*-test-{tag}")
            experiments.append(dict(name=tag, tag=tag, variant=None, role=ROLES[tag], models=models,
                                    complete=len(models) == len(SEGMENTATION_MODELS)))
        decision_path = _take(root, f"results/inference-variants/{tag}/validation/decision.json",
                              "inference_variant_decision", f"{tag}_variants", inputs)
        metrics_path = _take(root, f"results/inference-variants/{tag}/test/test_patient_metrics.csv",
                             "inference_variant_test_metrics", f"{tag}_variants", inputs)
        _take(root, f"results/inference-variants/{tag}/test/variant_summary.csv", "inference_variant_summary",
              f"{tag}_variants", inputs, "hash_only")
        manifest_path = _take(root, f"results/inference-variants/{tag}/test/manifest.json",
                              "inference_variant_test_manifest", f"{tag}_variants", inputs, "cross_check_skipped")
        adopted = None
        if decision_path is not None:
            decision = _completed_json(decision_path)
            adopted = decision.get("adopted")
            if not adopted or decision.get("tag", tag) != tag:
                raise ValueError(f"{decision_path}: no adopted variant for tag {tag}")
        variants[tag] = dict(adopted=adopted, decision=decision_path is not None, grouped=None)
        if metrics_path is None:
            continue
        grouped = load_variant_metrics(metrics_path)
        for model in SEGMENTATION_MODELS:
            if (model, "raw") not in grouped:
                raise ValueError(f"{metrics_path}: raw variant rows are required for {model}")
            if model in models:
                _same_metrics(grouped[(model, "raw")], models[model], f"{metrics_path} [{model}/raw]")
        variants[tag]["grouped"] = grouped
        if adopted is None:
            continue
        if any((model, adopted) not in grouped for model in SEGMENTATION_MODELS):
            raise ValueError(f"{metrics_path}: adopted variant {adopted!r} is not present for every model")
        if manifest_path is not None and _completed_json(manifest_path).get("adopted_variant") != adopted:
            raise ValueError(f"{manifest_path}: adopted variant differs from {decision_path}")
        if adopted == "raw":
            continue
        name = f"{tag}_{adopted}"
        variant_models = {model: grouped[(model, adopted)] for model in SEGMENTATION_MODELS}
        table5b = _take(root, f"results/inference-variants/{tag}/test/table5b/table5.json", "table5b_report", name,
                        inputs, "cross_check_skipped")
        if table5b is not None:
            _check_table5(table5b, variant_models, name)
        experiments.append(dict(name=name, tag=tag, variant=adopted, role=ROLES[f"{tag}_variant"],
                                models=variant_models, complete=True))
    return experiments, variants


def _collect_survival(root, inputs):
    experiments = []
    for tag in SURVIVAL_TAGS:
        path = _take(root, f"results/survival/{tag}/survival_predictions.csv", "survival_predictions", tag, inputs)
        table7 = _take(root, f"results/tables/{tag}/survival/table7.json", "table7_report", tag, inputs,
                       "cross_check_skipped")
        if path is None:
            continue
        groups = load_survival_predictions(path)
        if table7 is not None:
            _check_table7(table7, groups, str(path))
        experiments.append(dict(name=tag, tag=tag, role=ROLES[tag], groups=groups))
    return experiments


def _check_cohort(segmentation, survival):
    """Every present per-patient set must describe the same frozen test patients and the same survival truth."""
    reference = None
    for experiment in segmentation:
        for model, values in experiment["models"].items():
            current = (f"segmentation {experiment['name']}/{model}", set(_patients(values)))
            reference = reference or current
            if current[1] != reference[1]:
                raise ValueError(f"{current[0]} covers different patients than {reference[0]}")
    truth = None
    for experiment in survival:
        current = (f"survival {experiment['name']}", set(experiment["groups"]["image"]))
        reference = reference or current
        if current[1] != reference[1]:
            raise ValueError(f"{current[0]} covers different patients than {reference[0]}")
        clinical = {patient: (record["y_true"], record["age"])
                    for patient, record in experiment["groups"]["image"].items()}
        truth = truth or (current[0], clinical)
        if clinical != truth[1]:
            raise ValueError(f"{current[0]}: y_true or age differs from {truth[0]}")
    if reference is None:
        raise ValueError("No completed experiment inputs were found under results/; nothing to report")
    return sorted(reference[1])


def _segmentation_rows(experiments, seeds, n_resamples):
    rows = []
    for experiment in experiments:
        for model in SEGMENTATION_MODELS:
            if model not in experiment["models"]:
                continue
            values = experiment["models"][model]
            patients = _patients(values)
            for region in REGIONS:
                row = dict(experiment=experiment["name"], role=experiment["role"], model=model, region=region,
                           n_patients=len(patients))
                for measure in MEASURES:
                    observations = [values[(patient, region)][measure] for patient in patients]
                    summary = descriptive(observations)
                    ci = bca_mean_ci(observations, seed=next(seeds), n_resamples=n_resamples)
                    row.update({f"{measure}_mean": summary["mean"], f"{measure}_sd": summary["sd"],
                                f"{measure}_median": summary["median"], f"{measure}_n": summary["n_valid"],
                                f"{measure}_ci_low": ci["low"], f"{measure}_ci_high": ci["high"],
                                f"{measure}_ci_status": ci["status"]})
                rows.append(row)
    return rows


def _variant_order(names):
    return sorted(names, key=lambda name: (INFERENCE_VARIANTS.index(name) if name in INFERENCE_VARIANTS
                                           else len(INFERENCE_VARIANTS), name))


def _variant_rows(variants, seeds, n_resamples):
    """Every inference variant of every model, descriptive plus a WT Dice interval against raw; no tests."""
    rows = []
    for tag, item in variants.items():
        grouped = item["grouped"]
        if grouped is None:
            continue
        for model in SEGMENTATION_MODELS:
            raw = grouped[(model, "raw")]
            patients = _patients(raw)
            raw_wt = {patient: raw[(patient, "WT")]["dice"] for patient in patients}
            for variant in _variant_order({name for owner, name in grouped if owner == model}):
                values = grouped[(model, variant)]
                row = dict(experiment_tag=tag, model=model, variant=variant, decision_available=item["decision"],
                           adopted=variant == item["adopted"], n_patients=len(patients))
                for region in REGIONS:
                    for measure in MEASURES:
                        summary = descriptive(values[(patient, region)][measure] for patient in patients)
                        row.update({f"{region}_{measure}_mean": summary["mean"], f"{region}_{measure}_sd": summary["sd"],
                                    f"{region}_{measure}_median": summary["median"],
                                    f"{region}_{measure}_n": summary["n_valid"]})
                row.update(WT_dice_diff_vs_raw_mean=None, WT_dice_diff_ci_low=None, WT_dice_diff_ci_high=None,
                           WT_dice_diff_ci_status=None, bootstrap_seed=None)
                if variant != "raw":
                    seed = next(seeds)
                    difference = paired_difference({patient: values[(patient, "WT")]["dice"] for patient in patients},
                                                   raw_wt, seed=seed, n_resamples=n_resamples)
                    row.update(WT_dice_diff_vs_raw_mean=difference["summary"]["mean"],
                               WT_dice_diff_ci_low=difference["mean_ci"]["low"],
                               WT_dice_diff_ci_high=difference["mean_ci"]["high"],
                               WT_dice_diff_ci_status=difference["mean_ci"]["status"], bootstrap_seed=seed)
                row["analysis_role"] = "descriptive_all_variants_interval_only_no_test"
                rows.append(row)
    return rows


def _paired_row(left, right, *, family, comparison, experiment, model, region, measure, left_label, right_label,
                role, favourable, claim, seeds, n_resamples, test=True):
    """ID-aligned left minus right with a paired BCa interval and, when requested, a two-sided Wilcoxon test."""
    seed = next(seeds)
    difference = paired_difference(left, right, seed=seed, n_resamples=n_resamples)
    wilcoxon = paired_wilcoxon(left, right) if test else None
    summary, ci = difference["summary"], difference["mean_ci"]
    joint = [abs(left[key] - right[key]) for key in left if left[key] is not None and right[key] is not None]
    return dict(family=family, family_label=FAMILIES.get(family, family), comparison=comparison, experiment=experiment,
                model=model, region=region, measure=measure, left=left_label, right=right_label,
                direction="left_minus_right", favourable_sign=favourable, n=summary["n_valid"],
                mean_difference=summary["mean"], sd_difference=summary["sd"], median_difference=summary["median"],
                ci_low=ci["low"], ci_high=ci["high"], ci_status=ci["status"],
                max_abs_difference=max(joint) if joint else None,
                wilcoxon_n=None if wilcoxon is None else wilcoxon["n"],
                wilcoxon_statistic=None if wilcoxon is None else wilcoxon["statistic"],
                wilcoxon_status=None if wilcoxon is None else wilcoxon["status"],
                raw_p=None if wilcoxon is None else wilcoxon["p_value"], holm_p=None, analysis_role=role, claim=claim,
                bootstrap_seed=seed, n_resamples=n_resamples)


def _holm_within_families(rows):
    families = defaultdict(dict)
    for row in rows:
        if row["raw_p"] is not None:
            if row["comparison"] in families[row["family"]]:
                raise ValueError(f"Duplicate tested comparison {row['comparison']} in family {row['family']}")
            families[row["family"]][row["comparison"]] = row["raw_p"]
    adjusted = {family: holm_adjust(p_values) for family, p_values in families.items()}
    for row in rows:
        if row["raw_p"] is not None:
            row["holm_p"] = adjusted[row["family"]][row["comparison"]]


def _segmentation_specs(experiments):
    by_name = {experiment["name"]: experiment for experiment in experiments}
    specs = []
    for experiment in experiments:
        if experiment["complete"]:
            name = experiment["name"]
            specs.append(dict(family="A", comparison=f"A:{name}:csfinet_minus_unet", experiment=name,
                              model="csfinet_vs_unet", left=experiment["models"]["csfinet"],
                              right=experiment["models"]["unet"], left_label=f"csfinet@{name}",
                              right_label=f"unet@{name}", claim=f"CSFINet has higher WT Dice than U-Net in {name}"))
    for experiment in experiments:
        if experiment["variant"] is None or experiment["tag"] not in by_name:
            continue
        raw, tag, variant = by_name[experiment["tag"]], experiment["tag"], experiment["variant"]
        for model in SEGMENTATION_MODELS:
            if model in raw["models"] and model in experiment["models"]:
                specs.append(dict(family="C", comparison=f"C:{tag}:{model}:{variant}_minus_raw", experiment=tag,
                                  model=model, left=experiment["models"][model], right=raw["models"][model],
                                  left_label=f"{model}@{experiment['name']}", right_label=f"{model}@{tag}",
                                  claim=f"the adopted inference variant {variant} raises WT Dice over raw "
                                        f"inference for {model} in {tag}"))
    return specs


def _segmentation_comparisons(experiments, seeds, n_resamples):
    """Families A and C on WT Dice with Holm adjustment; TC/ET Dice and WT precision/recall as intervals only."""
    rows = []
    plan = (("WT", "dice", PRIMARY_SEGMENTATION, True), ("TC", "dice", EXPLORATORY_DICE, False),
            ("ET", "dice", EXPLORATORY_DICE, False), ("WT", "precision", EXPLORATORY_WT, False),
            ("WT", "recall", EXPLORATORY_WT, False))
    for spec in _segmentation_specs(experiments):
        for region, measure, role, test in plan:
            left = {patient: spec["left"][(patient, region)][measure] for patient in _patients(spec["left"])}
            right = {patient: spec["right"][(patient, region)][measure] for patient in _patients(spec["right"])}
            rows.append(_paired_row(left, right, family=spec["family"], comparison=spec["comparison"],
                                    experiment=spec["experiment"], model=spec["model"], region=region, measure=measure,
                                    left_label=spec["left_label"], right_label=spec["right_label"], role=role,
                                    favourable="positive", claim=spec["claim"], seeds=seeds, n_resamples=n_resamples,
                                    test=test))
    _holm_within_families(rows)
    return rows


def _arrays(group):
    patients = sorted(group)
    return tuple(np.asarray([group[patient][key] for patient in patients], dtype=float)
                 for key in ("y_true", "y_pred", "age"))


def _survival_rows(experiments, seeds, n_resamples):
    rows = []
    for experiment in experiments:
        for model in SURVIVAL_MODELS:
            group = experiment["groups"][model]
            observed, predicted, _ = _arrays(group)
            metrics = survival_metrics(observed, predicted)
            mae_ci = bca_mean_ci(metrics["absolute_error"], seed=next(seeds), n_resamples=n_resamples)
            rmse_ci = bca_statistic_ci(np.asarray(metrics["residual"]), lambda x: np.sqrt(np.mean(x ** 2)),
                                       seed=next(seeds), n_resamples=n_resamples, name="RMSE")
            median_ci = bca_statistic_ci(metrics["absolute_error"], np.median, seed=next(seeds),
                                         n_resamples=n_resamples, name="median_AE")
            seed = next(seeds)
            next(seeds)  # correlation_ci consumes seed and seed + 1
            correlations = correlation_ci(observed, predicted, seed=seed, n_resamples=n_resamples)
            pearson, spearman = correlations["pearson"], correlations["spearman"]
            rows.append(dict(experiment=experiment["name"], role=experiment["role"], model=model,
                             model_kind="neural" if model in SURVIVAL_NEURAL else "reference", n=metrics["n"],
                             mae=metrics["mae"], sd_ae=metrics["sd_ae"], mae_ci_low=mae_ci["low"],
                             mae_ci_high=mae_ci["high"], mae_ci_status=mae_ci["status"], rmse=metrics["rmse"],
                             rmse_ci_low=rmse_ci["low"], rmse_ci_high=rmse_ci["high"], median_ae=metrics["median_ae"],
                             median_ae_ci_low=median_ci["low"], median_ae_ci_high=median_ci["high"],
                             mean_residual=metrics["mean_residual"], sd_residual=metrics["sd_residual"],
                             pearson=pearson["coefficient"], pearson_ci_low=pearson["low"],
                             pearson_ci_high=pearson["high"], pearson_status=pearson["status"],
                             spearman=spearman["coefficient"], spearman_ci_low=spearman["low"],
                             spearman_ci_high=spearman["high"], spearman_status=spearman["status"],
                             mask_source="|".join(sorted({record["mask_source"] for record in group.values()}))))
    return rows


def _correlation(left, right):
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _degeneracy_rows(experiments):
    """Prediction spread and age tracking: a near-constant output or an age proxy is not image information."""
    rows = []
    for experiment in experiments:
        for model in SURVIVAL_MODELS:
            observed, predicted, age = _arrays(experiment["groups"][model])
            prediction = descriptive(predicted)
            truth = descriptive(observed)
            rows.append(dict(experiment=experiment["name"], model=model, n=len(predicted),
                             prediction_mean=prediction["mean"], prediction_sd=prediction["sd"],
                             prediction_min=float(predicted.min()), prediction_max=float(predicted.max()),
                             distinct_predictions=len(np.unique(np.round(predicted, 6))),
                             observed_sd=truth["sd"], prediction_sd_over_observed_sd=prediction["sd"] / truth["sd"],
                             prediction_age_r=_correlation(age, predicted), observed_age_r=_correlation(age, observed),
                             prediction_observed_r=_correlation(predicted, observed),
                             mae=float(np.mean(np.abs(predicted - observed))),
                             analysis_role="descriptive_degeneracy_diagnostic"))
    return rows


def _survival_comparisons(experiments, seeds, n_resamples):
    """Family E: neural models versus the two reference predictors."""
    def errors(experiment, model):
        return {patient: record["absolute_error"] for patient, record in experiment["groups"][model].items()}

    rows = []
    for experiment in experiments:
        for reference in SURVIVAL_REFERENCES:
            for model in SURVIVAL_NEURAL:
                rows.append(_paired_row(errors(experiment, model), errors(experiment, reference), family="E",
                    comparison=f"E:{model}_minus_{reference}", experiment=experiment["name"], model=model,
                    region=None, measure="absolute_error", left_label=f"{model}@{experiment['name']}",
                    right_label=f"{reference}@{experiment['name']}", role="neural_vs_reference_AE",
                    favourable="negative", claim=f"{model} has lower absolute error than the {reference} reference",
                    seeds=seeds, n_resamples=n_resamples))
    _holm_within_families(rows)
    return rows


def _fmt(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def _signed(value, digits=4):
    return "NA" if value is None else f"{value:+.{digits}f}"


def _interval(low, high, digits=4):
    return "NA" if low is None or high is None else f"[{_fmt(low, digits)}, {_fmt(high, digits)}]"


def _p(value):
    if value is None:
        return "NA"
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.1e}"


def _coefficient(value, low, high, digits=3):
    return "NA" if value is None else f"{_fmt(value, digits)} {_interval(low, high, digits)}"


def _digits(row):
    return 2 if row["measure"] == "absolute_error" else 4


DISCLOSURE_NOTE = ("This package reports the standalone study project on 47 frozen test patients. "
                   "Inference variants are selected on 38 development-validation patients. "
                   "A single split and seed do not establish generalization to an independent cohort.")


def _segmentation_markdown(rows, experiments, variants):
    lines = ["# Segmentation comparison — frozen 47-patient test set", "", DISCLOSURE_NOTE, "",
             "Mean ± sample SD (ddof=1) [95% BCa CI of the mean]; median; n = valid patients (undefined "
             "precision/recall keep their reduced n).", ""]
    for tag, item in variants.items():
        state = ("adopted variant: " + item["adopted"]) if item["adopted"] else "decision not available"
        lines.append(f"- Inference variants {tag}: {state}"
                     + ("" if item["grouped"] is not None else "; test metrics not available") + ".")
    partial = [experiment["name"] for experiment in experiments if not experiment["complete"]]
    if partial:
        lines.append(f"- Partial experiments (one model only): {', '.join(partial)}.")
    lines += ["", "| Experiment | Role | Model | Region | Dice | Precision | Recall |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    if not rows:
        lines.append("| (no segmentation inputs available) | | | | | | |")
    for row in rows:
        cells = [f"{_fmt(row[f'{m}_mean'])} ± {_fmt(row[f'{m}_sd'])} {_interval(row[f'{m}_ci_low'], row[f'{m}_ci_high'])}; "
                 f"median {_fmt(row[f'{m}_median'])} (n={row[f'{m}_n']})" for m in MEASURES]
        lines.append(f"| {row['experiment']} | {row['role']} | {row['model']} | {row['region']} | " + " | ".join(cells) + " |")
    return lines


def _paired_markdown(title, rows, note):
    lines = [f"# {title}", "", DISCLOSURE_NOTE, "", note, "",
             "| Family | Comparison | Experiment | Model | Region | Measure | n | Left − right [95% BCa] | "
             "Wilcoxon p | Holm p | Role |", "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |"]
    if not rows:
        lines.append("| (no paired comparisons available) | | | | | | | | | | |")
    for row in rows:
        digits = _digits(row)
        lines.append(f"| {row['family']} | {row['comparison']} | {row['experiment']} | {row['model']} | "
                     f"{row['region'] or '-'} | {row['measure']} | {row['n']} | "
                     f"{_signed(row['mean_difference'], digits)} {_interval(row['ci_low'], row['ci_high'], digits)} | "
                     f"{_p(row['raw_p'])} | {_p(row['holm_p'])} | {row['analysis_role']} |")
    return lines


def _survival_markdown(rows, degeneracy):
    lines = ["# Survival comparison — frozen 47-patient test set", "", DISCLOSURE_NOTE, "",
             "MAE ± SD(AE) uses the sample SD (ddof=1) of the 47 absolute errors; intervals are 95% BCa over "
             "patients except Pearson (Fisher z). Constant reference predictions have undefined correlations.", "",
             "| Experiment | Model | MAE ± SD(AE) [95% CI] | RMSE [95% CI] | Median AE [95% CI] | Pearson r [95% CI] | "
             "Spearman rho [95% CI] | Mask source |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    if not rows:
        lines.append("| (no survival inputs available) | | | | | | | |")
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['model']} | {_fmt(row['mae'], 2)} ± {_fmt(row['sd_ae'], 2)} "
                     f"{_interval(row['mae_ci_low'], row['mae_ci_high'], 2)} | {_fmt(row['rmse'], 2)} "
                     f"{_interval(row['rmse_ci_low'], row['rmse_ci_high'], 2)} | {_fmt(row['median_ae'], 2)} "
                     f"{_interval(row['median_ae_ci_low'], row['median_ae_ci_high'], 2)} | "
                     f"{_coefficient(row['pearson'], row['pearson_ci_low'], row['pearson_ci_high'])} | "
                     f"{_coefficient(row['spearman'], row['spearman_ci_low'], row['spearman_ci_high'])} | "
                     f"{row['mask_source']} |")
    lines += ["", "## Degeneracy diagnostic", "",
              "A prediction SD near zero means the network outputs almost the same survival for every patient; a "
              "|r(age, prediction)| near 1 means the output is an age proxy. Neither is evidence of image information.",
              "", "| Experiment | Model | Prediction SD (days) | Observed SD (days) | SD ratio | Distinct predictions | "
              "r(age, prediction) | r(age, observed) | MAE |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in degeneracy:
        lines.append(f"| {row['experiment']} | {row['model']} | {_fmt(row['prediction_sd'], 2)} | "
                     f"{_fmt(row['observed_sd'], 2)} | {_fmt(row['prediction_sd_over_observed_sd'], 3)} | "
                     f"{row['distinct_predictions']} | {_fmt(row['prediction_age_r'], 3)} | "
                     f"{_fmt(row['observed_age_r'], 3)} | {_fmt(row['mae'], 2)} |")
    return lines


def _verdict(row):
    """Classify a tested comparison against its claim from the Holm-adjusted Wilcoxon test and the interval."""
    mean, low, high, p = row["mean_difference"], row["ci_low"], row["ci_high"], row["holm_p"]
    digits = _digits(row)
    if mean is None or p is None:
        return "not_assessable", "the paired test or interval is undefined"
    favourable = mean > 0 if row["favourable_sign"] == "positive" else mean < 0
    excludes_zero = low is not None and high is not None and (low > 0 or high < 0)
    unit = " days" if row["measure"] == "absolute_error" else ""
    numbers = (f"mean difference {_signed(mean, digits)}{unit}, 95% BCa CI {_interval(low, high, digits)}, "
               f"Wilcoxon p = {_p(row['raw_p'])}, Holm-adjusted p = {_p(p)}, n = {row['n']}")
    if p < ALPHA and favourable:
        return "supported", numbers + ("" if excludes_zero else "; the interval nevertheless includes zero")
    if p < ALPHA:
        return "not_supported", numbers + "; the significant difference runs against the claimed direction"
    where = ("in the claimed direction" if favourable else "against the claimed direction" if mean != 0 else "zero")
    return "not_supported", (numbers + f"; the point estimate is {where} and not significant after Holm adjustment, "
                             "which is not evidence of equivalence")


def _methods_paragraph(seed, n_resamples):
    return (
        "The study project uses 235 patients: 188 training and 47 test, with 150 development-training and "
        "38 validation patients inside the training cohort. Models are refitted on 188 patients after epoch "
        "selection. The completed segmentation configuration uses Dice plus cross-entropy loss, per-patient "
        "accumulated optimizer updates, flip/intensity augmentation and a cosine learning-rate schedule. "
        "Inference variants are selected on validation data. Survival inputs contain a whole-tumor mask and "
        "four masked MRI channels; source hashes identify the actual frozen inputs of each saved run. "
        f"Patient-level summaries use sample SD (ddof=1) and 95% BCa intervals ({n_resamples:,} resamples, "
        f"base seed {seed}). Two-sided paired Wilcoxon tests use Holm correction within each family: "
        "A, CSFINet versus U-Net for raw and adopted inference; C, adopted versus raw inference for each model; "
        "E, four neural survival models versus age-only OLS and training mean. TC/ET Dice and WT precision/recall "
        "are exploratory interval estimates. Pearson intervals use Fisher z and Spearman intervals use paired "
        "BCa bootstrap. Single-seed patient dispersion is not training variability."
    )


def _manuscript_lines(context):
    seg_rows, seg_tests = context["segmentation_rows"], context["segmentation_tests"]
    surv_rows, surv_tests, degeneracy = context["survival_rows"], context["survival_tests"], context["degeneracy"]
    lines = ["# Statistical analysis report", "",
             "Generated from saved per-patient outputs only. Every number below is recomputed from those rows; "
             "nothing is filled in for experiments whose inputs are missing. Values belong to the frozen 47-patient "
             "held-out test set.", "",
             "## Methods — experiments, disclosure and statistics", "", _methods_paragraph(context["seed"], context["n_resamples"]),
             "", "## Results — segmentation (WT; TC and ET are in segmentation_comparison.md)", ""]
    wt_rows = [row for row in seg_rows if row["region"] == "WT"]
    if not wt_rows:
        lines.append("- Not available: no segmentation inputs were found.")
    for row in wt_rows:
        lines.append(f"- {row['experiment']} ({row['role']}), {row['model']}: WT Dice {_fmt(row['dice_mean'])} ± "
                     f"{_fmt(row['dice_sd'])} {_interval(row['dice_ci_low'], row['dice_ci_high'])}, median "
                     f"{_fmt(row['dice_median'])} (n = {row['dice_n']}); precision {_fmt(row['precision_mean'])} ± "
                     f"{_fmt(row['precision_sd'])} (n = {row['precision_n']}); recall {_fmt(row['recall_mean'])} ± "
                     f"{_fmt(row['recall_sd'])} (n = {row['recall_n']}).")
    lines += ["", "## Results — pre-specified segmentation comparisons (families A and C)", ""]
    primary = [row for row in seg_tests if row["analysis_role"] == PRIMARY_SEGMENTATION]
    if not primary:
        lines.append("- Not available: no complete pair of segmentation experiments was found.")
    for row in primary:
        extras = [f"{item['region']} {item['measure']} {_signed(item['mean_difference'])} "
                  f"{_interval(item['ci_low'], item['ci_high'])} (n = {item['n']})"
                  for item in seg_tests if item["comparison"] == row["comparison"] and item is not row]
        lines.append(f"- Family {row['family']} ({row['family_label']}), {row['left']} − {row['right']}: WT Dice "
                     f"{_signed(row['mean_difference'])} {_interval(row['ci_low'], row['ci_high'])} (n = {row['n']}); "
                     f"two-sided Wilcoxon p = {_p(row['raw_p'])}, Holm-adjusted p = {_p(row['holm_p'])}. "
                     f"Exploratory intervals, no test: {'; '.join(extras)}.")
    lines += ["", "## Results — survival", ""]
    if not surv_rows:
        lines.append("- Not available: no survival prediction inputs were found.")
    for row in surv_rows:
        lines.append(f"- {row['experiment']} ({row['role']}), {row['model']}: MAE {_fmt(row['mae'], 2)} ± "
                     f"{_fmt(row['sd_ae'], 2)} days {_interval(row['mae_ci_low'], row['mae_ci_high'], 2)}; RMSE "
                     f"{_fmt(row['rmse'], 2)} {_interval(row['rmse_ci_low'], row['rmse_ci_high'], 2)}; median AE "
                     f"{_fmt(row['median_ae'], 2)} {_interval(row['median_ae_ci_low'], row['median_ae_ci_high'], 2)}; "
                     f"Pearson r {_coefficient(row['pearson'], row['pearson_ci_low'], row['pearson_ci_high'])}; "
                     f"Spearman rho {_coefficient(row['spearman'], row['spearman_ci_low'], row['spearman_ci_high'])}.")
    lines += ["", "## Results — survival paired comparisons (family E) and degeneracy diagnostic", ""]
    tested = [row for row in surv_tests if row["raw_p"] is not None]
    if not tested:
        lines.append("- Not available: no survival comparison could be formed.")
    for row in tested:
        lines.append(f"- Family {row['family']} ({row['family_label']}), {row['left']} − {row['right']}: absolute "
                     f"error {_signed(row['mean_difference'], 2)} days {_interval(row['ci_low'], row['ci_high'], 2)} "
                     f"(n = {row['n']}); two-sided Wilcoxon p = {_p(row['raw_p'])}, Holm-adjusted p = "
                     f"{_p(row['holm_p'])}; negative values favour the first model.")
    for row in degeneracy:
        lines.append(f"- Degeneracy, {row['experiment']} {row['model']}: prediction SD {_fmt(row['prediction_sd'], 2)} "
                     f"days versus observed survival SD {_fmt(row['observed_sd'], 2)} days (ratio "
                     f"{_fmt(row['prediction_sd_over_observed_sd'], 3)}); {row['distinct_predictions']} distinct "
                     f"predictions; r(age, prediction) = {_fmt(row['prediction_age_r'], 3)}; r(age, observed) = "
                     f"{_fmt(row['observed_age_r'], 3)}.")
    lines += ["", "## Claims the data do / do not support", "",
              "Verdicts use the Holm-adjusted two-sided Wilcoxon p-value at alpha = 0.05 in the claimed direction; "
              "the paired BCa interval is reported alongside. Exploratory rows carry no verdict.", ""]
    verdicts = defaultdict(list)
    for row in [*primary, *tested]:
        status, detail = _verdict(row)
        verdicts[status].append(f"- {row['claim']} — {detail}.")
    for status, title in (("supported", "### Supported by the data"), ("not_supported", "### Not supported by the data"),
                          ("not_assessable", "### Not assessable from the available rows")):
        lines += [title, "", *(verdicts[status] or ["- None."]), ""]
    unavailable = [item for item in context["missing"] if item["effect"] == "experiment_unavailable"]
    lines += ["### Not assessable because inputs are missing", ""]
    lines += [f"- {item['experiment']}: `{item['path']}` ({item['role']})." for item in unavailable] or ["- None."]
    lines += ["", "### Standing caveats", "",
              "- A non-significant difference is not evidence of equivalence; no equivalence margin was pre-specified.",
              "- Single seed and single split: SDs are between-patient dispersion, not training stability; the study "
              "cannot establish the superiority of either architecture from one training run.",
              "- Exploratory rows (TC/ET Dice, WT precision/recall, all-variant descriptives) have intervals only "
              "and must not be reported with p-values.",
              "- Cross-checks: every present Table 5/5b/7 JSON report was verified to summarize exactly the "
              "per-patient rows used here, and raw inference-variant rows were verified to reproduce the plain "
              "test metrics.", "",
              "## Missing inputs", ""]
    lines += [f"- `{item['path']}` — {item['role']} ({item['experiment']}; {item['effect']})."
              for item in context["missing"]] or ["- None; every expected input was present."]
    lines += ["", "## Provenance", "",
              f"Source hashes: report-manifest.json (seed {context['seed']}, {context['n_resamples']:,} resamples)."]
    return lines


def _write_table(directory, name, rows):
    if not rows:
        return None
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"{name}: inconsistent row keys")
    path = directory / name
    write_csv(path, rows, columns)
    return path


def _write_text(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _git_commit(root):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(project_root=".", output="results/delivery/study", seed=20260914, n_resamples=10000):
    """Build the study comparison package from whatever completed inputs exist; refuse to overwrite."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {root}")
    if n_resamples < 2:
        raise ValueError("At least two bootstrap resamples are required")
    final_output = Path(output)
    final_output = (final_output if final_output.is_absolute() else root / final_output).resolve()
    if final_output.exists():
        raise FileExistsError(f"Report output exists; completed report packages are immutable: {final_output}")
    inputs = dict(present=[], missing=[])
    segmentation, variants = _collect_segmentation(root, inputs)
    survival = _collect_survival(root, inputs)
    patients = _check_cohort(segmentation, survival)
    for relative, role in PROVENANCE_FILES:
        _take(root, relative, role, "provenance", inputs, "provenance_hash_skipped")
    seeds = itertools.count(seed)
    segmentation_rows = _segmentation_rows(segmentation, seeds, n_resamples)
    variant_rows = _variant_rows(variants, seeds, n_resamples)
    segmentation_tests = _segmentation_comparisons(segmentation, seeds, n_resamples)
    survival_rows = _survival_rows(survival, seeds, n_resamples)
    degeneracy = _degeneracy_rows(survival)
    survival_tests = _survival_comparisons(survival, seeds, n_resamples)
    context = dict(seed=seed, n_resamples=n_resamples, segmentation_rows=segmentation_rows,
                   segmentation_tests=segmentation_tests, survival_rows=survival_rows, survival_tests=survival_tests,
                   degeneracy=degeneracy, missing=inputs["missing"])
    final_output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f".{final_output.name}.tmp-", dir=final_output.parent))
    try:
        written = [
            _write_table(directory, "segmentation_comparison.csv", segmentation_rows),
            _write_text(directory / "segmentation_comparison.md",
                        _segmentation_markdown(segmentation_rows, segmentation, variants)),
            _write_table(directory, "segmentation_paired_tests.csv", segmentation_tests),
            _write_text(directory / "segmentation_paired_tests.md", _paired_markdown(
                "Segmentation paired comparisons — frozen 47-patient test set", segmentation_tests,
                "Left minus right per patient; positive favours the left side. Holm adjustment is applied within "
                "each family to the WT Dice tests only; TC/ET Dice and WT precision/recall rows are exploratory "
                "intervals without a test.")),
            _write_table(directory, "inference_variants_all.csv", variant_rows),
            _write_table(directory, "survival_comparison.csv", survival_rows),
            _write_text(directory / "survival_comparison.md", _survival_markdown(survival_rows, degeneracy)),
            _write_table(directory, "survival_paired_tests.csv", survival_tests),
            _write_text(directory / "survival_paired_tests.md", _paired_markdown(
                "Survival paired comparisons — frozen 47-patient test set", survival_tests,
                "Absolute error, left minus right per patient; negative favours the left side. "
                "Family E applies Holm correction across eight neural-versus-reference comparisons.")),
            _write_table(directory, "survival_degeneracy.csv", degeneracy),
            _write_text(directory / "statistical-report.md", _manuscript_lines(context)),
        ]
        written = [path for path in written if path is not None]
        missing_results = [item for item in inputs["missing"] if item["effect"] != "provenance_hash_skipped"]
        manifest = dict(
            status="completed" if not missing_results else "completed_with_missing_inputs",
            scope="study_results_comparison_frozen_47_patient_test", generated_at=datetime.now(timezone.utc).isoformat(),
            project_root=str(root), git_commit=_git_commit(root), reporter_sha256=sha256(Path(__file__)),
            seed=seed, n_resamples=n_resamples, alpha=ALPHA,
            confidence_interval="95pct_BCa_pointwise_not_simultaneous", test_patients=len(patients), test_ids=patients,
            experiments=dict(
                segmentation=[dict(name=e["name"], tag=e["tag"], variant=e["variant"], role=e["role"],
                                   models=sorted(e["models"]), complete=e["complete"]) for e in segmentation],
                survival=[dict(name=e["name"], role=e["role"], models=sorted(e["groups"])) for e in survival]),
            adopted_variants={tag: item["adopted"] for tag, item in variants.items()},
            families=FAMILIES, inputs=inputs["present"], missing_inputs=inputs["missing"],
            counts=dict(segmentation_rows=len(segmentation_rows), segmentation_paired_rows=len(segmentation_tests),
                        inference_variant_rows=len(variant_rows), survival_rows=len(survival_rows),
                        survival_paired_rows=len(survival_tests), degeneracy_rows=len(degeneracy)),
            outputs={path.name: sha256(path) for path in written},
            disclosure=DISCLOSURE_NOTE)
        write_json(directory / "report-manifest.json", manifest)
        os.replace(directory, final_output)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m csfinet_repro.analysis_report", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="results/delivery/study",
                        help="Output directory; relative paths are resolved against --project-root")
    parser.add_argument("--seed", type=int, default=20260914, help="Bootstrap base seed; not a training seed")
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args(argv)
    manifest = build_report(args.project_root, args.output, args.seed, args.resamples)
    print(f"{manifest['status']}: {len(manifest['inputs'])} inputs hashed, {len(manifest['missing_inputs'])} missing; "
          f"{len(manifest['outputs'])} files written to {args.output}")
    for item in manifest["missing_inputs"]:
        print(f"missing ({item['effect']}): {item['path']}")
    return manifest


if __name__ == "__main__":
    main()
