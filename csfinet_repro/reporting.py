"""Generate manuscript figures and deterministic cases from completed artifacts."""

from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import linregress, pearsonr

from .files import read_csv, sha256, write_csv, write_json
from .metrics import steiger_overlapping_correlation

NEURAL_MODELS = ("image", "image_age", "image_resection", "image_age_resection")
ALL_MODELS = (*NEURAL_MODELS, "training_mean", "age_ols")
STATUSES = ("GTR", "STR", "NA")


def absolute_error_bin(value):
    value = float(value)
    if value < 0 or not np.isfinite(value):
        raise ValueError("Absolute error must be finite and nonnegative")
    for upper, label in ((180, "[0,180]"), (360, "(180,360]"),
                         (540, "(360,540]"), (720, "(540,720]")):
        if value <= upper:
            return label
    return "(720,inf)"


def residual_bin(value):
    """Continuous signed-error bins with zero as an explicit category."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Residual must be finite")
    if value < -720:
        return "(-inf,-720)"
    if value < -540:
        return "[-720,-540)"
    if value < -360:
        return "[-540,-360)"
    if value < -180:
        return "[-360,-180)"
    if value < 0:
        return "[-180,0)"
    if value == 0:
        return "{0}"
    if value <= 180:
        return "(0,180]"
    if value <= 360:
        return "(180,360]"
    if value <= 540:
        return "(360,540]"
    if value <= 720:
        return "(540,720]"
    return "(720,inf)"


def select_cases(rows, preferred="BraTS20_Training_199", count=3):
    if len(rows) < count or len({row["patient_id"] for row in rows}) != len(rows):
        raise ValueError("Case selection requires unique patients")
    ordered = sorted(rows, key=lambda row: (float(row["absolute_error"]), row["patient_id"]))
    selected = [next(row for row in rows if row["patient_id"] == preferred)] if any(
        row["patient_id"] == preferred for row in rows) else []
    for quantile in np.linspace(.25, .75, count - len(selected)):
        target = float(np.quantile([float(row["absolute_error"]) for row in ordered], quantile))
        candidate = min((row for row in ordered if row not in selected),
                        key=lambda row: (abs(float(row["absolute_error"]) - target), row["patient_id"]))
        selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def _normalized(image):
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / max(high - low, 1e-8), 0, 1)


def _validate_predictions(rows, test_ids):
    keyed = defaultdict(dict)
    for row in rows:
        patient, model = row["patient_id"], row["model"]
        if patient not in test_ids or model not in ALL_MODELS or patient in keyed[model]:
            raise ValueError("Survival predictions contain an invalid split, model, or duplicate")
        keyed[model][patient] = row
    if set(keyed) != set(ALL_MODELS) or any(set(group) != test_ids for group in keyed.values()):
        raise ValueError("Figures require six models with all 47 frozen test patients")
    for patient in test_ids:
        values = {(keyed[model][patient]["y_true"], keyed[model][patient]["age"],
                   keyed[model][patient]["resection_status"]) for model in ALL_MODELS}
        if len(values) != 1:
            raise ValueError(f"Clinical values differ across models for {patient}")
    return keyed


def _write_error_figures(rows, output, artifacts, tables):
    ae_labels = ("[0,180]", "(180,360]", "(360,540]", "(540,720]", "(720,inf)")
    residual_labels = ("(-inf,-720)", "[-720,-540)", "[-540,-360)", "[-360,-180)",
                       "[-180,0)", "{0}", "(0,180]", "(180,360]", "(360,540]", "(540,720]", "(720,inf)")
    specs = (("absolute_error", absolute_error_bin, ae_labels, "survival_absolute_error_bins"),
             ("residual", residual_bin, residual_labels, "survival_signed_residual_bins"))
    colors = {"GTR": "#0072B2", "STR": "#E69F00", "NA": "#999999"}
    for field, binning, labels, stem in specs:
        counts = defaultdict(int)
        for row in rows:
            if row["model"] in NEURAL_MODELS:
                counts[(row["model"], row["resection_status"], binning(row[field]))] += 1
        bin_rows = [dict(model=model, resection_status=status, **{f"{field}_bin": label},
                         count=counts[(model, status, label)])
                    for model in NEURAL_MODELS for status in STATUSES for label in labels]
        csv_path = output / f"{stem}.csv"
        write_csv(csv_path, bin_rows, list(bin_rows[0]))
        tables.append(csv_path)
        figure, axes = plt.subplots(2, 2, figsize=(14 if field == "residual" else 12, 8),
                                   sharey=True, constrained_layout=True)
        x = np.arange(len(labels))
        for axis, model in zip(axes.flat, NEURAL_MODELS):
            bottom = np.zeros(len(labels))
            for status in STATUSES:
                values = np.asarray([counts[(model, status, label)] for label in labels])
                axis.bar(x, values, bottom=bottom, label=status, color=colors[status])
                bottom += values
            axis.set_title(model.replace("_", " + ").title())
            axis.set_xticks(x, labels, rotation=35, ha="right")
            axis.set_ylabel("Patients")
        axes[0, 0].legend(frameon=False)
        path = output / f"{stem}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        artifacts.append(path)


def _write_age_analysis(rows, output, artifacts, tables):
    groups = {model: sorted((row for row in rows if row["model"] == model),
                            key=lambda row: row["patient_id"]) for model in ALL_MODELS}
    reference = groups[ALL_MODELS[0]]
    age = np.asarray([float(row["age"]) for row in reference])
    observed = np.asarray([float(row["y_true"]) for row in reference])
    series = {"observed": observed,
              **{model: np.asarray([float(row["y_pred"]) for row in groups[model]]) for model in ALL_MODELS}}
    analysis_rows = []
    for name, values in series.items():
        regression = linregress(age, values)
        item = dict(series=name, n=len(age), slope=float(regression.slope), intercept=float(regression.intercept),
                    pearson_r=None, pearson_ci_low=None, pearson_ci_high=None, pearson_p=None,
                    r_squared=None, status="undefined_constant_series" if np.ptp(values) == 0 else "ok")
        if np.ptp(values) != 0:
            correlation = pearsonr(age, values)
            interval = correlation.confidence_interval(.95)
            item.update(pearson_r=float(correlation.statistic), pearson_ci_low=float(interval.low),
                        pearson_ci_high=float(interval.high), pearson_p=float(correlation.pvalue),
                        r_squared=float(correlation.statistic ** 2))
        analysis_rows.append(item)
    analysis_path = output / "age_survival_analysis.csv"
    write_csv(analysis_path, analysis_rows, list(analysis_rows[0]))
    tables.append(analysis_path)
    steiger_rows = []
    for model in NEURAL_MODELS:
        item = steiger_overlapping_correlation(age, observed, series[model])
        steiger_rows.append(dict(comparison=f"age_observed_vs_age_{model}", **item))
    steiger_path = output / "age_survival_steiger.csv"
    write_csv(steiger_path, steiger_rows, list(steiger_rows[0]))
    tables.append(steiger_path)

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True, sharex=True, sharey=True)
    for axis, model in zip(axes.flat, NEURAL_MODELS):
        predicted = series[model]
        axis.scatter(age, observed, s=26, alpha=.7, label="Observed", color="#555555")
        axis.scatter(age, predicted, s=26, alpha=.7, label="Predicted", color="#009E73")
        order = np.argsort(age)
        fit = np.polyfit(age, predicted, 1)
        axis.plot(age[order], np.polyval(fit, age[order]), color="#D55E00", linewidth=1.5)
        axis.set_title(model.replace("_", " + ").title())
        axis.set_xlabel("Age (years)")
        axis.set_ylabel("Survival (days)")
    axes[0, 0].legend(frameon=False)
    path = output / "age_vs_survival.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    artifacts.append(path)


def generate_figures(patient_csv, raw_root, csfinet_predictions, shap_artifacts, survival_predictions,
                     csfinet_development, unet_development, output):
    patients = {row["patient_id"]: row for row in read_csv(
        patient_csv, ["patient_id", "age", "survival_days", "resection_status", "split"])}
    rows = read_csv(survival_predictions, ["patient_id", "model", "y_true", "y_pred", "residual",
                                              "absolute_error", "age", "resection_status"])
    test_ids = {patient for patient, row in patients.items() if row["split"] == "test"}
    if len(test_ids) != 47:
        raise ValueError("Expected the frozen 47-patient test split")
    keyed = _validate_predictions(rows, test_ids)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Figure output exists; use a new report directory")
    output.mkdir(parents=True)
    artifacts, tables = [], []

    _write_error_figures(rows, output, artifacts, tables)
    _write_age_analysis(rows, output, artifacts, tables)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for run_path, label, color in ((csfinet_development, "CSFINet", "#0072B2"),
                                   (unet_development, "U-Net", "#D55E00")):
        run = json.loads(Path(run_path).read_text(encoding="utf-8"))
        if run.get("status") != "completed" or not run.get("history"):
            raise ValueError(f"Completed development run required: {run_path}")
        epochs = [item["epoch"] for item in run["history"]]
        axes[0].plot(epochs, [item["mean_train_CE"] for item in run["history"]], label=label, color=color)
        axes[1].plot(epochs, [item["validation_WT_Dice"] for item in run["history"]], label=label, color=color)
    axes[0].set(title="Training cross-entropy", xlabel="Epoch", ylabel="CE")
    axes[1].set(title="Internal validation WT Dice", xlabel="Epoch", ylabel="Dice")
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=.2)
    path = output / "segmentation_training_curves.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    artifacts.append(path)

    cases = select_cases(list(keyed["image_age_resection"].values()))
    case_rows, case_prediction_rows = [], []
    for case in cases:
        patient = case["patient_id"]
        modalities = [nib.load(Path(raw_root) / patient / f"{patient}_{name}.nii.gz").get_fdata(dtype=np.float32)
                      for name in ("flair", "t1", "t1ce", "t2")]
        truth = nib.load(Path(raw_root) / patient / f"{patient}_seg.nii.gz").get_fdata(dtype=np.float32) > 0
        predicted = nib.load(Path(csfinet_predictions) / patient /
                             f"{patient}_csfinet_prediction.nii.gz").get_fdata(dtype=np.float32) > 0
        z = int(np.argmax(truth.reshape(240 * 240, 155).sum(axis=0)))
        figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for axis, image, name in zip(axes.flat[:4], modalities, ("FLAIR", "T1", "T1ce", "T2")):
            axis.imshow(_normalized(image[:, :, z]), cmap="gray", origin="lower")
            axis.set_title(name)
            axis.set_axis_off()
        axes.flat[4].imshow(_normalized(modalities[0][:, :, z]), cmap="gray", origin="lower")
        if truth[:, :, z].any():
            axes.flat[4].contour(truth[:, :, z], levels=[.5], colors=["#ffb000"], linewidths=1.2)
        if predicted[:, :, z].any():
            axes.flat[4].contour(predicted[:, :, z], levels=[.5], colors=["#00b7c7"], linewidths=1.2)
        axes.flat[4].set_title("GT / CSFINet")
        axes.flat[4].set_axis_off()
        shap = Path(shap_artifacts) / patient / "ground_truth_WT_region.npz"
        with np.load(shap, allow_pickle=False) as archive:
            indices, attribution = archive["slice_z"], archive["attribution"].astype(np.float32)
        position = int(np.argmin(np.abs(indices - z)))
        signed = attribution[position].sum(axis=0)
        scale = max(float(np.abs(signed).max()), 1e-8)
        axes.flat[5].imshow(_normalized(modalities[0][:, :, int(indices[position])]), cmap="gray", origin="lower")
        axes.flat[5].imshow(signed / scale, cmap="coolwarm", vmin=-1, vmax=1, alpha=.58, origin="lower")
        axes.flat[5].set_title(f"Gradient SHAP · z={int(indices[position])}")
        axes.flat[5].set_axis_off()
        figure.suptitle(f"{patient} · observed {float(case['y_true']):.0f} d · predicted {float(case['y_pred']):.1f} d")
        path = output / f"case_{patient}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        artifacts.append(path)
        case_rows.append(dict(patient_id=patient, selection_rule="illustrative_199_if_test_else_error_quantiles",
                              slice_z=z, observed_days=case["y_true"], predicted_days=case["y_pred"],
                              absolute_error=case["absolute_error"]))
        for model in ALL_MODELS:
            row = keyed[model][patient]
            case_prediction_rows.append(dict(patient_id=patient, model=model, observed_days=row["y_true"],
                                             predicted_days=row["y_pred"], residual=row["residual"],
                                             absolute_error=row["absolute_error"]))
    case_path = output / "case_selection.csv"
    case_predictions_path = output / "case_predictions.csv"
    write_csv(case_path, case_rows, list(case_rows[0]))
    write_csv(case_predictions_path, case_prediction_rows, list(case_prediction_rows[0]))
    tables.extend((case_path, case_predictions_path))
    manifest = dict(status="completed", scope="manuscript_figures_from_new_frozen_test_experiment",
                    split_sha256=sha256(patient_csv), survival_predictions_sha256=sha256(survival_predictions),
                    csfinet_development_sha256=sha256(csfinet_development),
                    unet_development_sha256=sha256(unet_development),
                    artifacts=[dict(path=item.name, sha256=sha256(item)) for item in artifacts],
                    tables=[dict(path=item.name, sha256=sha256(item)) for item in tables])
    write_json(output / "manifest.json", manifest)
    return manifest
