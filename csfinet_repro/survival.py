"""Survival input cache, training, inference, and engineering checks."""

from collections import defaultdict
import json
from pathlib import Path
import random
import subprocess
import time

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from .files import read_csv, sha256, write_csv, write_json
from .metrics import (bca_mean_ci, bca_statistic_ci, correlation_ci, holm_adjust, paired_difference,
                      paired_wilcoxon, survival_metrics)
from .models import SurvivalNet
from .training import seed_everything

VARIANTS = {"image": 0, "image_age": 1, "image_resection": 3, "image_age_resection": 4}


def fit_clinical(training_rows):
    ids = [row["patient_id"] for row in training_rows]
    if not ids or len(ids) != len(set(ids)) or any(row["split"] != "train" for row in training_rows):
        raise ValueError("Clinical fit requires unique training patients; test patients forbidden")
    ages = np.array([float(row["age"]) for row in training_rows])
    if not np.isfinite(ages).all() or (ages <= 0).any() or ages.std(ddof=0) == 0:
        raise ValueError("Age scaling requires finite positive ages with nonzero variation")
    return dict(training_ids=sorted(ids), age_mean=float(ages.mean()), age_sd=float(ages.std(ddof=0)),
                ddof=0, resection_order=["GTR", "STR", "NA"])


def clinical_features(rows, fitted, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown survival variant")
    result = []
    for row in rows:
        age = float(row["age"])
        if not np.isfinite(age) or age <= 0 or row["resection_status"] not in fitted["resection_order"]:
            raise ValueError("Invalid clinical features")
        features = []
        if variant in ("image_age", "image_age_resection"):
            features.append((age - fitted["age_mean"]) / fitted["age_sd"])
        if variant in ("image_resection", "image_age_resection"):
            features.extend(float(row["resection_status"] == status) for status in fitted["resection_order"])
        result.append(features)
    return np.asarray(result, dtype=np.float32).reshape(len(rows), VARIANTS[variant])


def load_wt(path):
    image = nib.load(path)
    labels = image.get_fdata(dtype=np.float32)
    if labels.shape != (240, 240, 155) or not np.isin(labels, [0, 1, 2, 4]).all():
        raise ValueError("Expected native 240x240x155 BraTS tissue labels")
    slices = torch.from_numpy((labels > 0).transpose(2, 0, 1).copy()).float().unsqueeze(1)
    resized = F.interpolate(slices, size=(128, 128), mode="nearest")
    return resized.permute(1, 0, 2, 3).contiguous()


def save_wt_cache(mask, target):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != (1, 155, 128, 128) or not np.isin(mask, [0, 1]).all():
        raise ValueError("Survival cache requires binary 1x155x128x128 WT")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".part.npz")
    np.savez_compressed(temporary, packed=np.packbits(mask.reshape(-1)), shape=np.asarray(mask.shape, dtype=np.int16))
    temporary.replace(target)
    return sha256(target)


def load_wt_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["shape"])
        packed = archive["packed"]
    if shape != (1, 155, 128, 128):
        raise ValueError("Cached WT shape mismatch")
    size = int(np.prod(shape))
    mask = np.unpackbits(packed, count=size).reshape(shape).astype(np.float32)
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Cached WT is not binary")
    return torch.from_numpy(mask)


_MEMORY = {}


def load_survival_cache(path):
    """Load a bit-packed WT cache or a float16 MRI cache; keep a read-only in-memory copy."""
    key = str(Path(path).resolve())
    if key not in _MEMORY:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
        if "volume" in names:
            from .survival_mri import load_mri_cache
            _MEMORY[key] = load_mri_cache(path)
        else:
            _MEMORY[key] = load_wt_cache(path)
    return _MEMORY[key]


def create_survival_cache(patient_csv, raw_root, prediction_root, segmentation_manifest, output):
    rows = read_csv(patient_csv, ["patient_id", "split"])
    if len(rows) != 235 or sum(row["split"] == "test" for row in rows) != 47:
        raise ValueError("Expected frozen 188/47 patient split")
    inference = json.loads(Path(segmentation_manifest).read_text(encoding="utf-8"))
    if (inference.get("status") != "completed" or inference.get("model") != "csfinet"
            or inference.get("split_sha256") != sha256(patient_csv)
            or set(inference.get("test_ids", [])) != {row["patient_id"] for row in rows if row["split"] == "test"}):
        raise ValueError("Completed CSFINet frozen-test inference manifest required")
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed survival cache already exists")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for position, row in enumerate(rows, 1):
        patient = row["patient_id"]
        if row["split"] == "train":
            source = Path(raw_root) / patient / f"{patient}_seg.nii.gz"
            source_kind = "ground_truth_WT"
        else:
            source = Path(prediction_root) / patient / f"{patient}_csfinet_prediction.nii.gz"
            source_kind = "predicted_CSFINet_WT"
        mask = load_wt(source).numpy()
        target = output / f"{patient}.npz"
        if target.exists():
            if not np.array_equal(load_wt_cache(target).numpy(), mask):
                raise ValueError(f"Existing partial cache differs from source: {target}")
            cache_sha = sha256(target)
        else:
            cache_sha = save_wt_cache(mask, target)
        artifacts.append(dict(patient_id=patient, split=row["split"], source_kind=source_kind,
                              source_path=str(source), source_sha256=sha256(source), cache_file=target.name,
                              cache_sha256=cache_sha, foreground_voxels=int(mask.sum())))
        if position % 25 == 0 or position == len(rows):
            print(f"Survival WT cache {position}/235", flush=True)
    manifest = dict(status="completed", patients=235, training_mask="ground_truth_WT",
                    test_mask="predicted_CSFINet_WT", interpolation="nearest",
                    tensor_shape_NCDHW=[1, 1, 155, 128, 128], dtype="uint8_bitpacked_npz",
                    split_sha256=sha256(patient_csv), segmentation_manifest_sha256=sha256(segmentation_manifest),
                    artifacts=artifacts)
    write_json(output / "manifest.json", manifest)
    return manifest


def _survival_checkpoint(model, optimizer, rng, epoch, best_epoch, best_score, history, identity, fitted):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), shuffle_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), epoch=epoch,
                best_epoch=best_epoch, best_score=best_score, history=history, identity=identity, fitted_clinical=fitted)


def _atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    temporary.replace(path)
    return sha256(path)


def _batch_predictions(model, rows, fitted, variant, cache_root, batch_size, device, training):
    model.train(training)
    losses, predictions, truths = [], [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            image = torch.stack([load_survival_cache(Path(cache_root) / f"{row['patient_id']}.npz") for row in group]).to(device)
            clinical = (torch.from_numpy(clinical_features(group, fitted, variant)).to(device)
                        if VARIANTS[variant] else None)
            truth = torch.tensor([float(row["survival_days"]) for row in group], dtype=torch.float32, device=device)
            predicted = model(image, clinical)
            if training:
                yield image, clinical, truth, predicted
            else:
                predictions.extend(float(value) for value in predicted.cpu())
                truths.extend(float(value) for value in truth.cpu())
    if not training:
        yield predictions, truths


def train_survival(config_path, patient_csv, cache_root, output, variant, phase, epochs=None, selection=None, resume=False):
    if variant not in VARIANTS or phase not in {"pilot", "development", "refit"}:
        raise ValueError("Invalid survival variant or phase")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    settings = config["survival"]
    rows = read_csv(patient_csv, ["patient_id", "age", "survival_days", "resection_status", "split", "development_split"])
    manifest_path = Path(cache_root) / "manifest.json"
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("status") != "completed" or cache_manifest.get("split_sha256") != sha256(patient_csv):
        raise ValueError("Completed provenance-matched survival cache required")
    training = [row for row in rows if row["split"] == "train" and (phase == "refit" or row["development_split"] == "train")]
    validation = [] if phase == "refit" else [row for row in rows if row["development_split"] == "validation"]
    if len(training) != (188 if phase == "refit" else 150) or len(validation) != (0 if phase == "refit" else 38):
        raise ValueError("Survival split counts do not match protocol")
    selection_sha = None
    if phase == "pilot":
        training, validation = training[:4], validation[:2]
        budget = 2 if epochs is None else epochs
    elif phase == "development":
        budget = settings["max_epochs"] if epochs is None else epochs
    else:
        if selection is None or epochs is not None:
            raise ValueError("Survival refit requires completed development run.json")
        selected = json.loads(Path(selection).read_text(encoding="utf-8"))
        if (selected.get("status") != "completed" or selected.get("phase") != "development"
                or selected.get("variant") != variant or selected.get("config_sha256") != sha256(config_path)
                or selected.get("split_sha256") != sha256(patient_csv)):
            raise ValueError("Survival selection provenance mismatch")
        budget, selection_sha = selected["best_epoch"], sha256(selection)
    if not isinstance(budget, int) or not 1 <= budget <= settings["max_epochs"]:
        raise ValueError("Invalid survival epoch budget")
    expected_hash = {item["patient_id"]: item["cache_sha256"] for item in cache_manifest["artifacts"]}
    for row in training + validation:
        path = Path(cache_root) / f"{row['patient_id']}.npz"
        if sha256(path) != expected_hash.get(row["patient_id"]):
            raise ValueError(f"Survival cache hash mismatch: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("Survival protocol requires CUDA")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Survival run directory exists")
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(variant=variant, phase=phase, config_sha256=sha256(config_path), split_sha256=sha256(patient_csv),
                    cache_manifest_sha256=sha256(manifest_path), selection_sha256=selection_sha, max_epochs=budget,
                    code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    seed_everything(config["seed"])
    device = torch.device("cuda")
    model = SurvivalNet(VARIANTS[variant], tuple(settings["cnn_channels"]), settings["dropout"],
                        in_channels=settings.get("input_channels", 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"], foreach=False)
    rng = np.random.default_rng(config["seed"])
    fitted = fit_clinical(training)
    first_epoch, best_epoch, best_score, history = 1, 0, float("inf"), []
    run = dict(**identity, status="running", seed=config["seed"], fitted_clinical=fitted,
               git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               training_ids=[row["patient_id"] for row in training], validation_ids=[row["patient_id"] for row in validation],
               mask_policy="GT_WT_for_all_training_and_internal_validation", batch_size=settings["batch_size"],
               **({"input_channels": settings["input_channels"], "input_kind": settings.get("input_kind")}
                  if "input_channels" in settings else {}),
               torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), research_result=phase != "pilot")
    if resume:
        previous = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if previous.get("last_checkpoint_sha256") != sha256(output / "last.pt"):
            raise ValueError("Survival checkpoint hash mismatch")
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != identity or saved["fitted_clinical"] != fitted:
            raise ValueError("Survival resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["shuffle_rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        first_epoch, best_epoch, best_score = saved["epoch"] + 1, saved["best_epoch"], saved["best_score"]
        history, run = saved["history"], previous
    write_json(output / "config.json", config)
    write_json(output / "run.json", run)
    try:
        for epoch in range(first_epoch, budget + 1):
            started, losses = time.perf_counter(), []
            shuffled = [training[index] for index in rng.permutation(len(training))]
            for image, clinical, truth, predicted in _batch_predictions(model, shuffled, fitted, variant, cache_root,
                                                                         settings["batch_size"], device, True):
                optimizer.zero_grad(set_to_none=True)
                loss = F.huber_loss(predicted, truth, delta=settings["huber_delta"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite survival training loss")
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError("Non-finite survival gradients")
                optimizer.step()
                losses.append(float(loss.detach()))
            if validation:
                predicted_values, true_values = next(_batch_predictions(model, validation, fitted, variant, cache_root,
                                                                         settings["batch_size"], device, False))
                score = float(np.mean(np.abs(np.asarray(predicted_values) - np.asarray(true_values))))
            else:
                score = None
            improved = score is not None and score < best_score
            if improved:
                best_score, best_epoch = score, epoch
                run["best_checkpoint_sha256"] = _atomic_torch(output / "best.pt", model.state_dict())
            history.append(dict(epoch=epoch, mean_train_huber=float(np.mean(losses)), validation_MAE=score,
                                elapsed_seconds=time.perf_counter() - started))
            saved = _survival_checkpoint(model, optimizer, rng, epoch, best_epoch, best_score, history, identity, fitted)
            run["last_checkpoint_sha256"] = _atomic_torch(output / "last.pt", saved)
            run.update(last_epoch=epoch, best_epoch=best_epoch, best_validation_MAE=best_score if validation else None, history=history)
            write_json(output / "run.json", run)
            print(f"Survival {variant}/{phase} epoch={epoch} Huber={np.mean(losses):.4f} validation_MAE={score}", flush=True)
            if validation and epoch - best_epoch >= settings["patience"]:
                break
        if phase == "refit":
            run["final_checkpoint_sha256"] = _atomic_torch(output / "final.pt", model.state_dict())
        run["status"] = "completed"
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        write_json(output / "run.json", run)
    return run


def infer_survival(config_path, patient_csv, cache_root, run_dirs, output):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    rows = read_csv(patient_csv, ["patient_id", "age", "survival_days", "resection_status", "split", "development_split"])
    training = [row for row in rows if row["split"] == "train"]
    tests = [row for row in rows if row["split"] == "test"]
    if len(training) != 188 or len(tests) != 47:
        raise ValueError("Expected frozen 188/47 split")
    cache_manifest_path = Path(cache_root) / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if (cache_manifest.get("status") != "completed" or cache_manifest.get("split_sha256") != sha256(patient_csv)
            or any(item["source_kind"] != "predicted_CSFINet_WT" for item in cache_manifest["artifacts"] if item["split"] == "test")):
        raise ValueError("Test cache must contain predicted CSFINet WT for all test patients")
    if set(run_dirs) != set(VARIANTS):
        raise ValueError("Provide one refit run for each of four survival variants")
    if Path(output).exists():
        raise FileExistsError("Survival prediction output already exists")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows, lineage = [], {}
    for variant in VARIANTS:
        run_dir = Path(run_dirs[variant])
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        checkpoint = run_dir / "final.pt"
        if (run.get("status") != "completed" or run.get("phase") != "refit" or run.get("variant") != variant
                or run.get("config_sha256") != sha256(config_path) or run.get("split_sha256") != sha256(patient_csv)
                or run.get("cache_manifest_sha256") != sha256(cache_manifest_path)
                or run.get("final_checkpoint_sha256") != sha256(checkpoint)):
            raise ValueError(f"Refit provenance mismatch: {variant}")
        model = SurvivalNet(VARIANTS[variant], tuple(config["survival"]["cnn_channels"]), config["survival"]["dropout"],
                            in_channels=config["survival"].get("input_channels", 1)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        fitted = run["fitted_clinical"]
        predicted, true = next(_batch_predictions(model, tests, fitted, variant, cache_root,
                                                   config["survival"]["batch_size"], device, False))
        for row, observed, estimate in zip(tests, true, predicted):
            prediction_rows.append(dict(run_id=config["protocol"], patient_id=row["patient_id"], model=variant,
                                        y_true=observed, y_pred=estimate, residual=estimate - observed,
                                        absolute_error=abs(estimate - observed), age=float(row["age"]),
                                        resection_status=row["resection_status"], mask_source="predicted_CSFINet_WT"))
        lineage[variant] = dict(run_sha256=sha256(run_dir / "run.json"), checkpoint_sha256=sha256(checkpoint),
                                fitted_clinical=fitted)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    truth = np.asarray([float(row["survival_days"]) for row in training])
    mean_prediction = float(truth.mean())
    age = np.asarray([float(row["age"]) for row in training])
    design = np.column_stack((np.ones(len(age)), age))
    coefficients = np.linalg.lstsq(design, truth, rcond=None)[0]
    for row in tests:
        observed = float(row["survival_days"])
        references = {"training_mean": mean_prediction,
                      "age_ols": float(coefficients[0] + coefficients[1] * float(row["age"]))}
        for name, estimate in references.items():
            prediction_rows.append(dict(run_id=config["protocol"], patient_id=row["patient_id"], model=name,
                                        y_true=observed, y_pred=estimate, residual=estimate - observed,
                                        absolute_error=abs(estimate - observed), age=float(row["age"]),
                                        resection_status=row["resection_status"], mask_source="reference_no_mask"))
    write_csv(output, prediction_rows, list(prediction_rows[0]))
    manifest = dict(status="completed", scope="frozen_47_patient_test", split_sha256=sha256(patient_csv),
                    cache_manifest_sha256=sha256(cache_manifest_path), predictions_sha256=sha256(output),
                    neural_lineage=lineage, reference_models={"training_mean": mean_prediction,
                    "age_ols_intercept": float(coefficients[0]), "age_ols_slope": float(coefficients[1]),
                    "fit_patient_ids": [row["patient_id"] for row in training]}, test_ids=[row["patient_id"] for row in tests])
    write_json(Path(output).with_suffix(".manifest.json"), manifest)
    return manifest


def summarize_survival_table(input_path, output, seed=20260914, n_resamples=10000):
    rows = read_csv(input_path, ["run_id", "patient_id", "model", "y_true", "y_pred", "residual", "absolute_error"])
    groups, truths = defaultdict(dict), {}
    for row in rows:
        patient, model = row["patient_id"], row["model"]
        if patient in groups[model]:
            raise ValueError("Duplicate survival patient/model row")
        observed, estimate = float(row["y_true"]), float(row["y_pred"])
        if patient in truths and truths[patient] != observed:
            raise ValueError("Survival truth differs across models")
        truths[patient] = observed
        groups[model][patient] = (observed, estimate)
    required = set(VARIANTS) | {"training_mean", "age_ols"}
    if set(groups) != required or len(truths) != 47 or any(set(group) != set(truths) for group in groups.values()):
        raise ValueError("Table 7 requires six models over identical 47 patients")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary, table, index = {}, [], 0
    for model in sorted(groups):
        pairs = [groups[model][patient] for patient in sorted(truths)]
        observed = [pair[0] for pair in pairs]
        predicted = [pair[1] for pair in pairs]
        metrics = survival_metrics(observed, predicted)
        correlations = correlation_ci(observed, predicted, seed=seed + index * 2,
                                      n_resamples=n_resamples)
        mae_ci = bca_mean_ci(metrics["absolute_error"], seed=seed + 100 + index, n_resamples=n_resamples)
        residuals = np.asarray(metrics["residual"])
        rmse_ci = bca_statistic_ci(residuals, lambda x: np.sqrt(np.mean(x ** 2)),
                                   seed=seed + 120 + index, n_resamples=n_resamples, name="RMSE")
        median_ae_ci = bca_statistic_ci(metrics["absolute_error"], np.median,
                                        seed=seed + 140 + index, n_resamples=n_resamples, name="median_AE")
        summary[model] = dict(metrics={key: value for key, value in metrics.items() if key not in {"residual", "absolute_error"}},
                              mae_ci=mae_ci, rmse_ci=rmse_ci, median_ae_ci=median_ae_ci, correlations=correlations)
        table.append(dict(model=model, mae=metrics["mae"], sd_ae=metrics["sd_ae"], rmse=metrics["rmse"],
                          mae_ci_low=mae_ci["low"], mae_ci_high=mae_ci["high"],
                          rmse_ci_low=rmse_ci["low"], rmse_ci_high=rmse_ci["high"],
                          median_ae=metrics["median_ae"], median_ae_ci_low=median_ae_ci["low"],
                          median_ae_ci_high=median_ae_ci["high"], pearson=correlations["pearson"]["coefficient"],
                          pearson_ci_low=correlations["pearson"]["low"], pearson_ci_high=correlations["pearson"]["high"],
                          spearman=correlations["spearman"]["coefficient"], spearman_ci_low=correlations["spearman"]["low"],
                          spearman_ci_high=correlations["spearman"]["high"], n=metrics["n"]))
        index += 1
    baseline = {patient: abs(pair[1] - pair[0]) for patient, pair in groups["image"].items()}
    comparisons, raw_p = {}, {}
    for index, model in enumerate(("image_age", "image_resection", "image_age_resection")):
        candidate = {patient: abs(pair[1] - pair[0]) for patient, pair in groups[model].items()}
        test = paired_wilcoxon(candidate, baseline)
        raw_p[model] = test["p_value"]
        comparisons[model] = dict(direction=f"{model}_AE_minus_image_AE",
                                  difference=paired_difference(candidate, baseline, seed=seed + 200 + index,
                                                               n_resamples=n_resamples), wilcoxon=test)
    adjusted = holm_adjust(raw_p)
    for model, value in adjusted.items():
        comparisons[model]["wilcoxon"]["holm_adjusted_p"] = value
    table_path = output / "table7.csv"
    write_csv(table_path, table, list(table[0]))
    markdown = ["# Table 7 — New frozen 47-patient experiment", "",
                "AE SD is sample SD (ddof=1); correlation intervals are paired 95% BCa over patients.", "",
                "| Model | MAE ± SD(AE) [95% CI of MAE] | RMSE [95% CI] | Median AE [95% CI] | Pearson r [95% CI] | Spearman ρ [95% CI] |",
                "| --- | --- | --- | --- | --- | --- |"]
    latex = [r"\begin{tabular}{lrrrrr}", r"Model & MAE $\pm$ SD(AE) & RMSE & Median AE & Pearson & Spearman \\", r"\hline"]
    def interval(coefficient, low, high):
        return "NA" if coefficient is None or low is None or high is None else f"{coefficient:.3f} [{low:.3f}, {high:.3f}]"
    def days_interval(value, low, high):
        return f"{value:.2f} [NA]" if low is None or high is None else f"{value:.2f} [{low:.2f}, {high:.2f}]"
    for row in table:
        pearson = interval(row["pearson"], row["pearson_ci_low"], row["pearson_ci_high"])
        spearman = interval(row["spearman"], row["spearman_ci_low"], row["spearman_ci_high"])
        mae = days_interval(row["mae"], row["mae_ci_low"], row["mae_ci_high"])
        rmse = days_interval(row["rmse"], row["rmse_ci_low"], row["rmse_ci_high"])
        median = days_interval(row["median_ae"], row["median_ae_ci_low"], row["median_ae_ci_high"])
        markdown.append(f"| {row['model']} | {mae} ±SD {row['sd_ae']:.2f} | {rmse} | {median} | {pearson} | {spearman} |")
        latex.append(f"{row['model']} & {mae} $\\pm$SD {row['sd_ae']:.2f} & {rmse} & {median} & {pearson} & {spearman} " + r"\\")
    latex.append(r"\end{tabular}")
    (output / "table7.md").write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    (output / "table7.tex").write_text("\n".join(latex) + "\n", encoding="utf-8", newline="\n")
    report = dict(status="completed", scope="new_frozen_47_patient_experiment", input_sha256=sha256(input_path), seed=seed,
                  n_resamples=n_resamples, table_sha256=sha256(table_path), models=summary,
                  comparisons_to_image_only=comparisons)
    write_json(output / "table7.json", report)
    return report


def smoke_survival(config_path, patient_csv, root, output):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    rows = [r for r in read_csv(patient_csv, ["patient_id", "age", "survival_days", "resection_status", "split", "development_split"])
            if r["development_split"] == "train"]
    fitted = fit_clinical(rows)
    selected = rows[:2]
    images = torch.stack([load_wt(Path(root) / r["patient_id"] / f"{r['patient_id']}_seg.nii.gz") for r in selected]).cuda()
    truth = torch.tensor([float(r["survival_days"]) for r in selected], device="cuda")
    report = dict(kind="engineering_smoke_not_research_result", patient_ids=[r["patient_id"] for r in selected],
                  image_shape=list(images.shape), mask_source="GT_WT", dtype="float32",
                  config_sha256=sha256(config_path), split_sha256=sha256(patient_csv), fitted_clinical=fitted,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), variants={})
    for variant, count in VARIANTS.items():
        seed_everything(config["seed"])
        settings = config["survival"]
        model = SurvivalNet(count, tuple(settings["cnn_channels"]), settings["dropout"]).cuda()
        clinical = torch.from_numpy(clinical_features(selected, fitted, variant)).cuda() if count else None
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        before = model.head[0].weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started, losses = time.perf_counter(), []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images, clinical)
            loss = F.huber_loss(prediction, truth, delta=settings["huber_delta"])
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(f"Non-finite survival loss/gradient: {variant}")
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        if torch.equal(before, model.head[0].weight.detach()) or (prediction <= 0).any():
            raise RuntimeError("Survival optimizer or Softplus contract failed")
        report["variants"][variant] = dict(status="passed", parameters=sum(p.numel() for p in model.parameters()),
                                            huber_loss=losses, optimizer_steps=2, elapsed_seconds=time.perf_counter() - started,
                                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        print(f"Survival smoke {variant}: passed", flush=True)
        del model, optimizer, clinical, prediction, loss, before
        torch.cuda.empty_cache()
    write_json(output, report)
    return report
