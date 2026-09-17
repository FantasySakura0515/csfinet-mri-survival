"""Synthetic-data tests for the study comparison reporter; no GPU, no real results, deterministic fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest

from csfinet_repro.files import read_csv, sha256, write_csv, write_json
from csfinet_repro.analysis_report import (MEASURES, REGIONS, SEGMENTATION_COLUMNS, SURVIVAL_MODELS, VARIANT_COLUMNS,
                                     build_report, main)

PATIENTS = [f"BraTS20_Training_{index:03d}" for index in range(1, 48)]
REGION_OFFSET = {"WT": 0.0, "TC": -.05, "ET": -.08}
BLANK = (PATIENTS[3], "ET")  # one undefined precision exercises the None path and the reduced valid n
OUTPUTS = ("segmentation_comparison.csv", "segmentation_comparison.md", "segmentation_paired_tests.csv",
           "segmentation_paired_tests.md", "inference_variants_all.csv", "survival_comparison.csv",
           "survival_comparison.md", "survival_paired_tests.csv", "survival_paired_tests.md",
           "survival_degeneracy.csv", "statistical-report.md", "report-manifest.json")


def dice_value(model, tag, index):
    """CSFINet minus U-Net alternates +/-0.02 (null on average); study adds 0.05 with small patient-level jitter."""
    value = .55 + .3 * index / 46
    if model == "unet":
        value -= .02 if index % 2 == 0 else -.02
    if tag == "study":
        value += .05 + .005 * ((index % 5) - 2)
    return value


def metric_rows(model, tag, variant=None, patients=PATIENTS):
    rows = []
    for index, patient in enumerate(patients):
        for region, offset in REGION_OFFSET.items():
            dice = dice_value(model, tag, index) + offset
            if variant == "lcc":
                dice += .01 + .002 * ((index % 3) - 1)
            elif variant == "tta":
                dice -= .01 + .002 * ((index % 3) - 1)
            blank = model == "csfinet" and tag == "study" and (patient, region) == BLANK
            row = dict(run_id=f"{model}-refit-test-{tag}", patient_id=patient, model=model)
            if variant is not None:
                row["variant"] = variant
            row.update(region=region, tp=100, fp=10, fn=5, dice=dice,
                       precision="" if blank else round(min(1.0, dice + .02), 6), recall=max(0.0, dice - .02),
                       iou=dice / (2 - dice))
            rows.append(row)
    return rows


def survival_rows(tag, patients=PATIENTS, truth_shift=0.0):
    """Synthetic neural outputs and two clinical reference models."""
    rows = []
    for index, patient in enumerate(patients):
        truth = 150.0 + 30.0 * index + truth_shift
        age = 40.0 + .5 * index
        predictions = {
            "image": truth + 80 * np.sin(index),
            "image_age": truth + 100 * np.cos(index + 1),
            "image_resection": truth + 400 * np.sin(index),
            "image_age_resection": truth + 150 * np.sin(2 * index) + 5 * (-1) ** index,
            "training_mean": 330.0,
            "age_ols": 1200.0 - 15.0 * age,
        }
        for model, predicted in predictions.items():
            predicted = float(max(0.0, predicted))
            rows.append(dict(run_id=f"survival-refit-test-{tag}", patient_id=patient, model=model, y_true=truth,
                             y_pred=predicted, residual=predicted - truth, absolute_error=abs(predicted - truth),
                             age=age, resection_status="GTR" if index % 2 else "STR",
                             mask_source="reference_no_mask" if model in ("training_mean", "age_ols")
                             else "predicted_CSFINet_WT"))
    return rows


def mean_of(values):
    return float(np.asarray([value for value in values if value not in ("", None)], dtype=float).mean())


def table5_json(rows_by_model):
    keyed = {(row["model"], row["patient_id"], row["region"]): row for rows in rows_by_model.values() for row in rows}
    report = dict(status="completed", direction="csfinet_minus_unet", models={})
    for model in rows_by_model:
        report["models"][model] = {
            region: {measure: dict(summary=dict(mean=mean_of(keyed[(model, patient, region)][measure]
                                                             for patient in PATIENTS))) for measure in MEASURES}
            for region in REGIONS}
    differences = [keyed[("csfinet", patient, "WT")]["dice"] - keyed[("unet", patient, "WT")]["dice"]
                   for patient in PATIENTS]
    report["paired"] = {"WT": {"dice": {"difference": {"summary": {"mean": float(np.mean(differences))}}}}}
    return report


def table7_json(rows):
    return dict(status="completed", models={
        model: dict(metrics=dict(mae=float(np.mean([row["absolute_error"] for row in rows if row["model"] == model]))))
        for model in SURVIVAL_MODELS})


def build_root(root, *, segmentation_tags=("study",), variant_tags=("study",), survival_tags=("survival",),
               adopted="lcc"):
    root = Path(root)
    for tag in segmentation_tags:
        rows_by_model = {}
        for model in ("csfinet", "unet"):
            rows_by_model[model] = metric_rows(model, tag)
            write_csv(root / f"results/segmentation/{model}-test-{tag}/segmentation_patient_metrics.csv",
                      rows_by_model[model], SEGMENTATION_COLUMNS)
        write_json(root / f"results/tables/{tag}/segmentation/table5.json", table5_json(rows_by_model))
    for tag in variant_tags:
        rows, adopted_rows = [], {}
        for model in ("csfinet", "unet"):
            for variant in ("raw", "lcc", "tta"):
                variant_rows = metric_rows(model, tag, variant)
                rows.extend(variant_rows)
                if variant == adopted:
                    adopted_rows[model] = variant_rows
        directory = root / f"results/inference-variants/{tag}"
        write_json(directory / "validation/decision.json", dict(status="completed", tag=tag, adopted=adopted))
        write_csv(directory / "test/test_patient_metrics.csv", rows, VARIANT_COLUMNS)
        write_csv(directory / "test/variant_summary.csv", [dict(model="csfinet", variant="raw")], ["model", "variant"])
        write_json(directory / "test/manifest.json", dict(status="completed", adopted_variant=adopted))
        if adopted != "raw":
            write_json(directory / "test/table5b/table5.json", table5_json(adopted_rows))
    for tag in survival_tags:
        rows = survival_rows(tag)
        write_csv(root / f"results/survival/{tag}/survival_predictions.csv", rows, list(rows[0]))
        write_json(root / f"results/tables/{tag}/survival/table7.json", table7_json(rows))
    for relative in ("configs/segmentation.json", "configs/survival.json",
                     "docs/protocol.md"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("fixture\n", encoding="utf-8")
    return root


def rows_of(path):
    return read_csv(path, [])


def test_full_report_recovers_constructed_directions_and_writes_every_file(tmp_path):
    root = build_root(tmp_path / "project")
    out = tmp_path / "out"
    manifest = build_report(root, out, seed=5, n_resamples=200)
    assert manifest["status"] == "completed" and manifest["missing_inputs"] == []
    assert manifest["test_ids"] == PATIENTS
    assert manifest["adopted_variants"] == {"study": "lcc"}
    assert set(manifest["outputs"]) == set(OUTPUTS) - {"report-manifest.json"}
    for name, digest in manifest["outputs"].items():
        assert sha256(out / name) == digest
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".out.tmp-")]
    table = {(r["experiment"], r["model"], r["region"]): r for r in rows_of(out / "segmentation_comparison.csv")}
    assert {key[0] for key in table} == {"study", "study_lcc"} and len(table) == 12
    assert table[("study", "csfinet", "ET")]["precision_n"] == "46"
    assert table[("study", "csfinet", "ET")]["dice_n"] == "47"
    paired = rows_of(out / "segmentation_paired_tests.csv")
    assert len(paired) == 20
    primary = {r["comparison"]: r for r in paired if r["analysis_role"] == "prespecified_primary_WT_dice"}
    assert {r["family"] for r in primary.values()} == {"A", "C"} and len(primary) == 4
    null = primary["A:study:csfinet_minus_unet"]
    assert float(null["ci_low"]) < 0 < float(null["ci_high"]) and float(null["holm_p"]) > .05
    variant = primary["C:study:csfinet:lcc_minus_raw"]
    assert .008 <= float(variant["mean_difference"]) <= .012 and float(variant["ci_low"]) > 0
    assert float(variant["raw_p"]) <= float(variant["holm_p"]) < .05
    exploratory = [r for r in paired if r["analysis_role"] != "prespecified_primary_WT_dice"]
    assert len(exploratory) == 16 and all(r["raw_p"] == "" and r["holm_p"] == "" for r in exploratory)
    variants = rows_of(out / "inference_variants_all.csv")
    assert len(variants) == 6 and sum(r["adopted"] == "True" for r in variants) == 2
    assert all(r["WT_dice_diff_vs_raw_mean"] == "" for r in variants if r["variant"] == "raw")
    survival = rows_of(out / "survival_comparison.csv")
    assert len(survival) == 6 and {r["experiment"] for r in survival} == {"survival"}
    comparisons = rows_of(out / "survival_paired_tests.csv")
    assert len(comparisons) == 8 and {r["family"] for r in comparisons} == {"E"}
    assert all(float(r["holm_p"]) >= float(r["raw_p"]) for r in comparisons)
    degeneracy = {r["model"]: r for r in rows_of(out / "survival_degeneracy.csv")}
    assert degeneracy["training_mean"]["prediction_age_r"] == ""
    text = (out / "statistical-report.md").read_text(encoding="utf-8")
    assert "not evidence of equivalence" in text
    assert "None; every expected input was present." in text
    assert set(manifest["families"]) == {"A", "C", "E"}
    assert {e["name"] for e in manifest["experiments"]["segmentation"]} == {"study", "study_lcc"}


def test_missing_survival_inputs_are_listed_without_other_experiments(tmp_path):
    root = build_root(tmp_path / "project", survival_tags=())
    manifest = build_report(root, "results/delivery/study", seed=5, n_resamples=200)
    out = root / "results/delivery/study"
    assert manifest["status"] == "completed_with_missing_inputs"
    missing = {item["path"]: item for item in manifest["missing_inputs"]}
    assert missing["results/survival/survival/survival_predictions.csv"]["effect"] == "experiment_unavailable"
    assert missing["results/tables/survival/survival/table7.json"]["effect"] == "cross_check_skipped"
    assert len(missing) == 2
    assert [e["name"] for e in manifest["experiments"]["segmentation"]] == ["study", "study_lcc"]
    assert manifest["experiments"]["survival"] == []
    assert not (out / "survival_paired_tests.csv").exists()
    text = (out / "statistical-report.md").read_text(encoding="utf-8")
    assert "Not available: no survival prediction inputs" in text
    with pytest.raises(FileExistsError):
        build_report(root, "results/delivery/study", seed=5, n_resamples=200)


def test_inconsistent_inputs_raise_before_anything_is_written(tmp_path):
    root = build_root(tmp_path / "segmentation", variant_tags=(), survival_tags=())
    (root / "results/tables/study/segmentation/table5.json").unlink()
    write_csv(root / "results/segmentation/unet-test-study/segmentation_patient_metrics.csv",
              metric_rows("unet", "study", patients=[*PATIENTS[:-1], "BraTS20_Training_999"]), SEGMENTATION_COLUMNS)
    with pytest.raises(ValueError, match="different patients"):
        build_report(root, tmp_path / "out1", seed=5, n_resamples=100)

    root = build_root(tmp_path / "survival_patients")
    rows = survival_rows("survival", patients=[*PATIENTS[:-1], "BraTS20_Training_999"])
    write_csv(root / "results/survival/survival/survival_predictions.csv", rows, list(rows[0]))
    write_json(root / "results/tables/survival/survival/table7.json", table7_json(rows))
    with pytest.raises(ValueError, match="different patients"):
        build_report(root, tmp_path / "out2", seed=5, n_resamples=100)

    root = build_root(tmp_path / "truth")
    rows = survival_rows("survival")
    rows[0]["y_true"] += 1.0
    write_csv(root / "results/survival/survival/survival_predictions.csv", rows, list(rows[0]))
    write_json(root / "results/tables/survival/survival/table7.json", table7_json(rows))
    with pytest.raises(ValueError, match="y_true or age"):
        build_report(root, tmp_path / "out3", seed=5, n_resamples=100)

    root = build_root(tmp_path / "raw")
    path = root / "results/inference-variants/study/test/test_patient_metrics.csv"
    rows = read_csv(path, VARIANT_COLUMNS)
    assert rows[0]["variant"] == "raw"
    rows[0]["dice"] = str(float(rows[0]["dice"]) + .001)
    write_csv(path, rows, VARIANT_COLUMNS)
    with pytest.raises(ValueError, match="raw variant"):
        build_report(root, tmp_path / "out4", seed=5, n_resamples=100)

    root = build_root(tmp_path / "table")
    path = root / "results/tables/study/segmentation/table5.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["models"]["unet"]["WT"]["dice"]["summary"]["mean"] += .01
    write_json(path, report)
    with pytest.raises(ValueError, match="does not match"):
        build_report(root, tmp_path / "out5", seed=5, n_resamples=100)

    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No completed experiment inputs"):
        build_report(tmp_path / "empty", tmp_path / "out6", seed=5, n_resamples=100)
    assert not any((tmp_path / f"out{index}").exists() for index in range(1, 7))


def test_cli_main_is_deterministic_for_a_fixed_seed(tmp_path, capsys):
    root = build_root(tmp_path / "project", segmentation_tags=("study",), variant_tags=(), survival_tags=("survival",))
    first = main(["--project-root", str(root), "--output", "first", "--seed", "11", "--resamples", "150"])
    second = main(["--project-root", str(root), "--output", str(tmp_path / "second"), "--seed", "11",
                   "--resamples", "150"])
    assert (root / "first" / "report-manifest.json").exists() and (tmp_path / "second" / "report-manifest.json").exists()
    assert first["status"] == "completed_with_missing_inputs" and first["outputs"] == second["outputs"]
    assert "inference_variants_all.csv" not in first["outputs"]
    printed = capsys.readouterr().out
    assert "completed_with_missing_inputs" in printed
    assert "missing (experiment_unavailable): results/inference-variants/study/validation/decision.json" in printed
