import numpy as np
import pytest
import csv
from pathlib import Path

from csfinet_repro.survival import (clinical_features, fit_clinical, load_wt_cache,
                                    save_wt_cache, summarize_survival_table)


def test_clinical_fit_excludes_test_and_preserves_na_category():
    training = [dict(patient_id="a", age=20, split="train", resection_status="GTR"),
                dict(patient_id="b", age=40, split="train", resection_status="STR")]
    fitted = fit_clinical(training)
    test = dict(patient_id="c", age=100, split="test", resection_status="NA")
    np.testing.assert_array_equal(clinical_features([test], fitted, "image_age_resection"), [[7, 0, 0, 1]])
    assert fitted["age_mean"] == 30 and fitted["age_sd"] == 10
    assert clinical_features(training, fitted, "image").shape == (2, 0)
    with pytest.raises(ValueError, match="test"):
        fit_clinical(training + [test])


def test_bitpacked_survival_cache_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    mask = (rng.random((1, 155, 128, 128)) < .03).astype(np.uint8)
    path = tmp_path / "mask.npz"
    assert len(save_wt_cache(mask, path)) == 64
    np.testing.assert_array_equal(load_wt_cache(path).numpy(), mask)
    assert path.stat().st_size < mask.nbytes / 4


def write_survival_predictions(path):
    columns = ["run_id", "patient_id", "model", "y_true", "y_pred", "residual", "absolute_error"]
    models = ["image", "image_age", "image_resection", "image_age_resection", "training_mean", "age_ols"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for index in range(47):
            truth = 100 + index * 10
            for model_index, model in enumerate(models):
                if model == "training_mean":
                    predicted = 330
                else:
                    predicted = truth + np.sin(index + model_index) * (30 - model_index)
                writer.writerow(dict(run_id="r", patient_id=f"p{index:02}", model=model,
                                     y_true=truth, y_pred=predicted, residual=predicted - truth,
                                     absolute_error=abs(predicted - truth)))


def test_complete_table7_has_six_models_holm_and_constant_reference_na_correlation(tmp_path):
    predictions = tmp_path / "predictions.csv"
    write_survival_predictions(predictions)
    result = summarize_survival_table(predictions, tmp_path / "table", seed=8, n_resamples=200)
    assert set(result["models"]) == {"image", "image_age", "image_resection", "image_age_resection", "training_mean", "age_ols"}
    assert result["models"]["training_mean"]["correlations"]["pearson"]["coefficient"] is None
    assert all(0 <= item["wilcoxon"]["holm_adjusted_p"] <= 1 for item in result["comparisons_to_image_only"].values())
    for name in ("table7.csv", "table7.json", "table7.md", "table7.tex"):
        assert (tmp_path / "table" / name).exists()
