import numpy as np
import pytest

from csfinet_repro.metrics import (binary_metrics, region_metrics, descriptive, bca_mean_ci,
                                    paired_difference, paired_wilcoxon, holm_adjust, correlation_ci,
                                    steiger_overlapping_correlation, survival_metrics)


def test_known_confusion_counts_and_patient_aggregation():
    first = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (first["tp"], first["fp"], first["fn"]) == (1, 1, 1)
    assert first["dice"] == first["precision"] == first["recall"] == .5
    assert first["iou"] == pytest.approx(1 / 3)
    second = binary_metrics([1], [1])
    assert descriptive([first["dice"], second["dice"]])["mean"] == .75
    pooled = binary_metrics([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert pooled["dice"] != .75


@pytest.mark.parametrize("truth,prediction,dice,precision,recall", [
    ([0], [0], 1, None, None), ([1], [0], 0, None, 0), ([0], [1], 0, 0, None)
])
def test_empty_region_conventions(truth, prediction, dice, precision, recall):
    result = binary_metrics(truth, prediction)
    assert (result["dice"], result["precision"], result["recall"]) == (dice, precision, recall)


def test_brats_regions_and_invalid_model_indices():
    metrics = region_metrics([0, 1, 2, 4], [0, 2, 2, 1])
    assert metrics["WT"]["dice"] == 1
    assert metrics["TC"]["dice"] == pytest.approx(2 / 3)
    assert metrics["ET"]["dice"] == 0
    with pytest.raises(ValueError):
        region_metrics([0, 3], [0, 1])


def test_sample_sd_and_explicit_missingness():
    result = descriptive([1, None, 3])
    assert (result["n_total"], result["n_valid"], result["mean"]) == (3, 2, 2)
    assert result["sd"] == pytest.approx(np.sqrt(2))
    assert descriptive([1])["sd"] is None
    assert descriptive([None])["mean"] is None
    with pytest.raises(ValueError):
        descriptive([1, np.nan])


def test_bootstrap_reproducibility_and_degeneracy():
    a = bca_mean_ci([.2, .4, .5, .8, .9], seed=17, n_resamples=500)
    assert a == bca_mean_ci([.2, .4, .5, .8, .9], seed=17, n_resamples=500)
    assert 0 <= a["low"] < a["high"] <= 1
    assert bca_mean_ci([1, 1, 1], seed=17)["low"] is None


def test_pairing_is_by_id_and_uses_joint_valid_patients():
    result = paired_difference({"b": 5, "a": 2, "c": None}, {"a": 1, "b": 3, "c": 4},
                               seed=17, n_resamples=500)
    assert result["summary"]["mean"] == 1.5
    assert result["summary"]["n_valid"] == 2
    with pytest.raises(ValueError):
        paired_difference({"a": 1}, {"b": 2}, seed=17)


def test_survival_ae_and_residual_sd_are_different():
    result = survival_metrics([10, 20, 30], [8, 22, 34])
    assert result["residual"] == [-2, 2, 4]
    assert result["mae"] == pytest.approx(8 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(8))
    assert result["sd_ae"] == pytest.approx(np.sqrt(4 / 3))
    assert result["sd_residual"] != result["sd_ae"]
    assert result["sd_ae"] ** 2 == pytest.approx(3 / 2 * (result["rmse"] ** 2 - result["mae"] ** 2))
    with pytest.raises(ValueError):
        survival_metrics([10], [float("inf")])


def test_wilcoxon_alignment_zero_case_and_holm_monotonicity():
    paired = paired_wilcoxon({"b": 3, "a": 1}, {"a": 0, "b": 2})
    assert paired["n"] == 2 and 0 <= paired["p_value"] <= 1
    assert paired_wilcoxon({"a": 1}, {"a": 1})["p_value"] == 1
    adjusted = holm_adjust({"a": .01, "b": .03, "c": .04})
    assert adjusted == {"a": pytest.approx(.03), "b": pytest.approx(.06), "c": pytest.approx(.06)}
    with pytest.raises(ValueError):
        holm_adjust({"bad": 2})


def test_correlation_coefficients_and_reproducible_paired_ci():
    truth = np.arange(12, dtype=float)
    prediction = truth + np.sin(truth)
    a = correlation_ci(truth, prediction, seed=4, n_resamples=300)
    b = correlation_ci(truth, prediction, seed=4, n_resamples=300)
    assert a == b
    assert .8 < a["pearson"]["coefficient"] <= 1
    assert a["pearson"]["low"] < a["pearson"]["high"]


def test_steiger_swap_symmetry_and_constant_input():
    shared = np.arange(20, dtype=float)
    left = shared + np.sin(shared)
    right = -shared + np.cos(shared)
    result = steiger_overlapping_correlation(shared, left, right)
    assert result["status"] == "ok"
    assert result["z"] > 0
    assert 0 <= result["p_value"] <= 1
    swapped = steiger_overlapping_correlation(shared, right, left)
    assert swapped["z"] == pytest.approx(-result["z"])
    assert swapped["p_value"] == pytest.approx(result["p_value"])
    assert steiger_overlapping_correlation(shared, np.ones(20), right)["status"] == "undefined_constant_input"


def test_steiger_exact_correlation_is_undefined_but_near_perfect_is_flagged():
    shared = np.arange(30, dtype=float)
    left = np.sin(shared) + shared / 5
    assert steiger_overlapping_correlation(shared, left, -shared)["status"] == "undefined_singular_correlation"
    result = steiger_overlapping_correlation(shared, left, -shared + np.cos(shared))
    assert result["status"] == "ok"
    assert np.isfinite(result["z"])
    assert "near_perfect" in result["diagnostic_warning"]


def test_steiger_identical_outcomes_are_not_clipped_into_finite_significance():
    x = np.arange(30, dtype=float)
    y = x + np.sin(x)
    result = steiger_overlapping_correlation(x, y, y)
    assert result["status"] == "undefined_singular_correlation"
    assert result["p_value"] is None
