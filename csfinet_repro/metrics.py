"""Patient-level metrics with explicit undefined values and bootstrap handling."""

import warnings

import numpy as np
from scipy.stats import bootstrap, norm, pearsonr, spearmanr, wilcoxon


def binary_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or not truth.size:
        raise ValueError("Masks must have the same nonempty shape")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Expected binary masks, not logits or tissue labels")
    truth, prediction = truth.astype(bool), prediction.astype(bool)
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    return dict(tp=tp, fp=fp, fn=fn,
                dice=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
                precision=tp / (tp + fp) if (tp + fp) else None,
                recall=tp / (tp + fn) if (tp + fn) else None,
                iou=tp / (tp + fp + fn) if (tp + fp + fn) else 1.0)


def region_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    for mask in (truth, prediction):
        if not np.isin(mask, [0, 1, 2, 4]).all():
            raise ValueError("Expected original BraTS labels 0, 1, 2, 4; remap model index 3 first")
    return {name: binary_metrics(np.isin(truth, labels), np.isin(prediction, labels))
            for name, labels in {"WT": [1, 2, 4], "TC": [1, 4], "ET": [4]}.items()}


def descriptive(values):
    """Only None denotes an intentional undefined observation; reject NaN/Inf."""
    values = list(values)
    valid = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.isfinite(valid).all() or valid.ndim != 1:
        raise ValueError("Expected finite scalar observations or explicit None")
    result = dict(n_total=len(values), n_valid=len(valid), mean=None, sd=None,
                  median=None, q1=None, q3=None, ddof=1)
    if len(valid):
        result.update(mean=float(valid.mean()), median=float(np.median(valid)),
                      q1=float(np.quantile(valid, .25)), q3=float(np.quantile(valid, .75)))
    if len(valid) > 1:
        result["sd"] = float(valid.std(ddof=1))
    return result


def bca_mean_ci(values, *, seed, n_resamples=10000):
    values = list(values)
    summary = descriptive(values)
    valid = np.asarray([v for v in values if v is not None], dtype=float)
    if n_resamples < 2:
        raise ValueError("At least two resamples required")
    result = dict(method="BCa", confidence_level=.95, n_resamples=n_resamples, seed=seed,
                  n_total=summary["n_total"], n_valid=len(valid), low=None, high=None, status="ok")
    if len(valid) < 2 or np.ptp(valid) == 0:
        result["status"] = "undefined_insufficient_or_constant_data"
        return result
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate = bootstrap((valid,), np.mean, method="BCa", confidence_level=.95,
                             n_resamples=n_resamples, rng=np.random.default_rng(seed), batch=512)
    low, high = estimate.confidence_interval
    result["warnings"] = sorted({str(w.message) for w in caught})
    if np.isfinite([low, high]).all():
        result.update(low=float(low), high=float(high))
    else:
        result["status"] = "undefined_degenerate_bootstrap"
    return result


def bca_statistic_ci(values, statistic, *, seed, n_resamples=10000, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("BCa statistic CI requires at least two finite observations")
    result = dict(method="BCa", statistic=name, confidence_level=.95, n_resamples=n_resamples,
                  seed=seed, low=None, high=None, status="ok")
    if np.ptp(values) == 0:
        result["status"] = "undefined_constant_data"
        return result
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate = bootstrap((values,), statistic, vectorized=False, method="BCa", confidence_level=.95,
                             n_resamples=n_resamples, rng=np.random.default_rng(seed), batch=512)
    low, high = estimate.confidence_interval
    result["warnings"] = sorted({str(w.message) for w in caught})
    if np.isfinite([low, high]).all():
        result.update(low=float(low), high=float(high))
    else:
        result["status"] = "undefined_degenerate_bootstrap"
    return result


def paired_difference(left, right, *, seed, n_resamples=10000):
    """ID-keyed left minus right; never pair rows by their incidental order."""
    if set(left) != set(right) or not left:
        raise ValueError("Paired inputs must contain the same nonempty patient ID set")
    descriptive(left.values())
    descriptive(right.values())
    differences = [None if left[k] is None or right[k] is None else float(left[k]) - float(right[k])
                   for k in sorted(left)]
    return dict(direction="left_minus_right", summary=descriptive(differences),
                mean_ci=bca_mean_ci(differences, seed=seed, n_resamples=n_resamples))


def paired_wilcoxon(left, right):
    """Two-sided paired Wilcoxon after ID alignment and joint missing-value removal."""
    if set(left) != set(right) or not left:
        raise ValueError("Paired inputs must contain the same nonempty patient ID set")
    pairs = [(left[key], right[key]) for key in sorted(left)
             if left[key] is not None and right[key] is not None]
    if not pairs:
        return dict(n=0, statistic=None, p_value=None, status="undefined_no_joint_observations")
    a, b = (np.asarray(values, dtype=float) for values in zip(*pairs))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Wilcoxon inputs must be finite or explicit None")
    if np.all(a == b):
        return dict(n=len(a), statistic=0.0, p_value=1.0, status="all_differences_zero")
    result = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto")
    return dict(n=len(a), statistic=float(result.statistic), p_value=float(result.pvalue), status="ok")


def holm_adjust(p_values):
    """Holm family-wise adjustment keyed by comparison name."""
    if not p_values:
        return {}
    if any(value is None or not np.isfinite(value) or not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("Holm adjustment requires finite p-values in [0,1]")
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted, running, count = {}, 0.0, len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (count - rank) * p_values[key])
        adjusted[key] = min(1.0, float(running))
    return adjusted


def correlation_ci(truth, prediction, *, seed, n_resamples=10000):
    """Pearson Fisher-z and Spearman paired-BCa intervals, per analysis plan."""
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or truth.shape != prediction.shape or len(truth) < 3:
        raise ValueError("Correlation requires matching vectors with at least three patients")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Correlation inputs must be finite")
    output = {}
    for index, (name, function) in enumerate((("pearson", pearsonr), ("spearman", spearmanr))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate = float(function(truth, prediction).statistic)
        method = "Fisher_z" if name == "pearson" else "paired_BCa"
        item = dict(coefficient=estimate if np.isfinite(estimate) else None, low=None, high=None, method=method,
                    confidence_level=.95, n_resamples=None if name == "pearson" else n_resamples,
                    seed=None if name == "pearson" else seed + index, status="ok")
        if not np.isfinite(estimate) or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
            item["status"] = "undefined_constant_input"
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if name == "pearson":
                    interval = pearsonr(truth, prediction).confidence_interval(.95)
                else:
                    interval = bootstrap((truth, prediction), lambda a, b: function(a, b).statistic,
                                         paired=True, vectorized=False, method="BCa", confidence_level=.95,
                                         n_resamples=n_resamples, rng=np.random.default_rng(seed + index), batch=512).confidence_interval
            low, high = interval
            item["warnings"] = sorted({str(w.message) for w in caught})
            if np.isfinite([low, high]).all():
                item.update(low=float(low), high=float(high))
            else:
                item["status"] = "undefined_degenerate_bootstrap"
        output[name] = item
    return output


def steiger_overlapping_correlation(shared, left, right):
    """Steiger (1980) z test for two dependent correlations sharing one variable.

    Compares corr(shared, left) with corr(shared, right) using formula 14 and
    the average-correlation covariance from formula 10. The p-value is
    two-sided. This test assumes complete paired observations.
    """
    shared, left, right = (np.asarray(values, dtype=float) for values in (shared, left, right))
    if shared.ndim != 1 or shared.shape != left.shape or shared.shape != right.shape or len(shared) < 4:
        raise ValueError("Steiger test requires three matching vectors with at least four observations")
    if not all(np.isfinite(values).all() for values in (shared, left, right)):
        raise ValueError("Steiger inputs must be finite")
    if any(np.ptp(values) == 0 for values in (shared, left, right)):
        return dict(n=len(shared), r_shared_left=None, r_shared_right=None, r_left_right=None,
                    z=None, p_value=None, status="undefined_constant_input",
                    method="Steiger_1980_overlapping_dependent_correlations")
    r_jk = float(pearsonr(shared, left).statistic)
    r_jh = float(pearsonr(shared, right).statistic)
    r_kh = float(pearsonr(left, right).statistic)
    matrix = np.asarray([[1, r_jk, r_jh], [r_jk, 1, r_kh], [r_jh, r_kh, 1]])
    eigen_min = float(np.linalg.eigvalsh(matrix).min())
    if max(abs(r_jk), abs(r_jh)) >= 1 - 1e-12 or eigen_min <= 1e-12:
        return dict(n=len(shared), r_shared_left=r_jk, r_shared_right=r_jh, r_left_right=r_kh,
                    z=None, p_value=None, status="undefined_singular_correlation",
                    method="Steiger_1980_overlapping_dependent_correlations")
    mean_r = (r_jk + r_jh) / 2
    denominator = (1 - mean_r ** 2) ** 2
    covariance = (r_kh * (1 - 2 * mean_r ** 2)
                  - .5 * mean_r ** 2 * (1 - 2 * mean_r ** 2 - r_kh ** 2)) / denominator
    variance = 2 - 2 * covariance
    if variance <= 0 or not np.isfinite(variance):
        return dict(n=len(shared), r_shared_left=r_jk, r_shared_right=r_jh, r_left_right=r_kh,
                    z=None, p_value=None, status="undefined_nonpositive_variance",
                    method="Steiger_1980_overlapping_dependent_correlations")
    fisher_left, fisher_right = np.arctanh(np.clip([r_jk, r_jh], -1 + 1e-15, 1 - 1e-15))
    z = float((fisher_left - fisher_right) * np.sqrt(len(shared) - 3) / np.sqrt(variance))
    return dict(n=len(shared), r_shared_left=r_jk, r_shared_right=r_jh, r_left_right=r_kh,
                z=z, p_value=float(2 * norm.sf(abs(z))), status="ok",
                diagnostic_warning=("near_perfect_shared_correlation; assess assumptions; not evidence of predictive value"
                                    if max(abs(r_jk), abs(r_jh)) >= .95 else None),
                method="Steiger_1980_overlapping_dependent_correlations")


def survival_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or truth.shape != prediction.shape or not truth.size:
        raise ValueError("Expected matching nonempty one-dimensional patient predictions")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Survival values and predictions must be finite")
    if np.any(truth <= 0) or np.any(prediction < 0):
        raise ValueError("Observed survival must be positive and predictions nonnegative")
    residual = prediction - truth
    ae = np.abs(residual)
    return dict(n=len(truth), mae=float(ae.mean()), sd_ae=descriptive(ae)["sd"],
                rmse=float(np.sqrt(np.mean(residual ** 2))), median_ae=float(np.median(ae)),
                mean_residual=float(residual.mean()), sd_residual=descriptive(residual)["sd"],
                residual=residual.tolist(), absolute_error=ae.tolist())
