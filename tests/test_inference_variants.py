import numpy as np
import pytest
import torch
from torch import nn

from csfinet_repro.inference_variants import (VARIANTS, calibration_ids, decide, largest_component,
                                              predict_labels, recalibrate_batchnorm)


def test_largest_component_keeps_one_26_connected_tumor_and_clears_the_rest():
    labels = np.zeros((6, 20, 20), dtype=np.uint8)
    labels[1:4, 2:9, 2:9] = 2          # large edema blob
    labels[2, 4:6, 4:6] = 4            # enhancing core inside it
    labels[4, 9, 9] = 1                # diagonally 26-connected voxel belongs to the big blob
    labels[5, 17:19, 17:19] = 1        # small distant false positive
    cleaned = largest_component(labels)
    assert cleaned[5].sum() == 0
    assert (cleaned[1:4] == labels[1:4]).all()
    assert cleaned[4, 9, 9] == 1
    assert cleaned.sum() == labels.sum() - labels[5].sum()
    single = labels.copy()
    single[5] = 0
    np.testing.assert_array_equal(largest_component(single), single)
    np.testing.assert_array_equal(largest_component(np.zeros((3, 4, 4), dtype=np.uint8)), 0)
    with pytest.raises(ValueError):
        largest_component(np.array([[[3]]], dtype=np.uint8))


class PositionalSegmenter(nn.Module):
    """Deliberately not flip-equivariant: the class depends on the column index only."""
    def forward(self, x):
        n, _, h, w = x.shape
        column = torch.arange(w).reshape(1, 1, w).expand(n, h, w)
        logits = torch.zeros(n, 4, h, w)
        for label in range(4):
            logits[:, label] = (column % 4 == label).float() * 5
        return logits + 0.01 * x[:, :1]


def _original(indices):
    indices = np.asarray(indices).copy()
    indices[indices == 3] = 4
    return indices


def test_predict_labels_inverts_each_flip_and_averages_the_flip_set():
    model = PositionalSegmenter().eval()
    images = np.random.default_rng(0).normal(size=(3, 4, 8, 8)).astype(np.float32)
    device = torch.device("cpu")
    raw = predict_labels(model, images, 2, device, ((),))
    np.testing.assert_array_equal(raw, _original(model(torch.from_numpy(images)).argmax(dim=1).numpy()))
    flipped_only = predict_labels(model, images, 2, device, ((-1,),))
    direct = torch.flip(model(torch.flip(torch.from_numpy(images), dims=[-1])).argmax(dim=1), dims=[-1]).numpy()
    np.testing.assert_array_equal(flipped_only, _original(direct))
    tta = predict_labels(model, images, 2, device, ((), (-1,), (-2,), (-2, -1)))
    assert tta.shape == raw.shape and np.isin(tta, [0, 1, 2, 4]).all()
    equivariant = nn.Conv2d(4, 4, 1).eval()
    np.testing.assert_array_equal(predict_labels(equivariant, images, 2, device, ((),)),
                                  predict_labels(equivariant, images, 2, device, ((), (-1,), (-2,), (-2, -1))))


def test_batchnorm_recalibration_changes_only_running_statistics():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU(), nn.Dropout(0.5),
                          nn.Conv2d(6, 4, 1))
    weights_before = {name: value.clone() for name, value in model.state_dict().items() if "running" not in name
                      and "num_batches" not in name}
    rng = np.random.default_rng(4)
    arrays = [rng.normal(loc=2.0, scale=3.0, size=(7, 4, 8, 8)).astype(np.float32) for _ in range(3)]
    layers = recalibrate_batchnorm(model, iter(arrays), 4, torch.device("cpu"))
    assert layers == 1
    assert not model.training and model[1].momentum == 0.1
    # momentum=None accumulates the plain average of per-batch means and unbiased variances.
    with torch.no_grad():
        means, variances = [], []
        for array in arrays:
            for start in range(0, len(array), 4):
                activation = model[0](torch.from_numpy(array[start:start + 4]))
                means.append(activation.mean(dim=(0, 2, 3)))
                variances.append(activation.var(dim=(0, 2, 3), unbiased=True))
    torch.testing.assert_close(model[1].running_mean, torch.stack(means).mean(dim=0), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(model[1].running_var, torch.stack(variances).mean(dim=0), rtol=1e-4, atol=1e-4)
    for name, value in model.state_dict().items():
        if name in weights_before:
            torch.testing.assert_close(value, weights_before[name], rtol=0, atol=0)
    with pytest.raises(ValueError):
        recalibrate_batchnorm(nn.Conv2d(4, 4, 1), iter(arrays), 4, torch.device("cpu"))


def test_calibration_subset_is_seeded_and_training_only():
    ids = [f"BraTS20_Training_{index:03d}" for index in range(1, 151)]
    subset = calibration_ids(list(reversed(ids)), 20260914, 30)
    assert subset == calibration_ids(ids, 20260914, 30)
    assert len(subset) == 30 and set(subset) < set(ids) and subset == sorted(subset)
    assert subset != calibration_ids(ids, 1, 30)
    with pytest.raises(ValueError):
        calibration_ids(ids, 1, 151)


def test_decision_rule_requires_per_model_non_inferiority_and_prefers_simpler_ties():
    def scores(**overrides):
        base = {variant: .80 for variant in VARIANTS}
        base.update(overrides)
        return base

    keep_raw = decide({"csfinet": scores(lcc=.79, tta=.81), "unet": scores(lcc=.81, tta=.79)})
    assert keep_raw["adopted"] == "raw"
    lcc = decide({"csfinet": scores(lcc=.81, tta=.81), "unet": scores(lcc=.81, tta=.81)})
    assert lcc["adopted"] == "lcc"
    best = decide({"csfinet": scores(lcc=.81, tta=.82, bn_tta_lcc=.85), "unet": scores(lcc=.81, bn_tta_lcc=.84)})
    assert best["adopted"] == "bn_tta_lcc"
    assert [c["variant"] for c in best["candidates"]] == list(VARIANTS[1:])
