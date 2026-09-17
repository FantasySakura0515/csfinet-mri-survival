"""Training options, checkpoint resume, and the standalone study configuration."""

import copy
import io
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from csfinet_repro import training
from csfinet_repro.segmentation import checkpoint_payload
from csfinet_repro.training import (LOSSES, UPDATE_POLICIES, augment_slices, dice_ce_loss, epoch_learning_rate,
                                    train_patient)

ROOT = Path(__file__).resolve().parents[1]
AUGMENTATION = {"flip_axes": [-2, -1], "intensity_scale": [0.9, 1.1], "intensity_shift": [-0.1, 0.1]}
CPU = torch.device("cpu")


def patient(seed, slices=155, size=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(slices, 4, size, size)).astype(np.float32)
    y = rng.integers(0, 4, size=(slices, size, size), dtype=np.int64)
    return x, y


def disabled_scaler():
    return torch.amp.GradScaler("cpu", enabled=False)


def test_dice_ce_loss_is_near_zero_for_perfect_one_hot_prediction_and_absent_classes():
    targets = torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]])
    logits = 60.0 * F.one_hot(targets, 4).permute(0, 3, 1, 2).float()
    assert dice_ce_loss(logits, targets).item() < 1e-6
    targets = torch.tensor([[[0, 1], [1, 0]]])  # classes 2 and 3 absent from prediction and target
    logits = 60.0 * F.one_hot(targets, 4).permute(0, 3, 1, 2).float()
    assert dice_ce_loss(logits, targets).item() < 1e-6


def test_dice_ce_loss_matches_hand_computed_uniform_prediction():
    targets = torch.ones(2, 3, 5, dtype=torch.long)  # 30 pixels, all class 1
    logits = torch.zeros(2, 4, 3, 5)  # softmax 0.25 everywhere
    pixels = 30
    present = 1 - (2 * 0.25 * pixels + 1) / (0.25 * pixels + pixels + 1)
    absent = 1 - 1 / (0.25 * pixels + 1)
    expected = np.log(4) + (present + 2 * absent) / 3
    assert dice_ce_loss(logits, targets).item() == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError):
        dice_ce_loss(logits, targets, smooth=0)
    with pytest.raises(ValueError):
        dice_ce_loss(logits[:, :1], targets)


def test_dice_ce_loss_stays_float32_under_autocast():
    model = torch.nn.Conv2d(4, 4, 1)
    x, y = patient(1, slices=3, size=4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = dice_ce_loss(model(torch.from_numpy(x)), torch.from_numpy(y))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.weight.grad).all()


def test_per_slice_batch_policy_updates_once_per_batch_and_returns_weighted_mean():
    torch.manual_seed(9)
    model = torch.nn.Conv2d(4, 4, 1)
    reference = copy.deepcopy(model)
    x, y = patient(10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    steps = []
    optimizer.register_step_post_hook(lambda *args: steps.append(1))
    stats = {}
    loss = train_patient(model, optimizer, disabled_scaler(), x, y, 20, CPU, amp=False,
                         update_policy="per_slice_batch", stats=stats)
    expected = 0.0
    for start in range(0, 155, 20):  # eight sequential SGD steps; the last batch holds 15 slices
        reference_optimizer.zero_grad()
        batch = F.cross_entropy(reference(torch.from_numpy(x[start:start + 20])), torch.from_numpy(y[start:start + 20]))
        batch.backward()
        reference_optimizer.step()
        expected += batch.item() * len(x[start:start + 20]) / 155
    assert len(steps) == 8
    assert stats == {"optimizer_steps": 8, "skipped_updates": 0}
    assert loss == pytest.approx(expected, rel=1e-6)
    for actual, wanted in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, wanted)


def test_per_slice_batch_skips_non_finite_gradient_batches_and_raises_on_non_finite_loss(monkeypatch):
    x, y = patient(5, slices=10)
    model = torch.nn.Conv2d(4, 4, 1)
    before = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(training, "_gradients_finite", lambda model: False)
    stats = {}
    loss = train_patient(model, torch.optim.SGD(model.parameters(), lr=0.1), disabled_scaler(), x, y, 5, CPU,
                         amp=False, loss_name="dice_ce", update_policy="per_slice_batch", stats=stats)
    assert np.isfinite(loss) and stats == {"optimizer_steps": 0, "skipped_updates": 2}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    monkeypatch.undo()
    x[0, 0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="dice_ce"):
        train_patient(model, torch.optim.SGD(model.parameters(), lr=0.1), disabled_scaler(), x, y, 5, CPU,
                      amp=False, loss_name="dice_ce", update_policy="per_slice_batch")
    for bad in (dict(loss_name="dice"), dict(update_policy="per_epoch"), dict(augmentation=AUGMENTATION)):
        with pytest.raises(ValueError):
            train_patient(model, torch.optim.SGD(model.parameters(), lr=0.1), disabled_scaler(), x, y, 5, CPU,
                          amp=False, **bad)


def test_keyword_defaults_reproduce_the_positional_call_bitwise():
    x, y = patient(3, slices=45)
    torch.manual_seed(4)
    model = torch.nn.Conv2d(4, 4, 1)
    twin = copy.deepcopy(model)
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    a = train_patient(model, torch.optim.Adam(model.parameters(), lr=0.01), disabled_scaler(), x, y, 20, CPU, amp=False)
    stats = {}
    b = train_patient(twin, torch.optim.Adam(twin.parameters(), lr=0.01), disabled_scaler(), x, y, 20, CPU, amp=False,
                      loss_name="four_class_cross_entropy", update_policy="per_patient", augmentation=None,
                      rng=rng, stats=stats)
    assert a == b
    assert stats == {"optimizer_steps": 1, "skipped_updates": 0}
    assert rng.bit_generator.state == state
    for p, q in zip(model.parameters(), twin.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_augment_slices_flips_labels_with_images_and_rescales_images_only():
    images = np.zeros((3, 4, 6, 8), dtype=np.float32)
    targets = np.zeros((3, 6, 8), dtype=np.int64)
    images[1, 2, 1, 2], targets[1, 1, 2] = 5.0, 3  # one marked pixel
    settings = {"flip_axes": [-2, -1], "intensity_scale": [2.0, 2.0], "intensity_shift": [1.0, 1.0]}
    positions = set()
    for seed in range(8):
        out_images, out_targets = augment_slices(images, targets, np.random.default_rng(seed), settings)
        assert out_images.shape == images.shape and out_targets.shape == targets.shape
        assert out_images.flags["C_CONTIGUOUS"] and out_targets.flags["C_CONTIGUOUS"]
        assert out_images.dtype == np.float32 and out_targets.dtype == np.int64
        image_position = np.unravel_index(np.argmax(out_images[1, 2]), out_images.shape[2:])
        label_position = np.unravel_index(np.argmax(out_targets[1]), out_targets.shape[1:])
        assert image_position == label_position
        assert out_images[1, 2][image_position] == pytest.approx(11.0)  # 5 * 2 + 1
        assert out_images[0].min() == out_images[0].max() == pytest.approx(1.0)  # 0 * 2 + 1
        assert set(np.unique(out_targets)) == {0, 3}
        torch.as_tensor(out_images)  # flipped views are made contiguous for torch
        positions.add(image_position)
    assert len(positions) > 1  # some seeds flipped
    assert images[1, 2, 1, 2] == 5.0 and targets[1, 1, 2] == 3  # inputs untouched
    with pytest.raises(ValueError, match="flip_axes"):
        augment_slices(images, targets, np.random.default_rng(0), {"flip_axes": [1]})
    with pytest.raises(ValueError, match="intensity_scale"):
        augment_slices(images, targets, np.random.default_rng(0), {"intensity_scale": [1.1, 0.9]})


def test_augment_slices_is_a_no_op_without_settings_and_consumes_four_draws_when_enabled():
    images, targets = patient(2, slices=4)
    for settings in (None, False, {}):
        rng = np.random.default_rng(7)
        state = rng.bit_generator.state
        out_images, out_targets = augment_slices(images, targets, rng, settings)
        assert out_images is images and out_targets is targets
        assert rng.bit_generator.state == state
    rng = np.random.default_rng(7)
    augment_slices(images, targets, rng, AUGMENTATION)
    expected = np.random.default_rng(7)
    expected.random(4)  # two flip Bernoullis, one scale, one shift
    assert rng.bit_generator.state == expected.bit_generator.state
    with pytest.raises(ValueError, match="Generator"):
        augment_slices(images, targets, None, AUGMENTATION)


def test_augmentation_draws_replay_from_restored_generator_state():
    images, targets = patient(21, slices=20)
    rng = np.random.default_rng(5)
    rng.permutation(150)  # the epoch permutation precedes the batch draws
    state = rng.bit_generator.state
    first = [augment_slices(images, targets, rng, AUGMENTATION) for _ in range(3)]
    rng.bit_generator.state = state
    second = [augment_slices(images, targets, rng, AUGMENTATION) for _ in range(3)]
    for (a, b), (c, d) in zip(first, second):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
    assert any(not np.array_equal(a, images) for a, _ in first)


def test_epoch_learning_rate_warmup_and_cosine_endpoints():
    assert epoch_learning_rate(4e-4, 7, 100, None) == 4e-4
    schedule = {"kind": "cosine", "warmup_epochs": 2, "final_fraction": 0.1}
    assert epoch_learning_rate(1.0, 1, 11, schedule) == pytest.approx(0.5)  # linear warmup
    assert epoch_learning_rate(1.0, 2, 11, schedule) == pytest.approx(1.0)  # end of warmup
    assert epoch_learning_rate(1.0, 3, 11, schedule) == pytest.approx(1.0)  # cosine starts at base
    assert epoch_learning_rate(1.0, 7, 11, schedule) == pytest.approx(0.55)  # midpoint
    assert epoch_learning_rate(1.0, 11, 11, schedule) == pytest.approx(0.1)  # final fraction at max_epochs
    rates = [epoch_learning_rate(1.0, epoch, 11, schedule) for epoch in range(3, 12)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    study = {"kind": "cosine", "warmup_epochs": 1, "final_fraction": 0.05}
    assert epoch_learning_rate(4e-4, 1, 30, study) == pytest.approx(4e-4)
    assert epoch_learning_rate(4e-4, 2, 30, study) == pytest.approx(4e-4)
    assert epoch_learning_rate(4e-4, 30, 30, study) == pytest.approx(2e-5)
    for bad in ({"kind": "linear"}, {"kind": "cosine", "warmup_epochs": 30}, {"kind": "cosine", "final_fraction": 2}):
        with pytest.raises(ValueError):
            epoch_learning_rate(4e-4, 1, 30, bad)
    with pytest.raises(ValueError):
        epoch_learning_rate(4e-4, 0, 30, study)


def test_restoring_checkpointed_rng_replays_an_augmented_per_slice_batch_epoch():
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 1), torch.nn.Dropout(0.2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = disabled_scaler()
    rng = np.random.default_rng(12)
    x, y = patient(13, slices=45)
    options = dict(amp=False, loss_name="dice_ce", update_policy="per_slice_batch", augmentation=AUGMENTATION, rng=rng)
    train_patient(model, optimizer, scaler, x, y, 20, CPU, **options)
    buffer = io.BytesIO()
    torch.save(checkpoint_payload(model, optimizer, scaler, rng, 1, 1, 0.4, [], {"test": True}), buffer)
    order = rng.permutation(len(x))
    stats = {}
    expected_loss = train_patient(model, optimizer, scaler, x[order], y[order], 20, CPU, stats=stats, **options)
    expected = copy.deepcopy(model.state_dict())
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    rng.bit_generator.state = saved["shuffle_rng"]
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    replayed_order = rng.permutation(len(x))
    assert np.array_equal(order, replayed_order)
    replay_stats = {}
    actual_loss = train_patient(model, optimizer, scaler, x[replayed_order], y[replayed_order], 20, CPU,
                                stats=replay_stats, **options)
    assert actual_loss == expected_loss
    assert stats == replay_stats == {"optimizer_steps": 3, "skipped_updates": 0}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_checkpoint_payload_adds_pending_stats_only_when_given():
    model = torch.nn.Linear(2, 1)
    arguments = (model, torch.optim.Adam(model.parameters()), disabled_scaler(), np.random.default_rng(3), 4, 2, .7,
                 [], {"run": "x"})
    assert "pending_stats" not in checkpoint_payload(*arguments, stage="validation_pending", pending_losses=[.9])
    payload = checkpoint_payload(*arguments, stage="validation_pending", pending_losses=[.9],
                                 pending_stats={"optimizer_steps": 8, "skipped_updates": 0})
    assert payload["pending_stats"] == {"optimizer_steps": 8, "skipped_updates": 0}


def test_completed_study_config_matches_the_frozen_cohort_and_declared_training_options():
    config = json.loads((ROOT / "configs/segmentation.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "data/splits/identity.json").read_text(encoding="utf-8"))
    assert config["seed"] == frozen["seed"]
    assert config["split"] == {"test_patients": frozen["test"],
                               "validation_patients": frozen["validation"],
                               "validation_seed": frozen["validation_seed"]}
    survival = json.loads((ROOT / "configs/survival.json").read_text(encoding="utf-8"))
    for key in ("nifti", "normalization"):
        assert config[key] == survival[key]
    settings = config["segmentation"]
    assert settings["loss"] == "dice_ce"
    assert settings["update_policy"] == "per_patient"
    assert settings["augmentation"] == AUGMENTATION
    assert settings["max_epochs"] == 30 and settings["patience"] == 8
    assert settings["lr_schedule"] == {"kind": "cosine", "warmup_epochs": 1, "final_fraction": 0.05}
    epoch_learning_rate(settings["learning_rate"], 30, 30, settings["lr_schedule"])
