import copy
import io
import random

import numpy as np
import torch

from csfinet_repro.training import train_patient
from csfinet_repro.segmentation import checkpoint_payload


def test_patient_accumulation_weights_last_15_slices_and_updates_once():
    torch.manual_seed(9)
    model = torch.nn.Conv2d(4, 4, 1)
    reference = copy.deepcopy(model)
    rng = np.random.default_rng(10)
    x = rng.normal(size=(155, 4, 2, 2)).astype(np.float32)
    y = rng.integers(0, 4, size=(155, 2, 2), dtype=np.int64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    steps = []
    optimizer.register_step_post_hook(lambda *args: steps.append(1))
    loss = train_patient(model, optimizer, torch.amp.GradScaler("cpu", enabled=False),
                         x, y, 20, torch.device("cpu"), amp=False)
    reference_loss = torch.nn.functional.cross_entropy(reference(torch.from_numpy(x)), torch.from_numpy(y))
    reference_loss.backward()
    reference_optimizer.step()
    assert len(steps) == 1
    np.testing.assert_allclose(loss, reference_loss.detach().numpy(), rtol=1e-6)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_restores_next_stochastic_adam_update():
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 1), torch.nn.Dropout(0.2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    rng = np.random.default_rng(12)
    x = rng.normal(size=(5, 4, 2, 2)).astype(np.float32)
    y = rng.integers(0, 4, size=(5, 2, 2), dtype=np.int64)
    train_patient(model, optimizer, scaler, x, y, 2, torch.device("cpu"), amp=False)
    buffer = io.BytesIO()
    torch.save(checkpoint_payload(model, optimizer, scaler, rng, 1, 1, 0.4, [], {"test": True}), buffer)
    order = rng.permutation(5)
    expected_loss = train_patient(model, optimizer, scaler, x[order], y[order], 2, torch.device("cpu"), amp=False)
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
    restored_order = rng.permutation(5)
    assert np.array_equal(order, restored_order)
    actual_loss = train_patient(model, optimizer, scaler, x[restored_order], y[restored_order], 2, torch.device("cpu"), amp=False)
    assert actual_loss == expected_loss
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_validation_pending_checkpoint_preserves_completed_training_losses():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    payload = checkpoint_payload(model, optimizer, scaler, np.random.default_rng(3), 4, 2, .7,
                                 [{"epoch": 3}], {"run": "x"},
                                 stage="validation_pending", pending_losses=[.9, .8])
    assert payload["stage"] == "validation_pending"
    assert payload["epoch"] == 4
    assert payload["pending_losses"] == [.9, .8]
    assert payload["history"] == [{"epoch": 3}]
