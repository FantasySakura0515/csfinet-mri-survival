import pytest
import torch
import copy
import io

from csfinet_repro.models import CSFINet, PIU, SurvivalNet, UNet, enable_activation_checkpoint, partition, restore, warp


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_warp_identity_and_known_horizontal_translation():
    image = torch.arange(30, dtype=torch.float32).reshape(1, 1, 5, 6)
    flow = torch.zeros(1, 2, 5, 6)
    torch.testing.assert_close(warp(image, flow), image)
    flow[:, 0] = 1
    shifted = warp(image, flow)
    torch.testing.assert_close(shifted[..., :-1], image[..., 1:])
    torch.testing.assert_close(shifted[..., -1], torch.zeros_like(image[..., -1]), atol=1e-5, rtol=0)


def test_piu_partition_roundtrip_rectangular_spatial_positions():
    x = torch.arange(2 * 12 * 20 * 3).reshape(2, 12, 20, 3)
    assert torch.equal(restore(partition(x, 4), 4), x)
    # First group must contain adjacent pixels, not every fourth pixel.
    assert torch.equal(partition(x, 4)[0, 0, 0], x[0, :4, :4].reshape(16, 3))


def test_piu_connects_all_three_scales():
    unit = PIU((8, 16, 32), dim=16, heads=4, dropout=0)
    inputs = [torch.randn(1, c, size, size, requires_grad=True) for c, size in [(8, 12), (16, 6), (32, 3)]]
    outputs = unit(inputs)
    assert [x.shape for x in outputs] == [x.shape for x in inputs]
    outputs[0].sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in inputs)


def test_checkpoint_preserves_gradients_dropout_and_batchnorm_state():
    torch.manual_seed(31)
    model = CSFINet((4, 8, 16, 32, 64), (2, 2, 2), (2, 4, 8), piu_dim=16, piu_heads=4)
    checked = enable_activation_checkpoint(copy.deepcopy(model))
    x = torch.randn(2, 4, 32, 48)
    torch.manual_seed(42)
    result = model(x)
    result.square().mean().backward()
    torch.manual_seed(42)
    actual = checked(x)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, result)
    for (name, expected), (name2, observed) in zip(model.named_parameters(), checked.named_parameters()):
        assert name == name2
        torch.testing.assert_close(observed.grad, expected.grad)
    for name, expected in model.named_buffers():
        torch.testing.assert_close(dict(checked.named_buffers())[name], expected)


def test_saved_weights_restore_identical_eval_prediction():
    model = CSFINet((4, 8, 16, 32, 64), (2, 2, 2), (2, 4, 8), piu_dim=16, piu_heads=4).eval()
    source = torch.randn(1, 4, 32, 48)
    with torch.no_grad():
        expected = model(source)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = copy.deepcopy(model)
    for parameter in restored.parameters():
        parameter.data.zero_()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(source), expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["csfinet", "unet"])
def test_four_modality_240_logits_and_backward(name):
    # Reduced width tests topology; the separate GPU smoke checks full paper configuration.
    model = (CSFINet((4, 8, 16, 32, 64), (2, 2, 6), (2, 4, 8), piu_dim=16, piu_heads=4)
             if name == "csfinet" else UNet((4, 8, 16, 32, 64)))
    x = torch.randn(1, 4, 240, 240)
    logits = model(x)
    assert logits.shape == (1, 4, 240, 240)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.randint(0, 4, (1, 240, 240))).backward()
    assert model.encoder.blocks[0][0].weight.grad.abs().sum() > 0
    assert model.head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("features", [0, 1, 3, 4])
def test_survival_ablation_has_positive_scalar_and_image_gradient(features):
    model = SurvivalNet(features, channels=(2, 4, 8))
    image = torch.rand(2, 1, 16, 16, 16, requires_grad=True)
    clinical = torch.rand(2, features) if features else None
    predicted = model(image, clinical)
    assert predicted.shape == (2,) and (predicted > 0).all()
    torch.nn.functional.huber_loss(predicted, torch.tensor([100., 700.]), delta=365).backward()
    assert image.grad.abs().sum() > 0
