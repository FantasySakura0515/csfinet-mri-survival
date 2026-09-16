import numpy as np
import pytest
import torch

from csfinet_repro.models import SurvivalNet
from csfinet_repro.survival import load_survival_cache, save_wt_cache
from csfinet_repro.survival_mri import CHANNELS, SHAPE, build_mri_volume, load_mri_cache, save_mri_cache


def test_mri_volume_keeps_mask_channel_and_zeroes_intensities_outside_wt():
    rng = np.random.default_rng(1)
    images = rng.normal(size=(155, 4, 240, 240)).astype(np.float32)
    mask = torch.zeros(1, 155, 128, 128)
    mask[:, 70:80, 40:60, 50:70] = 1
    volume = build_mri_volume(images, mask)
    assert volume.shape == SHAPE and volume.dtype == np.float16 and len(CHANNELS) == SHAPE[0]
    np.testing.assert_array_equal(volume[0], mask[0].numpy())
    outside = mask[0].numpy() == 0
    assert np.all(volume[1:][:, outside] == 0)
    assert volume[1:][:, ~outside].std() > 0
    constant = np.zeros((155, 4, 240, 240), dtype=np.float32)
    constant[:, 2] = 3.0
    full = build_mri_volume(constant, torch.ones(1, 155, 128, 128))
    np.testing.assert_allclose(full[3], 3.0, rtol=1e-3)
    np.testing.assert_array_equal(full[1], 0)
    with pytest.raises(ValueError):
        build_mri_volume(images[:10], mask)
    with pytest.raises(ValueError):
        build_mri_volume(images, mask * 2)


def test_mri_cache_round_trip_and_loader_dispatch(tmp_path):
    rng = np.random.default_rng(2)
    mask = (rng.random((1, 155, 128, 128)) < .02).astype(np.float32)
    volume = np.concatenate([mask, rng.normal(size=(4, 155, 128, 128)).astype(np.float32) * mask]).astype(np.float16)
    path = tmp_path / "patient.npz"
    assert len(save_mri_cache(volume, path)) == 64
    loaded = load_mri_cache(path)
    assert loaded.dtype == torch.float32 and tuple(loaded.shape) == SHAPE
    np.testing.assert_array_equal(loaded.numpy(), volume.astype(np.float32))
    assert load_survival_cache(path) is load_survival_cache(path)
    packed = tmp_path / "packed.npz"
    save_wt_cache(mask.astype(np.uint8), packed)
    assert tuple(load_survival_cache(packed).shape) == (1, 155, 128, 128)
    with pytest.raises(ValueError):
        save_mri_cache(volume[:4], tmp_path / "bad.npz")


@pytest.mark.parametrize("features", [0, 4])
def test_survival_net_accepts_five_input_channels(features):
    model = SurvivalNet(features, channels=(2, 4, 8), in_channels=5)
    image = torch.rand(2, 5, 16, 16, 16)
    clinical = torch.rand(2, 4) if features else None
    predicted = model(image, clinical)
    assert predicted.shape == (2,) and (predicted > 0).all()
    assert SurvivalNet(0, channels=(2, 4, 8)).in_channels == 1
    with pytest.raises(ValueError):
        SurvivalNet(0, in_channels=0)
