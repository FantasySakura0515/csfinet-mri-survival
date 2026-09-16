import numpy as np
import pytest

from csfinet_repro.nifti import normalize_volume
from csfinet_repro.splits import assign_split


def patients(n=235):
    return [dict(patient_id=f"BraTS20_Training_{i:03d}", age=40, survival_days=400, resection_status="NA")
            for i in range(1, n + 1)]


def test_patient_split_is_reproducible_order_independent_and_disjoint():
    source = patients()
    rows = assign_split(source, 20260914, 47, 38, 20260915)
    assert rows == assign_split(list(reversed(source)), 20260914, 47, 38, 20260915)
    train = {r["patient_id"] for r in rows if r["split"] == "train"}
    test = {r["patient_id"] for r in rows if r["split"] == "test"}
    validation = {r["patient_id"] for r in rows if r["development_split"] == "validation"}
    assert len(train) == 188 and len(test) == 47 and len(validation) == 38
    assert validation < train and not train & test
    assert len(train | test) == 235
    assert rows != assign_split(source, 1, 47, 38, 2)


def test_patient_split_rejects_duplicates_and_empty_training():
    with pytest.raises(ValueError, match="Duplicate"):
        assign_split(patients() + patients(1), 1, 47, 38, 2)
    with pytest.raises(ValueError, match="sizes"):
        assign_split(patients(), 1, 47, 188, 2)


def test_normalization_is_whole_volume_including_background():
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    normalized = normalize_volume(source)
    np.testing.assert_allclose(normalized.mean(), 0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(), 1, atol=1e-6)
    assert not np.allclose(normalized.mean(axis=(0, 1)), 0)
    assert normalized[0, 0, 0] < 0
    np.testing.assert_array_equal(normalize_volume(np.ones((2, 3, 4))), 0)
    with pytest.raises(ValueError):
        normalize_volume(np.full((2, 3, 4), np.nan))
