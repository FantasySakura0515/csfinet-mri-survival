import io
import zipfile

import h5py
import numpy as np
import pytest

from csfinet_repro.data import build_cohort, validate_slice_metadata, load_hdf5_slice, decode_exclusive_mask
from csfinet_repro.download import unpack_download
from csfinet_repro.files import write_csv
from csfinet_repro.__main__ import summarize_survival


def test_alive_record_remains_excluded_and_split_unassigned(tmp_path):
    ids = [f"BraTS20_Training_{n:03}" for n in [1, 84]]
    write_csv(tmp_path / "mapping.csv", [{"BraTS_2020_subject_ID": p} for p in ids], ["BraTS_2020_subject_ID"])
    records = [dict(Brats20ID=p, Age="61.6", Survival_days=days, Extent_of_Resection="NA")
               for p, days in zip(ids, ["289", "ALIVE (361 days later)"])]
    columns = ["Brats20ID", "Age", "Survival_days", "Extent_of_Resection"]
    write_csv(tmp_path / "survival.csv", records, columns)
    candidates, excluded = build_cohort(tmp_path / "survival.csv", tmp_path / "mapping.csv")
    assert len(candidates) == len(excluded) == 1
    assert candidates[0]["split"] == "unassigned"
    assert candidates[0]["hdf5_volume"] == ""
    assert candidates[0]["resection_status"] == "NA"
    assert excluded[0]["survival_raw"] == "ALIVE (361 days later)"
    write_csv(tmp_path / "survival.csv", records + records[:1], columns)
    with pytest.raises(ValueError, match="duplicate"):
        build_cohort(tmp_path / "survival.csv", tmp_path / "mapping.csv")


def test_metadata_requires_all_slices_without_duplicates(tmp_path):
    columns = ["slice_path", "target", "volume", "slice"]
    rows = [dict(slice_path=f"/content/data/volume_1_slice_{s}.h5", target=0, volume=1, slice=s)
            for s in range(155)]
    path = tmp_path / "meta.csv"
    write_csv(path, rows, columns)
    assert validate_slice_metadata(path) == {1: set(range(155))}
    write_csv(path, rows[:-1], columns)
    with pytest.raises(ValueError, match="155"):
        validate_slice_metadata(path)
    write_csv(path, rows + rows[:1], columns)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_slice_metadata(path)


def test_hdf5_preserves_data_without_implied_channel_reordering(tmp_path):
    path = tmp_path / "slice.h5"
    image = np.arange(240 * 240 * 4, dtype=float).reshape(240, 240, 4)
    mask = np.zeros((240, 240, 3), dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f["image"] = image
        f["mask"] = mask
    actual, actual_mask = load_hdf5_slice(path)
    np.testing.assert_array_equal(actual, image)
    np.testing.assert_array_equal(actual_mask, mask)


def test_mask_decoding_requires_explicit_exclusive_labels():
    mask = np.eye(3, dtype=np.uint8).reshape(1, 3, 3)
    np.testing.assert_array_equal(decode_exclusive_mask(mask, [4, 1, 2]), [[4, 1, 2]])
    with pytest.raises(ValueError):
        decode_exclusive_mask(mask, [0, 1, 2])
    mask[0, 0, 1] = 1
    with pytest.raises(ValueError, match="Overlapping"):
        decode_exclusive_mask(mask, [1, 2, 4])


def test_download_archive_is_read_without_extracting_paths():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("../../sample.csv", "a,b\n1,2\n")
    assert unpack_download(stream.getvalue(), "sample.csv", 100) == b"a,b\n1,2\n"
    with pytest.raises(ValueError):
        unpack_download(b"<html>login</html>", "sample.csv", 100)
    with pytest.raises(ValueError):
        unpack_download(stream.getvalue(), "sample.csv", 2)


def test_survival_csv_rejects_mixed_runs_and_missing_patient(tmp_path):
    columns = ["run_id", "patient_id", "model", "y_true", "y_pred"]
    rows = [dict(run_id="test_only", patient_id=p, model=m, y_true=y, y_pred=y + 1)
            for m in ["a", "b"] for p, y in [("p1", 10), ("p2", 20)]]
    path, output = tmp_path / "pred.csv", tmp_path / "summary.json"
    write_csv(path, rows, columns)
    result = summarize_survival(path, output, 2, 17)
    assert result["models"]["a"]["sd_ae"] == 0
    assert result["models"]["a"]["mae_ci"]["low"] is None
    write_csv(path, rows[:-1], columns)
    with pytest.raises(ValueError, match="same expected patient"):
        summarize_survival(path, output, 2, 17)
    rows[0]["run_id"] = "other_run"
    write_csv(path, rows, columns)
    with pytest.raises(ValueError, match="one nonempty run_id"):
        summarize_survival(path, output, 2, 17)
