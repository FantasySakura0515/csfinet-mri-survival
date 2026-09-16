import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from csfinet_repro.segmentation_evaluation import save_prediction, summarize_segmentation


def test_prediction_round_trip_keeps_native_geometry_and_original_labels(tmp_path):
    reference = tmp_path / "reference.nii.gz"
    affine = np.diag([1, 2, 3, 1])
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 2), dtype=np.uint8), affine), reference)
    prediction = np.array([[[0, 1, 2, 4], [4, 2, 1, 0], [0, 0, 0, 0]],
                           [[1, 1, 2, 2], [4, 4, 0, 0], [2, 1, 4, 0]]], dtype=np.uint8)
    target = tmp_path / "prediction.nii.gz"
    digest = save_prediction(reference, prediction, target)
    loaded = nib.load(target)
    assert loaded.shape == (3, 4, 2)
    np.testing.assert_array_equal(loaded.affine, affine)
    np.testing.assert_array_equal(loaded.get_fdata().transpose(2, 0, 1), prediction)
    assert len(digest) == 64


def write_metrics(path, model, offset):
    columns = ["run_id", "patient_id", "model", "region", "tp", "fp", "fn", "dice", "precision", "recall", "iou"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for patient in range(47):
            for index, region in enumerate(("WT", "TC", "ET")):
                value = .4 + offset + patient / 1000 + index / 100
                writer.writerow(dict(run_id="r", patient_id=f"p{patient:02}", model=model, region=region,
                                     tp=10, fp=2, fn=3, dice=value, precision=value + .01,
                                     recall=value - .01, iou=value - .1))


def test_table5_uses_47_matching_patients_and_creates_all_formats(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_metrics(a, "csfinet", .1)
    write_metrics(b, "unet", 0)
    report = summarize_segmentation([a, b], tmp_path / "out", seed=3, n_resamples=200)
    assert report["models"]["csfinet"]["WT"]["dice"]["summary"]["n_valid"] == 47
    assert report["paired"]["WT"]["dice"]["difference"]["summary"]["mean"] == pytest.approx(.1)
    for name in ("table5.csv", "table5.json", "table5.md", "table5.tex"):
        assert (tmp_path / "out" / name).exists()
