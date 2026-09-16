import csv
import json

import nibabel as nib
import numpy as np

from csfinet_repro.schematics import generate_schematics


def test_generate_seven_traceable_schematics(tmp_path):
    patient = "p001"
    patients = tmp_path / "patients.csv"
    with patients.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient_id", "split", "development_split"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({"patient_id": patient, "split": "train", "development_split": "train"})
    root = tmp_path / "raw" / patient
    root.mkdir(parents=True)
    for index, modality in enumerate(("flair", "t1", "t1ce", "t2")):
        values = np.arange(60, dtype=np.float32).reshape(3, 4, 5) + index
        nib.save(nib.Nifti1Image(values, np.eye(4)), root / f"{patient}_{modality}.nii.gz")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"segmentation": {"channels": [64, 128, 256, 512, 1024]},
                                  "survival": {"image_head": [128, 128, 64, 32, 1],
                                               "fusion_head": [144, 64, 32, 1]}}), encoding="utf-8")
    output = tmp_path / "figures"
    manifest = generate_schematics(config, patients, tmp_path / "raw", output)
    assert manifest["status"] == "completed"
    assert manifest["normalization_patient"] == patient
    assert len(manifest["artifacts"]) == 14
    assert {item["path"] for item in manifest["artifacts"]} == {
        f"fig{number}_{name}.{suffix}"
        for number, name in ((1, "pipeline"), (2, "csfinet"), (3, "piu"), (4, "normalization"),
                             (5, "fiu"), (6, "survival_variants"), (7, "survival_heads"))
        for suffix in ("png", "svg")}
    assert all((output / item["path"]).stat().st_size > 0 for item in manifest["artifacts"])
