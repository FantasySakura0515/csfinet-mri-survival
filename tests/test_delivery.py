from pathlib import Path

import pytest

from csfinet_repro.delivery import (
    _cohort_table,
    _inventory_file,
    _mean_sd,
    _validate_schematics,
)
from csfinet_repro.files import sha256, write_json


def test_delivery_inventory_requires_project_containment_and_matching_hash(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    inside = root / "result.csv"
    inside.write_text("a\n1\n", encoding="utf-8")
    inventory = []
    digest = _inventory_file(root, inside, inventory, role="test")
    assert inventory == [{"path": "result.csv", "role": "test", "bytes": inside.stat().st_size, "sha256": digest}]
    with pytest.raises(ValueError, match="Hash mismatch"):
        _inventory_file(root, inside, [], expected_sha="0" * 64)
    outside = tmp_path / "outside.csv"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="outside project"):
        _inventory_file(root, outside, [])
    logical = root / "final" / "result.csv"
    projected = []
    _inventory_file(root, inside, projected, logical_path=logical)
    assert projected[0]["path"] == "final/result.csv"
    with pytest.raises(ValueError, match="inventory path is outside"):
        _inventory_file(root, inside, [], logical_path=outside)


def test_delivery_cohort_table_uses_patient_sample_sd_and_explicit_na(tmp_path):
    rows = [
        {"patient_id": "a", "split": "train", "age": "20", "survival_days": "100", "resection_status": "GTR"},
        {"patient_id": "b", "split": "train", "age": "40", "survival_days": "300", "resection_status": "STR"},
        {"patient_id": "c", "split": "test", "age": "30", "survival_days": "200", "resection_status": "NA"},
        {"patient_id": "d", "split": "test", "age": "50", "survival_days": "400", "resection_status": "GTR"},
    ]
    table, files = _cohort_table(rows, tmp_path)
    assert table[0]["age_mean"] == 35
    assert table[1]["age_sd"] == pytest.approx(14.1421356)
    assert table[2]["GTR"] == 1 and table[2]["NA"] == 1
    assert all(Path(path).exists() for path in files)
    assert _mean_sd({"x_mean": "", "x_sd": ""}, "x") == "NA"


def test_delivery_requires_all_fourteen_lineage_matched_schematics(tmp_path):
    root = tmp_path
    config = root / "configs" / "segmentation.json"
    config.parent.mkdir()
    config.write_text("{}\n", encoding="utf-8")
    directory = root / "results" / "schematics" / "study"
    directory.mkdir(parents=True)
    artifacts = []
    for index in range(14):
        path = directory / f"figure-{index}.png"
        path.write_bytes(f"figure {index}".encode())
        artifacts.append({"path": path.name, "sha256": sha256(path)})
    split_sha = "b" * 64
    write_json(directory / "manifest.json", {
        "status": "completed", "scope": "implementation_matched_figures_1_to_7",
        "config_sha256": sha256(config), "split_sha256": split_sha, "artifacts": artifacts,
    })
    inventory = []
    manifest = _validate_schematics(root, "study", split_sha, inventory)
    assert manifest["status"] == "completed"
    assert len(inventory) == 15
    assert sum(row["role"] == "manuscript_schematic" for row in inventory) == 14
    manifest["split_sha256"] = "wrong"
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="lineage"):
        _validate_schematics(root, "study", split_sha, [])
