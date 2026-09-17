import json

import pytest

from csfinet_repro.files import sha256, write_json
from csfinet_repro.delivery_suite import completed_delivery
from csfinet_repro.pipeline_guard import should_restart
from csfinet_repro.postprocess_suite import completed
from csfinet_repro.survival_suite import validate_completed_run


def test_completed_requires_valid_json_status_and_expected_lineage(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{", encoding="utf-8")
    assert completed(path) is False
    write_json(path, {"status": "running", "split_sha256": "a"})
    assert completed(path) is False
    write_json(path, {"status": "completed", "split_sha256": "a"})
    assert completed(path, split_sha256="a") is True
    assert completed(path, split_sha256="b") is False


def test_survival_suite_validates_completed_run_and_checkpoint(tmp_path):
    directory = tmp_path / "survival-image-development-study"
    directory.mkdir()
    checkpoint = directory / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    run_path = directory / "run.json"
    run = {"status": "completed", "variant": "image", "phase": "development",
           "config_sha256": "config", "split_sha256": "split",
           "cache_manifest_sha256": "cache", "best_checkpoint_sha256": sha256(checkpoint)}
    write_json(run_path, run)
    assert validate_completed_run(run_path, "image", "development", "config", "split", "cache") == run
    run["split_sha256"] = "wrong"
    write_json(run_path, run)
    with pytest.raises(ValueError, match="lineage"):
        validate_completed_run(run_path, "image", "development", "config", "split", "cache")


def test_pipeline_guard_restarts_only_missing_unfinished_watchers():
    assert should_restart("waiting_for_segmentation", 10, lambda pid: pid == 10) is False
    assert should_restart("running", 10, lambda pid: False) is True
    assert should_restart("failed", 10, lambda pid: False) is True
    assert should_restart("packaging_release", 10, lambda pid: False) is True
    assert should_restart("completed", 10, lambda pid: False) is False
    with pytest.raises(ValueError, match="Unexpected"):
        should_restart("unknown", 10, lambda pid: False)
    assert should_restart("failed", None, lambda pid: False) is True


def test_delivery_suite_accepts_only_completed_matching_output(tmp_path):
    output = tmp_path / "delivery"
    output.mkdir()
    manifest = output / "delivery-manifest.csv"
    manifest.write_text("path,sha256\n", encoding="utf-8")
    audit = {"status": "completed", "tag": "study", "manifest_sha256": sha256(manifest)}
    write_json(output / "delivery-audit.json", audit)
    assert completed_delivery(output, "study") == audit
    audit["manifest_sha256"] = "bad"
    write_json(output / "delivery-audit.json", audit)
    with pytest.raises(ValueError, match="manifest validation"):
        completed_delivery(output, "study")
