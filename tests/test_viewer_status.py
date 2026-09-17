import json

from csfinet_repro.viewer_status import result_status


def test_completed_inference_overrides_stale_running_launcher(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "segmentation-suite-study.json").write_text('{"status":"running"}')
    for name in ("csfinet", "unet"):
        folder = tmp_path / "results/segmentation" / f"{name}-test-study"
        folder.mkdir(parents=True)
        (folder / "inference-manifest.json").write_text(json.dumps({"status": "completed"}))
        (folder / "segmentation_patient_metrics.csv").write_text("patient_id,dice\np1,0.8\n")
    assert result_status(tmp_path)[0] == "completed"
    (folder / "segmentation_patient_metrics.csv").unlink()
    assert result_status(tmp_path)[0] != "completed"


def test_missing_or_invalid_manifest_is_not_reported_as_training(tmp_path):
    assert result_status(tmp_path)[0] == "結果待確認"
    folder = tmp_path / "results/segmentation/csfinet-test-study"
    folder.mkdir(parents=True)
    (folder / "inference-manifest.json").write_text("invalid")
    assert result_status(tmp_path)[0] == "結果待確認"
