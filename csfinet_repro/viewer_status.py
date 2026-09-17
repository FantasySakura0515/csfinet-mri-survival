"""Resolve read-only viewer state from completed inference, not stale launchers."""

import json
from pathlib import Path


def result_status(project):
    root = Path(project)
    jobs = []
    for name in ("csfinet", "unet"):
        directory = root / "results" / "segmentation" / f"{name}-test-study"
        path = directory / "inference-manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            complete = value.get("status") == "completed" and (directory / "segmentation_patient_metrics.csv").is_file()
        except (OSError, ValueError, TypeError):
            complete = False
        jobs.append(dict(model=name, phase="test_inference", status="completed" if complete else "unavailable"))
    if all(job["status"] == "completed" for job in jobs):
        return "completed", jobs
    return "結果待確認", jobs
