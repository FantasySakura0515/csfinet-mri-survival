"""Sequential segmentation training queue; run with python -m csfinet_repro.run_suite."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .files import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation.json")
    parser.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    parser.add_argument("--audit", default="data/manifests/cohort/nifti-audit.json")
    parser.add_argument("--root", default="data/raw/brats2020-nifti")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--tag", default="study")
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("Tag must be alphanumeric")
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("patients") != 235 or audit.get("split_sha256") != sha256(args.patients):
        raise ValueError("Full cohort audit must pass before starting the queue")
    jobs = []
    for model in ("csfinet", "unet"):
        for phase in ("development", "refit"):
            output = Path(args.runs) / f"{model}-{phase}-{args.tag}"
            command = [sys.executable, "-u", "-m", "csfinet_repro", "train-segmentation", "--model", model,
                       "--phase", phase, "--output", str(output), "--config", args.config,
                       "--patients", args.patients, "--root", args.root]
            if phase == "refit":
                command += ["--selection", str(Path(args.runs) / f"{model}-development-{args.tag}" / "run.json")]
            if (output / "last.pt").exists() or (output / "pending.pt").exists():
                command += ["--resume"]
            jobs.append(dict(model=model, phase=phase, output=str(output), command=command, status="pending"))
    manifest = dict(scope="segmentation_development_and_refit_only", status="running", jobs=jobs,
                    config_sha256=sha256(args.config), split_sha256=sha256(args.patients), audit_sha256=sha256(args.audit))
    path = Path(args.runs) / f"segmentation-suite-{args.tag}.json"
    for job in jobs:
        run_path = Path(job["output"]) / "run.json"
        if run_path.exists():
            existing = json.loads(run_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                if (existing.get("model") != job["model"] or existing.get("phase") != job["phase"]
                        or existing.get("config_sha256") != manifest["config_sha256"]
                        or existing.get("split_sha256") != manifest["split_sha256"]):
                    raise ValueError("Completed segmentation run does not match suite lineage")
                job.update(status="completed", skipped_existing=True)
                write_json(path, manifest)
                continue
        print(f"Starting {job['model']} {job['phase']}", flush=True)
        process = subprocess.Popen(job["command"])
        job.update(status="running", pid=process.pid)
        write_json(path, manifest)
        code = process.wait()
        job.update(status="completed" if code == 0 else "failed", exit_code=code)
        if code:
            manifest["status"] = "failed"
            write_json(path, manifest)
            raise SystemExit(code)
        write_json(path, manifest)
    manifest["status"] = "completed"
    write_json(path, manifest)


if __name__ == "__main__":
    main()
