"""Sequential four-variant survival development/refit queue."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .files import sha256, write_json
from .survival import VARIANTS


def validate_completed_run(run_path, variant, phase, config_sha, split_sha, cache_sha):
    run_path = Path(run_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    expected = {"status": "completed", "variant": variant, "phase": phase,
                "config_sha256": config_sha, "split_sha256": split_sha,
                "cache_manifest_sha256": cache_sha}
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed survival run lineage mismatch: {variant}/{phase}")
    checkpoint_name = "final.pt" if phase == "refit" else "best.pt"
    checkpoint_key = "final_checkpoint_sha256" if phase == "refit" else "best_checkpoint_sha256"
    checkpoint = run_path.parent / checkpoint_name
    if not checkpoint.is_file() or sha256(checkpoint) != run.get(checkpoint_key):
        raise ValueError(f"Completed survival checkpoint mismatch: {variant}/{phase}")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/survival-v2.json")
    parser.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--tag", default="v2mri")
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("Tag must be alphanumeric")
    cache_manifest = Path(args.cache) / "manifest.json"
    cached = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if cached.get("status") != "completed" or cached.get("split_sha256") != sha256(args.patients):
        raise ValueError("Completed survival cache required")
    jobs = []
    for variant in VARIANTS:
        for phase in ("development", "refit"):
            output = Path(args.runs) / f"survival-{variant}-{phase}-{args.tag}"
            command = [sys.executable, "-u", "-m", "csfinet_repro", "train-survival", "--variant", variant,
                       "--phase", phase, "--output", str(output), "--config", args.config,
                       "--patients", args.patients, "--cache", args.cache]
            if phase == "refit":
                selection = Path(args.runs) / f"survival-{variant}-development-{args.tag}" / "run.json"
                command += ["--selection", str(selection)]
            if (output / "last.pt").exists():
                command.append("--resume")
            jobs.append(dict(variant=variant, phase=phase, output=str(output), command=command, status="pending"))
    manifest = dict(scope="four_survival_variants_development_and_refit", status="running", jobs=jobs,
                    config_sha256=sha256(args.config), split_sha256=sha256(args.patients),
                    cache_manifest_sha256=sha256(cache_manifest),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    code_sha256={path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))})
    path = Path(args.runs) / f"survival-suite-{args.tag}.json"
    for job in jobs:
        run_json = Path(job["output"]) / "run.json"
        if run_json.exists() and json.loads(run_json.read_text(encoding="utf-8")).get("status") == "completed":
            validate_completed_run(run_json, job["variant"], job["phase"], manifest["config_sha256"],
                                   manifest["split_sha256"], manifest["cache_manifest_sha256"])
            job.update(status="completed", skipped_existing=True)
            write_json(path, manifest)
            continue
        print(f"Starting survival {job['variant']} {job['phase']}", flush=True)
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
