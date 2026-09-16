"""Wait for segmentation refits, then create all Table 5/7 prerequisite outputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .files import sha256, write_json

_STATE_PATH = None


def completed(path, **expected):
    path = Path(path)
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return value.get("status") == "completed" and all(value.get(key) == item for key, item in expected.items())


def run(command, label):
    print(f"Starting {label}", flush=True)
    result = subprocess.run(command)
    if result.returncode:
        raise RuntimeError(f"{label} exited with {result.returncode}")


def main():
    global _STATE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reconstruction-v21.json")
    parser.add_argument("--patients", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--segmentation-suite", required=True)
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not args.tag.isalnum() or args.poll_seconds < 10:
        raise ValueError("Tag must be alphanumeric and polling interval >=10 seconds")
    state_path = Path(args.runs) / f"postprocess-suite-{args.tag}.json"
    _STATE_PATH = state_path
    state = dict(status="waiting_for_segmentation", segmentation_suite=args.segmentation_suite,
                 config_sha256=sha256(args.config), analysis_sha256=sha256("configs/analysis.json"),
                 split_sha256=sha256(args.patients),
                 git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 code_sha256={path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
                 stages=[])
    write_json(state_path, state)
    while True:
        suite_path = Path(args.segmentation_suite)
        if suite_path.exists():
            segmentation = json.loads(suite_path.read_text(encoding="utf-8"))
            if segmentation.get("status") == "failed":
                state.update(status="failed", error="segmentation suite failed")
                write_json(state_path, state)
                raise RuntimeError("Segmentation suite failed")
            if segmentation.get("status") == "completed":
                if (segmentation.get("config_sha256") != state["config_sha256"]
                        or segmentation.get("split_sha256") != state["split_sha256"]):
                    raise ValueError("Completed segmentation suite does not match postprocess config/split")
                break
        time.sleep(args.poll_seconds)
    state["status"] = "running"
    write_json(state_path, state)
    result_root, artifact_root, run_root, cache_root = map(Path, (args.results, args.artifacts, args.runs, args.cache))
    metric_files = []
    for model in ("csfinet", "unet"):
        run_dir = run_root / f"{model}-refit-{args.tag}"
        output = result_root / "segmentation" / f"{model}-test-{args.tag}"
        predictions = artifact_root / "segmentation" / f"{model}-test-{args.tag}"
        manifest = output / "inference-manifest.json"
        if not completed(manifest, model=model, split_sha256=sha256(args.patients)):
            command = [sys.executable, "-u", "-m", "csfinet_repro", "infer-segmentation", "--config", args.config,
                       "--patients", args.patients, "--root", args.root, "--run", str(run_dir),
                       "--output", str(output), "--predictions", str(predictions)]
            if output.exists() or predictions.exists():
                command.append("--resume")
            run(command, f"{model} frozen test inference")
        metric_files.append(output / "segmentation_patient_metrics.csv")
        state["stages"].append(dict(name=f"{model}_test", status="completed", manifest_sha256=sha256(manifest)))
        write_json(state_path, state)
    table5 = result_root / "tables" / args.tag / "segmentation"
    if not completed(table5 / "table5.json"):
        command = [sys.executable, "-m", "csfinet_repro", "summarize-segmentation",
                   "--input", str(metric_files[0]), "--input", str(metric_files[1]), "--output", str(table5)]
        run(command, "Table 5")
    state["stages"].append(dict(name="table5", status="completed", report_sha256=sha256(table5 / "table5.json")))
    write_json(state_path, state)
    survival_cache = cache_root / f"survival-{args.tag}"
    csfinet_output = result_root / "segmentation" / f"csfinet-test-{args.tag}"
    csfinet_predictions = artifact_root / "segmentation" / f"csfinet-test-{args.tag}"
    if not completed(survival_cache / "manifest.json", split_sha256=sha256(args.patients)):
        command = [sys.executable, "-u", "-m", "csfinet_repro", "cache-survival", "--patients", args.patients,
                   "--root", args.root, "--predictions", str(csfinet_predictions),
                   "--segmentation-manifest", str(csfinet_output / "inference-manifest.json"), "--output", str(survival_cache)]
        run(command, "survival WT cache")
    state["stages"].append(dict(name="survival_cache", status="completed",
                                manifest_sha256=sha256(survival_cache / "manifest.json")))
    write_json(state_path, state)
    run([sys.executable, "-u", "-m", "csfinet_repro.survival_suite", "--config", args.config,
         "--patients", args.patients, "--cache", str(survival_cache), "--runs", args.runs, "--tag", args.tag],
        "survival training suite")
    state["stages"].append(dict(name="survival_training", status="completed",
                                manifest_sha256=sha256(run_root / f"survival-suite-{args.tag}.json")))
    write_json(state_path, state)
    predictions = result_root / "survival" / args.tag / "survival_predictions.csv"
    prediction_manifest = predictions.with_suffix(".manifest.json")
    if not completed(prediction_manifest, split_sha256=sha256(args.patients)):
        command = [sys.executable, "-u", "-m", "csfinet_repro", "infer-survival", "--config", args.config,
                   "--patients", args.patients, "--cache", str(survival_cache), "--output", str(predictions)]
        for variant in ("image", "image_age", "image_resection", "image_age_resection"):
            command += [f"--{variant.replace('_', '-')}-run", str(run_root / f"survival-{variant}-refit-{args.tag}")]
        run(command, "six-model survival test inference")
    table7 = result_root / "tables" / args.tag / "survival"
    if not completed(table7 / "table7.json"):
        run([sys.executable, "-m", "csfinet_repro", "summarize-survival-table", "--input", str(predictions),
             "--output", str(table7)], "Table 7")
    state["stages"].append(dict(name="table7", status="completed", report_sha256=sha256(table7 / "table7.json")))
    write_json(state_path, state)
    explain_output = result_root / "explainability" / args.tag
    explain_artifacts = artifact_root / "explainability" / args.tag
    explain_manifest = explain_output / "manifest.json"
    if not completed(explain_manifest, split_sha256=sha256(args.patients)):
        command = [sys.executable, "-u", "-m", "csfinet_repro", "explain-segmentation", "--config", args.config,
                   "--analysis", "configs/analysis.json", "--patients", args.patients, "--root", args.root,
                   "--run", str(run_root / f"csfinet-refit-{args.tag}"), "--predictions", str(csfinet_predictions),
                   "--inference-manifest", str(csfinet_output / "inference-manifest.json"),
                   "--output", str(explain_output), "--artifacts", str(explain_artifacts)]
        if explain_output.exists() or explain_artifacts.exists():
            command.append("--resume")
        run(command, "47-patient Gradient SHAP")
    state["stages"].append(dict(name="gradient_shap", status="completed", manifest_sha256=sha256(explain_manifest)))
    write_json(state_path, state)
    randomization = explain_output / "parameter_randomization.csv"
    randomization_manifest = randomization.with_suffix(".manifest.json")
    if not completed(randomization_manifest, split_sha256=sha256(args.patients)):
        command = [sys.executable, "-u", "-m", "csfinet_repro", "randomize-segmentation", "--config", args.config,
                   "--analysis", "configs/analysis.json", "--patients", args.patients, "--root", args.root,
                   "--run", str(run_root / f"csfinet-refit-{args.tag}"), "--output", str(randomization)]
        if randomization.exists() or randomization.with_suffix(".start.json").exists():
            command.append("--resume")
        run(command, "parameter randomization sanity check")
    table9 = result_root / "tables" / args.tag / "shap"
    if not completed(table9 / "table9.json"):
        run([sys.executable, "-m", "csfinet_repro", "summarize-shap",
             "--shap", str(explain_output / "shap_patient_metrics.csv"),
             "--segmentation", str(csfinet_output / "segmentation_patient_metrics.csv"),
             "--randomization", str(randomization), "--output", str(table9)], "Table 9")
    state["stages"].append(dict(name="table9", status="completed", report_sha256=sha256(table9 / "table9.json")))
    write_json(state_path, state)
    figures = result_root / "figures" / args.tag
    if not completed(figures / "manifest.json", split_sha256=sha256(args.patients)):
        run([sys.executable, "-m", "csfinet_repro", "generate-figures", "--patients", args.patients,
             "--root", args.root, "--csfinet-predictions", str(csfinet_predictions),
             "--shap-artifacts", str(explain_artifacts), "--survival-predictions", str(predictions),
             "--csfinet-development", str(run_root / f"csfinet-development-{args.tag}" / "run.json"),
             "--unet-development", str(run_root / f"unet-development-{args.tag}" / "run.json"),
             "--output", str(figures)], "manuscript figures")
    state["stages"].append(dict(name="figures", status="completed", manifest_sha256=sha256(figures / "manifest.json")))
    state["status"] = "completed"
    write_json(state_path, state)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _STATE_PATH is not None and _STATE_PATH.exists():
            state = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            if state.get("status") != "completed":
                state.update(status="failed", error=f"{type(error).__name__}: {error}")
                write_json(_STATE_PATH, state)
        raise
