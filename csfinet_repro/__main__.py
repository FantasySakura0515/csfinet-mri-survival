"""Run from the repository root: python -m csfinet_repro --help."""

import argparse
from collections import defaultdict
import json

from .data import audit_sources
from .download import download_sources
from .files import read_csv, sha256, write_json
from .metrics import bca_mean_ci, survival_metrics


def summarize_survival(input_path, output_path, expected_patients, seed):
    rows = read_csv(input_path, ["run_id", "patient_id", "model", "y_true", "y_pred"])
    if not rows or len({r["run_id"] for r in rows}) != 1 or not rows[0]["run_id"].strip():
        raise ValueError("Input must contain exactly one nonempty run_id")
    if expected_patients < 2:
        raise ValueError("Expected patient count must be at least two")
    groups = defaultdict(dict)
    truths = {}
    for row in rows:
        patient, model = row["patient_id"].strip(), row["model"].strip()
        if not patient or not model or patient in groups[model]:
            raise ValueError("Blank identifier or duplicate patient/model pair")
        truth, prediction = float(row["y_true"]), float(row["y_pred"])
        if patient in truths and truths[patient] != truth:
            raise ValueError(f"Ground truth differs across models for {patient}")
        truths[patient] = truth
        groups[model][patient] = (truth, prediction)
    if len(truths) != expected_patients or any(set(group) != set(truths) for group in groups.values()):
        raise ValueError("Every model must cover exactly the same expected patient IDs")
    output = dict(run_id=rows[0]["run_id"], source_sha256=sha256(input_path),
                  scope="basic_survival_errors_and_MAE_CI_not_complete_Table7", models={})
    for model, group in sorted(groups.items()):
        pairs = [group[p] for p in sorted(group)]
        metrics = survival_metrics([t for t, _ in pairs], [p for _, p in pairs])
        ci = bca_mean_ci(metrics["absolute_error"], seed=seed)
        metrics.pop("absolute_error")
        metrics.pop("residual")
        output["models"][model] = dict(**metrics, mae_ci=ci)
    write_json(output_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-source", help="Fetch/check pinned CSVs and one HDF5 sample")
    download.add_argument("--config", default="configs/source-files.json")
    download.add_argument("--destination", default="data/raw/source-check")
    audit = commands.add_parser("audit", help="Build candidate cohort and inspect local HDF5 files")
    audit.add_argument("--source", default="data/raw/source-check")
    audit.add_argument("--output", default="data/manifests/source-audit")
    split = commands.add_parser("create-split", help="Freeze a new reproducible 188/47 split")
    split.add_argument("--source", default="data/raw/source-check")
    split.add_argument("--config", default="configs/segmentation.json")
    split.add_argument("--output", default="data/manifests/cohort")
    cohort = commands.add_parser("summarize-cohort", help="Describe the new 235-patient cohort and split")
    cohort.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    cohort.add_argument("--output", default="data/manifests/cohort/cohort-summary.json")
    nifti = commands.add_parser("download-nifti", help="Download version-pinned raw images for frozen patients")
    nifti.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    nifti.add_argument("--destination", default="data/raw/brats2020-nifti")
    nifti.add_argument("--manifest", default="data/manifests/cohort/nifti-downloads.json")
    nifti.add_argument("--workers", type=int, default=3)
    nifti.add_argument("--limit", type=int)
    audit_raw = commands.add_parser("audit-nifti", help="Verify all 235 patient images, labels, geometry and hashes")
    audit_raw.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    audit_raw.add_argument("--root", default="data/raw/brats2020-nifti")
    audit_raw.add_argument("--output", default="data/manifests/cohort/nifti-audit.json")
    smoke = commands.add_parser("smoke-segmentation", help="Run one real training-patient optimizer step on CUDA")
    smoke.add_argument("--config", default="configs/segmentation.json")
    smoke.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    smoke.add_argument("--root", default="data/raw/brats2020-nifti")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--model", choices=["csfinet", "unet"], default="csfinet")
    smoke.add_argument("--slice-batch", type=int)
    smoke.add_argument("--slices", type=int, default=155)
    smoke.add_argument("--steps", type=int, default=2)
    smoke_surv = commands.add_parser("smoke-survival", help="Check four survival variants with real training WT volumes")
    smoke_surv.add_argument("--config", default="configs/survival.json")
    smoke_surv.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    smoke_surv.add_argument("--root", default="data/raw/brats2020-nifti")
    smoke_surv.add_argument("--output", required=True)
    train = commands.add_parser("train-segmentation", help="Pilot, internal selection, or refit on all 188 patients")
    train.add_argument("--config", default="configs/segmentation.json")
    train.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    train.add_argument("--root", default="data/raw/brats2020-nifti")
    train.add_argument("--output", required=True)
    train.add_argument("--model", choices=["csfinet", "unet"], required=True)
    train.add_argument("--phase", choices=["pilot", "development", "refit"], required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--selection", help="Completed development run.json required for refit")
    train.add_argument("--resume", action="store_true")
    infer_seg = commands.add_parser("infer-segmentation", help="Run a completed refit model on the frozen 47-patient test set")
    infer_seg.add_argument("--config", default="configs/segmentation.json")
    infer_seg.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    infer_seg.add_argument("--root", default="data/raw/brats2020-nifti")
    infer_seg.add_argument("--run", required=True)
    infer_seg.add_argument("--output", required=True)
    infer_seg.add_argument("--predictions", required=True)
    infer_seg.add_argument("--resume", action="store_true")
    table5 = commands.add_parser("summarize-segmentation", help="Build Table 5 from two frozen-test patient metric CSVs")
    table5.add_argument("--input", action="append", required=True, help="Repeat once per model")
    table5.add_argument("--output", required=True)
    table5.add_argument("--seed", type=int, default=20260914)
    table5.add_argument("--resamples", type=int, default=10000)
    cache_surv = commands.add_parser("cache-survival", help="Create GT-train/predicted-test 155x128x128 WT inputs")
    cache_surv.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    cache_surv.add_argument("--root", default="data/raw/brats2020-nifti")
    cache_surv.add_argument("--predictions", required=True)
    cache_surv.add_argument("--segmentation-manifest", required=True)
    cache_surv.add_argument("--output", required=True)
    cache_mri = commands.add_parser("cache-survival-mri", help="Create study WT-mask plus masked-MRI 5x155x128x128 survival inputs")
    cache_mri.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    cache_mri.add_argument("--root", default="data/raw/brats2020-nifti")
    cache_mri.add_argument("--predictions", required=True)
    cache_mri.add_argument("--segmentation-manifest", required=True)
    cache_mri.add_argument("--output", required=True)
    train_surv = commands.add_parser("train-survival", help="Pilot, development selection, or 188-patient refit")
    train_surv.add_argument("--config", default="configs/survival.json")
    train_surv.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    train_surv.add_argument("--cache", required=True)
    train_surv.add_argument("--output", required=True)
    train_surv.add_argument("--variant", choices=["image", "image_age", "image_resection", "image_age_resection"], required=True)
    train_surv.add_argument("--phase", choices=["pilot", "development", "refit"], required=True)
    train_surv.add_argument("--epochs", type=int)
    train_surv.add_argument("--selection")
    train_surv.add_argument("--resume", action="store_true")
    infer_surv = commands.add_parser("infer-survival", help="Predict all four variants plus two reference models")
    infer_surv.add_argument("--config", default="configs/survival.json")
    infer_surv.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    infer_surv.add_argument("--cache", required=True)
    for variant in ("image", "image_age", "image_resection", "image_age_resection"):
        infer_surv.add_argument(f"--{variant.replace('_', '-')}-run", required=True)
    infer_surv.add_argument("--output", required=True)
    table7 = commands.add_parser("summarize-survival-table", help="Build complete Table 7 from six-model predictions")
    table7.add_argument("--input", required=True)
    table7.add_argument("--output", required=True)
    table7.add_argument("--seed", type=int, default=20260914)
    table7.add_argument("--resamples", type=int, default=10000)
    explain = commands.add_parser("explain-segmentation", help="Run 47-patient fixed-region Gradient SHAP")
    explain.add_argument("--config", default="configs/segmentation.json")
    explain.add_argument("--analysis", default="configs/analysis.json")
    explain.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    explain.add_argument("--root", default="data/raw/brats2020-nifti")
    explain.add_argument("--run", required=True)
    explain.add_argument("--predictions", required=True)
    explain.add_argument("--inference-manifest", required=True)
    explain.add_argument("--output", required=True)
    explain.add_argument("--artifacts", required=True)
    explain.add_argument("--resume", action="store_true")
    randomize = commands.add_parser("randomize-segmentation", help="Run cumulative 47-patient parameter-randomization sanity check")
    randomize.add_argument("--config", default="configs/segmentation.json")
    randomize.add_argument("--analysis", default="configs/analysis.json")
    randomize.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    randomize.add_argument("--root", default="data/raw/brats2020-nifti")
    randomize.add_argument("--run", required=True)
    randomize.add_argument("--output", required=True)
    randomize.add_argument("--resume", action="store_true")
    table9 = commands.add_parser("summarize-shap", help="Build complete Table 9 from 47-patient SHAP outputs")
    table9.add_argument("--shap", required=True)
    table9.add_argument("--segmentation", required=True)
    table9.add_argument("--randomization", required=True)
    table9.add_argument("--output", required=True)
    table9.add_argument("--seed", type=int, default=20260914)
    table9.add_argument("--resamples", type=int, default=10000)
    figures = commands.add_parser("generate-figures", help="Generate manuscript training, error, age, and case figures")
    figures.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    figures.add_argument("--root", default="data/raw/brats2020-nifti")
    figures.add_argument("--csfinet-predictions", required=True)
    figures.add_argument("--shap-artifacts", required=True)
    figures.add_argument("--survival-predictions", required=True)
    figures.add_argument("--csfinet-development", required=True)
    figures.add_argument("--unet-development", required=True)
    figures.add_argument("--output", required=True)
    delivery = commands.add_parser("audit-delivery", help="Validate study artifacts and assemble the results package")
    delivery.add_argument("--project-root", default=".")
    delivery.add_argument("--tag", default="study")
    delivery.add_argument("--output")
    schematics = commands.add_parser("generate-schematics", help="Generate implementation-matched Figures 1-7")
    schematics.add_argument("--config", default="configs/segmentation.json")
    schematics.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    schematics.add_argument("--root", default="data/raw/brats2020-nifti")
    schematics.add_argument("--output", required=True)
    release = commands.add_parser("package-release", help="Package tracked source, formal reports, and large-artifact hashes")
    release.add_argument("--project-root", default=".")
    release.add_argument("--tag", default="study")
    release.add_argument("--output")
    validate_release_parser = commands.add_parser("validate-release", help="Validate an existing portable release bundle")
    validate_release_parser.add_argument("--output", required=True)
    validate_release_parser.add_argument("--tag", default="study")
    validate_release_parser.add_argument("--split-sha256", required=True)
    variants_validate = commands.add_parser("validate-inference-variants", help="Decide LCC/TTA/BatchNorm inference variants on the 38 validation patients before any test use")
    variants_validate.add_argument("--config", default="configs/segmentation.json")
    variants_validate.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    variants_validate.add_argument("--root", default="data/raw/brats2020-nifti")
    variants_validate.add_argument("--runs", default="runs")
    variants_validate.add_argument("--output", required=True)
    variants_validate.add_argument("--tag", default="study")
    variants_validate.add_argument("--seed", type=int, default=20260914)
    variants_validate.add_argument("--resamples", type=int, default=10000)
    variants_validate.add_argument("--calibration-patients", type=int, default=30)
    variants_test = commands.add_parser("evaluate-inference-variants", help="Evaluate the recorded inference-variant decision once on the frozen 47 test patients")
    variants_test.add_argument("--config", default="configs/segmentation.json")
    variants_test.add_argument("--patients", default="data/manifests/cohort/patients.csv")
    variants_test.add_argument("--root", default="data/raw/brats2020-nifti")
    variants_test.add_argument("--runs", default="runs")
    variants_test.add_argument("--results", default="results")
    variants_test.add_argument("--artifacts", default="artifacts")
    variants_test.add_argument("--decision", required=True)
    variants_test.add_argument("--output", required=True)
    variants_test.add_argument("--tag", default="study")
    variants_test.add_argument("--seed", type=int, default=20260914)
    variants_test.add_argument("--resamples", type=int, default=10000)
    survival = commands.add_parser("summarize-survival", help="Summarize saved predictions from one run")
    survival.add_argument("--input", required=True)
    survival.add_argument("--output", required=True)
    survival.add_argument("--expected-patients", type=int, default=47)
    survival.add_argument("--seed", type=int, required=True, help="Bootstrap RNG seed; not a training seed")
    args = parser.parse_args()
    if args.command == "download-source":
        result = download_sources(args.config, args.destination)
        print(f"Verified {len(result['files'])} source files at dataset version {result['version']}")
    elif args.command == "audit":
        result = audit_sources(args.source, args.output)
        print(json.dumps({k: result[k] for k in ["candidate_count", "excluded_count", "metadata_volumes",
                                                "metadata_slices", "local_hdf5_slices", "local_complete_volumes"]}))
    elif args.command == "create-split":
        from .splits import create_split
        result = create_split(args.source, args.config, args.output)
        print(f"Frozen {result['train_patients']} train / {result['test_patients']} test patients")
    elif args.command == "summarize-cohort":
        from .splits import summarize_cohort
        summarize_cohort(args.patients, args.output)
    elif args.command == "download-nifti":
        from .nifti import download_cohort
        result = download_cohort(args.patients, args.destination, args.manifest, args.workers, args.limit)
        print(f"Verified {len(result['files'])} NIfTI files")
    elif args.command == "audit-nifti":
        from .nifti import audit_nifti
        audit_nifti(args.patients, args.root, args.output)
    elif args.command == "smoke-segmentation":
        from .training import smoke_segmentation
        smoke_segmentation(args.config, args.patients, args.root, args.output, args.model, args.slice_batch, args.slices, args.steps)
    elif args.command == "train-segmentation":
        from .segmentation import train_segmentation
        train_segmentation(args.config, args.patients, args.root, args.output, args.model, args.phase,
                           args.epochs, args.selection, args.resume)
    elif args.command == "infer-segmentation":
        from .segmentation_evaluation import evaluate_segmentation
        evaluate_segmentation(args.config, args.patients, args.root, args.run, args.output, args.predictions, args.resume)
    elif args.command == "summarize-segmentation":
        from .segmentation_evaluation import summarize_segmentation
        summarize_segmentation(args.input, args.output, args.seed, args.resamples)
    elif args.command == "cache-survival":
        from .survival import create_survival_cache
        create_survival_cache(args.patients, args.root, args.predictions, args.segmentation_manifest, args.output)
    elif args.command == "cache-survival-mri":
        from .survival_mri import create_survival_mri_cache
        create_survival_mri_cache(args.patients, args.root, args.predictions, args.segmentation_manifest, args.output)
    elif args.command == "train-survival":
        from .survival import train_survival
        train_survival(args.config, args.patients, args.cache, args.output, args.variant, args.phase,
                       args.epochs, args.selection, args.resume)
    elif args.command == "infer-survival":
        from .survival import infer_survival
        run_dirs = {variant: getattr(args, f"{variant}_run") for variant in
                    ("image", "image_age", "image_resection", "image_age_resection")}
        infer_survival(args.config, args.patients, args.cache, run_dirs, args.output)
    elif args.command == "summarize-survival-table":
        from .survival import summarize_survival_table
        summarize_survival_table(args.input, args.output, args.seed, args.resamples)
    elif args.command == "explain-segmentation":
        from .explainability import explain_test
        explain_test(args.config, args.analysis, args.patients, args.root, args.run, args.predictions,
                     args.inference_manifest, args.output, args.artifacts, args.resume)
    elif args.command == "randomize-segmentation":
        from .explainability import parameter_randomization
        parameter_randomization(args.config, args.analysis, args.patients, args.root, args.run, args.output, args.resume)
    elif args.command == "summarize-shap":
        from .explainability import summarize_table9
        summarize_table9(args.shap, args.segmentation, args.randomization, args.output, args.seed, args.resamples)
    elif args.command == "generate-figures":
        from .reporting import generate_figures
        generate_figures(args.patients, args.root, args.csfinet_predictions, args.shap_artifacts,
                         args.survival_predictions, args.csfinet_development, args.unet_development, args.output)
    elif args.command == "audit-delivery":
        from .delivery import audit_delivery
        audit_delivery(args.project_root, args.tag, args.output)
    elif args.command == "generate-schematics":
        from .schematics import generate_schematics
        generate_schematics(args.config, args.patients, args.root, args.output)
    elif args.command == "package-release":
        from .release import create_release
        create_release(args.project_root, args.tag, args.output)
    elif args.command == "validate-release":
        from .release import validate_release
        result = validate_release(args.output, args.tag, args.split_sha256)
        print(f"Validated release at commit {result['git_commit']}")
    elif args.command == "validate-inference-variants":
        from .inference_variants import validate_variants
        validate_variants(args.config, args.patients, args.root, args.runs, args.output, args.tag, args.seed,
                          args.resamples, args.calibration_patients)
    elif args.command == "evaluate-inference-variants":
        from .inference_variants import evaluate_test_variants
        evaluate_test_variants(args.config, args.patients, args.root, args.runs, args.results, args.artifacts,
                               args.decision, args.output, args.tag, args.seed, args.resamples)
    elif args.command == "smoke-survival":
        from .survival import smoke_survival
        smoke_survival(args.config, args.patients, args.root, args.output)
    else:
        result = summarize_survival(args.input, args.output, args.expected_patients, args.seed)
        print(f"Summarized {len(result['models'])} models; saved {args.output}")


if __name__ == "__main__":
    main()
