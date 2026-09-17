"""Patient-level study partition and cohort summaries."""

import json
from pathlib import Path

import numpy as np

from .data import build_cohort
from .files import read_csv, sha256, write_csv, write_json
from .metrics import descriptive


def assign_split(candidates, seed, test_patients, validation_patients, validation_seed):
    patients = sorted(candidates, key=lambda row: row["patient_id"])
    ids = [row["patient_id"] for row in patients]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate patient ID")
    if not 0 < test_patients < len(ids) or not 0 < validation_patients < len(ids) - test_patients:
        raise ValueError("Invalid split sizes")
    test = set(np.random.default_rng(seed).permutation(ids)[:test_patients])
    training = sorted(set(ids) - test)
    validation = set(np.random.default_rng(validation_seed).permutation(training)[:validation_patients])
    return [dict(patient_id=row["patient_id"], age=row["age"], survival_days=row["survival_days"],
                 resection_status=row["resection_status"],
                 split="test" if row["patient_id"] in test else "train",
                 development_split=("test" if row["patient_id"] in test else
                                    "validation" if row["patient_id"] in validation else "train"))
            for row in patients]


def create_split(source, config_path, output):
    source, output = Path(source), Path(output)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    candidates, excluded = build_cohort(source / "survival_info.csv", source / "name_mapping.csv")
    if len(candidates) != 235:
        raise ValueError(f"Expected 235 eligible patients; found {len(candidates)}")
    rows = assign_split(candidates, config["seed"], **config["split"])
    target = output / "patients.csv"
    if target.exists():
        raise FileExistsError(f"Frozen split already exists: {target}; use a new protocol directory for changes")
    write_csv(target, rows, list(rows[0]))
    (output / "split-config.json").write_bytes(Path(config_path).read_bytes())
    report = dict(protocol=config["protocol"], patient_disjoint=True, seed=config["seed"],
                  **config["split"], train_patients=188, development_train_patients=150,
                  rng="numpy.random.default_rng/PCG64", numpy_version=np.__version__,
                  patient_csv_sha256=sha256(target), config_sha256=sha256(config_path),
                  source_sha256={name: sha256(source / name) for name in ["survival_info.csv", "name_mapping.csv"]},
                  excluded=excluded, selection="development train/validation only; refit all 188 for selected epoch count")
    write_json(output / "split.json", report)
    return report


def summarize_cohort(patient_csv, output):
    rows = read_csv(patient_csv, ["patient_id", "age", "survival_days", "resection_status", "split"])
    if len(rows) != 235 or len({r["patient_id"] for r in rows}) != 235:
        raise ValueError("Expected 235 unique patients")
    groups = {"all": rows, **{split: [r for r in rows if r["split"] == split] for split in ("train", "test")}}
    report = dict(kind="cohort_summary", split_sha256=sha256(patient_csv), groups={})
    for name, group in groups.items():
        report["groups"][name] = dict(n=len(group), age=descriptive(float(r["age"]) for r in group),
                                      survival_days=descriptive(float(r["survival_days"]) for r in group),
                                      resection_counts={s: sum(r["resection_status"] == s for r in group) for s in ("GTR", "STR", "NA")})
    write_json(output, report)
    return report
