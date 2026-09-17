"""Validate source records without guessing split, modality order or patient IDs."""

from collections import Counter, defaultdict
from importlib.metadata import version
import math
from pathlib import Path
import platform
import re

import h5py
import numpy as np

from .files import read_csv, sha256, write_csv, write_json

PATIENT = re.compile(r"BraTS20_Training_\d{3}")
SLICE = re.compile(r"volume_(\d+)_slice_(\d+)\.h5")


def build_cohort(survival_path, mapping_path):
    """Candidate inclusion rule, explicitly not recovery of the author's cohort."""
    mappings = read_csv(mapping_path, ["BraTS_2020_subject_ID"])
    mapping_ids = [r["BraTS_2020_subject_ID"].strip() for r in mappings]
    if len(set(mapping_ids)) != len(mapping_ids):
        raise ValueError("Duplicate patient ID in name_mapping.csv")
    known_ids = set(mapping_ids)
    rows = read_csv(survival_path, ["Brats20ID", "Age", "Survival_days", "Extent_of_Resection"])
    seen, candidates, excluded = set(), [], []
    for row in rows:
        patient = row["Brats20ID"].strip()
        if not PATIENT.fullmatch(patient) or patient in seen:
            raise ValueError(f"Malformed or duplicate clinical patient ID: {patient}")
        seen.add(patient)
        raw_age = row["Age"].strip()
        raw_survival = row["Survival_days"].strip()
        status = row["Extent_of_Resection"].strip()
        reasons = []
        if patient not in known_ids:
            reasons.append("missing_from_name_mapping")
        if status not in {"GTR", "STR", "NA"}:
            reasons.append("unrecognized_resection_status")
        try:
            age = float(raw_age)
            if not math.isfinite(age) or age <= 0:
                raise ValueError
        except ValueError:
            reasons.append("invalid_age")
        try:
            survival = float(raw_survival)
            if not math.isfinite(survival) or survival <= 0:
                raise ValueError
        except ValueError:
            reasons.append("non_numeric_or_nonpositive_survival")
        if reasons:
            excluded.append(dict(patient_id=patient, age_raw=raw_age,
                                 survival_raw=raw_survival, resection_raw=status,
                                 exclusion_reason=";".join(reasons)))
        else:
            candidates.append(dict(patient_id=patient, age=age, survival_days=survival,
                                   resection_status=status, split="unassigned",
                                   hdf5_volume="", mapping_status="unverified"))
    return sorted(candidates, key=lambda r: r["patient_id"]), sorted(excluded, key=lambda r: r["patient_id"])


def validate_slice_metadata(path):
    rows = read_csv(path, ["slice_path", "target", "volume", "slice"])
    by_volume = defaultdict(set)
    for row in rows:
        volume, index = int(row["volume"]), int(row["slice"])
        if volume < 1 or index not in range(155):
            raise ValueError(f"Invalid volume/slice: {volume}/{index}")
        name = Path(row["slice_path"].replace("\\", "/")).name
        match = SLICE.fullmatch(name)
        if not match or tuple(map(int, match.groups())) != (volume, index):
            raise ValueError(f"Path and metadata disagree: {name}")
        if index in by_volume[volume]:
            raise ValueError(f"Duplicate slice: {volume}/{index}")
        if row["target"] not in {"0", "1"}:
            raise ValueError(f"Unexpected metadata target: {row['target']}")
        by_volume[volume].add(index)
    for volume, indices in by_volume.items():
        if indices != set(range(155)):
            raise ValueError(f"Volume {volume} does not contain all 155 slice indices")
    if not by_volume:
        raise ValueError("Empty slice metadata")
    return by_volume


def load_hdf5_slice(path):
    """Return stored HWC arrays unchanged. No semantic reordering/normalization."""
    with h5py.File(path, "r") as file:
        if not {"image", "mask"}.issubset(file.keys()):
            raise ValueError(f"{path}: expected image and mask datasets")
        if file["image"].shape != (240, 240, 4) or file["mask"].shape != (240, 240, 3):
            raise ValueError(f"{path}: unexpected stored slice shape")
        image, mask = file["image"][:], file["mask"][:]
    if not np.isfinite(image).all():
        raise ValueError(f"{path}: non-finite MRI values")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError(f"{path}: non-binary mask channels")
    return image, mask


def decode_exclusive_mask(mask, channel_labels):
    """Call only after external evidence establishes each channel's label."""
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.shape[-1] != 3 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected an HWC binary three-channel mask")
    labels = tuple(channel_labels)
    if len(labels) != 3 or set(labels) != {1, 2, 4}:
        raise ValueError("Provide an explicit permutation of BraTS labels 1, 2, 4")
    if np.any(mask.sum(axis=-1) > 1):
        raise ValueError("Overlapping channels cannot be decoded as exclusive tissue classes")
    return np.sum(mask * np.asarray(labels, dtype=np.uint8), axis=-1).astype(np.uint8)


def inspect_hdf5(path):
    image, mask = load_hdf5_slice(path)
    with h5py.File(path, "r") as file:
        attributes = {"root": sorted(file.attrs),
                      "image": sorted(file["image"].attrs), "mask": sorted(file["mask"].attrs)}
    return dict(file=Path(path).name, sha256=sha256(path), image_shape=list(image.shape),
                mask_shape=list(mask.shape), image_dtype=str(image.dtype), mask_dtype=str(mask.dtype),
                channel_mean=image.mean(axis=(0, 1)).tolist(), channel_sd=image.std(axis=(0, 1)).tolist(),
                image_min=float(image.min()), image_max=float(image.max()),
                mask_channel_pixels=mask.sum(axis=(0, 1)).tolist(),
                overlapping_pixels=int((mask.sum(axis=-1) > 1).sum()),
                attribute_names=attributes,
                modality_order="unverified", mask_channel_semantics="unverified",
                patient_mapping="unverified")


def audit_sources(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    candidates, excluded = build_cohort(source_dir / "survival_info.csv", source_dir / "name_mapping.csv")
    volumes = validate_slice_metadata(source_dir / "meta_data.csv")
    samples, found = [], defaultdict(set)
    for path in sorted(source_dir.rglob("*.h5")):
        match = SLICE.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected HDF5 filename: {path.name}")
        volume, index = map(int, match.groups())
        if volume not in volumes or index not in volumes[volume] or index in found[volume]:
            raise ValueError(f"Unknown or duplicate local slice: {path.name}")
        samples.append(inspect_hdf5(path))
        found[volume].add(index)
    report = dict(cohort_status="eligible_cohort", candidate_count=len(candidates),
                  excluded_count=len(excluded), exclusion_details=excluded,
                  candidate_status_counts=dict(sorted(Counter(r["resection_status"] for r in candidates).items())),
                  metadata_volumes=len(volumes), metadata_slices=sum(map(len, volumes.values())),
                  local_hdf5_slices=len(samples), local_complete_volumes=sum(len(v) == 155 for v in found.values()),
                  hdf5_samples=samples,
                  blockers=["Patient partition must be applied from the study specification", "HDF5 volume to patient mapping unverified",
                            "Modality order and mask channel semantics unverified",
                            "Stored normalization may differ from manuscript"],
                  sources={name: sha256(source_dir / name) for name in
                           ["survival_info.csv", "name_mapping.csv", "meta_data.csv"]},
                  environment=dict(python=platform.python_version(), platform=platform.platform(),
                                   packages={name: version(name) for name in ["numpy", "scipy", "h5py"]}),
                  code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    write_csv(output_dir / "cohort_candidates.csv", candidates,
              ["patient_id", "age", "survival_days", "resection_status", "split", "hdf5_volume", "mapping_status"])
    write_csv(output_dir / "cohort_excluded.csv", excluded,
              ["patient_id", "age_raw", "survival_raw", "resection_raw", "exclusion_reason"])
    write_json(output_dir / "audit.json", report)
    return report
