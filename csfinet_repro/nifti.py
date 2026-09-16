"""Download named raw BraTS volumes and apply manuscript volume normalization."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

from .data import PATIENT
from .download import unpack_download
from .files import read_csv, sha256, write_json

MODALITIES = ("flair", "t1", "t1ce", "t2")
DATASET = "awsaf49/brats20-dataset-training-validation"
# Some late-numbered patients use float32/float64 rather than int16 storage.
# 240*240*155*8 + a NIfTI header fits within this bounded allowance.
LIMIT = 80 * 1024 * 1024


def download_volume(patient, modality, destination):
    if not PATIENT.fullmatch(patient) or modality not in (*MODALITIES, "seg"):
        raise ValueError("Invalid patient or modality")
    name = f"{patient}_{modality}.nii"
    # Verified against Kaggle's file listing, retained in patient355-source-list.json.
    remote_name = "W39_1998.09.19_Segm.nii" if (patient, modality) == ("BraTS20_Training_355", "seg") else name
    remote = f"BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/{patient}/{remote_name}"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/{quote(remote, safe='')}?datasetVersionNumber=1"
    target = Path(destination) / patient / (name + ".gz")
    record_path = target.with_suffix(target.suffix + ".json")
    if target.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["gzip_sha256"] != sha256(target) or record["source_url"] != url:
            raise ValueError(f"Cached volume identity mismatch: {target}")
        return record
    if target.exists():
        raise ValueError(f"Volume exists without provenance: {target}")
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "CSFINet-reconstruction"}), timeout=90) as response:
                payload = response.read(LIMIT + 1)
            payload = unpack_download(payload, remote_name, LIMIT)
            if len(payload) < 352 or payload[344:348] != b"n+1\x00":
                raise ValueError("Expected a single-file NIfTI-1 image")
            packed = gzip.compress(payload, compresslevel=1, mtime=0)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(packed)
            temporary.replace(target)
            record = dict(patient_id=patient, modality=modality, dataset=DATASET, version=1,
                          source_url=url, nifti_sha256=hashlib.sha256(payload).hexdigest(),
                          gzip_sha256=hashlib.sha256(packed).hexdigest(),
                          nifti_bytes=len(payload), gzip_bytes=len(packed))
            write_json(record_path, record)
            return record
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(f"Download failed for {patient}/{modality}: {error}") from error
            time.sleep(2 ** attempt)


def download_cohort(patient_csv, destination, manifest, workers=3, limit=None):
    rows = read_csv(patient_csv, ["patient_id", "split"])
    ids = [row["patient_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not PATIENT.fullmatch(p) for p in ids):
        raise ValueError("Invalid or duplicated patient IDs")
    if not 1 <= workers <= 6 or (limit is not None and limit <= 0):
        raise ValueError("Workers must be 1..6 and limit must be positive")
    selected = ids if limit is None else ids[:limit]
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_volume, p, m, destination): (p, m)
                   for p in selected for m in (*MODALITIES, "seg")}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(dict(patient_id=futures[future][0], modality=futures[future][1], error=str(error)))
                print(f"Download failure: {futures[future]}: {error}", flush=True)
            if index % 25 == 0 or index == len(futures):
                print(f"NIfTI files {index}/{len(futures)}, successful={len(results)}, failed={len(errors)}", flush=True)
    report = dict(dataset=DATASET, version=1, split_sha256=sha256(patient_csv), requested_patients=len(selected),
                  complete=len(results) == len(selected) * 5, errors=errors,
                  files=sorted(results, key=lambda r: (r["patient_id"], r["modality"])))
    write_json(manifest, report)
    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed; rerun to resume; see {manifest}")
    return report


def normalize_volume(image, epsilon=1e-8):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("Expected finite 3D volume")
    mean, sd = image.mean(dtype=np.float64), image.std(dtype=np.float64, ddof=0)
    return ((image - mean) / (sd + epsilon)).astype(np.float32)


def load_patient(root, patient):
    import nibabel as nib

    if not PATIENT.fullmatch(patient):
        raise ValueError("Invalid patient ID")
    arrays, affine = [], None
    for modality in (*MODALITIES, "seg"):
        image = nib.load(Path(root) / patient / f"{patient}_{modality}.nii.gz")
        if image.shape != (240, 240, 155) or not np.allclose(image.header.get_zooms()[:3], [1, 1, 1]):
            raise ValueError(f"Unexpected shape/spacing: {patient}/{modality}")
        if affine is not None and not np.allclose(image.affine, affine):
            raise ValueError(f"Modalities have different affines: {patient}")
        affine = image.affine
        data = image.get_fdata(dtype=np.float32)
        if modality == "seg":
            if not np.isin(data, [0, 1, 2, 4]).all():
                raise ValueError(f"Unexpected segmentation labels: {patient}")
            target = data.astype(np.int64)
            target[target == 4] = 3
        else:
            arrays.append(normalize_volume(data))
    # Stored XYZ -> slices Z, channels C, X, Y; preserve native orientation consistently.
    return np.stack(arrays).transpose(3, 0, 1, 2).copy(), target.transpose(2, 0, 1).copy()


def audit_nifti(patient_csv, root, output):
    rows = read_csv(patient_csv, ["patient_id", "split"])
    if len(rows) != 235 or len({r["patient_id"] for r in rows}) != 235:
        raise ValueError("Expected the 235 unique frozen patients")
    records = []
    for index, row in enumerate(rows, 1):
        patient = row["patient_id"]
        hashes = {}
        for modality in (*MODALITIES, "seg"):
            path = Path(root) / patient / f"{patient}_{modality}.nii.gz"
            expected = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
            hashes[modality] = sha256(path)
            if hashes[modality] != expected["gzip_sha256"]:
                raise ValueError(f"Hash mismatch: {path}")
        images, targets = load_patient(root, patient)
        records.append(dict(patient_id=patient, split=row["split"], image_shape=list(images.shape),
                            target_shape=list(targets.shape), normalized_channel_means=images.mean(axis=(0, 2, 3), dtype=np.float64).tolist(),
                            normalized_channel_sd=images.std(axis=(0, 2, 3), dtype=np.float64).tolist(),
                            training_label_counts={str(label): int(np.count_nonzero(targets == label)) for label in range(4)},
                            gzip_sha256=hashes))
        del images, targets
        if index % 25 == 0 or index == len(rows):
            print(f"Validated MRI patients {index}/{len(rows)}", flush=True)
    report = dict(status="passed", patients=len(rows), files=len(rows) * 5, split_sha256=sha256(patient_csv),
                  checks=["per-file SHA256", "shape", "spacing", "within-patient affine agreement", "finite intensities",
                          "original labels 0,1,2,4", "whole-volume normalization"], records=records)
    write_json(output, report)
    return report
