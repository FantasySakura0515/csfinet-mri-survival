"""Survival input: WT mask plus four masked, volume-normalized MRI channels.

Training and internal validation use ground-truth WT masks. The 47 test patients
use frozen CSFINet predictions identified by the cache manifest.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .files import read_csv, sha256, write_json
from .nifti import MODALITIES, load_patient
from .survival import load_wt

CHANNELS = ("wt_mask", *(f"{modality}_in_wt" for modality in MODALITIES))
SHAPE = (len(CHANNELS), 155, 128, 128)


def build_mri_volume(images, mask):
    """images: (155, 4, 240, 240) normalized float32; mask: (1, 155, 128, 128) binary tensor."""
    images = np.asarray(images, dtype=np.float32)
    if images.shape != (155, 4, 240, 240) or not np.isfinite(images).all():
        raise ValueError("Expected 155 x 4 x 240 x 240 finite normalized slices")
    mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.shape != (1, 155, 128, 128) or not torch.isin(mask, torch.tensor([0.0, 1.0])).all():
        raise ValueError("Expected a binary 1 x 155 x 128 x 128 whole-tumor mask")
    intensities = F.interpolate(torch.from_numpy(images), size=(128, 128), mode="area").permute(1, 0, 2, 3)
    volume = torch.cat((mask, intensities * mask), dim=0)
    if volume.shape != SHAPE:
        raise ValueError("Unexpected survival input shape")
    return volume.to(torch.float16).numpy()


def save_mri_cache(volume, target):
    volume = np.asarray(volume, dtype=np.float16)
    if volume.shape != SHAPE or not np.isfinite(volume).all():
        raise ValueError("Survival MRI cache requires a finite 5 x 155 x 128 x 128 volume")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".part.npz")
    np.savez_compressed(temporary, volume=volume, shape=np.asarray(SHAPE, dtype=np.int16),
                        channels=np.asarray(CHANNELS))
    temporary.replace(target)
    return sha256(target)


def load_mri_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["shape"])
        volume = archive["volume"]
        channels = tuple(str(value) for value in archive["channels"])
    if shape != SHAPE or volume.shape != SHAPE or channels != CHANNELS:
        raise ValueError("Cached survival MRI volume has an unexpected shape or channel order")
    if not np.isfinite(volume).all() or not np.isin(volume[0], [0, 1]).all():
        raise ValueError("Cached survival MRI volume is corrupt")
    return torch.from_numpy(volume.astype(np.float32))


def create_survival_mri_cache(patient_csv, raw_root, prediction_root, segmentation_manifest, output):
    rows = read_csv(patient_csv, ["patient_id", "split"])
    if len(rows) != 235 or sum(row["split"] == "test" for row in rows) != 47:
        raise ValueError("Expected frozen 188/47 patient split")
    inference = json.loads(Path(segmentation_manifest).read_text(encoding="utf-8"))
    if (inference.get("status") != "completed" or inference.get("model") != "csfinet"
            or inference.get("split_sha256") != sha256(patient_csv)
            or set(inference.get("test_ids", [])) != {row["patient_id"] for row in rows if row["split"] == "test"}):
        raise ValueError("Completed CSFINet frozen-test inference manifest required")
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed survival MRI cache already exists")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for position, row in enumerate(rows, 1):
        patient = row["patient_id"]
        if row["split"] == "train":
            source = Path(raw_root) / patient / f"{patient}_seg.nii.gz"
            source_kind = "ground_truth_WT"
        else:
            source = Path(prediction_root) / patient / f"{patient}_csfinet_prediction.nii.gz"
            source_kind = "predicted_CSFINet_WT"
        mask = load_wt(source)
        images, _ = load_patient(raw_root, patient)
        volume = build_mri_volume(images, mask)
        target = output / f"{patient}.npz"
        if target.exists():
            if not np.array_equal(load_mri_cache(target).numpy(), volume.astype(np.float32)):
                raise ValueError(f"Existing partial cache differs from source: {target}")
            cache_sha = sha256(target)
        else:
            cache_sha = save_mri_cache(volume, target)
        modality_hashes = {modality: sha256(Path(raw_root) / patient / f"{patient}_{modality}.nii.gz")
                           for modality in MODALITIES}
        artifacts.append(dict(patient_id=patient, split=row["split"], source_kind=source_kind,
                              source_path=str(source), source_sha256=sha256(source), mri_sha256=modality_hashes,
                              cache_file=target.name, cache_sha256=cache_sha, foreground_voxels=int(mask.sum())))
        del images, mask, volume
        if position % 25 == 0 or position == len(rows):
            print(f"Survival MRI cache {position}/235", flush=True)
    manifest = dict(status="completed", patients=235, training_mask="ground_truth_WT",
                    test_mask="predicted_CSFINet_WT", input_kind="wt_mask_and_masked_mri",
                    channels=list(CHANNELS), mask_interpolation="nearest", intensity_interpolation="area",
                    normalization="per_patient_per_modality_full_volume_zscore_before_masking",
                    tensor_shape_NCDHW=[1, *SHAPE], dtype="float16_npz",
                    split_sha256=sha256(patient_csv), segmentation_manifest_sha256=sha256(segmentation_manifest),
                    artifacts=artifacts)
    write_json(output / "manifest.json", manifest)
    return manifest
