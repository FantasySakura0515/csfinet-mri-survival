"""Independent, read-only viewer for locally supplied CSFINet research artifacts.

This release viewer is newly implemented; it does not import the legacy viewers.
No patient data, images, predictions or weights are downloaded by this application.
"""

import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(os.environ.get("CSFINET_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
SURVIVAL_MODELS = {
    "image": "Image only",
    "image_age": "Image + Age",
    "image_resection": "Image + Resection Status",
    "image_age_resection": "Image + Age + Resection Status",
    "age_ols": "Age-only OLS (reference)",
    "training_mean": "Training mean (reference)",
}


@st.cache_data(show_spinner=False, max_entries=16)
def read_table(path, modified):
    return pd.read_csv(path, keep_default_na=False)


def table(relative, required):
    path = ROOT / relative
    if not path.is_file():
        return None
    try:
        frame = read_table(str(path), path.stat().st_mtime_ns)
        if not set(required).issubset(frame.columns):
            raise ValueError("required columns are absent")
        return frame
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Cannot read {relative}: {error}")
        return None


@st.cache_data(show_spinner=False, max_entries=8)
def read_volume(path, modified):
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("expected a three-dimensional NIfTI volume")
    return data


def volume(path):
    if not path.is_file():
        return None
    try:
        return read_volume(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
        st.warning(f"Cannot read {path.name}: {error}")
        return None


def nifti(directory, stem):
    compressed = directory / f"{stem}.nii.gz"
    return compressed if compressed.is_file() else directory / f"{stem}.nii"


@st.cache_data(show_spinner=False, max_entries=8)
def attribution(path, modified, index):
    with np.load(path, allow_pickle=False) as stored:
        slices = stored["slice_z"]
        matches = np.flatnonzero(slices == index)
        if not len(matches):
            return None
        values = stored["attribution"][int(matches[0])].astype(np.float32)
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("expected four modality attribution maps")
    signed = values.sum(axis=0)
    maximum = float(np.max(np.abs(signed)))
    return signed / maximum if maximum else np.zeros_like(signed)


def panel(background, title, mask=None, signed=None):
    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    finite = background[np.isfinite(background)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    axis.imshow(background.T, cmap="gray", origin="lower", vmin=low, vmax=high)
    if signed is not None:
        axis.imshow(signed.T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1,
                    alpha=np.clip(np.abs(signed.T), 0, 1) * 0.85)
    if mask is not None and mask.any() and not mask.all():
        axis.contour(mask.T, levels=[0.5], colors=["#ffcc33"], linewidths=1)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def main():
    st.set_page_config(page_title="CSFINet research viewer", layout="wide")
    st.title("CSFINet MRI and survival research viewer")
    st.caption("Raw v2 segmentation · fixed-weight v2-mask survival · precomputed CSFINet GradientSHAP")
    st.info("Research outputs only. Survival estimates and attribution maps are not clinical advice.")
    cohort = table("data/manifests/cohort/patients.csv", ["patient_id", "split"])
    if cohort is None:
        st.info("Local data have not been configured. The public repository contains code and aggregate results; patient records and images are supplied separately.")
        st.code("CSFINET_PROJECT_ROOT=/path/to/local/project\nstreamlit run web.py")
        st.write("Expected cohort file: data/manifests/cohort/patients.csv")
        st.write("Images: data/raw/brats2020-nifti/<patient>/<patient>_<modality>.nii.gz")
        st.write("Predictions and attribution maps use the documented results/ and artifacts/ directories.")
        return
    patients = sorted({str(value) for value in cohort.loc[cohort["split"] == "test", "patient_id"]
                       if re.fullmatch(r"BraTS20_Training_\d{3}", str(value))})
    if not patients:
        st.warning("The local cohort contains no valid test-partition patient identifiers.")
        return
    if len(patients) != 47:
        st.warning(f"This local file contains {len(patients)} test patients; the reported study used 47.")
    patient = st.sidebar.selectbox("Test patient", patients)
    modality = st.sidebar.selectbox("MRI modality", ["flair", "t1", "t1ce", "t2"])
    models = st.sidebar.multiselect("Raw segmentation models", ["CSFINet", "U-Net"], default=["CSFINet", "U-Net"])
    raw = ROOT / "data/raw/brats2020-nifti" / patient
    image = volume(nifti(raw, f"{patient}_{modality}"))
    row = cohort.loc[cohort["patient_id"] == patient].iloc[0]
    fields = [key for key in ("age", "resection_status", "survival_days") if key in cohort.columns]
    if fields:
        st.dataframe(pd.DataFrame([row[fields].to_dict()]), hide_index=True)
    st.subheader("Survival estimates (days)")
    predictions = table("results/survival/v2mask-fixed/survival_predictions.csv", ["patient_id", "model", "y_pred"])
    if predictions is None:
        st.info("Current v2-mask survival predictions are not present locally.")
    else:
        available = predictions[predictions["patient_id"] == patient]
        output = []
        for key, label in SURVIVAL_MODELS.items():
            match = available[available["model"] == key]
            entry = {"Model": label, "Prediction (days)": "Unavailable", "Absolute error (days)": "Unavailable"}
            if len(match) == 1:
                item = match.iloc[0]
                entry["Prediction (days)"] = f"{float(item['y_pred']):.3f}"
                if "absolute_error" in match.columns:
                    entry["Absolute error (days)"] = f"{float(item['absolute_error']):.3f}"
            output.append(entry)
        st.dataframe(pd.DataFrame(output), hide_index=True)
        st.caption("Source: results/survival/v2mask-fixed/survival_predictions.csv. Changing the segmentation display does not change the survival inputs, which use CSFINet masks.")
    scores = []
    for model in ("csfinet", "unet"):
        metrics = table(f"results/segmentation/{model}-test-v2/segmentation_patient_metrics.csv", ["patient_id", "region", "dice"])
        if metrics is not None:
            selected = metrics[(metrics["patient_id"] == patient) & (metrics["region"] == "WT")]
            if len(selected) == 1:
                scores.append({"Model": "CSFINet" if model == "csfinet" else "U-Net", "Whole-volume WT Dice": float(selected.iloc[0]["dice"])})
    if scores:
        st.dataframe(pd.DataFrame(scores), hide_index=True)
    if image is None:
        st.info(f"The selected MRI is missing: {patient}_{modality}.nii[.gz]. Supply it locally to enable the slice and attribution views.")
        return
    index = st.sidebar.slider("Axial slice (zero-based)", 0, image.shape[2] - 1, image.shape[2] // 2)
    masks = {"Ground truth": volume(nifti(raw, f"{patient}_seg"))}
    for label in models:
        key = "csfinet" if label == "CSFINet" else "unet"
        directory = ROOT / "artifacts/segmentation" / f"{key}-test-v2" / patient
        masks[label] = volume(nifti(directory, f"{patient}_{key}_prediction"))
    columns = st.columns(len(masks))
    for column, (label, values) in zip(columns, masks.items()):
        with column:
            valid = values is not None and values.shape == image.shape
            panel(image[:, :, index], label, (values[:, :, index] > 0) if valid else None)
            if not valid:
                st.caption("Mask unavailable or shape mismatch; MRI only.")
    st.subheader("CSFINet GradientSHAP")
    st.caption("These attributions explain CSFINet, including when U-Net is displayed above. Exact saved slices only; positive contributions are red and negative contributions blue. Each signed four-modality sum is independently scaled to [-1, 1].")
    for column, target in zip(st.columns(2), ("ground_truth_WT_region", "predicted_WT_region")):
        with column:
            path = ROOT / "artifacts/explainability/v2" / patient / f"{target}.npz"
            if not path.is_file():
                st.info(f"No local attribution archive: {target}.")
                continue
            try:
                signed = attribution(str(path), path.stat().st_mtime_ns, index)
                if signed is None:
                    st.info(f"No saved attribution for slice {index}: {target}.")
                elif signed.shape != image.shape[:2]:
                    st.warning("Attribution and MRI shapes differ; overlay is unavailable.")
                else:
                    panel(image[:, :, index], target.replace("_", " "), signed=signed)
            except (OSError, ValueError, KeyError) as error:
                st.warning(f"Cannot read attribution: {error}")
    st.caption("WT is the union of nonzero segmentation labels. Attributions over a predicted region are self-region explanations, not independent ground-truth agreement.")


if __name__ == "__main__":
    main()
