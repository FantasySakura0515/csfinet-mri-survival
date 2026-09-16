"""Generate implementation-matched manuscript schematics and normalization figure."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import nibabel as nib
import numpy as np

from .files import read_csv, sha256, write_json

INK = "#172126"
BLUE = "#0072B2"
TEAL = "#009E73"
AMBER = "#E69F00"
PURPLE = "#7B61A8"
PALE = "#F3F5F4"


def _canvas(figsize=(12, 6), title=None):
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    if title:
        axis.set_title(title, fontsize=18, fontweight="bold", color=INK, pad=14)
    return figure, axis


def _box(axis, xy, width, height, text, color=BLUE, fontsize=10, alpha=.12):
    x, y = xy
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.015",
                           facecolor=color, edgecolor=color, linewidth=1.5, alpha=alpha)
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              fontsize=fontsize, color=INK, linespacing=1.3)
    return patch


def _arrow(axis, start, end, color=INK):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                   linewidth=1.35, color=color, shrinkA=3, shrinkB=3))


def _save(figure, output, stem, artifacts):
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        figure.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        artifacts.append(dict(path=path.name, sha256=sha256(path), bytes=path.stat().st_size))
    plt.close(figure)


def _pipeline(output, artifacts):
    figure, axis = _canvas(title="Fig. 1 · Reconstruction pipeline")
    items = [(.03, "FLAIR · T1\nT1ce · T2", BLUE), (.25, "CSFINet\n2D segmentation", PURPLE),
             (.47, "Whole-tumor\n3D mask", AMBER), (.69, "3D CNN +\nclinical fusion", TEAL),
             (.88, "Survival\n(days)", BLUE)]
    widths = [.14, .15, .14, .15, .09]
    for (x, text, color), width in zip(items, widths):
        _box(axis, (x, .48), width, .18, text, color)
    for left, right in zip(items[:-1], items[1:]):
        _arrow(axis, (left[0] + widths[items.index(left)], .57), (right[0], .57))
    _box(axis, (.70, .18), .13, .12, "Age\nResection", AMBER)
    _arrow(axis, (.765, .30), (.765, .48))
    axis.text(.5, .82, "188 training patients · 47 frozen test patients", ha="center", fontsize=11, color=INK)
    axis.text(.5, .08, "Development: 150 / 38 for epoch selection · Refit: all 188", ha="center", fontsize=10, color=INK)
    _save(figure, output, "fig1_pipeline", artifacts)


def _csfinet(config, output, artifacts):
    figure, axis = _canvas((13, 7), "Fig. 2 · Implemented CSFINet")
    channels = config["segmentation"]["channels"]
    xs = [.03, .17, .31, .45, .59]
    for level, (x, channel) in enumerate(zip(xs, channels), 1):
        _box(axis, (x, .68), .10, .12, f"CNN E{level}\n{channel} ch", BLUE)
        if level > 1:
            _arrow(axis, (xs[level - 2] + .10, .74), (x, .74))
    _box(axis, (.27, .30), .16, .13, "Swin blocks\ndepths 2 / 2 / 6", PURPLE)
    _box(axis, (.50, .30), .14, .13, "PIU\nd = 512", PURPLE)
    _box(axis, (.72, .46), .13, .16, "Multi-stage FIU\nshared flow +\ndual warp", AMBER)
    _box(axis, (.89, .475), .08, .13, "1×1 head\n4 logits", TEAL)
    _arrow(axis, (.64, .365), (.72, .50))
    _arrow(axis, (.43, .365), (.50, .365))
    _arrow(axis, (.64, .68), (.35, .43), color=PURPLE)
    axis.plot([.08, .08, .68, .68], [.65, .55, .55, .52], color="#667176", linewidth=1.4)
    axis.text(.36, .575, "CNN skips E1–E4", ha="center", va="bottom", fontsize=9, color="#586469")
    _arrow(axis, (.68, .52), (.72, .52), color="#667176")
    _arrow(axis, (.85, .54), (.89, .54))
    axis.text(.5, .06, "Four-class cross-entropy · one optimizer update per 155-slice patient", ha="center", fontsize=10)
    _save(figure, output, "fig2_csfinet", artifacts)


def _piu(output, artifacts):
    figure, axis = _canvas((12, 5.5), "Fig. 3 · Pyramid Interaction Unit")
    for y, label, windows, color in ((.68, "60×60", "4×4 → 16 tokens", BLUE),
                                     (.45, "30×30", "2×2 → 4 tokens", TEAL),
                                     (.22, "15×15", "1×1 → 1 token", AMBER)):
        _box(axis, (.05, y), .13, .12, label, color)
        _box(axis, (.25, y), .18, .12, windows, color)
        _arrow(axis, (.18, y + .06), (.25, y + .06))
        _arrow(axis, (.43, y + .06), (.53, .5))
    _box(axis, (.53, .41), .17, .18, "21 aligned tokens\nper local group", PURPLE)
    _box(axis, (.76, .41), .17, .18, "LN → MHA → MLP\nresidual blocks", PURPLE)
    _arrow(axis, (.70, .50), (.76, .50))
    axis.text(.845, .25, "Split 16 / 4 / 1\nand restore each scale", ha="center", va="center", fontsize=10)
    _arrow(axis, (.845, .41), (.845, .31))
    _save(figure, output, "fig3_piu", artifacts)


def _normalization(patient_csv, raw_root, output, artifacts):
    rows = [row for row in read_csv(patient_csv, ["patient_id", "split", "development_split"])
            if row["development_split"] == "train"]
    patient = sorted(row["patient_id"] for row in rows)[0]
    figure, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    sources = []
    rng = np.random.default_rng(20260914)
    for column, modality in enumerate(("flair", "t1", "t1ce", "t2")):
        path = Path(raw_root) / patient / f"{patient}_{modality}.nii.gz"
        volume = nib.load(path).get_fdata(dtype=np.float32)
        flat = volume.reshape(-1)
        indices = rng.choice(len(flat), size=min(250000, len(flat)), replace=False)
        sample = flat[indices]
        normalized = (sample - float(volume.mean())) / (float(volume.std(ddof=0)) + 1e-8)
        axes[0, column].hist(sample, bins=100, color=BLUE, alpha=.82)
        axes[1, column].hist(normalized, bins=100, color=TEAL, alpha=.82)
        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("log")
        axes[0, column].set_title(modality.upper())
        axes[0, column].set_ylabel("Raw count (log)" if column == 0 else "")
        axes[1, column].set_ylabel("Z-score count (log)" if column == 0 else "")
        axes[1, column].set_xlabel("Intensity")
        sources.append(dict(path=str(path).replace("\\", "/"), sha256=sha256(path),
                            mean=float(volume.mean()), sd=float(volume.std(ddof=0))))
    figure.suptitle(f"Fig. 4 · Full-volume modality normalization · {patient}", fontsize=16, fontweight="bold")
    _save(figure, output, "fig4_normalization", artifacts)
    return patient, sources


def _fiu(output, artifacts):
    figure, axis = _canvas((12, 5.5), "Fig. 5 · Feature Interaction Unit")
    _box(axis, (.04, .62), .15, .14, "Encoder feature\n(high resolution)", BLUE)
    _box(axis, (.04, .25), .15, .14, "Decoder feature\n(low resolution)", TEAL)
    _box(axis, (.28, .43), .15, .15, "Shared flow\nprediction", PURPLE)
    _box(axis, (.52, .62), .14, .14, "Warp decoder\ntoward encoder", AMBER)
    _box(axis, (.52, .25), .14, .14, "Warp encoder\ntoward decoder", AMBER)
    _box(axis, (.75, .43), .17, .15, "Concatenate → CBR\ninteracted feature", TEAL)
    _arrow(axis, (.19, .69), (.28, .53)); _arrow(axis, (.19, .32), (.28, .48))
    _arrow(axis, (.43, .53), (.52, .69)); _arrow(axis, (.43, .48), (.52, .32))
    _arrow(axis, (.66, .69), (.75, .54)); _arrow(axis, (.66, .32), (.75, .47))
    axis.text(.5, .08, "One predicted flow field is used for both directed warps · align_corners=True", ha="center", fontsize=10)
    _save(figure, output, "fig5_fiu", artifacts)


def _survival_variants(output, artifacts):
    figure, axis = _canvas((13, 6), "Fig. 6 · Survival prediction variants")
    _box(axis, (.04, .56), .14, .16, "WT mask\n1×155×128×128", BLUE)
    _box(axis, (.25, .56), .14, .16, "3D CNN\n128 embedding", PURPLE)
    _arrow(axis, (.18, .64), (.25, .64))
    _box(axis, (.25, .20), .14, .14, "Clinical MLP\n32 → 16", AMBER)
    _box(axis, (.04, .20), .14, .14, "Age and/or\nGTR · STR · NA", AMBER)
    _arrow(axis, (.18, .27), (.25, .27))
    variants = ((.48, .72, "Image", "128"), (.48, .52, "Image + Age", "144"),
                (.48, .32, "Image + Resection", "144"), (.48, .12, "Image + Age + Resection", "144"))
    for x, y, name, width in variants:
        _box(axis, (x, y), .20, .12, f"{name}\nhead input {width}", TEAL)
        _arrow(axis, (.39, .64), (x, y + .06))
        if name != "Image":
            _arrow(axis, (.39, .27), (x, y + .06), color=AMBER)
        _box(axis, (.78, y), .13, .12, "Softplus\n≥ 0 days", BLUE)
        _arrow(axis, (.68, y + .06), (.78, y + .06))
    _save(figure, output, "fig6_survival_variants", artifacts)


def _heads(config, output, artifacts):
    figure, axis = _canvas((13, 5.5), "Fig. 7 · Implemented survival heads")
    image = config["survival"]["image_head"]
    fusion = config["survival"]["fusion_head"]
    for row, (title, widths, color) in enumerate((("Image-only", image, BLUE), ("Clinical fusion", fusion, TEAL))):
        y = .66 - row * .38
        axis.text(.04, y + .06, title, ha="left", va="center", fontsize=12, fontweight="bold")
        xs = np.linspace(.20, .87, len(widths))
        for index, (x, width) in enumerate(zip(xs, widths)):
            label = str(width) if index < len(widths) - 1 else "1\nSoftplus"
            _box(axis, (float(x), y), .08, .12, label, color)
            if index:
                _arrow(axis, (float(xs[index - 1]) + .08, y + .06), (float(x), y + .06))
        axis.text(.55, y - .06, "ReLU between hidden layers · dropout 0.3 after first hidden layer",
                  ha="center", fontsize=9, color="#586469")
    _save(figure, output, "fig7_survival_heads", artifacts)


def generate_schematics(config_path, patient_csv, raw_root, output):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output)
    if output.exists():
        raise FileExistsError("Schematic output exists; use a new immutable directory")
    output.mkdir(parents=True)
    artifacts = []
    _pipeline(output, artifacts)
    _csfinet(config, output, artifacts)
    _piu(output, artifacts)
    patient, sources = _normalization(patient_csv, raw_root, output, artifacts)
    _fiu(output, artifacts)
    _survival_variants(output, artifacts)
    _heads(config, output, artifacts)
    manifest = dict(status="completed", scope="implementation_matched_figures_1_to_7",
                    config_sha256=sha256(config_path), split_sha256=sha256(patient_csv),
                    normalization_patient=patient, normalization_sources=sources,
                    artifacts=artifacts)
    write_json(output / "manifest.json", manifest)
    return manifest
