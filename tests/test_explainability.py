import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from csfinet_repro.explainability import (FixedRegionWTScore, attribution_metrics,
                                          gradient_shap, perturbation_curves,
                                          summarize_table9)


class PixelSegmenter(nn.Module):
    def forward(self, image):
        background = -image[:, :1]
        return torch.cat((background, image[:, :1], image[:, 1:2], image[:, 2:3]), dim=1)


def test_fixed_region_gradient_shap_and_perturbation_contract():
    model = PixelSegmenter().eval()
    image = torch.zeros(1, 4, 240, 240)
    image[:, 0, 100:140, 100:140] = 2
    region = torch.zeros(1, 240, 240, dtype=torch.bool)
    region[:, 100:140, 100:140] = True
    score = FixedRegionWTScore(model)(image, region)
    assert score.shape == (1,) and .9 < score.item() < 1
    attribution = gradient_shap(model, image, region, n_samples=4, stdev=0, seed=3)
    assert attribution.shape == image.shape and torch.isfinite(attribution).all()
    magnitude, metrics = attribution_metrics(attribution, region)
    assert metrics["lesion_attribution_mass"] == pytest.approx(1)
    assert metrics["pointing_accuracy"] == 1
    curves = perturbation_curves(model, image, region, magnitude, steps=4)
    assert len(curves["fractions"]) == 5
    assert curves["deletion"][0] == pytest.approx(score.item())
    assert curves["insertion"][-1] == pytest.approx(score.item())


def test_table9_requires_matching_full_patient_outputs(tmp_path):
    shap_path = tmp_path / "shap.csv"
    with shap_path.open("w", newline="", encoding="utf-8") as stream:
        columns = ["patient_id", "target", "eligible_slices", "lesion_attribution_mass",
                   "pointing_accuracy", "deletion_auc", "insertion_auc"]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for patient in range(47):
            for target_index, target in enumerate(("ground_truth_WT_region", "predicted_WT_region")):
                writer.writerow(dict(patient_id=f"p{patient:02}", target=target, eligible_slices=20,
                                     lesion_attribution_mass=.4 + patient / 1000 + target_index / 100,
                                     pointing_accuracy=patient % 2, deletion_auc=.3 + patient / 2000,
                                     insertion_auc=.7 - patient / 2000))
    segmentation_path = tmp_path / "segmentation.csv"
    with segmentation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient_id", "model", "region", "dice"], lineterminator="\n")
        writer.writeheader()
        for patient in range(47):
            writer.writerow(dict(patient_id=f"p{patient:02}", model="csfinet", region="WT", dice=.5 + patient / 100))
    randomization_path = tmp_path / "randomization.csv"
    with randomization_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient_id", "stage", "spearman_attribution_similarity"], lineterminator="\n")
        writer.writeheader()
        for patient in range(47):
            for index, stage in enumerate(("head", "decoder", "piu_swin", "encoder")):
                writer.writerow(dict(patient_id=f"p{patient:02}", stage=stage,
                                     spearman_attribution_similarity=.8 - index / 5 + patient / 10000))
    result = summarize_table9(shap_path, segmentation_path, randomization_path, tmp_path / "table9",
                              seed=4, n_resamples=200)
    assert result["targets"]["ground_truth_WT_region"]["lesion_attribution_mass"]["summary"]["n_valid"] == 47
    assert len(result["cumulative_parameter_randomization"]) == 4
    assert (tmp_path / "table9" / "table9.md").exists()
