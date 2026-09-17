# CSFINet-Based Multimodal MRI Image Segmentation and Survival Prediction

**[English](#english) | [繁體中文](#繁體中文)**

[Website / 線上展示](http://csfi-bmmissp.know-eng.net/) · [Model weights / 模型權重](https://github.com/FantasySakura0515/csfinet-mri-survival/releases/tag/v2.0.2) · [Reproduction guide / 重現指南](docs/reproduction.md)

---

## English

[Results](#reported-results) · [Install](#install) · [Data and weights](#get-the-data-and-weights) · [Evaluate](#evaluate-and-inspect) · [繁體中文 ↓](#繁體中文)

**Online demo:** [CSFINet MRI and survival research website](http://csfi-bmmissp.know-eng.net/) — inspect MRI segmentation, survival predictions and Gradient SHAP results in your browser.

This repository accompanies a study of segmentation-guided multimodal representation learning for brain tumor delineation and survival estimation using BraTS 2020. The proposed analytical framework comprises two sequential stages: voxel-level tumor segmentation from four MRI modalities, followed by patient-level survival regression integrating tumor-restricted imaging representations with clinical covariates.

The segmentation stage adopts CSFINet to combine convolutional feature extraction with hierarchical Swin Transformer representations. Its Patch-wise Interaction Unit (PIU) facilitates information exchange among spatially corresponding feature patches across resolutions, while its Feature-wise Interaction Unit (FIU) performs learned flow-based alignment during feature reconstruction. Together, these operations provide a mechanism for integrating local spatial detail and contextual information across scales. U-Net serves as the comparator under a shared data partition and training protocol; [implementation details](docs/methods.md) document the architecture and reconstruction choices.

For survival estimation, a three-dimensional convolutional encoder processes a five-channel representation comprising the whole-tumor (WT) mask and four tumor-masked MRI volumes. Four configurations evaluate the incorporation of age and resection status alongside imaging features. Gradient SHAP characterizes attribution distributions within fixed WT regions for the segmentation model, and the research viewer supports joint inspection of segmentation, survival estimates and attribution maps.

The release provides source code, experiment configurations, a patient-identifier partition specification and pretrained checkpoints with recorded provenance. Model weights are distributed through the repository's GitHub Release. MRI volumes, individual clinical records, patient-level predictions and attribution arrays are acquired or generated locally and are excluded from the distributed artifacts.

### Reported results

Evaluation was conducted on a reconstructed cohort of 235 patients using a fixed patient-level partition of 188 training and 47 test cases. Within the training cohort, 150 patients were allocated to development training and 38 to internal validation for model selection. The selected training duration was subsequently used to refit each model on all 188 training patients. Segmentation performance is summarized by the mean patient-level WT Dice coefficient.

| Segmentation model | Raw WT Dice | Matched flip-TTA WT Dice |
| --- | ---: | ---: |
| U-Net | 0.7373 | 0.7412 |
| CSFINet | 0.7625 | 0.7664 |

Under the shared evaluation protocol, CSFINet achieved higher WT Dice than U-Net in both raw inference and matched test-time augmentation (TTA). The raw-inference paired mean difference was **0.0253** (95% BCa confidence interval: **0.0150–0.0402**; Holm-adjusted Wilcoxon *p* = **3.57 × 10⁻⁵**). The TTA configuration was selected on the internal validation partition and applied identically to both models. These findings support a segmentation advantage over the implemented U-Net comparator within the evaluated cohort and protocol. [Aggregate results and paired comparisons](results/reference/) provide the corresponding statistical summaries.

The `v2mask-fixed` survival analysis constitutes a post-hoc evaluation of segmentation-derived input substitution: the four `v2mri` refit models and their fitted clinical transformations were held fixed while raw v2 CSFINet test masks and the corresponding masked MRI channels were substituted. The Image + Age + Resection Status configuration yielded the lowest MAE among the four neural models (**256.50 days**); age-only ordinary least squares yielded **254.54 days**. None of the eight neural-versus-reference error comparisons reached statistical significance after Holm correction. The incremental predictive value of the evaluated imaging–clinical combinations therefore remains unestablished relative to these reference models. This analysis reuses the previously inspected 47-patient test partition and is interpreted as exploratory evidence.

The repository implements a documented reconstruction of the study protocol; the earlier manuscript's unavailable historical partition, checkpoints and execution environment have not been recovered. Reproducibility is supported through explicit configurations, fixed partition identities and checkpoint verification, although retraining across numerical environments does not guarantee bitwise equivalence. The present findings establish an internally evaluated research implementation; external generalizability and clinical utility remain to be assessed.

### Install

Python 3.10 is the tested Python version. Run commands from the repository root in a virtual environment.

```bash
git clone https://github.com/FantasySakura0515/csfinet-mri-survival.git
cd csfinet-mri-survival
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Install PyTorch 2.7.1 and torchvision 0.22.1 using the appropriate CPU/CUDA command from the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v271), then install the project:

```bash
python -m pip install -e ".[all]"
```

The release evaluator accepts CPU or CUDA. Full-cohort segmentation and SHAP can be slow on CPU; the supplied training commands require CUDA.

### Get the data and weights

Two version-pinned Kaggle sources serve different purposes:

| Source | Version | Used for |
| --- | --- | --- |
| [Brain Tumor Segmentation (BraTS2020)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) | 3 | Clinical/source CSVs, patient-name mapping and source checks |
| [BraTS20 Dataset: Training and Validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) | 1 | Native NIfTI MRI and segmentation labels |

Access the datasets under their applicable terms. If Kaggle requires authentication or acceptance of terms, complete that step through Kaggle. The small source files and native MRI come from different releases; an HDF5-only download does not replace the required NIfTI data.

```bash
python scripts/prepare_data.py --download-source
python -m csfinet_repro download-nifti --workers 3
python scripts/download_weights.py --component all
python scripts/download_weights.py --verify-only
```

`prepare_data.py` reconstructs the clinical cohort locally from the downloaded CSVs and checks the recorded source hashes and patient-ID split. It does not download the full MRI dataset. For already downloaded source files, use its default directory `data/raw/source-check/` and run `python scripts/prepare_data.py` without `--download-source`.

The weight downloader verifies the release manifest and extracts checkpoints into the repository's `runs/` directories. Use `--component segmentation` or `--component survival` for a partial download. Evaluation of survival also needs raw CSFINet masks generated by the segmentation stage.

### Evaluate and inspect

Run each stage in order:

```bash
python scripts/evaluate_release.py --stage segmentation --device auto
python scripts/evaluate_release.py --stage survival --device auto
python scripts/evaluate_release.py --stage shap --device auto
python -m streamlit run web.py
```

Alternatively, `python scripts/evaluate_release.py --stage all --device auto` runs all stages. The segmentation stage evaluates both raw and matched flip-TTA outputs. Survival uses the raw CSFINet masks; SHAP explains the raw CSFINet segmentation model.

**Weights and MRI alone are not sufficient to populate the viewer.** The evaluation stages generate the prediction masks, survival CSV and SHAP files it displays. Run all three stages for all views to be available. Full evaluation covers the frozen 47 test patients; `--limit 1` provides a segmentation or survival smoke run in separate `results/smoke/` and `artifacts/smoke/` directories. A smoke run is not a paper result, and the SHAP stage requires the full cohort.

See the [reproduction guide](docs/reproduction.md) for expected outputs, data checks, checkpoint provenance and training from scratch.

### Tests

```bash
python -m pytest -q
```

Tests check code behavior and validation rules. Reproducing the reported cohort metrics requires the separately downloaded data and weights.

### Attribution and license status

Project-authored code, documentation and configurations, together with the six project-trained model checkpoints, are distributed under the **[Apache License 2.0](LICENSE)**. Weight archives include the license and attribution notices. The grant covers project-contributor rights; datasets and third-party components retain their original terms. See [NOTICE.md](NOTICE.md) for the scope and provenance, [data-source licensing notes](docs/data-licensing.md) for dataset requirements, and [CITATION.cff](CITATION.cff) for scholarly citation information.

[繁體中文 ↓](#繁體中文) · [Back to English ↑](#english)

---

## 繁體中文

[研究結果](#論文目前的結果) · [安裝](#安裝) · [資料及權重](#取得資料及權重) · [評估與檢視](#評估與檢視) · [English ↑](#english)

**線上展示：** [CSFINet MRI 分割與生存預測研究網站](http://csfi-bmmissp.know-eng.net/)，可在瀏覽器查看 MRI 分割、生存預測與 Gradient SHAP 結果。

本儲存庫提供以 BraTS 2020 為研究基礎之分割引導多模態表徵學習實作，探討腦腫瘤區域辨識與病人生存天數估計。整體分析架構由兩個依序銜接的階段構成：首先利用四種 MRI 模態進行體素層級的腫瘤分割，再將腫瘤區域限定的影像表徵與臨床共變項整合，建立病人層級的生存迴歸模型。

分割階段採用 CSFINet，結合卷積特徵擷取與階層式 Swin Transformer 表徵。其區塊層級交互單元（Patch-wise Interaction Unit, PIU）促進不同解析度下、空間位置相對應之特徵區塊的資訊交換；特徵層級交互單元（Feature-wise Interaction Unit, FIU）則於特徵重建過程中，透過可學習流場進行空間對齊。上述機制用以整合局部空間細節與跨尺度上下文資訊。研究以 U-Net 作為比較模型，採取共同的資料切分與訓練流程；架構設定及重建選擇詳見[方法說明](docs/methods.md)。

生存估計階段以三維卷積編碼器處理五通道輸入，包含全腫瘤（whole tumor, WT）遮罩及四種腫瘤遮罩內的 MRI 影像，並透過四種配置評估年齡與切除狀態的納入方式。Gradient SHAP 用於分析分割模型在固定 WT 區域內的特徵歸因分布；研究檢視器則整合呈現分割結果、生存估計與歸因圖，以支援個案層級的結果檢視。

公開內容涵蓋原始程式碼、實驗設定、僅含病人識別碼的資料切分定義，以及具來源紀錄的預訓練模型。模型權重由同一儲存庫的 GitHub Release 提供；MRI 體積影像、個別臨床紀錄、逐病人預測及歸因陣列，均由使用者於本機取得或產生，不納入發布檔案。

### 論文目前的結果

本研究於重建之 235 位病人群體進行評估，採固定的病人層級切分，分別配置 188 位訓練病人與 47 位測試病人。訓練群體進一步劃分為 150 位開發訓練病人及 38 位內部驗證病人，據以選定訓練輪數，再以全部 188 位訓練病人重新擬合各模型。分割表現以逐病人 WT Dice 係數的平均值呈現。

| 分割模型 | 原始推論 WT Dice | 相同翻轉 TTA 的 WT Dice |
| --- | ---: | ---: |
| U-Net | 0.7373 | 0.7412 |
| CSFINet | 0.7625 | 0.7664 |

在共同評估流程下，CSFINet 於原始推論及採用相同測試時資料增強（test-time augmentation, TTA）的條件中，均取得較 U-Net 為高的 WT Dice。原始推論的配對平均差為 **0.0253**（95% BCa 信賴區間：**0.0150–0.0402**；Holm 校正後 Wilcoxon *p* = **3.57 × 10⁻⁵**）。TTA 配置由內部驗證資料選定，並一致套用於兩個模型。此結果支持 CSFINet 相較於本研究所實作 U-Net 的分割優勢，其推論範圍限定於所評估之研究群體與實驗流程。相關統計摘要詳見[彙整結果與配對比較](results/reference/)。

`v2mask-fixed` 生存分析採事後的分割衍生輸入替換設計：固定四個 `v2mri` 最終模型的權重與既有臨床特徵轉換參數，替換為原始 v2 CSFINet 測試遮罩，並重建相應的遮罩內 MRI 通道。Image + Age + Resection Status 配置取得四個神經網路模型中最低的 MAE（**256.50 天**）；僅使用年齡之普通最小平方法的 MAE 為 **254.54 天**。八項神經網路與參考模型的誤差比較，於 Holm 校正後均未達統計顯著。因此，所評估影像與臨床特徵組合相較於參考模型的增額預測效益，仍未獲確立。此分析沿用先前已檢視的 47 位測試病人，結果定位為探索性證據。

本儲存庫實作具明確紀錄的研究流程重建，原稿所對應之歷史資料切分、模型權重與執行環境仍未恢復。實驗可重現性透過明確設定、固定分組識別及權重校驗加以支援；跨數值環境重新訓練，仍不保證逐位元一致。目前成果為經內部評估的研究實作，其外部泛化能力與臨床效用尚待後續驗證。

### 安裝

已測試的 Python 版本為 3.10。請在儲存庫根目錄執行命令，並使用虛擬環境。

```bash
git clone https://github.com/FantasySakura0515/csfinet-mri-survival.git
cd csfinet-mri-survival
python -m venv .venv
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 啟用環境；Linux/macOS 使用 `source .venv/bin/activate`。先依照 [PyTorch 官方版本安裝表](https://pytorch.org/get-started/previous-versions/#v271)，選擇適合硬體的 CPU/CUDA 指令，安裝 PyTorch 2.7.1 及 torchvision 0.22.1，再安裝本專案：

```bash
python -m pip install -e ".[all]"
```

Release 評估工具支援 CPU 與 CUDA。完整分割及 SHAP 在 CPU 上可能相當慢；本專案提供的訓練命令需要 CUDA。

### 取得資料及權重

本研究使用兩個固定版本的 Kaggle 資料來源：

| 來源 | 版本 | 用途 |
| --- | --- | --- |
| [Brain Tumor Segmentation (BraTS2020)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) | 3 | 臨床與來源 CSV、病人名稱對照及資料檢查 |
| [BraTS20 Dataset: Training and Validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) | 1 | 原始 NIfTI MRI 與分割標註 |

請依各資料集的使用條款取得資料；若 Kaggle 要求登入或接受條款，請先在 Kaggle 完成。來源 CSV 與原始影像分屬不同資料集，只有 HDF5 切片無法取代所需的 NIfTI 影像。

```bash
python scripts/prepare_data.py --download-source
python -m csfinet_repro download-nifti --workers 3
python scripts/download_weights.py --component all
python scripts/download_weights.py --verify-only
```

`prepare_data.py` 會從下載的 CSV 在本機重建臨床資料與病人切分，並核對來源雜湊及固定病人識別碼。它不會下載整套 MRI。若已自行取得來源檔，請放在預設的 `data/raw/source-check/`，再執行不含 `--download-source` 的 `python scripts/prepare_data.py`。

權重工具會依 Release manifest 檢查檔案，解壓到儲存庫的 `runs/` 目錄。可用 `--component segmentation` 或 `--component survival` 分別下載。生存分析也需要先由分割階段產生原始 CSFINet 遮罩。

### 評估與檢視

依序執行：

```bash
python scripts/evaluate_release.py --stage segmentation --device auto
python scripts/evaluate_release.py --stage survival --device auto
python scripts/evaluate_release.py --stage shap --device auto
python -m streamlit run web.py
```

也可執行 `python scripts/evaluate_release.py --stage all --device auto` 一次完成三個階段。分割階段會評估原始推論及兩個模型相同的翻轉 TTA；生存與 SHAP 都使用原始 CSFINet 結果。

**只有權重與 MRI，還不足以在檢視器顯示完整結果。** 上述評估階段會在本機產生預測遮罩、生存 CSV 與 SHAP 檔案，檢視器再讀取這些產物。完整評估使用固定的 47 位測試病人；分割或生存階段可加 `--limit 1` 做單人試跑，輸出放在獨立的 `results/smoke/` 與 `artifacts/smoke/`。試跑不等於論文結果，SHAP 階段須使用完整群體。

詳細輸出位置、來源檢查、權重沿革及從頭訓練指令，請見[重現指南](docs/reproduction.md)。

### 測試

```bash
python -m pytest -q
```

測試用於檢查程式行為與驗證規則。若要重現論文的群體指標，仍須另外取得資料及權重。

### 來源與授權狀態

本專案自行撰寫的程式、文件與設定，以及六個自行訓練的模型權重，採 **[Apache License 2.0](LICENSE)** 發布。權重壓縮檔內附授權全文與來源聲明；授權範圍限於專案貢獻者所持有的權利，資料集與第三方元件仍適用各自原有條款。授權範圍與實作來源詳見 [NOTICE.md](NOTICE.md)，資料使用要求見[資料來源授權說明](docs/data-licensing.md)，學術引用資訊則由 [CITATION.cff](CITATION.cff) 提供。

[回到繁體中文 ↑](#繁體中文) · [English ↑](#english)
