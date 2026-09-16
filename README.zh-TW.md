# CSFINet-Based Multimodal MRI Image Segmentation and Survival Prediction

[English](README.md) · [重現指南](docs/reproduction.md) · [v2.0.1 模型權重](https://github.com/FantasySakura0515/csfinet-mri-survival/releases/tag/v2.0.1)

本專案實作 BraTS 2020 的兩階段研究流程：先以 CSFINet 或 U-Net 分割四模態 MRI，再將全腫瘤遮罩、四個遮罩內 MRI 通道及可選的臨床特徵用於生存天數迴歸。Gradient SHAP 與本機 Streamlit 檢視器提供結果檢查功能。

公開儲存庫包含程式碼、設定、僅含病人識別碼的資料切分定義，以及權重下載工具。模型權重放在同一儲存庫的 GitHub Release。MRI、臨床 CSV、逐病人預測與歸因圖須在本機取得或產生，不包含在公開儲存庫或權重壓縮檔內。

## 論文目前的結果

重建的研究群體共 235 人，分成 188 位訓練病人及 47 位測試病人。188 人再分成 150/38 人供訓練及內部驗證選模，最後用全部 188 人重新擬合模型。

| 分割模型 | 原始推論 WT Dice | 相同翻轉 TTA 的 WT Dice |
| --- | ---: | ---: |
| U-Net | 0.7373 | 0.7412 |
| CSFINet | 0.7625 | 0.7664 |

兩個模型採用相同資料切分及修訂後的訓練流程。TTA 由內部驗證資料選定，再同時套用至兩個模型。表中為逐病人全腫瘤 Dice 的平均值，不能直接視為對外部競賽方法的排名。

目前的 `v2mask-fixed` 生存分析使用四個既有 `v2mri` 最終模型，固定權重後換入原始 v2 CSFINet 測試遮罩。Image + Age + Resection Status 的 MAE 為 **256.50 天**，是四個神經網路配置中最低的點估計；僅使用年齡的普通最小平方法為 **254.54 天**。八項神經網路與參考模型的誤差比較在 Holm 校正後均未達顯著。這是使用同一批 47 位測試病人的事後輸入替換分析，不是重新訓練，也不是獨立驗證。

本專案重建可查核的研究流程，並未恢復原始論文已缺失的歷史切分、權重或訓練執行環境。重新訓練或使用不同數值環境，不保證產生逐位元相同的權重。本模型供研究使用，尚未經臨床驗證。

## 安裝

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

## 取得資料及權重

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

## 評估與檢視

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

## 測試

```bash
python -m pytest -q
```

測試用於檢查程式行為與驗證規則。若要重現論文的群體指標，仍須另外取得資料及權重。

## 來源與授權狀態

論文作者、原始 CSFINet 引用、實作來源與套件授權說明見 [NOTICE.md](NOTICE.md)。本專案公開研究程式與權重，目前尚未選定開源再利用授權。GitHub 引用功能使用 [CITATION.cff](CITATION.cff)。
