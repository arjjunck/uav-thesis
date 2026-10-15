# Anomaly-Aware UAV Energy Prediction and Remaining Flight-Time Estimation Using PX4 Flight Telemetry

**MSc Data Science Thesis — Munster Technological University (MTU Cork), 2026**
**Author:** Arjun Krishna Krishnakumar

---

## Overview

Drones depend entirely on battery power. Running out mid-flight causes a crash. Accurate energy prediction can help prevent this, but most existing research only looks at normal operating conditions. This project asks a different question: **what happens when the drone is not behaving normally?**

The thesis builds a two-stage pipeline:

1. **Anomaly detection** — train an autoencoder on normal flights only and use its reconstruction error as an anomaly score
2. **Energy prediction** — test whether feeding that anomaly score into the energy model as an extra feature improves accuracy

The dataset is built entirely from public PX4 Flight Review `.ulg` logs, covering 162 real quadrotor flights split into normal and abnormal groups.

---

## Research Question

> Will integrating an anomaly score into a UAV energy prediction model improve its accuracy, particularly on abnormal flights?

---

## Dataset Pipeline

Raw `.ulg` files and processed datasets are not included in this repo due to file size. The full pipeline can be reproduced using the scripts below.

### Steps

| Step | Description |
|------|-------------|
| 1 | Download PX4 public metadata from Flight Review |
| 2 | Filter for quadrotor logs only |
| 3 | Separate normal and abnormal flight logs |
| 4 | Download selected `.ulg` files |
| 5 | Validate battery telemetry completeness |
| 6 | Convert ULog data to processed CSV format |
| 7 | Resample to 10 Hz |
| 8 | Create energy prediction target variable |
| 9 | Run EDA and leakage-safe train/validation/test split |

**Final dataset:** ~1 million processed rows from 162 quadrotor flights (126 normal, 36 abnormal)

**Target variable:** `target_energy_next_60s_wh` — energy consumed in the next 60 seconds (Wh)

### Main Scripts

- `px4_download_and_convert_pipeline.py` — initial download, validation, and conversion
- `px4_incremental_download_process_clean.py` — incremental download and expanded dataset cleaning
- `uav_px4_dataset_project/notebooks/eda_and_data_split.ipynb` — EDA and leakage-safe train/test split

---

## Methodology

### Stage 1 — Anomaly Detection

Three autoencoder architectures were compared, all trained exclusively on normal flights. Reconstruction error was used as the anomaly score.

| Model | Description |
|-------|-------------|
| LSTM-AE | LSTM-based autoencoder |
| CNN-LSTM-AE | Convolutional + LSTM autoencoder |
| Transformer-AE | Transformer-based autoencoder |

**Evaluation metrics:** AUROC and AUPRC across 5 stratified folds. AUPRC was the primary metric due to class imbalance (126 normal vs 36 abnormal flights).

### Stage 2 — Energy Prediction

Five models were tested in two configurations: baseline features only (16 features) and baseline + anomaly score (17 features).

| Model | Type |
|-------|------|
| Linear Regression | Classical |
| Random Forest | Ensemble |
| XGBoost | Gradient boosting |
| LSTM | Recurrent neural network |
| BiLSTM | Bidirectional recurrent neural network |

**Evaluation metrics:** MAE, RMSE, R² across 5 folds. Results split by normal vs abnormal flight subgroups. Statistical significance tested with the Wilcoxon signed-rank test and Cohen's d. SHAP used for feature importance across all five models.

---

## Results

### Anomaly Detection

| Model | AUROC | AUPRC |
|-------|-------|-------|
| LSTM-AE | — | — |
| CNN-LSTM-AE | — | — |
| **Transformer-AE** | **0.767** | **0.652** |

The Transformer Autoencoder was the best-performing anomaly detector across all folds.

### Energy Prediction — Baseline (no anomaly score)

| Model | MAE (Wh) | RMSE (Wh) | R² |
|-------|----------|-----------|-----|
| Linear Regression | 0.563 ± 0.154 | 1.077 ± 0.488 | 0.818 ± 0.109 |
| Random Forest | 0.815 ± 0.253 | 2.057 ± 0.820 | 0.126 ± 0.863 |
| XGBoost | 0.722 ± 0.271 | 1.780 ± 0.979 | 0.534 ± 0.289 |
| **LSTM** | **0.520 ± 0.248** | **1.173 ± 0.771** | **0.848 ± 0.048** |
| BiLSTM | 0.578 ± 0.285 | 1.246 ± 0.800 | 0.796 ± 0.103 |

LSTM had the best overall baseline performance. Tree-based models (Random Forest, XGBoost) struggled significantly on abnormal flights.

### Effect of Adding the Anomaly Score

| Model | Baseline MAE | +Anomaly Score MAE | Change |
|-------|-------------|-------------------|--------|
| BiLSTM | 0.578 | ~0.536 | **−7.2%** |
| LSTM | 0.520 | ~0.519 | −0.2% |
| Random Forest | 0.815 | higher | worse |
| XGBoost | 0.722 | higher | worse |
| Linear Regression | 0.563 | ~0.563 | no change |

BiLSTM showed the most improvement with the anomaly score added (7.2% MAE reduction). However, this did not reach statistical significance (Wilcoxon signed-rank test, p = 0.219, Cohen's d = modest).

### Remaining Flight-Time Estimation

Remaining flight time (RFT) was derived from energy prediction error rather than predicted directly:

```
RFT error (s) = MAE (Wh) ÷ Mean active-phase power (W) × 3600
```

| Model | Baseline RFT error | +Anomaly Score | Change |
|-------|-------------------|----------------|--------|
| BiLSTM | 10.74 s | 9.98 s | **−0.76 s** |
| LSTM | — | — | −0.2 s |
| Random Forest | — | — | higher |
| XGBoost | — | — | higher |

Note: RFT is a first-order estimate. Ground truth remaining flight-time data were not available in the PX4 dataset.

---

## Key Findings

**1. The Transformer Autoencoder was the best anomaly detector**, outperforming LSTM-AE and CNN-LSTM-AE on both AUROC and AUPRC.

**2. LSTM was the strongest energy predictor** at baseline, beating tree-based models and BiLSTM without the anomaly score.

**3. The anomaly score provided a modest improvement for BiLSTM** (7.2% MAE reduction) but the result was not statistically significant, likely due to the small abnormal flight sample size.

**4. 51% of abnormal flights were electrically indistinguishable from normal ones.** Their power draw looked the same even though something was wrong — motor degradation or sensor drift that did not manifest in battery current. This is a fundamental limitation of reconstruction-error-based anomaly detection from electrical telemetry alone.

**5. Instantaneous power was the dominant feature across all models**, with a Pearson correlation of r = 0.948 against the energy target. This was confirmed by SHAP analysis.

---

## Limitations

- Small abnormal flight sample (36 flights) limits statistical power
- Anomaly labels were sourced from PX4 Flight Review metadata, not manually verified
- RFT estimation is first-order only — no ground truth RFT data was available
- Reconstruction error from electrical signals may miss anomalies that do not affect power draw

---

## Tech Stack

- Python (pandas, numpy, scikit-learn, PyTorch)
- LSTM, BiLSTM, Transformer Autoencoder
- XGBoost
- SHAP
- PX4 ULog / pyulog
- Jupyter Notebooks

---

## Installation

```bash
git clone https://github.com/arjjunck/uav-thesis.git
cd uav-thesis
pip install -r requirements.txt
```

To reproduce the dataset pipeline, run the scripts in order:

```bash
python px4_download_and_convert_pipeline.py
python px4_incremental_download_process_clean.py
```

Then open the notebook for EDA and splitting:

```
uav_px4_dataset_project/notebooks/eda_and_data_split.ipynb
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
