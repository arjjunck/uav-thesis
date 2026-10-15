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
| 7 | Resample to 10 Hz using forward-fill interpolation |
| 8 | Create energy prediction target variable (`target_energy_next_60s_wh`) |
| 9 | Run EDA and leakage-safe stratified group K-fold train/validation/test split |

**Final dataset:** 922,895 rows from 162 quadrotor flights (126 normal, 36 abnormal)

**Target variable:** `target_energy_next_60s_wh` — total energy consumed over the next 60 seconds (Wh), defined at each timestep as the sum over the following 600 rows at 10 Hz

### Data Cleaning Summary

| Cleaning stage | Rows remaining | Flights remaining |
|---------------|---------------|------------------|
| Raw dataset | ~1,020,000 | 172 |
| After high-energy outlier removal | 997,929 | 171 |
| After temporal gap removal | 971,822 | 168 |
| After impossible-voltage removal | ~961,125 | 166 |
| After high-power outlier removal | ~951,432 | 165 |
| After 12-cell aircraft removal | 922,895 | 163 |
| After zero-cell-count removal | 918,998 | 162 |

### Main Scripts

- `px4_download_and_convert_pipeline.py` — initial download, validation, and conversion
- `px4_incremental_download_process_clean.py` — incremental download and expanded dataset cleaning
- `uav_px4_dataset_project/notebooks/eda_and_data_split.ipynb` — EDA and leakage-safe train/test split

---

## Features

378 raw columns were reduced to 16 battery telemetry features through correlation analysis, mutual information scoring, and feature engineering.

| No | Feature | Description |
|----|---------|-------------|
| 1 | `battery_status__cell_count` | Number of battery cells |
| 2 | `battery_status__current_a` | Instantaneous current (A) |
| 3 | `battery_status__current_filtered_a` | Filtered current (A) |
| 4 | `battery_status__discharged_mah` | Cumulative charge discharged (mAh) |
| 5 | `battery_status__remaining` | Remaining charge fraction (0 to 1) |
| 6 | `battery_status__scale` | Battery capacity correction factor |
| 7 | `battery_status__voltage_filtered_v` | Filtered pack voltage (V) |
| 8 | `battery_status__voltage_v` | Raw pack voltage (V) |
| 9 | `cumulative_energy_wh` | Total energy consumed since takeoff (Wh) |
| 10 | `power_w` | Instantaneous power draw (W) |
| 11 | `time_s` | Elapsed flight time (s) |
| 12 | `is_idle` | Binary: 1 if instantaneous power < 10 W |
| 13 | `power_w_rolling_std_3s` | Rolling std of power over 3 seconds |
| 14 | `voltage_per_cell` | Pack voltage ÷ cell count (V per cell) |
| 15 | `voltage_filtered_per_cell` | Filtered pack voltage ÷ cell count (V per cell) |
| 16 | `voltage_per_cell_suspicious` | Binary: 1 if per-cell voltage outside safe LiPo range (2.5–4.35 V) |

**Key finding from feature selection:** `power_w` had a Pearson correlation of **0.948** with the target, confirming the task is largely linear in the dominant input. Raw pack voltage was excluded due to inconsistency across aircraft with different cell counts; per-cell normalisation was applied instead.

---

## Methodology

### Stage 1 — Anomaly Detection

Three autoencoder architectures were compared, all trained exclusively on normal-labelled flights. Input windows: 100 rows (10 seconds at 10 Hz), stride 20 during training, stride 10 at scoring time. Reconstruction MSE was used as the anomaly score. The threshold was set at the 95th percentile of training reconstruction scores.

| Model | Description |
|-------|-------------|
| LSTM-AE | Two-layer LSTM encoder/decoder, 32-dimensional latent vector |
| CNN-LSTM-AE | 1-D convolutional layers followed by LSTM encoder/decoder |
| Transformer-AE | Self-attention mechanism replacing LSTM layers |

**Primary metric:** AUPRC (due to class imbalance — ~25% abnormal flights; random baseline ≈ 0.25)

### Stage 2 — Energy Prediction

Five models tested in two configurations: baseline (16 features) and anomaly-aware (17 features, with Transformer-AE score appended). The anomaly score was assigned at flight level — a single value per flight assigned to every row in that flight.

| Model | Type | Input window |
|-------|------|-------------|
| Linear Regression | Classical | Row-level |
| Random Forest | Ensemble (300 trees, max depth 15) | Row-level |
| XGBoost | Gradient boosting (1000 estimators, depth 5, lr 0.01) | Row-level |
| LSTM | Recurrent (2 layers, 64 hidden units) | 50 rows (5 seconds) |
| BiLSTM | Bidirectional recurrent (2 layers, 64 hidden units) | 50 rows (5 seconds) |

**Evaluation:** MAE (primary), RMSE, R² across 5 stratified group K-folds. Statistical significance via Wilcoxon signed-rank test and Cohen's d. Feature importance via SHAP.

---

## Results

### Anomaly Detection

| Architecture | AUROC (mean ± σ) | AUPRC (mean ± σ) | Separation ratio (median) |
|-------------|-----------------|-----------------|--------------------------|
| **Transformer-AE** | **0.767 ± 0.144** | **0.652 ± 0.091** | **100×** |
| CNN-LSTM-AE | 0.591 ± 0.198 | 0.466 ± 0.112 | 30× |
| LSTM-AE | 0.533 ± 0.224 | 0.459 ± 0.120 | 4× |

The Transformer-AE outperformed both architectures on every metric and showed stronger inter-fold consistency. A Mann-Whitney U test confirmed statistically significant separation between normal and abnormal reconstruction scores (p < 0.0001 across all folds). Cohen's d = 0.324 (small-to-medium effect), reflecting wide within-group variance rather than model weakness.

### Energy Prediction — Baseline (16 features, no anomaly score)

| Model | MAE (Wh) | RMSE (Wh) | R² |
|-------|----------|-----------|-----|
| Linear Regression | 0.563 ± 0.154 | 1.077 ± 0.488 | 0.818 ± 0.109 |
| Random Forest | 0.815 ± 0.253 | 2.057 ± 0.820 | 0.126 ± 0.863† |
| XGBoost | 0.722 ± 0.271 | 1.780 ± 0.979 | 0.534 ± 0.289 |
| **LSTM** | **0.520 ± 0.248** | **1.173 ± 0.771** | **0.848 ± 0.048** |
| BiLSTM | 0.578 ± 0.285 | 1.246 ± 0.800 | 0.796 ± 0.103 |

† Random Forest R² includes fold_04 where RF produced R² = −1.467 due to atypical test-set composition (75% indistinguishable abnormal flights).

**Naive baselines for reference:** mean predictor MAE = 2.07 Wh, R² ≈ 0.33. Power persistence baseline MAE = 0.41 Wh, R² ≈ 0.85.

### Effect of Adding the Anomaly Score (17 features)

| Model | Baseline MAE (Wh) | +Anomaly Score MAE (Wh) | ΔMAE (Wh) | Direction |
|-------|-------------------|------------------------|-----------|-----------|
| Linear Regression | 0.563 ± 0.154 | 0.563 ± 0.154 | +0.0003 | Marginal improvement |
| Random Forest | 0.815 ± 0.253 | 0.892 ± 0.240 | −0.077 | Degraded (9.4% worse) |
| XGBoost | 0.722 ± 0.271 | 0.775 ± 0.236 | −0.052 | Degraded |
| LSTM | 0.520 ± 0.248 | 0.519 ± 0.246 | +0.002 | Negligible |
| **BiLSTM** | **0.578 ± 0.285** | **0.536 ± 0.230** | **+0.042** | **Modest improvement (7.2%)** |

BiLSTM showed the most consistent improvement. Statistical significance testing was limited by five folds: Wilcoxon signed-rank test p = 0.219, Cohen's d = modest. The result did not reach significance, likely due to small sample size. In a per-flight case study (flight d24243e7), adding the anomaly score reduced MAE by 58% by correcting systematic overprediction in the post-burst phase.

### Remaining Flight-Time Estimation

RFT was derived from energy prediction error rather than predicted directly:

```
RFT error (s) = MAE (Wh) ÷ Mean active-phase power (W) × 3600
```

Mean active-phase power varied substantially across folds (97.4 W in fold_04 to 284.8 W in fold_02), so a separate value was computed per fold.

| Model | Baseline RFT error (s) | +Anomaly Score RFT error (s) | Change |
|-------|----------------------|----------------------------|--------|
| Linear Regression | — | — | ~no change |
| Random Forest | — | — | higher (worse) |
| XGBoost | — | — | higher (worse) |
| LSTM | — | — | −0.2 s |
| **BiLSTM** | **10.74** | **9.98** | **−0.76 s** |

Note: RFT is a first-order linear approximation (assumes constant mean power). Ground truth remaining flight-time data were not available in the PX4 dataset, so this is an estimate only.

---

## Key Findings

**1. The Transformer Autoencoder was the best anomaly detector** (AUROC 0.767, AUPRC 0.652), outperforming CNN-LSTM-AE and LSTM-AE on all metrics with stronger cross-fold consistency.

**2. LSTM was the strongest energy predictor at baseline** (MAE 0.520 Wh, R² 0.848), with the lowest inter-fold variance of any model (±0.048).

**3. The anomaly score gave a modest improvement for BiLSTM** (7.2% MAE reduction), but the result was not statistically significant (p = 0.219), likely due to the small abnormal flight sample.

**4. 51% of abnormal flights were electrically indistinguishable from normal ones.** Their power draw, current, and per-cell voltage all fell within the normal range. These flights had been labelled abnormal due to GPS signal loss, motor imbalance, or unstable attitude control — issues that arose in the flight control layer and left no trace in battery telemetry. This is a fundamental limitation of reconstruction-error-based anomaly detection from electrical signals alone.

**5. Instantaneous power was the dominant feature across all models**, Pearson r = 0.948 with the target, confirmed by SHAP analysis.

**6. Tree-based models struggled most on abnormal flights** — Random Forest abnormal-to-normal MAE ratio was 4.13×, XGBoost 5.26×, versus LSTM 3.36× and BiLSTM 3.17×.

---

## Limitations

- Small abnormal flight sample (36 flights) limits statistical power for significance testing
- Anomaly labels sourced from PX4 Flight Review pilot ratings, not manually verified
- 51% of abnormal flights electrically indistinguishable — reconstruction error cannot detect anomalies that do not manifest in battery telemetry
- RFT estimation is a first-order approximation only — no ground truth flight-time data available
- Anomaly score is a per-flight constant, not a timestep-level signal

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
