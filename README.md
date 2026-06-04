# UAV Battery and Energy Prediction Using PX4 Flight Logs

This project builds a UAV battery and energy prediction pipeline using public PX4 Flight Review `.ulg` logs.

## Project Overview

The project focuses on predicting UAV energy consumption using battery and flight telemetry. The dataset was built from PX4 quadrotor logs and separated into normal and abnormal flight groups.

## Main Steps

1. Download PX4 public metadata
2. Filter quadrotor logs
3. Separate normal and abnormal logs
4. Download selected `.ulg` files
5. Validate battery telemetry
6. Convert ULog data into processed CSV format
7. Resample data to 10 Hz
8. Create energy prediction target
9. Perform EDA and leakage-safe train/validation/test splitting

## Important Note

The raw `.ulg` files and processed datasets are not included in this repository because they are large. The code can be used to reproduce the dataset creation pipeline.

## Main Files

- `px4_download_and_convert_pipeline.py`  
  Initial data download, validation, and conversion pipeline.

- `px4_incremental_download_process_clean.py`  
  Incremental download and expanded dataset cleaning pipeline.

- `uav_px4_dataset_project/notebooks/eda_and_data_split.ipynb`  
  Exploratory data analysis and leakage-safe train/test split notebook.

## Dataset

The final clean dataset used in the project contained approximately 1 million processed rows from PX4 quadrotor flight logs.

## Target Variable

The main target variable is:

```text
target_energy_next_60s_wh