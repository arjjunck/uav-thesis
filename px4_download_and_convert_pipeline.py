#!/usr/bin/env python3

"""
PX4 UAV Dataset Download + Validation + CSV Conversion Pipeline

This script:
1. Downloads PX4 public metadata
2. Filters real quadrotor logs
3. Separates normal and abnormal datasets
4. Applies duration and Simulation/HIL/SIH removal filters
5. Selects duration-aware download candidates
6. Downloads .ulg files safely
7. Validates actual .ulg contents
8. Converts selected topics to raw CSV
9. Builds processed 10 Hz CSV files for battery/energy prediction
"""

from pathlib import Path
import ast
import json
import time
import re
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyulog import ULog


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path("uav_px4_dataset_project")

METADATA_DIR = PROJECT_DIR / "metadata"
SELECTION_DIR = PROJECT_DIR / "selection_tables"
RAW_ULG_DIR = PROJECT_DIR / "raw_ulg"
RAW_CSV_DIR = PROJECT_DIR / "extracted_topic_csv"
PROCESSED_DIR = PROJECT_DIR / "processed_10hz"
VALIDATION_DIR = PROJECT_DIR / "validation_reports"

for folder in [
    METADATA_DIR,
    SELECTION_DIR,
    RAW_ULG_DIR,
    RAW_CSV_DIR,
    PROCESSED_DIR,
    VALIDATION_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


DBINFO_URL = "https://review.px4.io/dbinfo"
FALLBACK_DOWNLOAD_URL = "https://review.px4.io/download"

SAMPLING_RATE_HZ = 10
PREDICTION_HORIZON_SECONDS = 60
MIN_DURATION_SECONDS = 300

DELAY_SECONDS = 6
MAX_RETRIES = 5
REFRESH_METADATA = False

NORMAL_RATINGS = ["good", "great"]
ABNORMAL_RATINGS = ["unsatisfactory", "crash_sw_hw", "crash_pilot"]

MAV_TYPE_REQUIRED = "quadrotor"

SIMULATION_KEYWORDS = [
    "simulation",
    "hil",
    "sih",
    "hardware in the loop",
    "software in the loop",
]

DURATION_BINS = [300, 480, 720, 1200, float("inf")]
DURATION_LABELS = ["5-8 min", "8-12 min", "12-20 min", "20+ min"]

# Normal dataset: 225 download candidates
NORMAL_DOWNLOAD_QUOTA = {
    "5-8 min": 80,
    "8-12 min": 85,
    "12-20 min": 40,
    "20+ min": 20,
}

# Abnormal dataset: smaller evaluation subset
ABNORMAL_DOWNLOAD_QUOTA = {
    "5-8 min": 15,
    "8-12 min": 15,
    "12-20 min": 10,
    "20+ min": 5,
}

RANDOM_STATE = 42

TOPICS_TO_EXTRACT = [
    "battery_status",
    "vehicle_local_position",
    "vehicle_gps_position",
    "vehicle_attitude",
    "vehicle_status",
    "sensor_combined",
    "vehicle_angular_velocity",
    "vehicle_acceleration",
    "actuator_outputs",
    "actuator_motors",
    "vehicle_air_data",
    "airspeed",
]

AUXILIARY_TOPICS = [
    "vehicle_local_position",
    "vehicle_gps_position",
    "vehicle_attitude",
    "sensor_combined",
    "vehicle_angular_velocity",
    "vehicle_acceleration",
    "actuator_outputs",
    "actuator_motors",
    "vehicle_air_data",
    "airspeed",
]

MIN_AUXILIARY_TOPICS = 2

SAVE_RAW_TOPIC_CSVS = True
SAVE_PROCESSED_10HZ_CSVS = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_flight_duration_seconds(value):
    """
    Parses PX4 metadata flight_mode_durations.

    Expected examples:
    [[4, 300]]
    '[[4, 300], [3, 120]]'
    """
    if value is None:
        return np.nan

    if isinstance(value, float) and np.isnan(value):
        return np.nan

    if isinstance(value, (list, tuple, np.ndarray)):
        durations = value

    elif isinstance(value, str):
        value = value.strip()

        if value == "" or value.lower() in ["nan", "none", "null"]:
            return np.nan

        try:
            durations = ast.literal_eval(value)
        except Exception:
            try:
                durations = json.loads(value)
            except Exception:
                return np.nan
    else:
        return np.nan

    total_duration = 0.0

    try:
        for item in durations:
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                total_duration += float(item[1])
        return total_duration
    except Exception:
        return np.nan


def safe_filename(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text)
    return text


# ============================================================
# STEP 1: DOWNLOAD / LOAD METADATA
# ============================================================

def load_or_download_metadata():
    metadata_file = METADATA_DIR / "px4_public_logs_metadata.csv"

    if metadata_file.exists() and not REFRESH_METADATA:
        print("Loading existing metadata CSV...")
        logs_df = pd.read_csv(metadata_file)
    else:
        print("Downloading PX4 public metadata...")
        response = requests.get(DBINFO_URL, timeout=300)
        response.raise_for_status()

        logs_json = response.json()
        logs_df = pd.DataFrame(logs_json)

        logs_df.to_csv(metadata_file, index=False)
        print("Saved metadata to:", metadata_file)

    print("Metadata shape:", logs_df.shape)
    return logs_df


# ============================================================
# STEP 2: FILTER NORMAL AND ABNORMAL CANDIDATES
# ============================================================

def prepare_candidates(logs_df):
    df = logs_df.copy()

    df = df.dropna(subset=["log_id"]).copy()

    for col in [
        "log_id",
        "mav_type",
        "rating",
        "airframe_type",
        "airframe_name",
        "description",
        "feedback",
        "vehicle_name",
        "source",
        "download_url",
    ]:
        if col not in df.columns:
            df[col] = ""

        df[col] = df[col].apply(clean_text)

    df["rating_clean"] = df["rating"].str.lower()
    df["mav_type_clean"] = df["mav_type"].str.lower()

    df["metadata_duration_s"] = df["flight_mode_durations"].apply(
        parse_flight_duration_seconds
    )
    df["metadata_duration_min"] = df["metadata_duration_s"] / 60

    df["duration_bin"] = pd.cut(
        df["metadata_duration_s"],
        bins=DURATION_BINS,
        labels=DURATION_LABELS,
        right=False,
    )

    text_cols = [
        "airframe_name",
        "airframe_type",
        "description",
        "feedback",
        "vehicle_name",
    ]

    combined_text = ""

    for col in text_cols:
        combined_text = combined_text + " " + df[col].astype(str).str.lower()

    simulation_mask = combined_text.str.contains(
        "|".join(SIMULATION_KEYWORDS),
        na=False,
        regex=True,
    )

    base_filter = (
        (df["mav_type_clean"] == MAV_TYPE_REQUIRED)
        & (df["metadata_duration_s"] >= MIN_DURATION_SECONDS)
        & (~simulation_mask)
        & (df["duration_bin"].notna())
    )

    normal_candidates = df[
        base_filter & df["rating_clean"].isin(NORMAL_RATINGS)
    ].copy()

    abnormal_candidates = df[
        base_filter & df["rating_clean"].isin(ABNORMAL_RATINGS)
    ].copy()

    print("\nCandidate summary after filters:")
    print("Normal candidates:", len(normal_candidates))
    print("Abnormal candidates:", len(abnormal_candidates))

    normal_candidates.to_csv(SELECTION_DIR / "normal_candidates_all.csv", index=False)
    abnormal_candidates.to_csv(SELECTION_DIR / "abnormal_candidates_all.csv", index=False)

    return normal_candidates, abnormal_candidates


# ============================================================
# STEP 3: STRATIFIED SELECTION
# ============================================================

def sample_by_duration_quota(df, quota, dataset_label):
    selected_parts = []

    for bin_name, required_count in quota.items():
        bin_df = df[df["duration_bin"].astype(str) == bin_name].copy()

        available_count = len(bin_df)
        sample_count = min(required_count, available_count)

        if sample_count == 0:
            print(f"Warning: no logs available for {dataset_label} / {bin_name}")
            continue

        sampled = bin_df.sample(
            n=sample_count,
            random_state=RANDOM_STATE,
            replace=False,
        )

        selected_parts.append(sampled)

        print(
            f"{dataset_label} | {bin_name}: "
            f"selected {sample_count} from {available_count}"
        )

    if not selected_parts:
        return pd.DataFrame()

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    selected["dataset_label"] = dataset_label

    print(f"\n{dataset_label} selected logs:", len(selected))
    print(
        f"{dataset_label} estimated raw rows at {SAMPLING_RATE_HZ} Hz:",
        int(selected["metadata_duration_s"].sum() * SAMPLING_RATE_HZ),
    )

    return selected


# ============================================================
# STEP 4: DOWNLOAD .ULG FILES
# ============================================================

def get_download_url(row):
    download_url = clean_text(row.get("download_url", ""))

    if download_url and download_url.lower() not in ["nan", "none", "null"]:
        return download_url

    log_id = clean_text(row["log_id"])
    return f"{FALLBACK_DOWNLOAD_URL}?log={log_id}"


def download_with_retry(url, output_path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return True, "already_exists"

    headers = {
        "User-Agent": "PX4-UAV-thesis-dataset-script/1.0"
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=600,
                headers=headers,
            )

            if response.status_code == 200:
                with open(output_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            file.write(chunk)

                if output_path.exists() and output_path.stat().st_size > 0:
                    time.sleep(DELAY_SECONDS)
                    return True, "downloaded"

                time.sleep(DELAY_SECONDS)
                return False, "empty_file"

            if response.status_code in [429, 503]:
                wait_time = min(30 * (2 ** attempt), 300)
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            if response.status_code == 404:
                time.sleep(DELAY_SECONDS)
                return False, "not_found_404"

            time.sleep(DELAY_SECONDS)
            return False, f"http_status_{response.status_code}"

        except Exception as error:
            wait_time = min(10 * (attempt + 1), 60)
            print(f"Download error: {error}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)

    return False, "failed_after_retries"


def download_dataset(selection_df, dataset_label):
    dataset_dir = RAW_ULG_DIR / dataset_label
    dataset_dir.mkdir(parents=True, exist_ok=True)

    download_records = []

    for _, row in tqdm(selection_df.iterrows(), total=len(selection_df), desc=f"Downloading {dataset_label}"):
        log_id = clean_text(row["log_id"])
        output_path = dataset_dir / f"{log_id}.ulg"

        url = get_download_url(row)

        success, status = download_with_retry(url, output_path)

        file_size_mb = (
            output_path.stat().st_size / (1024 * 1024)
            if output_path.exists()
            else 0
        )

        download_records.append({
            "dataset_label": dataset_label,
            "log_id": log_id,
            "download_success": success,
            "download_status": status,
            "file_path": str(output_path),
            "file_size_mb": round(file_size_mb, 3),
            "metadata_duration_s": row["metadata_duration_s"],
            "duration_bin": row["duration_bin"],
            "rating": row["rating"],
        })

    download_df = pd.DataFrame(download_records)
    download_df.to_csv(
        VALIDATION_DIR / f"{dataset_label}_download_report.csv",
        index=False,
    )

    return download_df


# ============================================================
# STEP 5: VALIDATE .ULG CONTENTS
# ============================================================

def inspect_ulg_file(ulg_path, dataset_label):
    result = {
        "dataset_label": dataset_label,
        "log_id": ulg_path.stem,
        "file_name": ulg_path.name,
        "read_success": False,
        "error": "",
        "topics_found": "",
        "important_topic_count": 0,
        "has_battery_status": False,
        "has_voltage_v": False,
        "has_current_a": False,
        "has_timestamp": False,
        "has_remaining": False,
        "has_discharged_mah": False,
        "battery_rows": 0,
        "actual_battery_duration_s": 0.0,
        "estimated_rows_10hz": 0,
        "valid_for_processing": False,
    }

    try:
        ulog = ULog(str(ulg_path), message_name_filter_list=TOPICS_TO_EXTRACT)
        result["read_success"] = True

        topics = sorted(set([data.name for data in ulog.data_list]))
        result["topics_found"] = ", ".join(topics)
        result["important_topic_count"] = sum(topic in topics for topic in AUXILIARY_TOPICS)

        battery_dfs = []

        for data in ulog.data_list:
            if data.name == "battery_status":
                battery_df = pd.DataFrame(data.data)
                battery_df["multi_id"] = data.multi_id
                battery_dfs.append(battery_df)

        if not battery_dfs:
            return result

        battery = pd.concat(battery_dfs, ignore_index=True)

        result["has_battery_status"] = True
        result["battery_rows"] = len(battery)

        result["has_voltage_v"] = "voltage_v" in battery.columns
        result["has_current_a"] = "current_a" in battery.columns
        result["has_timestamp"] = "timestamp" in battery.columns
        result["has_remaining"] = "remaining" in battery.columns
        result["has_discharged_mah"] = "discharged_mah" in battery.columns

        if result["has_timestamp"] and len(battery) > 1:
            duration_s = (
                battery["timestamp"].max() - battery["timestamp"].min()
            ) / 1_000_000

            result["actual_battery_duration_s"] = max(float(duration_s), 0.0)
            result["estimated_rows_10hz"] = int(
                result["actual_battery_duration_s"] * SAMPLING_RATE_HZ
            )

        result["valid_for_processing"] = (
            result["has_battery_status"]
            and result["has_voltage_v"]
            and result["has_current_a"]
            and result["has_timestamp"]
            and result["battery_rows"] > 100
            and result["actual_battery_duration_s"] >= MIN_DURATION_SECONDS
            and result["important_topic_count"] >= MIN_AUXILIARY_TOPICS
        )

        return result

    except Exception as error:
        result["error"] = str(error)
        return result


def validate_downloaded_files(dataset_label):
    dataset_dir = RAW_ULG_DIR / dataset_label
    ulg_files = sorted(dataset_dir.glob("*.ulg"))

    validation_records = []

    for ulg_path in tqdm(ulg_files, desc=f"Validating {dataset_label}"):
        validation_records.append(inspect_ulg_file(ulg_path, dataset_label))

    validation_df = pd.DataFrame(validation_records)

    validation_df.to_csv(
        VALIDATION_DIR / f"{dataset_label}_ulg_validation_report.csv",
        index=False,
    )

    print(f"\n{dataset_label} validation summary:")
    print("Downloaded files:", len(validation_df))
    print("Valid files:", validation_df["valid_for_processing"].sum())

    return validation_df


# ============================================================
# STEP 6: RAW TOPIC CSV CONVERSION
# ============================================================

def extract_topics_to_csv(ulg_path, dataset_label):
    output_dir = RAW_CSV_DIR / dataset_label / ulg_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ulog = ULog(str(ulg_path), message_name_filter_list=TOPICS_TO_EXTRACT)

        saved_count = 0

        for data in ulog.data_list:
            topic_name = data.name
            multi_id = data.multi_id

            df = pd.DataFrame(data.data)

            if df.empty:
                continue

            df.insert(0, "log_id", ulg_path.stem)
            df.insert(1, "dataset_label", dataset_label)
            df.insert(2, "topic_name", topic_name)
            df.insert(3, "multi_id", multi_id)

            file_name = f"{safe_filename(topic_name)}__multi_{multi_id}.csv"
            output_path = output_dir / file_name

            df.to_csv(output_path, index=False)
            saved_count += 1

        return True, saved_count, ""

    except Exception as error:
        return False, 0, str(error)


def convert_valid_ulgs_to_raw_csv(validation_df, dataset_label):
    valid_logs = validation_df[validation_df["valid_for_processing"] == True].copy()

    records = []

    for _, row in tqdm(valid_logs.iterrows(), total=len(valid_logs), desc=f"Raw CSV conversion {dataset_label}"):
        ulg_path = RAW_ULG_DIR / dataset_label / f"{row['log_id']}.ulg"

        success, saved_count, error = extract_topics_to_csv(ulg_path, dataset_label)

        records.append({
            "dataset_label": dataset_label,
            "log_id": row["log_id"],
            "raw_csv_success": success,
            "topics_saved": saved_count,
            "error": error,
        })

    report_df = pd.DataFrame(records)
    report_df.to_csv(
        VALIDATION_DIR / f"{dataset_label}_raw_csv_conversion_report.csv",
        index=False,
    )

    return report_df


# ============================================================
# STEP 7: PROCESSED 10 HZ DATASET CREATION
# ============================================================

def resample_topic_df(df, topic_key, log_start_timestamp, base_index, rule):
    if "timestamp" not in df.columns or df.empty:
        return pd.DataFrame(index=base_index)

    df = df.copy()

    df = df[df["timestamp"] >= log_start_timestamp].copy()

    if df.empty:
        return pd.DataFrame(index=base_index)

    df["time_s"] = (df["timestamp"] - log_start_timestamp) / 1_000_000
    df["time_td"] = pd.to_timedelta(df["time_s"], unit="s")

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    numeric_df = numeric_df.drop(
        columns=["timestamp", "time_s", "multi_id"],
        errors="ignore",
    )

    if numeric_df.empty:
        return pd.DataFrame(index=base_index)

    numeric_df.index = df["time_td"]
    numeric_df = numeric_df.groupby(level=0).mean().sort_index()

    resampled = numeric_df.resample(rule).mean()

    resampled = resampled.reindex(base_index)
    resampled = resampled.interpolate(method="linear", limit_direction="both")
    resampled = resampled.ffill().bfill()

    resampled.columns = [f"{topic_key}__{col}" for col in resampled.columns]

    return resampled


def build_processed_10hz_for_log(ulg_path, dataset_label):
    try:
        ulog = ULog(str(ulg_path), message_name_filter_list=TOPICS_TO_EXTRACT)

        topic_dfs = {}

        for data in ulog.data_list:
            topic_name = data.name
            multi_id = data.multi_id
            topic_key = topic_name if multi_id == 0 else f"{topic_name}_{multi_id}"

            df = pd.DataFrame(data.data)
            df["multi_id"] = multi_id

            if df.empty:
                continue

            topic_dfs[topic_key] = df

        if "battery_status" not in topic_dfs:
            return False, pd.DataFrame(), "battery_status missing"

        battery = topic_dfs["battery_status"]

        required_cols = ["timestamp", "voltage_v", "current_a"]

        if not all(col in battery.columns for col in required_cols):
            return False, pd.DataFrame(), "battery_status missing timestamp/voltage/current"

        log_start_timestamp = battery["timestamp"].min()
        log_end_timestamp = battery["timestamp"].max()

        duration_s = (log_end_timestamp - log_start_timestamp) / 1_000_000

        if duration_s < MIN_DURATION_SECONDS:
            return False, pd.DataFrame(), "duration below minimum"

        rule_ms = int(round(1000 / SAMPLING_RATE_HZ))
        rule = f"{rule_ms}ms"

        number_of_steps = int(duration_s * SAMPLING_RATE_HZ) + 1
        base_index = pd.timedelta_range(
            start="0s",
            periods=number_of_steps,
            freq=rule,
        )

        resampled_frames = []

        for topic_key, df_topic in topic_dfs.items():
            resampled = resample_topic_df(
                df=df_topic,
                topic_key=topic_key,
                log_start_timestamp=log_start_timestamp,
                base_index=base_index,
                rule=rule,
            )

            if not resampled.empty:
                resampled_frames.append(resampled)

        if not resampled_frames:
            return False, pd.DataFrame(), "no resampled topics"

        merged = pd.concat(resampled_frames, axis=1)

        merged = merged.reset_index()
        merged = merged.rename(columns={"index": "time_td"})
        merged["time_s"] = merged["time_td"].dt.total_seconds()

        merged.insert(0, "log_id", ulg_path.stem)
        merged.insert(1, "dataset_label", dataset_label)

        voltage_col = "battery_status__voltage_v"
        current_col = "battery_status__current_a"

        if voltage_col not in merged.columns or current_col not in merged.columns:
            return False, pd.DataFrame(), "processed voltage/current missing"

        merged["power_w"] = merged[voltage_col] * merged[current_col].abs()

        merged["dt_s"] = 1 / SAMPLING_RATE_HZ
        merged["energy_increment_wh"] = merged["power_w"] * merged["dt_s"] / 3600
        merged["cumulative_energy_wh"] = merged["energy_increment_wh"].cumsum()

        horizon_steps = PREDICTION_HORIZON_SECONDS * SAMPLING_RATE_HZ

        merged["future_cumulative_energy_wh"] = merged["cumulative_energy_wh"].shift(
            -horizon_steps
        )

        merged["target_energy_next_60s_wh"] = (
            merged["future_cumulative_energy_wh"] - merged["cumulative_energy_wh"]
        )

        return True, merged, ""

    except Exception as error:
        return False, pd.DataFrame(), str(error)


def create_processed_dataset(validation_df, dataset_label):
    valid_logs = validation_df[validation_df["valid_for_processing"] == True].copy()

    processed_output_dir = PROCESSED_DIR / dataset_label
    processed_output_dir.mkdir(parents=True, exist_ok=True)

    combined_parts = []
    records = []

    for _, row in tqdm(valid_logs.iterrows(), total=len(valid_logs), desc=f"Processed 10Hz {dataset_label}"):
        log_id = row["log_id"]
        ulg_path = RAW_ULG_DIR / dataset_label / f"{log_id}.ulg"

        success, processed_df, error = build_processed_10hz_for_log(
            ulg_path,
            dataset_label,
        )

        if success and not processed_df.empty:
            output_file = processed_output_dir / f"{log_id}_processed_10hz.csv"
            processed_df.to_csv(output_file, index=False)

            usable_ml_rows = processed_df["target_energy_next_60s_wh"].notna().sum()

            combined_parts.append(processed_df)

            records.append({
                "dataset_label": dataset_label,
                "log_id": log_id,
                "processed_success": True,
                "rows_raw_10hz": len(processed_df),
                "rows_with_60s_target": int(usable_ml_rows),
                "error": "",
            })

        else:
            records.append({
                "dataset_label": dataset_label,
                "log_id": log_id,
                "processed_success": False,
                "rows_raw_10hz": 0,
                "rows_with_60s_target": 0,
                "error": error,
            })

    report_df = pd.DataFrame(records)

    report_df.to_csv(
        VALIDATION_DIR / f"{dataset_label}_processed_10hz_report.csv",
        index=False,
    )

    if combined_parts:
        combined_df = pd.concat(combined_parts, ignore_index=True)

        combined_file = PROCESSED_DIR / f"{dataset_label}_combined_processed_10hz.csv"
        combined_df.to_csv(combined_file, index=False)

        ml_ready_df = combined_df.dropna(subset=["target_energy_next_60s_wh"]).copy()

        ml_ready_file = PROCESSED_DIR / f"{dataset_label}_ml_ready_10hz.csv"
        ml_ready_df.to_csv(ml_ready_file, index=False)

        print(f"\nSaved combined {dataset_label} processed file:", combined_file)
        print(f"Saved ML-ready {dataset_label} file:", ml_ready_file)
        print(f"{dataset_label} raw 10Hz rows:", len(combined_df))
        print(f"{dataset_label} ML-ready rows:", len(ml_ready_df))

        return combined_df, ml_ready_df, report_df

    return pd.DataFrame(), pd.DataFrame(), report_df


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("\n========== PX4 UAV DATASET PIPELINE STARTED ==========\n")

    logs_df = load_or_download_metadata()

    normal_candidates, abnormal_candidates = prepare_candidates(logs_df)

    normal_selection = sample_by_duration_quota(
        normal_candidates,
        NORMAL_DOWNLOAD_QUOTA,
        dataset_label="normal",
    )

    abnormal_selection = sample_by_duration_quota(
        abnormal_candidates,
        ABNORMAL_DOWNLOAD_QUOTA,
        dataset_label="abnormal",
    )

    normal_selection.to_csv(
        SELECTION_DIR / "normal_download_selection.csv",
        index=False,
    )

    abnormal_selection.to_csv(
        SELECTION_DIR / "abnormal_download_selection.csv",
        index=False,
    )

    print("\n========== DOWNLOADING NORMAL DATASET ==========")
    normal_download_report = download_dataset(normal_selection, "normal")

    print("\n========== DOWNLOADING ABNORMAL DATASET ==========")
    abnormal_download_report = download_dataset(abnormal_selection, "abnormal")

    print("\n========== VALIDATING NORMAL ULG FILES ==========")
    normal_validation = validate_downloaded_files("normal")

    print("\n========== VALIDATING ABNORMAL ULG FILES ==========")
    abnormal_validation = validate_downloaded_files("abnormal")

    if SAVE_RAW_TOPIC_CSVS:
        print("\n========== CONVERTING NORMAL ULG TOPICS TO RAW CSV ==========")
        convert_valid_ulgs_to_raw_csv(normal_validation, "normal")

        print("\n========== CONVERTING ABNORMAL ULG TOPICS TO RAW CSV ==========")
        convert_valid_ulgs_to_raw_csv(abnormal_validation, "abnormal")

    if SAVE_PROCESSED_10HZ_CSVS:
        print("\n========== CREATING NORMAL PROCESSED 10HZ DATASET ==========")
        normal_combined, normal_ml_ready, normal_processed_report = create_processed_dataset(
            normal_validation,
            "normal",
        )

        print("\n========== CREATING ABNORMAL PROCESSED 10HZ DATASET ==========")
        abnormal_combined, abnormal_ml_ready, abnormal_processed_report = create_processed_dataset(
            abnormal_validation,
            "abnormal",
        )

        if not normal_ml_ready.empty or not abnormal_ml_ready.empty:
            final_mixed_parts = []

            if not normal_ml_ready.empty:
                final_mixed_parts.append(normal_ml_ready)

            if not abnormal_ml_ready.empty:
                final_mixed_parts.append(abnormal_ml_ready)

            final_mixed_df = pd.concat(final_mixed_parts, ignore_index=True)

            final_mixed_file = PROCESSED_DIR / "final_mixed_normal_abnormal_ml_ready_10hz.csv"
            final_mixed_df.to_csv(final_mixed_file, index=False)

            print("\n========== FINAL MIXED DATASET CREATED ==========")
            print("Final mixed ML-ready file:", final_mixed_file)
            print("Final mixed ML-ready rows:", len(final_mixed_df))
            print("Normal rows:", len(normal_ml_ready))
            print("Abnormal rows:", len(abnormal_ml_ready))

    print("\n========== PIPELINE COMPLETE ==========\n")

    print("Important output folders:")
    print("Metadata:", METADATA_DIR)
    print("Selections:", SELECTION_DIR)
    print("Raw ULG:", RAW_ULG_DIR)
    print("Raw topic CSV:", RAW_CSV_DIR)
    print("Processed 10Hz CSV:", PROCESSED_DIR)
    print("Validation reports:", VALIDATION_DIR)


if __name__ == "__main__":
    main()