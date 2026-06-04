#!/usr/bin/env python3

"""
Incremental PX4 UAV Dataset Pipeline

This script:
1. Keeps your existing downloaded data.
2. Downloads extra normal and abnormal PX4 .ulg logs.
3. Skips logs already downloaded.
4. Validates actual ULG contents.
5. Converts valid ULG logs into processed 10 Hz CSV.
6. Creates final mixed normal + abnormal dataset.
7. Removes unusable zero-current / zero-power flights.
8. Saves a larger clean ML-ready dataset.
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
PROCESSED_DIR = PROJECT_DIR / "processed_10hz"
VALIDATION_DIR = PROJECT_DIR / "validation_reports"

for folder in [
    METADATA_DIR,
    SELECTION_DIR,
    RAW_ULG_DIR,
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

MAV_TYPE_REQUIRED = "quadrotor"

NORMAL_RATINGS = ["good", "great"]
ABNORMAL_RATINGS = ["unsatisfactory", "crash_sw_hw", "crash_pilot"]

SIMULATION_KEYWORDS = [
    "simulation",
    "hil",
    "sih",
    "hardware in the loop",
    "software in the loop",
]

DURATION_BINS = [300, 480, 720, 1200, float("inf")]
DURATION_LABELS = ["5-8 min", "8-12 min", "12-20 min", "20+ min"]

# Extra files to download now
# Existing downloaded logs will be skipped automatically.
NORMAL_EXTRA_DOWNLOAD_QUOTA = {
    "5-8 min": 25,
    "8-12 min": 30,
    "12-20 min": 20,
    "20+ min": 5,
}

ABNORMAL_EXTRA_DOWNLOAD_QUOTA = {
    "5-8 min": 20,
    "8-12 min": 25,
    "12-20 min": 15,
    "20+ min": 10,
}

RANDOM_STATE = 99

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

# Flight-quality cleaning thresholds
MAX_ZERO_CURRENT_PERCENT = 90
MIN_TOTAL_ENERGY_WH = 0.1
MIN_MAX_POWER_W = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_filename(text):
    text = str(text)
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text)


def parse_flight_duration_seconds(value):
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


# ============================================================
# STEP 1: LOAD OR DOWNLOAD PX4 METADATA
# ============================================================

def load_or_download_metadata():
    metadata_file = METADATA_DIR / "px4_public_logs_metadata.csv"

    if metadata_file.exists():
        print("Loading existing metadata CSV...")
        logs_df = pd.read_csv(metadata_file, low_memory=False)
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
# STEP 2: PREPARE NORMAL AND ABNORMAL CANDIDATES
# ============================================================

def prepare_candidates(logs_df):
    df = logs_df.copy()

    df = df.dropna(subset=["log_id"]).copy()

    needed_cols = [
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
        "flight_mode_durations",
    ]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

        if col != "flight_mode_durations":
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

    combined_text = (
        df["airframe_name"].astype(str).str.lower()
        + " "
        + df["airframe_type"].astype(str).str.lower()
        + " "
        + df["description"].astype(str).str.lower()
        + " "
        + df["feedback"].astype(str).str.lower()
        + " "
        + df["vehicle_name"].astype(str).str.lower()
    )

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

    normal_candidates.to_csv(SELECTION_DIR / "normal_candidates_all.csv", index=False)
    abnormal_candidates.to_csv(SELECTION_DIR / "abnormal_candidates_all.csv", index=False)

    print("\nCandidate summary:")
    print("Normal candidates:", len(normal_candidates))
    print("Abnormal candidates:", len(abnormal_candidates))

    return normal_candidates, abnormal_candidates


# ============================================================
# STEP 3: SELECT EXTRA LOGS ONLY
# ============================================================

def select_extra_logs(df, quota, dataset_label):
    dataset_dir = RAW_ULG_DIR / dataset_label
    dataset_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = {file.stem for file in dataset_dir.glob("*.ulg")}

    df = df[~df["log_id"].astype(str).isin(existing_ids)].copy()

    print(f"\n{dataset_label.upper()} DATASET")
    print("Already downloaded files:", len(existing_ids))
    print("Remaining candidates after excluding existing:", len(df))

    selected_parts = []

    for bin_name, required_count in quota.items():
        bin_df = df[df["duration_bin"].astype(str) == bin_name].copy()

        available_count = len(bin_df)
        sample_count = min(required_count, available_count)

        if sample_count == 0:
            print(f"{bin_name}: no available logs")
            continue

        sampled = bin_df.sample(
            n=sample_count,
            random_state=RANDOM_STATE,
            replace=False,
        )

        selected_parts.append(sampled)

        print(f"{bin_name}: selected {sample_count} from {available_count}")

    if not selected_parts:
        print(f"No new {dataset_label} logs selected.")
        return pd.DataFrame()

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    selected["dataset_label"] = dataset_label

    print(f"New {dataset_label} logs selected:", len(selected))
    print(
        f"Estimated new rows at {SAMPLING_RATE_HZ} Hz:",
        int(selected["metadata_duration_s"].sum() * SAMPLING_RATE_HZ),
    )

    selected.to_csv(
        SELECTION_DIR / f"{dataset_label}_extra_download_selection.csv",
        index=False,
    )

    return selected


# ============================================================
# STEP 4: DOWNLOAD EXTRA ULG FILES
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
    if selection_df.empty:
        print(f"No new {dataset_label} logs to download.")
        return pd.DataFrame()

    dataset_dir = RAW_ULG_DIR / dataset_label
    dataset_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for _, row in tqdm(selection_df.iterrows(), total=len(selection_df), desc=f"Downloading {dataset_label}"):
        log_id = clean_text(row["log_id"])
        url = get_download_url(row)
        output_path = dataset_dir / f"{log_id}.ulg"

        success, status = download_with_retry(url, output_path)

        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0

        records.append({
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

    report_df = pd.DataFrame(records)

    report_df.to_csv(
        VALIDATION_DIR / f"{dataset_label}_extra_download_report.csv",
        index=False,
    )

    return report_df


# ============================================================
# STEP 5: VALIDATE ALL DOWNLOADED ULG FILES
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


def validate_all_downloaded_files(dataset_label):
    dataset_dir = RAW_ULG_DIR / dataset_label
    ulg_files = sorted(dataset_dir.glob("*.ulg"))

    records = []

    for ulg_path in tqdm(ulg_files, desc=f"Validating all {dataset_label} logs"):
        records.append(inspect_ulg_file(ulg_path, dataset_label))

    validation_df = pd.DataFrame(records)

    validation_df.to_csv(
        VALIDATION_DIR / f"{dataset_label}_all_ulg_validation_report.csv",
        index=False,
    )

    print(f"\n{dataset_label} validation:")
    print("Downloaded files:", len(validation_df))
    print("Valid files:", int(validation_df["valid_for_processing"].sum()))

    return validation_df


# ============================================================
# STEP 6: PROCESS VALID ULG FILES TO 10 HZ CSV
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

    # Forward-fill only to avoid using future values.
    # Remaining NaNs can be handled later during modelling.
    resampled = resampled.ffill(limit=10)

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

            df_topic = pd.DataFrame(data.data)
            df_topic["multi_id"] = multi_id

            if not df_topic.empty:
                topic_dfs[topic_key] = df_topic

        if "battery_status" not in topic_dfs:
            return False, pd.DataFrame(), "battery_status missing"

        battery = topic_dfs["battery_status"]

        required_cols = ["timestamp", "voltage_v", "current_a"]

        if not all(col in battery.columns for col in required_cols):
            return False, pd.DataFrame(), "battery missing timestamp/voltage/current"

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

        frames = []

        for topic_key, topic_df in topic_dfs.items():
            resampled = resample_topic_df(
                df=topic_df,
                topic_key=topic_key,
                log_start_timestamp=log_start_timestamp,
                base_index=base_index,
                rule=rule,
            )

            if not resampled.empty:
                frames.append(resampled)

        if not frames:
            return False, pd.DataFrame(), "no resampled topics"

        merged = pd.concat(frames, axis=1)

        merged = merged.reset_index()
        merged = merged.rename(columns={"index": "time_td"})
        merged["time_s"] = merged["time_td"].dt.total_seconds()

        merged.insert(0, "log_id", ulg_path.stem)
        merged.insert(1, "dataset_label", dataset_label)

        voltage_col = "battery_status__voltage_v"
        current_col = "battery_status__current_a"

        if voltage_col not in merged.columns or current_col not in merged.columns:
            return False, pd.DataFrame(), "voltage/current missing after processing"

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


def process_all_valid_logs(validation_df, dataset_label):
    valid_logs = validation_df[validation_df["valid_for_processing"] == True].copy()

    processed_output_dir = PROCESSED_DIR / dataset_label
    processed_output_dir.mkdir(parents=True, exist_ok=True)

    combined_parts = []
    records = []

    for _, row in tqdm(valid_logs.iterrows(), total=len(valid_logs), desc=f"Processing {dataset_label} logs"):
        log_id = row["log_id"]
        ulg_path = RAW_ULG_DIR / dataset_label / f"{log_id}.ulg"

        success, processed_df, error = build_processed_10hz_for_log(
            ulg_path=ulg_path,
            dataset_label=dataset_label,
        )

        if success and not processed_df.empty:
            output_file = processed_output_dir / f"{log_id}_processed_10hz.csv"
            processed_df.to_csv(output_file, index=False)

            ml_rows = processed_df["target_energy_next_60s_wh"].notna().sum()

            combined_parts.append(processed_df)

            records.append({
                "dataset_label": dataset_label,
                "log_id": log_id,
                "processed_success": True,
                "rows_raw_10hz": len(processed_df),
                "rows_with_60s_target": int(ml_rows),
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
        VALIDATION_DIR / f"{dataset_label}_all_processed_10hz_report.csv",
        index=False,
    )

    if combined_parts:
        combined_df = pd.concat(combined_parts, ignore_index=True)

        combined_file = PROCESSED_DIR / f"{dataset_label}_combined_processed_10hz.csv"
        combined_df.to_csv(combined_file, index=False)

        ml_ready_df = combined_df.dropna(subset=["target_energy_next_60s_wh"]).copy()

        ml_ready_file = PROCESSED_DIR / f"{dataset_label}_ml_ready_10hz.csv"
        ml_ready_df.to_csv(ml_ready_file, index=False)

        print(f"\n{dataset_label} raw rows:", len(combined_df))
        print(f"{dataset_label} ML-ready rows:", len(ml_ready_df))

        return combined_df, ml_ready_df, report_df

    return pd.DataFrame(), pd.DataFrame(), report_df


# ============================================================
# STEP 7: CLEAN FINAL MIXED DATASET
# ============================================================

def clean_zero_current_flights(final_mixed_df):
    current_col = "battery_status__current_a"

    flight_summary = (
        final_mixed_df.groupby(["log_id", "dataset_label"])
        .agg(
            rows=("time_s", "count"),
            duration_s=("time_s", "max"),
            mean_current_a=(current_col, "mean"),
            max_current_a=(current_col, "max"),
            mean_power_w=("power_w", "mean"),
            max_power_w=("power_w", "max"),
            total_energy_wh=("cumulative_energy_wh", "max"),
            mean_target_60s=("target_energy_next_60s_wh", "mean"),
        )
        .reset_index()
    )

    zero_stats = (
        final_mixed_df.assign(
            zero_current=final_mixed_df[current_col].abs() < 0.01,
            zero_power=final_mixed_df["power_w"].abs() < 0.01,
        )
        .groupby("log_id")
        .agg(
            zero_current_percent=("zero_current", "mean"),
            zero_power_percent=("zero_power", "mean"),
        )
        .reset_index()
    )

    zero_stats["zero_current_percent"] *= 100
    zero_stats["zero_power_percent"] *= 100

    flight_summary = flight_summary.merge(zero_stats, on="log_id", how="left")

    usable_flights = flight_summary[
        (flight_summary["zero_current_percent"] < MAX_ZERO_CURRENT_PERCENT)
        & (flight_summary["total_energy_wh"] > MIN_TOTAL_ENERGY_WH)
        & (flight_summary["max_power_w"] > MIN_MAX_POWER_W)
    ].copy()

    removed_flights = flight_summary[
        ~flight_summary["log_id"].isin(usable_flights["log_id"])
    ].copy()

    clean_df = final_mixed_df[
        final_mixed_df["log_id"].isin(usable_flights["log_id"])
    ].copy()

    flight_summary.to_csv(
        VALIDATION_DIR / "expanded_flight_summary_before_cleaning.csv",
        index=False,
    )

    usable_flights.to_csv(
        VALIDATION_DIR / "expanded_usable_flights_after_cleaning.csv",
        index=False,
    )

    removed_flights.to_csv(
        VALIDATION_DIR / "expanded_removed_flights_after_cleaning.csv",
        index=False,
    )

    clean_file = PROCESSED_DIR / "clean_ml_ready_10hz_expanded.csv"
    clean_df.to_csv(clean_file, index=False)

    print("\n========== CLEANING SUMMARY ==========")
    print("Original flights:", flight_summary["log_id"].nunique())
    print("Usable flights after cleaning:", usable_flights["log_id"].nunique())
    print("Removed flights:", removed_flights["log_id"].nunique())

    print("\nUsable flights by label:")
    print(usable_flights["dataset_label"].value_counts())

    print("\nRows before cleaning:", len(final_mixed_df))
    print("Rows after cleaning:", len(clean_df))
    print("Removed rows:", len(final_mixed_df) - len(clean_df))

    print("\nClean expanded dataset saved to:")
    print(clean_file)

    return clean_df, flight_summary, usable_flights, removed_flights


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========== INCREMENTAL PX4 PIPELINE STARTED ==========\n")

    logs_df = load_or_download_metadata()

    normal_candidates, abnormal_candidates = prepare_candidates(logs_df)

    normal_extra_selection = select_extra_logs(
        df=normal_candidates,
        quota=NORMAL_EXTRA_DOWNLOAD_QUOTA,
        dataset_label="normal",
    )

    abnormal_extra_selection = select_extra_logs(
        df=abnormal_candidates,
        quota=ABNORMAL_EXTRA_DOWNLOAD_QUOTA,
        dataset_label="abnormal",
    )

    print("\n========== DOWNLOADING EXTRA NORMAL LOGS ==========")
    download_dataset(normal_extra_selection, "normal")

    print("\n========== DOWNLOADING EXTRA ABNORMAL LOGS ==========")
    download_dataset(abnormal_extra_selection, "abnormal")

    print("\n========== VALIDATING ALL NORMAL LOGS ==========")
    normal_validation = validate_all_downloaded_files("normal")

    print("\n========== VALIDATING ALL ABNORMAL LOGS ==========")
    abnormal_validation = validate_all_downloaded_files("abnormal")

    print("\n========== PROCESSING ALL VALID NORMAL LOGS ==========")
    normal_combined, normal_ml_ready, _ = process_all_valid_logs(
        normal_validation,
        "normal",
    )

    print("\n========== PROCESSING ALL VALID ABNORMAL LOGS ==========")
    abnormal_combined, abnormal_ml_ready, _ = process_all_valid_logs(
        abnormal_validation,
        "abnormal",
    )

    mixed_parts = []

    if not normal_ml_ready.empty:
        mixed_parts.append(normal_ml_ready)

    if not abnormal_ml_ready.empty:
        mixed_parts.append(abnormal_ml_ready)

    if not mixed_parts:
        print("No ML-ready data created. Check validation reports.")
        return

    final_mixed_df = pd.concat(mixed_parts, ignore_index=True)

    final_mixed_file = PROCESSED_DIR / "final_mixed_normal_abnormal_ml_ready_10hz_expanded.csv"
    final_mixed_df.to_csv(final_mixed_file, index=False)

    print("\n========== FINAL MIXED DATASET CREATED ==========")
    print("Final mixed expanded file:", final_mixed_file)
    print("Rows:", len(final_mixed_df))
    print("Flights:", final_mixed_df["log_id"].nunique())
    print("Label distribution:")
    print(final_mixed_df["dataset_label"].value_counts())

    print("\n========== CLEANING ZERO-CURRENT / ZERO-POWER FLIGHTS ==========")
    clean_zero_current_flights(final_mixed_df)

    print("\n========== INCREMENTAL PIPELINE COMPLETE ==========\n")


if __name__ == "__main__":
    main()