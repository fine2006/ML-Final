"""
Phase 1: Data Investigation

Objectives:
1. Classify Bhatagaon September 2025 PM2.5 spikes (>500 ug/m3) as
   real events vs likely sensor errors.
2. Attribute data loss sources for a reproducible preprocessing baseline.
3. Quantify region imbalance and compute region weights for Phase 5.

Outputs:
- data/raw/pollution_data_raw.csv
- data/raw/pollution_data_hourly_unique.csv
- data/raw/phase1_investigation_results.json
- visualizations/phase_1_data_investigation/*.png
- logs/data_investigation/data_investigation_*.log
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_SOURCE_DIR = PROJECT_ROOT / "Pollution Data Raipur"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOGS_DIR = PROJECT_ROOT / "logs" / "data_investigation"
VIZ_DIR = PROJECT_ROOT / "visualizations" / "phase_1_data_investigation"


REGION_DIRS: dict[str, Path] = {
    "Bhatagaon": RAW_SOURCE_DIR / "Bhatagaon DCR",
    "AIIMS": RAW_SOURCE_DIR / "DCR AIIMS",
    "IGKV": RAW_SOURCE_DIR / "IGKV DCR",
    "SILTARA": RAW_SOURCE_DIR / "SILTARA DCR",
}


MISSING_TOKENS = {
    "",
    "DATA F",
    "DATA FAIL",
    "DATAFAIL",
    "MAINT",
    "MAINT.",
    "INVALID",
    "PW.OFF",
    "POWER OFF",
    "LINK F",
    "LINK FAIL",
    "CALIB.",
    "CALIBRATION",
    "WARM UP",
    "Z.REF",
    "ZERO REF",
    "NAN",
    "NONE",
    "NULL",
    "NA",
    "N/A",
    "-",
    "--",
}
MISSING_TOKENS_NORMALIZED = {
    re.sub(r"[^A-Z0-9]+", "", token.upper()) for token in MISSING_TOKENS
}

STOP_ROW_TOKENS = {
    "MIN",
    "MAX",
    "AVG",
    "STDEV",
    "NUM",
    "STATUS",
    "DCR",
    "DONE BY",
    "COMMENT",
}


NUMERIC_COLUMNS = [
    "pm25",
    "pm10",
    "no2",
    "o3",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
]


@dataclass
class SpikeClusterResult:
    start: str
    end: str
    peak_time: str
    peak_pm25: float
    n_points_above_500: int
    temporal_pattern: str
    other_regions_correlated: bool
    weather_supports_stagnation: bool
    classification: str
    action: str
    evidence: str


def setup_logging() -> logging.Logger:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("data_investigation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"data_investigation_{timestamp}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger.info("=" * 72)
    logger.info("PHASE 1: DATA INVESTIGATION")
    logger.info("Random seed: %s", RANDOM_SEED)
    logger.info("=" * 72)
    return logger


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]+", "", text)


def choose_engine(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".xls":
        return "xlrd"
    if suffix == ".xlsb":
        return "pyxlsb"
    return "openpyxl"


def parse_numeric(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return np.nan
        if normalize_text(cleaned) in MISSING_TOKENS_NORMALIZED:
            return np.nan
        cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def parse_timestamp(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value

    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        try:
            return pd.Timestamp(value)
        except Exception:
            pass

    text = str(value).strip()
    if not text:
        return pd.NaT

    if text.upper() in STOP_ROW_TOKENS:
        return pd.NaT

    add_day = False
    if "24:00" in text:
        text = text.replace("24:00", "00:00")
        add_day = True

    timestamp = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(timestamp):
        return pd.NaT
    if add_day:
        timestamp += pd.Timedelta(days=1)
    return timestamp


def extract_year_from_path(path: Path) -> int | None:
    years = re.findall(r"20\d{2}", str(path))
    if not years:
        return None
    return int(years[0])


def discover_source_files(
    region_dir: Path,
    include_quarterly: bool = True,
) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for candidate in region_dir.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".xls", ".xlsx", ".xlsb"}:
            continue
        if candidate.name.startswith("~$"):
            continue

        name_lower = candidate.name.lower()
        if "sample" in name_lower or "blank" in name_lower:
            continue

        is_quarterly = (
            "quat" in name_lower
            or "quarter" in name_lower
            or "qtr" in name_lower
            or "15min" in name_lower
            or "15-min" in name_lower
            or "15 min" in name_lower
        )
        is_hourly = "hourly" in name_lower
        if not is_hourly and not is_quarterly:
            continue
        if is_quarterly and not include_quarterly:
            continue

        year = extract_year_from_path(candidate)
        if year is not None and not (2022 <= year <= 2025):
            continue

        granularity = "quarterly" if is_quarterly else "hourly"
        files.append((candidate, granularity))

    return sorted(files, key=lambda item: str(item[0]))


def find_header_row(sheet: pd.DataFrame) -> int | None:
    max_scan = min(len(sheet), 30)
    for row_index in range(max_scan):
        row = sheet.iloc[row_index]
        if any(
            isinstance(value, str) and "DATE & TIME" in value.upper() for value in row
        ):
            return row_index
    return None


def locate_columns(header_values: list[Any]) -> dict[str, int | None]:
    normalized = [normalize_text(value) for value in header_values]

    def find_index(predicate) -> int | None:
        for column_index, column_name in enumerate(normalized):
            if predicate(column_name):
                return column_index
        return None

    return {
        "timestamp": find_index(lambda x: "DATE" in x and "TIME" in x),
        "pm25": find_index(lambda x: "PM25" in x),
        "pm10": find_index(lambda x: "PM10" in x),
        "no2": find_index(lambda x: "NO2" in x),
        "o3": find_index(lambda x: x == "O3" or x.startswith("O3")),
        "temperature": find_index(lambda x: x.startswith("TEMP")),
        "humidity": find_index(lambda x: x.startswith("HUM")),
        "wind_speed": find_index(lambda x: x == "WS" or x.startswith("WS")),
        "wind_direction": find_index(lambda x: x == "WD" or x.startswith("WD")),
    }


def extract_records_from_sheet(
    sheet_df: pd.DataFrame,
    region: str,
    source_file: Path,
    sheet_name: str,
    source_granularity: str,
) -> list[dict[str, Any]]:
    header_row = find_header_row(sheet_df)
    if header_row is None:
        return []

    columns = locate_columns(sheet_df.iloc[header_row].tolist())
    if columns["timestamp"] is None or columns["pm25"] is None:
        return []

    records: list[dict[str, Any]] = []
    for row_index in range(header_row + 2, len(sheet_df)):
        row = sheet_df.iloc[row_index]
        raw_timestamp = row.iloc[columns["timestamp"]]

        if (
            isinstance(raw_timestamp, str)
            and raw_timestamp.strip().upper() in STOP_ROW_TOKENS
        ):
            break

        timestamp = parse_timestamp(raw_timestamp)
        if pd.isna(timestamp):
            continue

        record: dict[str, Any] = {
            "region": region,
            "timestamp": timestamp,
            "source_file": str(source_file.relative_to(RAW_SOURCE_DIR)),
            "source_sheet": str(sheet_name),
            "source_granularity": source_granularity,
        }

        for column_name in [
            "pm25",
            "pm10",
            "no2",
            "o3",
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
        ]:
            column_index = columns[column_name]
            if column_index is None or column_index >= len(row):
                record[column_name] = np.nan
            else:
                record[column_name] = parse_numeric(row.iloc[column_index])

        records.append(record)

    return records


def load_raw_hourly_dataset(
    logger: logging.Logger,
    include_quarterly: bool = True,
) -> pd.DataFrame:
    all_records: list[dict[str, Any]] = []

    for region, region_dir in REGION_DIRS.items():
        files = discover_source_files(
            region_dir=region_dir,
            include_quarterly=include_quarterly,
        )
        hourly_file_count = sum(1 for _, g in files if g == "hourly")
        quarterly_file_count = sum(1 for _, g in files if g == "quarterly")
        logger.info(
            "%s: discovered %d hourly + %d quarterly files",
            region,
            hourly_file_count,
            quarterly_file_count,
        )

        extracted_for_region = 0
        for file_path, granularity in files:
            try:
                workbook = pd.ExcelFile(file_path, engine=choose_engine(file_path))
            except Exception as exc:
                logger.warning("Failed to open workbook %s: %s", file_path, exc)
                continue

            for sheet_name in workbook.sheet_names:
                if not re.fullmatch(r"\d{1,2}", str(sheet_name).strip()):
                    continue
                try:
                    sheet_df = pd.read_excel(
                        workbook, sheet_name=sheet_name, header=None
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to read %s [%s]: %s", file_path.name, sheet_name, exc
                    )
                    continue

                extracted = extract_records_from_sheet(
                    sheet_df=sheet_df,
                    region=region,
                    source_file=file_path,
                    sheet_name=str(sheet_name),
                    source_granularity=granularity,
                )
                extracted_for_region += len(extracted)
                all_records.extend(extracted)

        logger.info("%s: extracted %d rows", region, extracted_for_region)

    if not all_records:
        raise RuntimeError("No rows extracted from hourly workbooks.")

    raw_df = pd.DataFrame(all_records)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    raw_df = raw_df.dropna(subset=["timestamp"]).copy()
    raw_df = raw_df[
        (raw_df["timestamp"].dt.year >= 2022) & (raw_df["timestamp"].dt.year <= 2025)
    ]
    raw_df = raw_df.sort_values(["region", "timestamp"]).reset_index(drop=True)

    by_granularity = raw_df["source_granularity"].value_counts().to_dict()
    logger.info(
        "Raw rows (2022-2025): %d | by granularity: %s", len(raw_df), by_granularity
    )
    return raw_df


def deduplicate_hourly_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    canonical_df = raw_df.copy()
    canonical_df["timestamp"] = pd.to_datetime(
        canonical_df["timestamp"], errors="coerce"
    )
    canonical_df = canonical_df.dropna(subset=["timestamp"])
    canonical_df["timestamp"] = canonical_df["timestamp"].dt.floor("h")

    if "source_granularity" not in canonical_df.columns:
        dedup_df = (
            canonical_df.groupby(["region", "timestamp"], as_index=False)[
                NUMERIC_COLUMNS
            ]
            .mean()
            .sort_values(["region", "timestamp"])
            .reset_index(drop=True)
        )
        return dedup_df

    hourly_df = canonical_df[canonical_df["source_granularity"] == "hourly"]
    quarterly_df = canonical_df[canonical_df["source_granularity"] == "quarterly"]

    hourly_agg = (
        hourly_df.groupby(["region", "timestamp"], as_index=False)[NUMERIC_COLUMNS]
        .mean()
        .set_index(["region", "timestamp"])
    )
    quarterly_agg = (
        quarterly_df.groupby(["region", "timestamp"], as_index=False)[NUMERIC_COLUMNS]
        .mean()
        .set_index(["region", "timestamp"])
    )

    # Prefer hourly values when both exist, fill remaining gaps from quarterly.
    dedup_df = (
        hourly_agg.combine_first(quarterly_agg)
        .reset_index()
        .sort_values(["region", "timestamp"])
        .reset_index(drop=True)
    )
    return dedup_df


def compute_source_contribution(
    raw_df: pd.DataFrame, dedup_df: pd.DataFrame
) -> dict[str, Any]:
    if "source_granularity" not in raw_df.columns:
        return {}

    normalized = raw_df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"])
    normalized["timestamp"] = normalized["timestamp"].dt.floor("h")

    rows_by_granularity = {
        key: int(value)
        for key, value in normalized["source_granularity"]
        .value_counts()
        .to_dict()
        .items()
    }

    hourly_df = normalized[normalized["source_granularity"] == "hourly"]
    quarterly_df = normalized[normalized["source_granularity"] == "quarterly"]

    hourly_agg = (
        hourly_df.groupby(["region", "timestamp"], as_index=False)[NUMERIC_COLUMNS]
        .mean()
        .set_index(["region", "timestamp"])
    )
    quarterly_agg = (
        quarterly_df.groupby(["region", "timestamp"], as_index=False)[NUMERIC_COLUMNS]
        .mean()
        .set_index(["region", "timestamp"])
    )

    hourly_idx = hourly_agg.index
    quarterly_idx = quarterly_agg.index
    overlap_idx = hourly_idx.intersection(quarterly_idx)
    quarterly_only_idx = quarterly_idx.difference(hourly_idx)
    combined_idx = hourly_idx.union(quarterly_idx)

    hourly_pm25 = hourly_agg["pm25"]
    combined_pm25 = hourly_pm25.combine_first(quarterly_agg["pm25"])
    pm25_from_hourly = int(hourly_pm25.notna().sum())
    pm25_combined_non_null = int(combined_pm25.notna().sum())
    pm25_from_quarterly = int(max(0, pm25_combined_non_null - pm25_from_hourly))
    pm25_missing_both = int(len(combined_pm25) - pm25_combined_non_null)

    per_region: dict[str, dict[str, int]] = {}
    for region in REGION_DIRS.keys():
        region_hourly = int((hourly_idx.get_level_values(0) == region).sum())
        region_quarterly = int((quarterly_idx.get_level_values(0) == region).sum())
        region_quarterly_only = int(
            (quarterly_only_idx.get_level_values(0) == region).sum()
        )
        region_combined = int((combined_idx.get_level_values(0) == region).sum())
        per_region[region] = {
            "hourly_unique_hours": region_hourly,
            "quarterly_unique_hours": region_quarterly,
            "quarterly_only_hours": region_quarterly_only,
            "combined_unique_hours": region_combined,
        }

    return {
        "rows_by_granularity": rows_by_granularity,
        "unique_hours": {
            "hourly": int(len(hourly_idx)),
            "quarterly": int(len(quarterly_idx)),
            "overlap": int(len(overlap_idx)),
            "quarterly_only": int(len(quarterly_only_idx)),
            "combined": int(len(combined_idx)),
            "combined_saved_rows": int(len(dedup_df)),
        },
        "pm25_availability": {
            "from_hourly": pm25_from_hourly,
            "from_quarterly_fill": pm25_from_quarterly,
            "combined_non_null": pm25_combined_non_null,
            "missing_both": pm25_missing_both,
        },
        "per_region": per_region,
    }


def find_spike_clusters(extreme_rows: pd.DataFrame) -> list[pd.DataFrame]:
    if extreme_rows.empty:
        return []

    clusters: list[pd.DataFrame] = []
    current_cluster = [extreme_rows.iloc[0]]

    for row_index in range(1, len(extreme_rows)):
        previous_time = current_cluster[-1]["timestamp"]
        current_row = extreme_rows.iloc[row_index]
        current_time = current_row["timestamp"]

        if current_time - previous_time <= pd.Timedelta(hours=1):
            current_cluster.append(current_row)
        else:
            clusters.append(pd.DataFrame(current_cluster))
            current_cluster = [current_row]

    clusters.append(pd.DataFrame(current_cluster))
    return clusters


def classify_cluster(
    dedup_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> SpikeClusterResult:
    cluster_df = cluster_df.sort_values("timestamp").reset_index(drop=True)
    start_time = pd.Timestamp(cluster_df.loc[0, "timestamp"])
    end_time = pd.Timestamp(cluster_df.loc[len(cluster_df) - 1, "timestamp"])
    peak_row = cluster_df.loc[cluster_df["pm25"].idxmax()]
    peak_time = pd.Timestamp(peak_row["timestamp"])
    peak_value = float(peak_row["pm25"])

    bhat_df = dedup_df[dedup_df["region"] == "Bhatagaon"].set_index("timestamp")

    before_value = bhat_df["pm25"].get(start_time - pd.Timedelta(hours=1), np.nan)
    after_value = bhat_df["pm25"].get(end_time + pd.Timedelta(hours=1), np.nan)

    low_before = pd.notna(before_value) and before_value < 100
    low_after = pd.notna(after_value) and after_value < 100
    abrupt = (
        peak_value > 500
        and cluster_df["pm25"].max() > 1000
        and (end_time - start_time) <= pd.Timedelta(hours=3)
        and low_before
        and low_after
    )

    if abrupt and len(cluster_df) == 1:
        temporal_pattern = "single_point_spike"
    elif abrupt:
        temporal_pattern = "abrupt_multi_hour_spike"
    else:
        temporal_pattern = "ramp_up_and_decay"

    region_at_peak = dedup_df[dedup_df["timestamp"] == peak_time]
    other_regions = region_at_peak[region_at_peak["region"] != "Bhatagaon"]
    other_regions_correlated = bool((other_regions["pm25"] > 150).sum() >= 2)

    weather_supports_stagnation = bool(
        pd.notna(peak_row.get("humidity"))
        and pd.notna(peak_row.get("wind_speed"))
        and float(peak_row.get("humidity", np.nan)) >= 85
        and float(peak_row.get("wind_speed", np.nan)) <= 0.5
    )

    if (
        temporal_pattern == "ramp_up_and_decay"
        and other_regions_correlated
        and weather_supports_stagnation
    ):
        classification = "REAL EVENT"
        action = "KEEP"
    elif (
        temporal_pattern in {"single_point_spike", "abrupt_multi_hour_spike"}
        and not other_regions_correlated
    ):
        classification = "SENSOR ERROR"
        action = "REMOVE"
    else:
        classification = "UNCERTAIN"
        action = "MANUAL REVIEW"

    pre_text = f"{float(before_value):.2f}" if pd.notna(before_value) else "nan"
    post_text = f"{float(after_value):.2f}" if pd.notna(after_value) else "nan"
    other_max = (
        float(other_regions["pm25"].max()) if not other_regions.empty else np.nan
    )
    other_text = f"{other_max:.2f}" if pd.notna(other_max) else "nan"

    evidence = (
        f"Duration={int((end_time - start_time) / pd.Timedelta(hours=1)) + 1}h, "
        f"pre={pre_text}, "
        f"post={post_text}, "
        f"other_regions_max={other_text}"
    )

    return SpikeClusterResult(
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        peak_time=peak_time.isoformat(),
        peak_pm25=peak_value,
        n_points_above_500=int(len(cluster_df)),
        temporal_pattern=temporal_pattern,
        other_regions_correlated=other_regions_correlated,
        weather_supports_stagnation=weather_supports_stagnation,
        classification=classification,
        action=action,
        evidence=evidence,
    )


def create_spike_visualizations(
    dedup_df: pd.DataFrame,
    spike_clusters: list[pd.DataFrame],
    cluster_results: list[SpikeClusterResult],
) -> None:
    if not spike_clusters:
        return

    # 1) Temporal pattern
    n_clusters = len(spike_clusters)
    fig, axes = plt.subplots(
        n_clusters, 1, figsize=(14, 3.5 * n_clusters), sharex=False
    )
    if n_clusters == 1:
        axes = [axes]

    bhat_df = dedup_df[dedup_df["region"] == "Bhatagaon"].sort_values("timestamp")
    for index, (cluster_df, result) in enumerate(
        zip(spike_clusters, cluster_results, strict=False)
    ):
        axis = axes[index]
        peak_time = pd.Timestamp(result.peak_time)
        window = bhat_df[
            (bhat_df["timestamp"] >= peak_time - pd.Timedelta(hours=12))
            & (bhat_df["timestamp"] <= peak_time + pd.Timedelta(hours=12))
        ]
        axis.plot(window["timestamp"], window["pm25"], marker="o", linewidth=1.5)
        axis.axvspan(
            pd.Timestamp(result.start), pd.Timestamp(result.end), color="red", alpha=0.2
        )
        axis.axvline(peak_time, color="red", linestyle="--", linewidth=1)
        axis.set_ylabel("PM2.5")
        axis.set_title(
            f"Cluster {index + 1}: peak={result.peak_pm25:.2f} | {result.classification}"
        )
        axis.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "bhatagaon_spike_temporal_pattern.png", dpi=160)
    plt.close()

    # 2) Regional correlation around each peak
    fig, axes = plt.subplots(
        n_clusters, 1, figsize=(14, 3.5 * n_clusters), sharex=False
    )
    if n_clusters == 1:
        axes = [axes]

    for index, result in enumerate(cluster_results):
        axis = axes[index]
        peak_time = pd.Timestamp(result.peak_time)
        window = dedup_df[
            (dedup_df["timestamp"] >= peak_time - pd.Timedelta(hours=6))
            & (dedup_df["timestamp"] <= peak_time + pd.Timedelta(hours=6))
        ]
        for region, region_window in window.groupby("region"):
            axis.plot(
                region_window["timestamp"],
                region_window["pm25"],
                marker="o",
                label=region,
            )
        axis.axvline(peak_time, color="black", linestyle="--", linewidth=1)
        axis.set_ylabel("PM2.5")
        axis.set_title(f"Cluster {index + 1} peak at {peak_time}")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "bhatagaon_spike_regional_correlation.png", dpi=160)
    plt.close()

    # 3) Weather context for highest peak cluster
    top_result = max(cluster_results, key=lambda item: item.peak_pm25)
    top_peak_time = pd.Timestamp(top_result.peak_time)
    window = bhat_df[
        (bhat_df["timestamp"] >= top_peak_time - pd.Timedelta(hours=24))
        & (bhat_df["timestamp"] <= top_peak_time + pd.Timedelta(hours=24))
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(window["timestamp"], window["pm25"], color="tab:red")
    axes[0].axvline(top_peak_time, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("PM2.5")
    axes[0].grid(alpha=0.25)

    axes[1].plot(window["timestamp"], window["temperature"], color="tab:orange")
    axes[1].axvline(top_peak_time, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Temp")
    axes[1].grid(alpha=0.25)

    axes[2].plot(window["timestamp"], window["wind_speed"], color="tab:blue")
    axes[2].axvline(top_peak_time, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Wind")
    axes[2].grid(alpha=0.25)

    axes[3].plot(window["timestamp"], window["humidity"], color="tab:green")
    axes[3].axvline(top_peak_time, color="black", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Humidity")
    axes[3].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "bhatagaon_spike_weather_context.png", dpi=160)
    plt.close()


def count_sequence_ready_points(series: pd.Series) -> int:
    is_valid = series.notna().astype(int).to_numpy()
    contiguous_lengths: list[int] = []
    current = 0
    for value in is_valid:
        if value:
            current += 1
        elif current:
            contiguous_lengths.append(current)
            current = 0
    if current:
        contiguous_lengths.append(current)
    return int(sum(max(length - 1, 0) for length in contiguous_lengths))


def compute_data_loss_analysis(dedup_df: pd.DataFrame) -> dict[str, Any]:
    region_rows: list[dict[str, Any]] = []

    for region, region_df in dedup_df.groupby("region"):
        region_df = region_df.sort_values("timestamp").copy()
        raw_count = len(region_df)

        pm25_series = region_df.set_index("timestamp")["pm25"]
        pm25_after_impossible = pm25_series.where(
            (pm25_series >= 0) & (pm25_series <= 5000)
        )
        after_impossible = int(pm25_after_impossible.notna().sum())

        full_index = pd.date_range(
            pm25_after_impossible.index.min(),
            pm25_after_impossible.index.max(),
            freq="h",
        )
        pm25_hourly = pm25_after_impossible.reindex(full_index)
        pm25_after_missing = pm25_hourly.interpolate(
            method="time", limit=5, limit_direction="both"
        )
        after_missing = int(pm25_after_missing.notna().sum())
        interpolated_gain = int(max(0, after_missing - after_impossible))
        unresolved_missing_after_interp = int(max(0, len(full_index) - after_missing))

        pm25_after_outlier = pm25_after_missing.where(
            (pm25_after_missing >= 0) & (pm25_after_missing <= 300)
        )
        after_outliers = int(pm25_after_outlier.notna().sum())

        after_sequence = count_sequence_ready_points(pm25_after_outlier)

        observed_gaps = region_df["timestamp"].diff().dropna()
        gap_bins = {
            "lt_6h": int((observed_gaps < pd.Timedelta(hours=6)).sum()),
            "h6_to_h24": int(
                (
                    (observed_gaps >= pd.Timedelta(hours=6))
                    & (observed_gaps < pd.Timedelta(hours=24))
                ).sum()
            ),
            "h24_to_d7": int(
                (
                    (observed_gaps >= pd.Timedelta(hours=24))
                    & (observed_gaps < pd.Timedelta(days=7))
                ).sum()
            ),
            "gt_7d": int((observed_gaps >= pd.Timedelta(days=7)).sum()),
            "max_gap_hours": float(observed_gaps.max() / pd.Timedelta(hours=1))
            if not observed_gaps.empty
            else 0.0,
        }

        region_rows.append(
            {
                "region": region,
                "raw": int(raw_count),
                "after_impossible": after_impossible,
                "after_missing": after_missing,
                "after_outliers": after_outliers,
                "after_sequence": int(after_sequence),
                "full_hours": int(len(full_index)),
                "loss_impossible": int(raw_count - after_impossible),
                "loss_missing": int(max(0, after_impossible - after_missing)),
                "interpolated_gain": interpolated_gain,
                "unresolved_missing_after_interp": unresolved_missing_after_interp,
                "loss_outliers": int(after_missing - after_outliers),
                "loss_sequence": int(after_outliers - after_sequence),
                "gap_stats": gap_bins,
            }
        )

    region_table = (
        pd.DataFrame(region_rows).sort_values("region").reset_index(drop=True)
    )
    totals = {
        "raw": int(region_table["raw"].sum()),
        "after_impossible": int(region_table["after_impossible"].sum()),
        "after_missing": int(region_table["after_missing"].sum()),
        "after_outliers": int(region_table["after_outliers"].sum()),
        "after_sequence": int(region_table["after_sequence"].sum()),
        "loss_impossible": int(region_table["loss_impossible"].sum()),
        "loss_missing": int(region_table["loss_missing"].sum()),
        "interpolated_gain": int(region_table["interpolated_gain"].sum()),
        "unresolved_missing_after_interp": int(
            region_table["unresolved_missing_after_interp"].sum()
        ),
        "loss_outliers": int(region_table["loss_outliers"].sum()),
        "loss_sequence": int(region_table["loss_sequence"].sum()),
    }

    # Recovery sensitivity: what if interpolation threshold is relaxed?
    recovery_12h_total = 0
    recovery_24h_total = 0
    for _, region_df in dedup_df.groupby("region"):
        series = region_df.sort_values("timestamp").set_index("timestamp")["pm25"]
        series = series.where((series >= 0) & (series <= 5000))
        full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
        series = series.reindex(full_index)

        interp_12 = series.interpolate(method="time", limit=11, limit_direction="both")
        out_12 = interp_12.where((interp_12 >= 0) & (interp_12 <= 300))
        recovery_12h_total += count_sequence_ready_points(out_12)

        interp_24 = series.interpolate(method="time", limit=23, limit_direction="both")
        out_24 = interp_24.where((interp_24 >= 0) & (interp_24 <= 300))
        recovery_24h_total += count_sequence_ready_points(out_24)

    recovery_analysis = {
        "baseline_after_sequence": totals["after_sequence"],
        "if_gap_threshold_12h": int(recovery_12h_total),
        "if_gap_threshold_24h": int(recovery_24h_total),
        "recovery_12h_abs": int(recovery_12h_total - totals["after_sequence"]),
        "recovery_24h_abs": int(recovery_24h_total - totals["after_sequence"]),
        "recovery_12h_pct_of_baseline": float(
            100.0
            * (recovery_12h_total - totals["after_sequence"])
            / max(totals["after_sequence"], 1)
        ),
        "recovery_24h_pct_of_baseline": float(
            100.0
            * (recovery_24h_total - totals["after_sequence"])
            / max(totals["after_sequence"], 1)
        ),
    }

    return {
        "region_table": region_table,
        "totals": totals,
        "recovery_analysis": recovery_analysis,
    }


def create_loss_and_imbalance_visualizations(
    data_loss: dict[str, Any],
    dedup_df: pd.DataFrame,
    region_weights: dict[str, float],
) -> None:
    region_table: pd.DataFrame = data_loss["region_table"]
    totals = data_loss["totals"]

    # 1) Data loss breakdown per region
    fig, axis = plt.subplots(figsize=(10, 6))
    regions = region_table["region"].tolist()
    retained = region_table["after_sequence"].to_numpy()
    loss_missing = region_table["loss_missing"].to_numpy()
    loss_outliers = region_table["loss_outliers"].to_numpy()
    loss_sequence = region_table["loss_sequence"].to_numpy()

    axis.bar(regions, retained, label="Retained", color="#2a9d8f")
    axis.bar(
        regions, loss_missing, bottom=retained, label="Missing/Gaps", color="#e9c46a"
    )
    axis.bar(
        regions,
        loss_outliers,
        bottom=retained + loss_missing,
        label="Outliers",
        color="#f4a261",
    )
    axis.bar(
        regions,
        loss_sequence,
        bottom=retained + loss_missing + loss_outliers,
        label="Sequence",
        color="#e76f51",
    )
    axis.set_ylabel("Record count")
    axis.set_title("Data Retention by Region (Raw -> Sequence-ready)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "data_loss_breakdown_per_region.png", dpi=160)
    plt.close()

    # 2) Data loss source attribution
    fig, axis = plt.subplots(figsize=(10, 3.6))
    components = [
        "Impossible values",
        "Missing/gaps",
        "Outlier removal",
        "Sequence boundary",
    ]
    values = [
        totals["loss_impossible"],
        totals["loss_missing"],
        totals["loss_outliers"],
        totals["loss_sequence"],
    ]

    left = 0
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261"]
    total_raw = max(totals["raw"], 1)
    for component, value, color in zip(components, values, colors, strict=True):
        axis.barh(
            ["Loss attribution"],
            [value],
            left=left,
            label=f"{component}: {value}",
            color=color,
        )
        left += value
    axis.set_title(
        f"Data Loss Attribution ({totals['raw']:,} -> {totals['after_sequence']:,})"
    )
    axis.set_xlabel("Records lost")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    axis.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "data_loss_sources_stacked_bar.png", dpi=160)
    plt.close()

    # 3) Region imbalance distribution (pie + bar)
    raw_counts = dedup_df.groupby("region").size().reindex(regions)
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    axes[0].pie(
        raw_counts.values,
        labels=[
            f"{r} ({100 * v / raw_counts.sum():.1f}%)" for r, v in raw_counts.items()
        ],
        autopct="%1.1f%%",
        startangle=130,
    )
    axes[0].set_title("Data Distribution Across Regions (Hourly unique)")

    axes[1].bar(raw_counts.index.tolist(), raw_counts.values, color="#457b9d")
    ratio = raw_counts.max() / max(raw_counts.min(), 1)
    axes[1].set_title(f"Imbalance Ratio: {ratio:.2f}x | Weights: {region_weights}")
    axes[1].set_ylabel("Record count")
    axes[1].grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "region_imbalance_distribution.png", dpi=160)
    plt.close()


def compute_region_imbalance(
    dedup_df: pd.DataFrame, data_loss: dict[str, Any]
) -> dict[str, Any]:
    raw_counts = dedup_df.groupby("region").size().sort_values(ascending=False)
    raw_total = int(raw_counts.sum())
    raw_distribution = {
        region: {
            "count": int(count),
            "percent": float(100.0 * count / raw_total),
        }
        for region, count in raw_counts.items()
    }

    post_counts = (
        data_loss["region_table"]
        .set_index("region")["after_sequence"]
        .sort_values(ascending=False)
    )
    post_total = int(post_counts.sum())
    post_distribution = {
        region: {
            "count": int(count),
            "percent": float(100.0 * count / post_total),
        }
        for region, count in post_counts.items()
    }

    quality = {}
    for region, region_df in dedup_df.groupby("region"):
        n_rows = len(region_df)
        missing_pct = 100.0 * float(region_df["pm25"].isna().sum()) / max(n_rows, 1)
        outlier_pct = (
            100.0
            * float(((region_df["pm25"] > 300) | (region_df["pm25"] < 0)).sum())
            / max(n_rows, 1)
        )
        quality[region] = {
            "missing_pct": missing_pct,
            "outlier_pct": outlier_pct,
            "effective_pct": max(0.0, 100.0 - missing_pct - outlier_pct),
        }

    yearly_distribution = {}
    for year in [2022, 2023, 2024, 2025]:
        year_df = dedup_df[dedup_df["timestamp"].dt.year == year]
        if year_df.empty:
            continue
        year_counts = year_df.groupby("region").size()
        year_total = int(year_counts.sum())
        yearly_distribution[str(year)] = {
            region: float(100.0 * year_counts.get(region, 0) / year_total)
            for region in REGION_DIRS.keys()
        }

    fractions = {
        region: value["percent"] / 100.0 for region, value in post_distribution.items()
    }
    region_weights = {
        region: float((0.25 / fraction) if fraction > 0 else 0.0)
        for region, fraction in fractions.items()
    }

    return {
        "raw_distribution": raw_distribution,
        "post_distribution": post_distribution,
        "raw_imbalance_ratio": float(raw_counts.max() / max(raw_counts.min(), 1)),
        "post_imbalance_ratio": float(post_counts.max() / max(post_counts.min(), 1)),
        "quality": quality,
        "yearly_distribution": yearly_distribution,
        "region_weights": region_weights,
    }


def analyze_bhatagaon_spike(dedup_df: pd.DataFrame) -> dict[str, Any]:
    bhat_sept = dedup_df[
        (dedup_df["region"] == "Bhatagaon")
        & (dedup_df["timestamp"].dt.year == 2025)
        & (dedup_df["timestamp"].dt.month == 9)
    ].sort_values("timestamp")

    extreme_rows = bhat_sept[bhat_sept["pm25"] > 500].copy()
    clusters = find_spike_clusters(extreme_rows)
    cluster_results = [classify_cluster(dedup_df, cluster) for cluster in clusters]

    summary = {
        "extreme_points": int(len(extreme_rows)),
        "cluster_count": int(len(cluster_results)),
        "real_event_clusters": int(
            sum(item.classification == "REAL EVENT" for item in cluster_results)
        ),
        "sensor_error_clusters": int(
            sum(item.classification == "SENSOR ERROR" for item in cluster_results)
        ),
        "uncertain_clusters": int(
            sum(item.classification == "UNCERTAIN" for item in cluster_results)
        ),
        "max_pm25": float(extreme_rows["pm25"].max())
        if not extreme_rows.empty
        else float("nan"),
    }

    return {
        "rows": bhat_sept,
        "extreme_rows": extreme_rows,
        "clusters": clusters,
        "cluster_results": cluster_results,
        "summary": summary,
    }


def save_outputs(
    raw_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    spike_analysis: dict[str, Any],
    data_loss: dict[str, Any],
    imbalance: dict[str, Any],
    source_contribution: dict[str, Any],
    logger: logging.Logger,
) -> Path:
    raw_path = DATA_RAW_DIR / "pollution_data_raw.csv"
    dedup_path = DATA_RAW_DIR / "pollution_data_hourly_unique.csv"
    results_path = DATA_RAW_DIR / "phase1_investigation_results.json"

    raw_df.to_csv(raw_path, index=False)
    dedup_df.to_csv(dedup_path, index=False)

    serializable_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED,
            "raw_rows": int(len(raw_df)),
            "hourly_unique_rows": int(len(dedup_df)),
            "timestamp_min": str(dedup_df["timestamp"].min()),
            "timestamp_max": str(dedup_df["timestamp"].max()),
            "source_contribution": source_contribution,
        },
        "bhatagaon_spike": {
            "summary": spike_analysis["summary"],
            "clusters": [
                asdict(result) for result in spike_analysis["cluster_results"]
            ],
        },
        "data_loss": {
            "totals": data_loss["totals"],
            "region_table": data_loss["region_table"].to_dict(orient="records"),
            "recovery_analysis": data_loss["recovery_analysis"],
        },
        "region_imbalance": imbalance,
    }

    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(serializable_results, handle, indent=2)

    logger.info("Saved raw dataset: %s", raw_path)
    logger.info("Saved deduplicated dataset: %s", dedup_path)
    logger.info("Saved analysis JSON: %s", results_path)
    return results_path


def main() -> None:
    logger = setup_logging()

    raw_df = load_raw_hourly_dataset(logger)
    dedup_df = deduplicate_hourly_rows(raw_df)
    source_contribution = compute_source_contribution(raw_df, dedup_df)
    logger.info("Hourly unique rows (region+timestamp): %d", len(dedup_df))
    if source_contribution:
        unique_hours = source_contribution.get("unique_hours", {})
        pm25_contrib = source_contribution.get("pm25_availability", {})
        logger.info(
            "Combined unique hours: %s (quarterly-only=%s, overlap=%s)",
            unique_hours.get("combined"),
            unique_hours.get("quarterly_only"),
            unique_hours.get("overlap"),
        )
        logger.info(
            "PM2.5 combined coverage: hourly=%s, quarterly_fill=%s, missing_both=%s",
            pm25_contrib.get("from_hourly"),
            pm25_contrib.get("from_quarterly_fill"),
            pm25_contrib.get("missing_both"),
        )

    spike_analysis = analyze_bhatagaon_spike(dedup_df)
    create_spike_visualizations(
        dedup_df=dedup_df,
        spike_clusters=spike_analysis["clusters"],
        cluster_results=spike_analysis["cluster_results"],
    )

    data_loss = compute_data_loss_analysis(dedup_df)
    imbalance = compute_region_imbalance(dedup_df, data_loss)
    create_loss_and_imbalance_visualizations(
        data_loss=data_loss,
        dedup_df=dedup_df,
        region_weights=imbalance["region_weights"],
    )

    results_path = save_outputs(
        raw_df=raw_df,
        dedup_df=dedup_df,
        spike_analysis=spike_analysis,
        data_loss=data_loss,
        imbalance=imbalance,
        source_contribution=source_contribution,
        logger=logger,
    )

    logger.info("\nPhase 1 complete.")
    logger.info(
        "Bhatagaon extreme points (>500): %d",
        spike_analysis["summary"]["extreme_points"],
    )
    logger.info(
        "Data loss total raw -> sequence: %d -> %d",
        data_loss["totals"]["raw"],
        data_loss["totals"]["after_sequence"],
    )
    logger.info(
        "Post-preprocessing imbalance ratio: %.3fx", imbalance["post_imbalance_ratio"]
    )
    logger.info("Region weights: %s", imbalance["region_weights"])
    logger.info("Results JSON: %s", results_path)


if __name__ == "__main__":
    main()
