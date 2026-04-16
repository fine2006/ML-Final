"""
Shared utilities for Phase 3 preprocessing scripts.

This module enforces the same canonical merge logic as Phase 1:
- parse hourly and quarterly sources
- floor timestamps to hourly resolution
- prefer hourly values where overlap exists
- fill missing hourly coverage from quarterly values
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


RAW_DATA_PATH = DATA_RAW_DIR / "pollution_data_raw.csv"
CANONICAL_DATA_PATH = DATA_RAW_DIR / "pollution_data_hourly_unique.csv"
PHASE1_RESULTS_PATH = DATA_RAW_DIR / "phase1_investigation_results.json"


SENSOR_SPIKE_DEFAULT = {
    "Bhatagaon": [
        pd.Timestamp("2025-09-01 23:00:00"),
        pd.Timestamp("2025-09-02 00:00:00"),
        pd.Timestamp("2025-09-10 08:00:00"),
        pd.Timestamp("2025-09-10 09:00:00"),
        pd.Timestamp("2025-09-11 03:00:00"),
        pd.Timestamp("2025-09-11 04:00:00"),
        pd.Timestamp("2025-09-11 05:00:00"),
    ]
}


POLLUTANT_COLUMNS = ["pm25", "pm10", "no2", "o3"]
WEATHER_COLUMNS = ["temperature", "humidity", "wind_speed", "wind_direction"]


def sanitize_impossible_values(
    df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convert sentinel/impossible measurements to NaN so downstream interpolation
    and gap policies can handle them consistently.
    """
    out = df.copy()
    stats: dict[str, dict[str, int]] = {
        col: {
            "sentinel_to_nan": 0,
            "invalid_to_nan": 0,
            "high_clip_to_nan": 0,
        }
        for col in POLLUTANT_COLUMNS + WEATHER_COLUMNS
    }

    for col in POLLUTANT_COLUMNS + WEATHER_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

        sentinel_mask = out[col] <= -1_000_000_000
        sentinel_count = int(sentinel_mask.sum())
        if sentinel_count:
            out.loc[sentinel_mask, col] = np.nan
            stats[col]["sentinel_to_nan"] = sentinel_count

    # Pollutants cannot be negative concentrations.
    for col in POLLUTANT_COLUMNS:
        invalid_mask = out[col] < 0
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            out.loc[invalid_mask, col] = np.nan
            stats[col]["invalid_to_nan"] += invalid_count

    # PM2.5 hard sensor saturation upper bound.
    pm25_high = out["pm25"] > 5000
    pm25_high_count = int(pm25_high.sum())
    if pm25_high_count:
        out.loc[pm25_high, "pm25"] = np.nan
        stats["pm25"]["high_clip_to_nan"] += pm25_high_count

    # Weather physical constraints.
    humidity_invalid = (out["humidity"] < 0) | (out["humidity"] > 100)
    humidity_invalid_count = int(humidity_invalid.sum())
    if humidity_invalid_count:
        out.loc[humidity_invalid, "humidity"] = np.nan
        stats["humidity"]["invalid_to_nan"] += humidity_invalid_count

    wind_speed_invalid = out["wind_speed"] < 0
    wind_speed_invalid_count = int(wind_speed_invalid.sum())
    if wind_speed_invalid_count:
        out.loc[wind_speed_invalid, "wind_speed"] = np.nan
        stats["wind_speed"]["invalid_to_nan"] += wind_speed_invalid_count

    wind_dir_invalid = (out["wind_direction"] < 0) | (out["wind_direction"] > 360)
    wind_dir_invalid_count = int(wind_dir_invalid.sum())
    if wind_dir_invalid_count:
        out.loc[wind_dir_invalid, "wind_direction"] = np.nan
        stats["wind_direction"]["invalid_to_nan"] += wind_dir_invalid_count

    total_fixed = 0
    for col, payload in stats.items():
        changed = int(sum(payload.values()))
        total_fixed += changed
        if changed:
            logger.info(
                "Sanitized %s values in %s (sentinel=%d invalid=%d high_clip=%d)",
                changed,
                col,
                payload["sentinel_to_nan"],
                payload["invalid_to_nan"],
                payload["high_clip_to_nan"],
            )
    if total_fixed:
        logger.info(
            "Total impossible/sentinel values converted to NaN: %d", total_fixed
        )

    return out, {
        "sentinel_threshold": -1_000_000_000,
        "rules": {
            "pollutants_non_negative": True,
            "pm25_max": 5000,
            "humidity_range": [0, 100],
            "wind_speed_min": 0,
            "wind_direction_range": [0, 360],
        },
        "stats": stats,
    }


def _load_phase1_results() -> dict[str, Any]:
    if not PHASE1_RESULTS_PATH.exists():
        return {}
    with open(PHASE1_RESULTS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _phase1_has_quarterly_merge(results: dict[str, Any]) -> bool:
    source = results.get("metadata", {}).get("source_contribution", {})
    rows_by_granularity = source.get("rows_by_granularity", {})
    unique_hours = source.get("unique_hours", {})
    return (
        rows_by_granularity.get("quarterly", 0) > 0
        and unique_hours.get("quarterly_only", 0) > 0
    )


def ensure_canonical_dataset(
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load canonical dataset produced by Phase 1.

    If missing (or missing quarterly merge metadata), rebuild using the exact
    Phase 1 loader + merge functions from scripts/data_investigation.py.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    phase1_results = _load_phase1_results()

    needs_rebuild = not CANONICAL_DATA_PATH.exists()
    if (
        not needs_rebuild
        and phase1_results
        and not _phase1_has_quarterly_merge(phase1_results)
    ):
        logger.info(
            "Canonical dataset exists but lacks quarterly-merge metadata; rebuilding."
        )
        needs_rebuild = True

    if needs_rebuild:
        logger.info("Building canonical dataset using Phase 1 merge logic...")
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from scripts.data_investigation import (  # local import to avoid cycles at module import time
            deduplicate_hourly_rows,
            load_raw_hourly_dataset,
        )

        raw_df = load_raw_hourly_dataset(logger=logger, include_quarterly=True)
        canonical_df = deduplicate_hourly_rows(raw_df)
        raw_df.to_csv(RAW_DATA_PATH, index=False)
        canonical_df.to_csv(CANONICAL_DATA_PATH, index=False)
        logger.info("Saved rebuilt raw dataset: %s", RAW_DATA_PATH)
        logger.info("Saved rebuilt canonical dataset: %s", CANONICAL_DATA_PATH)

        phase1_results = _load_phase1_results()

    canonical_df = pd.read_csv(CANONICAL_DATA_PATH, parse_dates=["timestamp"])
    canonical_df["timestamp"] = pd.to_datetime(
        canonical_df["timestamp"], errors="coerce"
    )
    canonical_df = canonical_df.dropna(subset=["timestamp"]).copy()
    canonical_df["timestamp"] = canonical_df["timestamp"].dt.floor("h")
    canonical_df = canonical_df.sort_values(["region", "timestamp"]).reset_index(
        drop=True
    )
    return canonical_df, phase1_results


def get_sensor_error_timestamps(
    phase1_results: dict[str, Any],
) -> dict[str, set[pd.Timestamp]]:
    spikes: dict[str, set[pd.Timestamp]] = {
        region: set() for region in SENSOR_SPIKE_DEFAULT
    }

    for region, timestamps in SENSOR_SPIKE_DEFAULT.items():
        spikes.setdefault(region, set()).update(timestamps)

    clusters = phase1_results.get("bhatagaon_spike", {}).get("clusters", [])
    for cluster in clusters:
        if cluster.get("classification") != "SENSOR ERROR":
            continue
        start = pd.to_datetime(cluster.get("start"), errors="coerce")
        end = pd.to_datetime(cluster.get("end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        for ts in pd.date_range(start.floor("h"), end.floor("h"), freq="h"):
            spikes.setdefault("Bhatagaon", set()).add(ts)

    return spikes


def remove_sensor_error_rows(
    df: pd.DataFrame,
    sensor_spikes: dict[str, set[pd.Timestamp]],
    logger: logging.Logger,
) -> pd.DataFrame:
    if not sensor_spikes:
        return df

    filtered = df.copy()
    removed_total = 0
    for region, timestamp_set in sensor_spikes.items():
        if not timestamp_set:
            continue
        mask = (filtered["region"] == region) & (
            filtered["timestamp"].isin(timestamp_set)
        )
        removed = int(mask.sum())
        if removed:
            logger.info("Removing %d sensor-error rows for region=%s", removed, region)
            filtered = filtered.loc[~mask].copy()
            removed_total += removed

    if removed_total:
        logger.info("Removed total sensor-error rows: %d", removed_total)
    return filtered


def split_timestamps(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, pd.Timestamp]:
    unique_timestamps = pd.Index(sorted(df["timestamp"].dropna().unique()))
    if len(unique_timestamps) < 3:
        raise ValueError("Insufficient timestamps to create train/val/test split.")

    train_end_idx = max(0, int(len(unique_timestamps) * train_ratio) - 1)
    val_end_idx = max(
        train_end_idx + 1, int(len(unique_timestamps) * (train_ratio + val_ratio)) - 1
    )
    val_end_idx = min(val_end_idx, len(unique_timestamps) - 2)

    train_end = pd.Timestamp(unique_timestamps[train_end_idx])
    val_end = pd.Timestamp(unique_timestamps[val_end_idx])

    return {
        "train_end": train_end,
        "val_end": val_end,
        "test_start": val_end + pd.Timedelta(hours=1),
    }


def apply_time_split(
    df: pd.DataFrame, boundaries: dict[str, pd.Timestamp]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["timestamp"] <= boundaries["train_end"]].copy()
    val_df = df[
        (df["timestamp"] > boundaries["train_end"])
        & (df["timestamp"] <= boundaries["val_end"])
    ].copy()
    test_df = df[df["timestamp"] > boundaries["val_end"]].copy()
    return train_df, val_df, test_df


def load_region_weights(phase1_results: dict[str, Any]) -> dict[str, float]:
    weights = phase1_results.get("region_imbalance", {}).get("region_weights", {})
    if weights:
        return {str(k): float(v) for k, v in weights.items()}
    return {"Bhatagaon": 1.0, "AIIMS": 1.0, "IGKV": 1.0, "SILTARA": 1.0}
