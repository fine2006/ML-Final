"""
Phase 3: LSTM preprocessing pipeline.

Uses the canonical hourly+quarterly merge contract from Phase 1.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_BOOTSTRAP = SCRIPT_DIR.parent
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.preprocess_common import (
    PROJECT_ROOT,
    apply_time_split,
    ensure_canonical_dataset,
    get_sensor_error_timestamps,
    load_region_weights,
    remove_sensor_error_rows,
    sanitize_impossible_values,
    split_timestamps,
)


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "preprocessed_lstm_v1"
LOG_DIR = PROJECT_ROOT / "logs" / "preprocess_lstm"


FEATURE_COLUMNS = [
    "pm25",
    "pm10",
    "no2",
    "o3",
    "temperature_lag_1",
    "humidity_lag_1",
    "wind_speed_lag_1",
    "wind_direction_lag_1",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "region_id",
]


HORIZONS = [1, 12, 24, 168, 672]


def setup_logging() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("preprocess_lstm")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"preprocess_lstm_{stamp}.log"

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
    return logger


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out["timestamp"].dt.hour
    day = out["timestamp"].dt.dayofweek
    month = out["timestamp"].dt.month

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_sin"] = np.sin(2 * np.pi * day / 7)
    out["day_cos"] = np.cos(2 * np.pi * day / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def add_region_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    regions = sorted(out["region"].dropna().unique().tolist())
    mapping = {region: idx for idx, region in enumerate(regions)}
    out["region_id"] = out["region"].map(mapping).astype(float)
    return out, mapping


def add_weather_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["region", "timestamp"]).copy()
    for col in ["temperature", "humidity", "wind_speed", "wind_direction"]:
        out[f"{col}_lag_1"] = out.groupby("region")[col].shift(1)
    return out


def apply_lstm_missing_policy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pieces: list[pd.DataFrame] = []
    stats: dict[str, Any] = {}

    for region, region_df in df.groupby("region"):
        region_df = region_df.sort_values("timestamp").copy()
        full_index = pd.date_range(
            region_df["timestamp"].min(),
            region_df["timestamp"].max(),
            freq="h",
        )

        reindexed = (
            region_df.set_index("timestamp")
            .reindex(full_index)
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )
        reindexed["region"] = region

        numeric_cols = [
            "pm25",
            "pm10",
            "no2",
            "o3",
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
        ]
        for col in numeric_cols:
            reindexed[col] = reindexed[col].interpolate(
                method="linear",
                limit=5,
                limit_direction="both",
            )

        before_drop = len(reindexed)
        reindexed = reindexed.dropna(subset=numeric_cols).copy()
        dropped = before_drop - len(reindexed)

        stats[region] = {
            "hours_total": int(before_drop),
            "hours_after_interpolation": int(len(reindexed)),
            "hours_dropped_due_to_long_gaps": int(dropped),
        }
        pieces.append(reindexed)

    merged = pd.concat(pieces, ignore_index=True)
    merged = merged.sort_values(["region", "timestamp"]).reset_index(drop=True)
    return merged, stats


def robust_scale(
    train_df: pd.DataFrame, apply_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = apply_df.copy()
    scaler: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLUMNS:
        if col == "region_id":
            out[col] = out[col].astype(float)
            scaler[col] = {"median": 0.0, "iqr": 1.0}
            continue

        median = float(train_df[col].median())
        q1 = float(train_df[col].quantile(0.25))
        q3 = float(train_df[col].quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            iqr = 1.0

        out[col] = (out[col] - median) / iqr
        scaler[col] = {"median": median, "iqr": float(iqr)}
    return out, scaler


def robust_scale_from_reference(
    reference_df: pd.DataFrame, apply_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = apply_df.copy()
    scaler: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLUMNS:
        if col == "region_id":
            out[col] = out[col].astype(float)
            scaler[col] = {"median": 0.0, "iqr": 1.0}
            continue

        median = float(reference_df[col].median())
        q1 = float(reference_df[col].quantile(0.25))
        q3 = float(reference_df[col].quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            iqr = 1.0

        out[col] = (out[col] - median) / iqr
        scaler[col] = {"median": median, "iqr": float(iqr)}
    return out, scaler


def add_horizon_targets(
    df: pd.DataFrame, horizons: list[int]
) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values(["region", "timestamp"]).copy()
    target_columns: list[str] = []
    for pollutant in ["pm25", "pm10", "no2", "o3"]:
        for horizon in horizons:
            col = f"target_{pollutant}_h{horizon}"
            out[col] = out.groupby("region")[pollutant].shift(-horizon)
            target_columns.append(col)
    return out, target_columns


def estimate_sequence_counts(
    df: pd.DataFrame, horizons: list[int]
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for horizon in horizons:
        seq_len = max(2 * horizon, 2)
        total = 0
        by_region: dict[str, int] = {}
        for region, region_df in df.groupby("region"):
            n = len(region_df)
            count = max(0, n - seq_len - horizon)
            by_region[region] = int(count)
            total += count
        stats[f"h{horizon}"] = {
            "seq_len": int(seq_len),
            "total": int(total),
            **{f"region_{k}": int(v) for k, v in by_region.items()},
        }
    return stats


def save_split_table(prefix: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT_DIR / f"{prefix}.csv", index=False)


def main() -> None:
    logger = setup_logging()
    logger.info("Starting LSTM preprocessing with shared canonical merge logic")

    canonical_df, phase1_results = ensure_canonical_dataset(logger)
    logger.info("Canonical rows loaded: %d", len(canonical_df))

    sensor_spikes = get_sensor_error_timestamps(phase1_results)
    canonical_df = remove_sensor_error_rows(canonical_df, sensor_spikes, logger)

    canonical_df, impossible_stats = sanitize_impossible_values(canonical_df, logger)

    # LSTM policy keeps outliers except impossible values already filtered.
    canonical_df = canonical_df[
        (canonical_df["pm25"].isna())
        | ((canonical_df["pm25"] >= 0) & (canonical_df["pm25"] <= 5000))
    ].copy()

    missing_handled_df, gap_stats = apply_lstm_missing_policy(canonical_df)
    missing_handled_df = add_time_features(missing_handled_df)
    missing_handled_df, region_map = add_region_ids(missing_handled_df)
    missing_handled_df = add_weather_lags(missing_handled_df)

    boundaries = split_timestamps(missing_handled_df)
    train_df, val_df, test_df = apply_time_split(missing_handled_df, boundaries)
    logger.info(
        "Time split rows -> train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    train_targeted_raw, target_columns = add_horizon_targets(train_df, HORIZONS)
    val_targeted_raw, _ = add_horizon_targets(val_df, HORIZONS)
    test_targeted_raw, _ = add_horizon_targets(test_df, HORIZONS)

    train_targeted, scaler = robust_scale(train_df, train_targeted_raw)
    val_targeted, _ = robust_scale_from_reference(train_targeted_raw, val_targeted_raw)
    test_targeted, _ = robust_scale_from_reference(
        train_targeted_raw, test_targeted_raw
    )

    required_cols = FEATURE_COLUMNS + target_columns
    train_targeted = train_targeted.dropna(subset=required_cols).copy()
    val_targeted = val_targeted.dropna(subset=required_cols).copy()
    test_targeted = test_targeted.dropna(subset=required_cols).copy()

    save_split_table("train", train_targeted)
    save_split_table("val", val_targeted)
    save_split_table("test", test_targeted)

    sequence_estimate = {
        "train": estimate_sequence_counts(train_targeted, HORIZONS),
        "val": estimate_sequence_counts(val_targeted, HORIZONS),
        "test": estimate_sequence_counts(test_targeted, HORIZONS),
    }

    metadata: dict[str, Any] = {
        "pipeline": "lstm",
        "version": "v1",
        "random_seed": RANDOM_SEED,
        "created_at": datetime.now().isoformat(),
        "source": {
            "canonical_path": str(
                (
                    PROJECT_ROOT / "data" / "raw" / "pollution_data_hourly_unique.csv"
                ).resolve()
            ),
            "merge_policy": "hourly_preferred_quarterly_fill",
            "floor_policy": "timestamp.floor('h') for hourly and quarterly",
            "source_contribution": phase1_results.get("metadata", {}).get(
                "source_contribution", {}
            ),
        },
        "sensor_error_removal": {
            "enabled": True,
            "spike_counts": {
                region: len(values) for region, values in sensor_spikes.items()
            },
        },
        "impossible_value_policy": impossible_stats,
        "split_boundaries": {k: str(v) for k, v in boundaries.items()},
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": target_columns,
        "horizons": HORIZONS,
        "region_mapping": region_map,
        "region_weights": load_region_weights(phase1_results),
        "gap_policy": {
            "interpolate_limit_hours": 5,
            "drop_after_interpolation_if_nan": True,
            "stats": gap_stats,
        },
        "leakage_policy": {
            "weather_features_use_lagged_values_only": True,
            "weather_lags_included": [1],
            "forbidden": [
                "temperature[t]",
                "humidity[t]",
                "wind_speed[t]",
                "wind_direction[t]",
            ],
        },
        "sequence_materialization": {
            "representation": "rowwise_features_with_targets",
            "note": "Sequences are materialized on-the-fly during training using seq_len=2*horizon.",
            "estimated_counts": sequence_estimate,
        },
        "scaler": scaler,
        "shapes": {
            "train_rows": int(len(train_targeted)),
            "val_rows": int(len(val_targeted)),
            "test_rows": int(len(test_targeted)),
            "train_columns": int(train_targeted.shape[1]),
            "val_columns": int(val_targeted.shape[1]),
            "test_columns": int(test_targeted.shape[1]),
        },
    }

    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Saved LSTM preprocessed outputs to %s", OUT_DIR)
    logger.info(
        "Row counts -> train=%d val=%d test=%d",
        len(train_targeted),
        len(val_targeted),
        len(test_targeted),
    )


if __name__ == "__main__":
    main()
