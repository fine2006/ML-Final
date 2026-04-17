"""
Phase 3: XGB preprocessing pipeline.

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
OUT_DIR = DATA_DIR / "preprocessed_xgb_v1"
LOG_DIR = PROJECT_ROOT / "logs" / "preprocess_xgb"


POLLUTANTS = ["pm25", "pm10", "no2", "o3"]
WEATHER = ["temperature", "humidity", "wind_speed", "wind_direction"]
LAGS = [1, 3, 6, 12, 24, 48, 168]
WEATHER_LAGS = [1, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24]
HORIZONS = [1, 24, 168]

# Apply explicit outlier clipping policy consistently across all pollutants.
# These are XGB-only caps (LSTM keeps plausible outliers by design).
XGB_POLLUTANT_CAPS = {
    "pm25": 300.0,
    "pm10": 600.0,
    "no2": 250.0,
    "o3": 150.0,
}


def setup_logging() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("preprocess_xgb")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"preprocess_xgb_{stamp}.log"

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


def _missingness_features(observed: pd.Series) -> pd.DataFrame:
    observed = observed.fillna(False).astype(bool).reset_index(drop=True)
    n = len(observed)

    interpolation_flag = (~observed).astype(int).to_numpy()
    hours_since = np.zeros(n, dtype=np.float32)
    interpolation_distance = np.zeros(n, dtype=np.float32)
    consecutive_gaps = np.zeros(n, dtype=np.float32)

    gap_lengths: list[int] = []
    start = None
    for idx, flag in enumerate(observed.to_numpy()):
        if flag:
            if start is not None:
                gap_lengths.append(idx - start)
                start = None
        else:
            if start is None:
                start = idx
    if start is not None:
        gap_lengths.append(n - start)

    max_gap = max(gap_lengths) if gap_lengths else 1
    measurement_frequency = float(observed.mean() * 100.0)

    last_seen = None
    idx = 0
    arr = observed.to_numpy()
    while idx < n:
        if arr[idx]:
            last_seen = idx
            idx += 1
            continue

        gap_start = idx
        while idx < n and not arr[idx]:
            idx += 1
        gap_end = idx
        gap_len = gap_end - gap_start

        for offset, pos in enumerate(range(gap_start, gap_end), start=1):
            if last_seen is None:
                hours_since[pos] = float(offset)
            else:
                hours_since[pos] = float(pos - last_seen)
            interpolation_distance[pos] = float(gap_len)
            consecutive_gaps[pos] = float(offset)

    gap_ratio = np.where(
        interpolation_flag == 1, (hours_since / float(max_gap)) * 100.0, 0.0
    )

    return pd.DataFrame(
        {
            "hours_since_measurement": hours_since,
            "interpolation_distance": interpolation_distance,
            "consecutive_gaps": consecutive_gaps,
            "gap_ratio": gap_ratio.astype(np.float32),
            "measurement_frequency": np.full(
                n, measurement_frequency, dtype=np.float32
            ),
            "interpolation_flag": interpolation_flag.astype(np.float32),
        }
    )


def apply_xgb_missing_policy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pieces: list[pd.DataFrame] = []
    stats: dict[str, Any] = {}

    for region, region_df in df.groupby("region"):
        region_df = region_df.sort_values("timestamp").copy()
        full_index = pd.date_range(
            region_df["timestamp"].min(),
            region_df["timestamp"].max(),
            freq="h",
        )

        base = (
            region_df.set_index("timestamp")
            .reindex(full_index)
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )
        base["region"] = region

        observed_pm25 = base["pm25"].notna().copy()
        missing_features = _missingness_features(observed_pm25)

        numeric_cols = POLLUTANTS + WEATHER
        for col in numeric_cols:
            base[col] = base[col].interpolate(method="linear").ffill().bfill()

        merged = pd.concat([base, missing_features], axis=1)
        pieces.append(merged)

        stats[region] = {
            "hours_total": int(len(merged)),
            "original_pm25_missing_hours": int((~observed_pm25).sum()),
            "measurement_frequency_pct": float(observed_pm25.mean() * 100.0),
        }

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(["region", "timestamp"]).reset_index(drop=True)
    return out, stats


def add_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["region", "timestamp"]).copy()

    for weather_col in WEATHER:
        out[weather_col] = out.groupby("region")[weather_col].shift(1)

    for pollutant in POLLUTANTS:
        for lag in LAGS:
            out[f"{pollutant}_lag_{lag}"] = out.groupby("region")[pollutant].shift(lag)
        for window in ROLLING_WINDOWS:
            out[f"{pollutant}_roll_mean_{window}"] = out.groupby("region")[
                pollutant
            ].transform(lambda s: s.rolling(window=window, min_periods=window).mean())
            out[f"{pollutant}_roll_std_{window}"] = out.groupby("region")[
                pollutant
            ].transform(lambda s: s.rolling(window=window, min_periods=window).std())

    for weather_col in WEATHER:
        for lag in WEATHER_LAGS:
            out[f"{weather_col}_lag_{lag}"] = out.groupby("region")[weather_col].shift(
                lag
            )

    out["hour"] = out["timestamp"].dt.hour.astype(int)
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype(int)
    out["day_of_month"] = out["timestamp"].dt.day.astype(int)
    out["month"] = out["timestamp"].dt.month.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["season"] = ((out["month"] % 12) // 3).astype(int)

    region_dummies = pd.get_dummies(out["region"], prefix="region")
    out = pd.concat([out, region_dummies], axis=1)
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for pollutant in POLLUTANTS:
        for horizon in HORIZONS:
            out[f"target_{pollutant}_h{horizon}"] = out.groupby("region")[
                pollutant
            ].shift(-horizon)
    return out


def apply_xgb_outlier_caps(
    df: pd.DataFrame,
    caps: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    n_before = len(out)

    keep_mask = pd.Series(True, index=out.index)
    removal_by_pollutant: dict[str, int] = {}

    for pollutant, cap in caps.items():
        invalid = (~out[pollutant].isna()) & (
            (out[pollutant] < 0) | (out[pollutant] > float(cap))
        )
        removal_by_pollutant[pollutant] = int(invalid.sum())
        keep_mask &= ~invalid

    out = out.loc[keep_mask].copy()
    n_after = len(out)

    stats = {
        "caps": {k: float(v) for k, v in caps.items()},
        "rows_before": int(n_before),
        "rows_after": int(n_after),
        "rows_removed_total": int(n_before - n_after),
        "rows_flagged_by_pollutant": removal_by_pollutant,
    }
    return out, stats


def split_feature_target_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [
        c
        for c in df.columns
        if c not in {"timestamp", "region"} and not c.startswith("target_")
    ]
    return feature_cols, target_cols


def persist_split(name: str, X: pd.DataFrame, y: pd.DataFrame) -> None:
    X.to_csv(OUT_DIR / f"X_{name}.csv", index=False)
    y.to_csv(OUT_DIR / f"y_{name}.csv", index=False)


def main() -> None:
    logger = setup_logging()
    logger.info("Starting XGB preprocessing with shared canonical merge logic")

    canonical_df, phase1_results = ensure_canonical_dataset(logger)
    logger.info("Canonical rows loaded: %d", len(canonical_df))

    sensor_spikes = get_sensor_error_timestamps(phase1_results)
    canonical_df = remove_sensor_error_rows(canonical_df, sensor_spikes, logger)

    canonical_df, impossible_stats = sanitize_impossible_values(canonical_df, logger)

    canonical_df = canonical_df[
        (canonical_df["pm25"].isna())
        | ((canonical_df["pm25"] >= 0) & (canonical_df["pm25"] <= 5000))
    ].copy()

    xgb_df, missing_stats = apply_xgb_missing_policy(canonical_df)

    xgb_df, outlier_stats = apply_xgb_outlier_caps(xgb_df, XGB_POLLUTANT_CAPS)
    logger.info(
        "Removed %d rows by XGB pollutant caps policy",
        outlier_stats["rows_removed_total"],
    )
    logger.info(
        "Rows flagged by pollutant caps: %s",
        outlier_stats["rows_flagged_by_pollutant"],
    )

    xgb_df = add_xgb_features(xgb_df)
    boundaries = split_timestamps(xgb_df)
    train_df, val_df, test_df = apply_time_split(xgb_df, boundaries)

    logger.info(
        "Time split rows -> train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    train_df = add_targets(train_df)
    val_df = add_targets(val_df)
    test_df = add_targets(test_df)

    feature_cols, target_cols = split_feature_target_columns(train_df)

    train_df = train_df.dropna(subset=feature_cols + target_cols).copy()
    val_df = val_df.dropna(subset=feature_cols + target_cols).copy()
    test_df = test_df.dropna(subset=feature_cols + target_cols).copy()

    X_train = train_df[["timestamp", "region"] + feature_cols].copy()
    y_train = train_df[["timestamp", "region"] + target_cols].copy()
    X_val = val_df[["timestamp", "region"] + feature_cols].copy()
    y_val = val_df[["timestamp", "region"] + target_cols].copy()
    X_test = test_df[["timestamp", "region"] + feature_cols].copy()
    y_test = test_df[["timestamp", "region"] + target_cols].copy()

    persist_split("train", X_train, y_train)
    persist_split("val", X_val, y_val)
    persist_split("test", X_test, y_test)

    metadata: dict[str, Any] = {
        "pipeline": "xgb",
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
            "phase1_all_pollutants_loss_totals": {
                pollutant: payload.get("totals", {})
                for pollutant, payload in phase1_results.get(
                    "all_pollutants_loss", {}
                ).items()
            },
            "phase1_all_pollutants_outlier_summary": {
                pollutant: {
                    "quantiles": payload.get("quantiles", {}),
                    "thresholds": payload.get("thresholds", {}),
                    "fraction_above": payload.get("fraction_above", {}),
                }
                for pollutant, payload in phase1_results.get(
                    "all_pollutants_outlier_analysis", {}
                )
                .get("pollutants", {})
                .items()
            },
        },
        "sensor_error_removal": {
            "enabled": True,
            "spike_counts": {
                region: len(values) for region, values in sensor_spikes.items()
            },
        },
        "impossible_value_policy": impossible_stats,
        "split_boundaries": {k: str(v) for k, v in boundaries.items()},
        "pollutants": POLLUTANTS,
        "horizons": HORIZONS,
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "region_weights": load_region_weights(phase1_results),
        "missing_policy": {
            "interpolate_all_gaps": True,
            "missingness_feature_count": 6,
            "stats": missing_stats,
        },
        "leakage_policy": {
            "weather_features_use_lagged_values_only": True,
            "weather_current_values_shifted_by": 1,
            "forbidden": [
                "temperature[t]",
                "humidity[t]",
                "wind_speed[t]",
                "wind_direction[t]",
            ],
        },
        "outlier_policy": outlier_stats,
        "shapes": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val.shape),
            "y_val": list(y_val.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
    }

    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Saved XGB preprocessed outputs to %s", OUT_DIR)
    logger.info(
        "Row counts after feature+target filtering -> train=%d val=%d test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )


if __name__ == "__main__":
    main()
