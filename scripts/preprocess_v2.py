"""
DECISION LOG:
- Per-region outlier handling with IQR thresholds
- Comprehensive feature engineering (extended lags, rolling stats, cyclical time)
- Preserve maximum data through flexible imputation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Optional, List
import warnings

warnings.filterwarnings("ignore")


def get_outlier_bounds(df: pd.DataFrame, column: str = "pm25") -> Tuple[float, float]:
    """Calculate IQR-based outlier bounds per region."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = max(0, Q1 - 1.5 * IQR)  # PM2.5 cannot be negative
    upper = Q3 + 1.5 * IQR
    return lower, upper


def apply_per_region_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply region-specific outlier thresholds."""
    df = df.copy()

    region_bounds = {}
    for region in df["region"].unique():
        region_data = df[df["region"] == region]
        lower, upper = get_outlier_bounds(region_data)
        region_bounds[region] = (lower, upper)
        print(f"  {region}: bounds ({lower:.1f}, {upper:.1f})")

    mask = pd.Series([False] * len(df), index=df.index)
    for region, (lower, upper) in region_bounds.items():
        region_mask = df["region"] == region
        valid = (df["pm25"] >= lower) & (df["pm25"] <= upper)
        mask = mask | (region_mask & valid)

    return df[mask].copy()


def handle_missing_values(df: pd.DataFrame, region: str = None) -> pd.DataFrame:
    """Per-region missing value handling strategies."""
    df = df.copy()

    if region:
        regions = [region]
    else:
        regions = df["region"].unique()

    for reg in regions:
        reg_mask = df["region"] == reg
        reg_df = df[reg_mask]["pm25"]

        # Step 1: Linear interpolation for gaps < 6 hours
        df.loc[reg_mask, "pm25"] = reg_df.interpolate(method="linear")

        # Step 2: Forward fill for remaining gaps
        df.loc[reg_mask, "pm25"] = df.loc[reg_mask, "pm25"].ffill()

        # Step 3: Backward fill for leading/trailing gaps
        df.loc[reg_mask, "pm25"] = df.loc[reg_mask, "pm25"].bfill()

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create extended lag features (Decision: extended lags for weekly patterns)."""
    lags = [1, 3, 6, 12, 24, 48, 168]  # Extended to include weekly

    for lag in lags:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling statistics features."""
    windows = {
        "rolling_6h_mean": 6,
        "rolling_12h_mean": 12,
        "rolling_24h_mean": 24,
        "rolling_6h_std": 6,
        "rolling_24h_std": 24,
    }

    for name, window in windows.items():
        if "mean" in name:
            df[name] = df["pm25"].rolling(window=window, min_periods=1).mean()
        else:
            df[name] = df["pm25"].rolling(window=window, min_periods=1).std()

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical time encoding features."""
    df = df.copy()

    # Hour of day (cyclical)
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Day of week (cyclical)
    df["dow"] = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Month (cyclical)
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Day of year (cyclical)
    df["doy"] = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

    return df


def create_region_features(df: pd.DataFrame, method: str = "onehot") -> pd.DataFrame:
    """Create region encoding features."""
    df = df.copy()

    if method == "onehot":
        # One-hot encoding
        regions = df["region"].unique()
        for region in regions:
            df[f"region_{region.replace(' ', '_')}"] = (df["region"] == region).astype(
                int
            )
    else:
        # Label encoding for LSTM
        le = LabelEncoder()
        df["region_encoded"] = le.fit_transform(df["region"])

    return df


def preprocess_hourly(df: pd.DataFrame, add_features: bool = True) -> pd.DataFrame:
    """Complete preprocessing pipeline for hourly data."""
    print("\n[Preprocessing Hourly Data]")

    df = df.copy()

    # Handle column naming (pm25_raw -> pm25)
    if "pm25_raw" in df.columns:
        df = df.rename(columns={"pm25_raw": "pm25"})

    # Step 1: Set datetime index
    df = df.set_index("datetime").sort_index()
    df = df[["pm25", "region"]]

    # Step 2: Resample to hourly (handle quarter-hourly input)
    df = df.groupby("region").resample("h").mean().reset_index(level=0)
    df = df.sort_index()

    # Step 3: Handle missing values (preserve data)
    print("  Handling missing values...")
    df = handle_missing_values(df)
    print(f"    After missing handling: {len(df)} records")

    # Step 4: Per-region outlier filtering
    print("  Applying per-region outlier thresholds...")
    df = apply_per_region_outliers(df)
    print(f"    After outlier removal: {len(df)} records")

    if add_features:
        # Step 5: Create lag features
        print("  Creating lag features...")
        df = create_lag_features(df)

        # Step 6: Create rolling features
        print("  Creating rolling features...")
        df = create_rolling_features(df)

        # Step 7: Create time features
        print("  Creating time features...")
        df = create_time_features(df)

        # Step 8: Create region features
        print("  Creating region features...")
        df = create_region_features(df)

    # Step 9: Drop NaN rows (from lag features)
    df = df.dropna()
    print(f"  Final: {len(df)} records")

    return df


def preprocess_quarter_hourly(
    df: pd.DataFrame, add_features: bool = True
) -> pd.DataFrame:
    """Complete preprocessing pipeline for quarter-hourly data."""
    print("\n[Preprocessing Quarter-Hourly Data]")

    df = df.copy()

    # Step 1: Set datetime index
    df = df.set_index("datetime").sort_index()
    df = df[["pm25", "region"]]

    # Step 2: Resample to 15-min (already quarter-hourly)
    # Keep as is, just handle missing values

    # Step 3: Handle missing values
    print("  Handling missing values...")
    for region in df["region"].unique():
        reg_mask = df["region"] == region
        reg_df = df[reg_mask]["pm25"]
        df.loc[reg_mask, "pm25"] = reg_df.interpolate(method="linear").ffill().bfill()
    print(f"    After missing handling: {len(df)} records")

    # Step 4: Per-region outlier filtering
    print("  Applying per-region outlier thresholds...")
    df = apply_per_region_outliers(df)
    print(f"    After outlier removal: {len(df)} records")

    if add_features:
        # Create features adapted for 15-min data
        print("  Creating 15-min features...")

        # Lags: 1, 4 (1hr), 12 (3hr), 24 (6hr), 48 (12hr), 96 (24hr)
        lags = [1, 4, 12, 24, 48, 96]
        for lag in lags:
            df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

        # Rolling features (adapted for 15-min)
        df["rolling_4h_mean"] = df["pm25"].rolling(window=4, min_periods=1).mean()
        df["rolling_24h_mean"] = df["pm25"].rolling(window=24, min_periods=1).mean()
        df["rolling_96h_mean"] = df["pm25"].rolling(window=96, min_periods=1).mean()

        # Time features
        df["minute"] = df.index.minute
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

        df["hour"] = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["dow"] = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

        # Region features
        df = create_region_features(df)

    # Drop NaN
    df = df.dropna()
    print(f"  Final: {len(df)} records")

    return df


def create_horizon_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create target variable for specific prediction horizon."""
    df = df.copy()

    if horizon == 1:
        # t+1: Next hour
        df["target"] = df["pm25"].shift(-1)
    elif horizon == 6:
        # t+6: 6 hours ahead
        df["target"] = df["pm25"].shift(-6)
    elif horizon == 24:
        # t+24: 24 hours ahead
        df["target"] = df["pm25"].shift(-24)
    else:
        raise ValueError(f"Unknown horizon: {horizon}")

    # Remove rows where target is NaN (end of series)
    df = df.dropna(subset=["target"])

    return df


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_scaler: Optional[StandardScaler] = None,
) -> Tuple:
    """Scale features using StandardScaler (fit on train only)."""
    feature_cols = [c for c in train.columns if c not in ["pm25", "target", "region"]]

    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    train_scaled[feature_cols] = scaler.transform(train[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])

    # Also scale target if provided
    if target_scaler is not None:
        target_scaler.fit(train[["target"]])
        train_scaled["target"] = target_scaler.transform(train[["target"]]).ravel()
        val_scaled["target"] = target_scaler.transform(val[["target"]]).ravel()
        test_scaled["target"] = target_scaler.transform(test[["target"]]).ravel()

    return train_scaled, val_scaled, test_scaled, scaler


def prepare_xy(
    df: pd.DataFrame, exclude_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare X and y arrays."""
    if len(df) == 0:
        raise ValueError("Empty DataFrame provided to prepare_xy")

    if exclude_cols is None:
        exclude_cols = ["pm25", "target", "region"]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    y = df["target"].values

    return X, y


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from scripts.load_data_v2 import load_all_regions, load_quarter_hourly_data

    print("=" * 60)
    print("Loading Data")
    print("=" * 60)

    df_hourly = load_all_regions([2022, 2023, 2024, 2025])
    print(f"Hourly: {len(df_hourly)} records")

    # Preprocess hourly
    df_processed = preprocess_hourly(df_hourly)
    print(f"\nProcessed hourly: {len(df_processed)}")
    print(f"Columns: {df_processed.columns.tolist()}")
    print(f"\nFeature summary:")
    print(df_processed.describe())
