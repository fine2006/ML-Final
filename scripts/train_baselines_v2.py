"""
DECISION LOG:
- Expanded baseline models: Linear Regression (Ridge), Random Forest, XGBoost, LightGBM
- Comprehensive hyperparameter configuration with tqdm progress
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def train_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    alpha: float = 1.0,
) -> Ridge:
    """Train Ridge Regression model with L2 regularization."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train Random Forest model with increased complexity."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        max_samples=0.8,  # Use 80% of samples
        max_features=0.7,  # Use 70% of features
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    n_estimators: int = 800,
    learning_rate: float = 0.03,
    max_depth: int = 8,
    random_state: int = 42,
    early_stopping_rounds: int = 50,
) -> xgb.XGBRegressor:
    """Train XGBoost model with tuned hyperparameters."""

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    if X_val is not None and y_val is not None:
        params["early_stopping_rounds"] = early_stopping_rounds

    model = xgb.XGBRegressor(**params)

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    n_estimators: int = 800,
    learning_rate: float = 0.03,
    num_leaves: int = 63,
    random_state: int = 42,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMRegressor:
    """Train LightGBM model with tuned hyperparameters."""

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }

    callbacks = []
    if X_val is not None and y_val is not None:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    model = lgb.LGBMRegressor(**params)

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    else:
        model.fit(X_train, y_train)

    return model


def evaluate_model(
    model, X: np.ndarray, y: np.ndarray, model_name: str = "Model"
) -> Dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    return {
        "model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "predictions": y_pred,
        "actuals": y,
    }


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """Train all baseline models and return results with progress bars."""

    results = []

    print("\n[Training Baseline Models]")

    # Ridge Regression with tqdm
    print("\n  Training Ridge Regression...")
    with tqdm(total=1, desc="Ridge Regression") as pbar:
        ridge = train_ridge_regression(X_train, y_train)
        ridge_result = evaluate_model(ridge, X_test, y_test, "Ridge Regression")
        results.append(ridge_result)
        pbar.update(1)
    print(f"    Test RMSE: {ridge_result['RMSE']:.4f}")

    # Random Forest with tqdm
    print("\n  Training Random Forest...")
    with tqdm(total=1, desc="Random Forest (300 trees)") as pbar:
        rf = train_random_forest(X_train, y_train, n_estimators=300, max_depth=None)
        rf_result = evaluate_model(rf, X_test, y_test, "Random Forest")
        results.append(rf_result)
        pbar.update(1)
    print(f"    Test RMSE: {rf_result['RMSE']:.4f}")

    # XGBoost with tqdm
    print("\n  Training XGBoost...")
    with tqdm(total=1, desc="XGBoost (800 rounds)") as pbar:
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        xgb_result = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        results.append(xgb_result)
        pbar.update(1)
    print(f"    Test RMSE: {xgb_result['RMSE']:.4f}")

    # LightGBM with tqdm
    print("\n  Training LightGBM...")
    with tqdm(total=1, desc="LightGBM (800 rounds)") as pbar:
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
        lgb_result = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
        results.append(lgb_result)
        pbar.update(1)
    print(f"    Test RMSE: {lgb_result['RMSE']:.4f}")

    return {r["model"]: r for r in results}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from scripts.load_data_v2 import load_all_regions
    from scripts.preprocess_v2 import (
        preprocess_hourly,
        create_horizon_targets,
        chronological_split,
        scale_features,
        prepare_xy,
    )

    print("Loading data...")
    df = load_all_regions([2022, 2023, 2024, 2025])

    print("Preprocessing...")
    df_proc = preprocess_hourly(df)
    df_t1 = create_horizon_targets(df_proc, 1)
    train, val, test = chronological_split(df_t1)

    # Scale
    train_scaled, val_scaled, test_scaled, scaler = scale_features(train, val, test)
    X_train, y_train = prepare_xy(train_scaled)
    X_val, y_val = prepare_xy(val_scaled)
    X_test, y_test = prepare_xy(test_scaled)

    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train all baselines
    results = train_all_baselines(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n=== Results ===")
    for name, res in results.items():
        print(
            f"{name}: RMSE={res['RMSE']:.4f}, MAE={res['MAE']:.4f}, R2={res['R2']:.4f}"
        )
