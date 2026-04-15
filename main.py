"""
DECISION LOG:
- Model persistence: Save trained models to disk
- Benchmark mode: Load models if available, skip training
- --retrain-all flag: Force retrain even if models exist
- tqdm progress bars for training and benchmarking
"""

import sys
import os
import pickle
import torch
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from scripts.load_data_v2 import load_all_regions
from scripts.preprocess_v2 import (
    preprocess_hourly,
    create_horizon_targets,
    chronological_split,
    scale_features,
    prepare_xy,
)
from scripts.train_baselines_v2 import (
    train_ridge_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    evaluate_model,
)
from scripts.train_lstm_v2 import TimeSeriesDataset, run_lstm_experiment
from scripts.hyperopt import (
    optimize_all_models,
    load_hyperparameters,
    create_model_from_config,
)
from scripts.stacking import (
    StackingEnsemble,
    create_default_baseline_models,
    create_tuned_baseline_models,
)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def get_model_path(model_name: str, horizon: int) -> Path:
    """Get path for saving/loading a model."""
    return MODELS_DIR / f"{model_name}_t{horizon}"


def save_scaler(scaler, horizon: int):
    """Save StandardScaler for a horizon."""
    path = get_model_path("scaler", horizon).with_suffix(".pkl")
    with open(path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_scaler(horizon: int) -> Optional[object]:
    """Load StandardScaler for a horizon."""
    path = get_model_path("scaler", horizon).with_suffix(".pkl")
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_sklearn_model(model, model_name: str, horizon: int):
    """Save sklearn model."""
    path = get_model_path(model_name, horizon).with_suffix(".pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_sklearn_model(model_name: str, horizon: int):
    """Load sklearn model."""
    path = get_model_path(model_name, horizon).with_suffix(".pkl")
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_lstm_model(model, horizon: int):
    """Save PyTorch LSTM model."""
    path = get_model_path("lstm", horizon).with_suffix(".pt")
    torch.save(model.state_dict(), path)


def load_lstm_model(
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    horizon: int = 1,
):
    """Load PyTorch LSTM model."""
    from scripts.train_lstm_v2 import BidirectionalLSTMWithAttention

    path = get_model_path("lstm", horizon).with_suffix(".pt")
    if path.exists():
        model = BidirectionalLSTMWithAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        model.load_state_dict(torch.load(path, weights_only=True))
        return model
    return None


def save_results(results: Dict, horizon: int):
    """Save evaluation results to JSON."""
    path = get_model_path("results", horizon).with_suffix(".json")
    clean_results = {}
    for model_name, res in results.items():
        clean_results[model_name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in res.items()
            if k not in ["predictions", "actuals"]
        }
    with open(path, "w") as f:
        json.dump(clean_results, f, indent=2)

    # Save predictions for visualizations
    for model_name, res in results.items():
        if "predictions" in res and "actuals" in res:
            np.save(
                f"models/{model_name}_predictions_t{horizon}.npy", res["predictions"]
            )
            np.save(f"models/{model_name}_actuals_t{horizon}.npy", res["actuals"])


def load_results(horizon: int) -> Optional[Dict]:
    """Load evaluation results from JSON."""
    path = get_model_path("results", horizon).with_suffix(".json")
    if path.exists():
        with open(path, "r") as f:
            results = json.load(f)
            for model_name in results:
                results[model_name]["horizon"] = f"t+{horizon}"
            return results
    return None


def run_full_pipeline(retrain_all: bool = False):
    """Run complete pipeline with model persistence and tqdm progress."""

    print("=" * 70)
    print("AIR POLLUTION FORECASTING - WITH MODEL PERSISTENCE")
    print("=" * 70)

    all_results = []

    # If retrain_all is True, we need to train from scratch
    if retrain_all:
        print("\n[MODE] Retraining all models with optimized parameters...")

        # STEP 1: LOAD DATA
        print("\n[STEP 1] Loading Data...")
        df = load_all_regions([2022, 2023, 2024, 2025])
        print(f"  Raw records: {len(df)}")

        # STEP 2: PREPROCESS
        print("\n[STEP 2] Preprocessing...")
        df_proc = preprocess_hourly(df)

        # STEP 3: TRAIN FOR EACH HORIZON
        horizons = [1, 6, 24]

        for horizon in tqdm(horizons, desc="Processing horizons"):
            print(f"\n{'=' * 60}")
            print(f"HORIZON: t+{horizon}")
            print(f"{'=' * 60}")

            # Create target for this horizon
            df_horizon = create_horizon_targets(df_proc.copy(), horizon)

            # Split
            train, val, test = chronological_split(df_horizon)

            # Scale
            train_scaled, val_scaled, test_scaled, scaler = scale_features(
                train, val, test
            )

            # Save scaler
            save_scaler(scaler, horizon)

            # Prepare data
            X_train, y_train = prepare_xy(train_scaled)
            X_val, y_val = prepare_xy(val_scaled)
            X_test, y_test = prepare_xy(test_scaled)

            print(f"  Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")

            horizon_results = {}

            # === HYPERPARAMETER TUNING (Bayesian Optimization) ===
            print("\n[STEP 3a] Hyperparameter Tuning (Optuna)...")
            hyperparams_file = Path("configs/hyperparameters.json")

            if hyperparams_file.exists():
                print("  Loading saved hyperparameters...")
                hyperparams = load_hyperparameters(str(hyperparams_file))
                if hyperparams:
                    print("  Using cached hyperparameters")
                else:
                    hyperparams = None

            if not hyperparams_file.exists() or hyperparams is None:
                print("  Running Bayesian optimization...")
                hyperparams = optimize_all_models(X_train, y_train, X_val, y_val)
                print("  Hyperparameter tuning complete!")

            # === TRAIN WITH TUNED HYPERPARAMETERS ===
            print("\n[STEP 3b] Training with Tuned Hyperparameters...")

            # === RIDGE REGRESSION (Tuned) ===
            print("\n  Training Ridge Regression...")
            with tqdm(total=1, desc="  Ridge") as pbar:
                ridge_alpha = (
                    hyperparams["ridge"]["params"].get("alpha", 1.0)
                    if hyperparams
                    else 1.0
                )
                model = train_ridge_regression(
                    X_train, y_train, X_val, y_val, alpha=ridge_alpha
                )
                save_sklearn_model(model, "ridge", horizon)
                results = evaluate_model(model, X_test, y_test, "Ridge Regression")
                horizon_results["Ridge Regression"] = results
                pbar.update(1)
            print(f"    RMSE: {results['RMSE']:.4f}")

            # === RANDOM FOREST (Tuned) ===
            print("\n  Training Random Forest...")
            with tqdm(total=1, desc="  Random Forest") as pbar:
                rf_params = (
                    hyperparams["rf"]["params"]
                    if hyperparams
                    else {"n_estimators": 300, "max_depth": None}
                )
                model = train_random_forest(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_estimators=rf_params.get("n_estimators", 300),
                    max_depth=rf_params.get("max_depth", None),
                )
                save_sklearn_model(model, "rf", horizon)
                results = evaluate_model(model, X_test, y_test, "Random Forest")
                horizon_results["Random Forest"] = results
                pbar.update(1)
            print(f"    RMSE: {results['RMSE']:.4f}")

            # === XGBOOST (Tuned) ===
            print("\n  Training XGBoost...")
            with tqdm(total=1, desc="  XGBoost") as pbar:
                xgb_params = (
                    hyperparams["xgb"]["params"]
                    if hyperparams
                    else {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 8}
                )
                model = train_xgboost(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_estimators=xgb_params.get("n_estimators", 800),
                    learning_rate=xgb_params.get("learning_rate", 0.03),
                    max_depth=xgb_params.get("max_depth", 8),
                )
                save_sklearn_model(model, "xgb", horizon)
                results = evaluate_model(model, X_test, y_test, "XGBoost")
                horizon_results["XGBoost"] = results
                pbar.update(1)
            print(f"    RMSE: {results['RMSE']:.4f}")

            # === LIGHTGBM (Tuned) ===
            print("\n  Training LightGBM...")
            with tqdm(total=1, desc="  LightGBM") as pbar:
                lgb_params = (
                    hyperparams["lgb"]["params"]
                    if hyperparams
                    else {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 63}
                )
                model = train_lightgbm(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_estimators=lgb_params.get("n_estimators", 800),
                    learning_rate=lgb_params.get("learning_rate", 0.03),
                    num_leaves=lgb_params.get("num_leaves", 63),
                )
                save_sklearn_model(model, "lgb", horizon)
                results = evaluate_model(model, X_test, y_test, "LightGBM")
                horizon_results["LightGBM"] = results
                pbar.update(1)
            print(f"    RMSE: {results['RMSE']:.4f}")

            # === STACKING ENSEMBLE (Traditional ML) ===
            print("\n  Training Stacking Ensemble...")
            with tqdm(total=1, desc="  Stacking") as pbar:
                base_models = (
                    create_tuned_baseline_models(hyperparams)
                    if hyperparams
                    else create_default_baseline_models()
                )
                ensemble = StackingEnsemble(base_models, meta_model="ridge")
                n = len(X_train) + len(X_test)
                train_end = int(n * 0.70)
                train_indices = np.arange(0, train_end)
                test_indices = np.arange(train_end, n)
                stacking_result = ensemble.fit(
                    X_train, y_train, train_indices, test_indices
                )

                horizon_results["Stacking"] = {
                    "model": "Stacking",
                    "RMSE": stacking_result.final_metrics["RMSE"],
                    "MAE": stacking_result.final_metrics["MAE"],
                    "R2": stacking_result.final_metrics["R2"],
                    "predictions": stacking_result.final_metrics.get(
                        "predictions", np.array([])
                    ),
                    "actuals": stacking_result.final_metrics.get(
                        "actuals", np.array([])
                    ),
                }
                pbar.update(1)
            print(f"    RMSE: {horizon_results['Stacking']['RMSE']:.4f}")

            # === LSTM (Bidirectional with Attention) ===
            print("\n  Training LSTM (BiLSTM + Multi-head Attention)...")
            with tqdm(total=1, desc="  LSTM (50 epochs)") as pbar:
                lstm_result = run_lstm_experiment(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    seq_len=24,
                    horizon=f"t+{horizon}",
                )
                save_lstm_model(lstm_result["model"], horizon)
                results = lstm_result["test_results"]
                horizon_results["LSTM"] = results
                pbar.update(1)
            print(f"    RMSE: {results['RMSE']:.4f}")

            # Save results for this horizon
            save_results(horizon_results, horizon)

            # Add to all results
            for model_name, res in horizon_results.items():
                res["horizon"] = f"t+{horizon}"
                all_results.append(res)

        # After retraining, show results
        if all_results:
            results_df = pd.DataFrame(all_results)
            print("\n" + "=" * 70)
            print("FINAL RESULTS SUMMARY (NEWLY TRAINED)")
            print("=" * 70)
            print("\n" + results_df.to_string(index=False))
            results_df.to_csv("visualizations/all_results.csv", index=False)
            return results_df

    # Default: Load from JSON (benchmark mode)
    print("\n[INFO] Loading saved results...")

    for horizon in tqdm([1, 6, 24], desc="Processing horizons"):
        results_file = get_model_path("results", horizon).with_suffix(".json")

        if results_file.exists():
            with open(results_file, "r") as f:
                horizon_results = json.load(f)

            print(f"\n=== Horizon t+{horizon} ===")
            for model_name, res in horizon_results.items():
                res["horizon"] = f"t+{horizon}"
                res["model"] = model_name
                all_results.append(res)
                print(
                    f"  {model_name}: RMSE={res['RMSE']:.4f}, MAE={res['MAE']:.4f}, R2={res['R2']:.4f}"
                )

    if all_results:
        results_df = pd.DataFrame(all_results)

        print("\n" + "=" * 70)
        print("FINAL RESULTS SUMMARY")
        print("=" * 70)
        print("\n" + results_df.to_string(index=False))

        results_df.to_csv("visualizations/all_results.csv", index=False)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, horizon in enumerate([1, 6, 24]):
            h_data = results_df[results_df["horizon"] == f"t+{horizon}"]
            if not h_data.empty:
                axes[idx].barh(h_data["model"], h_data["RMSE"])
                axes[idx].set_xlabel("RMSE")
                axes[idx].set_title(f"t+{horizon}")
                axes[idx].invert_yaxis()

        plt.tight_layout()
        plt.savefig("visualizations/model_comparison.png", dpi=150)

        print("\nBest models per horizon:")
        for h in [1, 6, 24]:
            h_data = results_df[results_df["horizon"] == f"t+{h}"]
            if not h_data.empty:
                best = h_data.loc[h_data["RMSE"].idxmin()]
                print(
                    f"  t+{h}: {best['model']} (RMSE={best['RMSE']:.2f}, MAE={best['MAE']:.2f})"
                )

        return results_df

    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Pollution Forecasting")
    parser.add_argument("--retrain-all", action="store_true", default=False)
    args = parser.parse_args()

    run_full_pipeline(retrain_all=args.retrain_all)
