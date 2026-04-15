"""
Visualizations for PM2.5 Air Pollution Forecasting Project

Generates comprehensive visualizations for ML research project presentation.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

MODELS_DIR = Path("models")
VIS_DIR = Path("visualizations")
VIS_DIR.mkdir(exist_ok=True)

COLORS = {
    "Ridge": "#1f77b4",
    "Random Forest": "#2ca02c",
    "XGBoost": "#ff7f0e",
    "LightGBM": "#d62728",
    "LSTM": "#9467bd",
    "Actual": "#1a1a1a",
}


def plot_predictions_vs_actual():
    """Plot predictions vs actual time series for each horizon."""
    print("\n[1/6] Generating Prediction vs Actual plots...")

    horizons = [1, 12, 24]
    model_name = "LSTM"

    for horizon in horizons:
        pred_file = MODELS_DIR / f"{model_name}_predictions_t{horizon}.npy"
        actual_file = MODELS_DIR / f"{model_name}_actuals_t{horizon}.npy"

        if not pred_file.exists() or not actual_file.exists():
            print(f"  Skipping t+{horizon}: prediction files not found")
            continue

        predictions = np.load(pred_file)
        actuals = np.load(actual_file)

        n_samples = min(500, len(predictions))

        fig, ax = plt.subplots(figsize=(14, 5))
        x = range(n_samples)

        ax.plot(
            x,
            actuals[:n_samples],
            color=COLORS["Actual"],
            label="Actual",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.plot(
            x,
            predictions[:n_samples],
            color=COLORS["LSTM"],
            label=f"{model_name} Predicted",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.fill_between(
            x,
            actuals[:n_samples],
            predictions[:n_samples],
            alpha=0.15,
            color=COLORS["LSTM"],
        )

        ax.set_xlabel("Sample Index (Test Set)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"LSTM Predictions vs Actual - Horizon t+{horizon}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            VIS_DIR / f"predictions_t{horizon}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        print(f"  Saved predictions_t{horizon}.png (RMSE: {rmse:.2f})")


def plot_model_comparison():
    """Enhanced model comparison bar chart."""
    print("\n[2/6] Generating Model Comparison plot...")

    csv_file = VIS_DIR / "all_results.csv"
    if not csv_file.exists():
        print("  Skipping: all_results.csv not found")
        return

    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, horizon in enumerate([1, 24]):
        ax = axes[idx]
        h_data = df[df["horizon"] == f"t+{horizon}"]

        if h_data.empty:
            continue

        models = h_data["model"].tolist()
        rmse_vals = h_data["RMSE"].tolist()
        colors = [COLORS.get(m, "#333333") for m in models]

        bars = ax.barh(range(len(models)), rmse_vals, color=colors)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel("RMSE")
        ax.set_title(f"Horizon t+{horizon}")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        for bar, val in zip(bars, rmse_vals):
            ax.text(
                val + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=9,
            )

    plt.suptitle("Model Comparison - RMSE by Horizon", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "model_comparison_enhanced.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved model_comparison_enhanced.png")


def plot_loss_curves():
    """LSTM training loss curves."""
    print("\n[3/6] Generating Loss Curves...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, horizon in enumerate([1, 24]):
        ax = axes[idx]
        history_file = MODELS_DIR / f"loss_history_t{horizon}.json"

        if not history_file.exists():
            ax.text(
                0.5,
                0.5,
                f"No history for t+{horizon}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"t+{horizon}")
            continue

        with open(history_file, "r") as f:
            history = json.load(f)

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        if not train_loss:
            ax.text(
                0.5,
                0.5,
                f"Empty history for t+{horizon}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        epochs = range(1, len(train_loss) + 1)
        ax.plot(
            epochs, train_loss, label="Train Loss", color=COLORS["Ridge"], linewidth=2
        )
        ax.plot(
            epochs, val_loss, label="Val Loss", color=COLORS["XGBoost"], linewidth=2
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"t+{horizon}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("LSTM Training Convergence", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved loss_curves.png")


def plot_feature_importance():
    """Random Forest feature importance."""
    print("\n[4/6] Generating Feature Importance plot...")

    pred_file = MODELS_DIR / "rf_predictions_t1.npy"
    if not pred_file.exists():
        print("  Skipping: RF model not available")
        return

    from scripts.preprocess_v2 import preprocess_hourly
    from scripts.load_data_v2 import load_all_regions
    from scripts.preprocess_v2 import (
        create_horizon_targets,
        chronological_split,
        scale_features,
        prepare_xy,
    )

    print("  Loading data for feature importance...")
    df = load_all_regions([2022, 2023, 2024, 2025])
    df_proc = preprocess_hourly(df)
    df_t1 = create_horizon_targets(df_proc, 1)
    train, val, test = chronological_split(df_t1)
    train_scaled, val_scaled, test_scaled, scaler = scale_features(train, val, test)
    X_train, y_train = prepare_xy(train_scaled)

    feature_names = [
        c for c in train_scaled.columns if c not in ["pm25", "target", "region"]
    ]

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[-15:][::-1]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(top_features))
    ax.barh(y_pos, top_importances, color=COLORS["Random Forest"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top 15 Feature Importance - Random Forest")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(VIS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved feature_importance.png")


def plot_residuals():
    """Residual distributions."""
    print("\n[5/6] Generating Residual plots...")

    model_name = "LSTM"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, horizon in enumerate([1, 24]):
        ax = axes[idx]
        pred_file = MODELS_DIR / f"{model_name}_predictions_t{horizon}.npy"
        actual_file = MODELS_DIR / f"{model_name}_actuals_t{horizon}.npy"

        if not pred_file.exists() or not actual_file.exists():
            ax.text(
                0.5,
                0.5,
                f"No data for t+{horizon}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"t+{horizon}")
            continue

        predictions = np.load(pred_file)
        actuals = np.load(actual_file)
        residuals = actuals - predictions

        ax.hist(residuals, bins=50, color=COLORS["LSTM"], alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero")
        ax.axvline(
            x=np.mean(residuals),
            color="orange",
            linestyle="-",
            linewidth=1.5,
            label=f"Mean: {np.mean(residuals):.2f}",
        )

        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"t+{horizon}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Residual Distributions - LSTM", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "residuals.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved residuals.png")


def plot_horizon_degradation():
    """RMSE by horizon line chart."""
    print("\n[6/6] Generating Horizon Degradation plot...")

    csv_file = VIS_DIR / "all_results.csv"
    if not csv_file.exists():
        print("  Skipping: all_results.csv not found")
        return

    df = pd.read_csv(csv_file)
    models = df["model"].unique()
    horizons = [1, 12, 24]

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models:
        model_data = df[df["model"] == model]
        rmse_vals = []
        for h in horizons:
            h_data = model_data[model_data["horizon"] == f"t+{h}"]
            if not h_data.empty:
                rmse_vals.append(h_data["RMSE"].values[0])
            else:
                rmse_vals.append(np.nan)

        ax.plot(
            horizons,
            rmse_vals,
            marker="o",
            linewidth=2,
            label=model,
            color=COLORS.get(model, "#333333"),
            markersize=8,
        )

    ax.set_xlabel("Prediction Horizon (hours)")
    ax.set_ylabel("RMSE")
    ax.set_title("Model Performance Degradation by Horizon")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"t+{h}" for h in horizons])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIS_DIR / "horizon_degradation.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved horizon_degradation.png")


def generate_all_visualizations():
    """Generate all visualizations."""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_predictions_vs_actual()
    plot_model_comparison()
    plot_loss_curves()
    plot_feature_importance()
    plot_residuals()
    plot_horizon_degradation()

    print("\n" + "=" * 60)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"Output directory: {VIS_DIR}")


if __name__ == "__main__":
    generate_all_visualizations()
