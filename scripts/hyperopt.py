"""
Bayesian Hyperparameter Optimization using Optuna

This module implements TPE (Tree-structured Parzen Estimator) for finding
optimal hyperparameters for each model.
"""

import optuna
import numpy as np
from typing import Dict, Callable, Any, Optional, List
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


@dataclass
class BestParams:
    """Container for best hyperparameters."""

    model: str
    params: Dict[str, Any]
    value: float
    n_trials: int


def create_ridge_study(X_train, y_train, X_val, y_val, n_trials: int = 30):
    """Optimize Ridge regression."""

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)

        from sklearn.linear_model import Ridge

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        return rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return BestParams(
        model="ridge",
        params={"alpha": study.best_params["alpha"]},
        value=study.best_value,
        n_trials=n_trials,
    )


def create_rf_study(X_train, y_train, X_val, y_val, n_trials: int = 30):
    """Optimize Random Forest."""

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        return rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return BestParams(
        model="rf",
        params={
            "n_estimators": study.best_params["n_estimators"],
            "max_depth": study.best_params["max_depth"],
            "min_samples_split": study.best_params["min_samples_split"],
        },
        value=study.best_value,
        n_trials=n_trials,
    )


def create_xgb_study(X_train, y_train, X_val, y_val, n_trials: int = 40):
    """Optimize XGBoost."""

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 200, 800)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        max_depth = trial.suggest_int("max_depth", 4, 10)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 5)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        return rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return BestParams(
        model="xgb",
        params={
            "n_estimators": study.best_params["n_estimators"],
            "learning_rate": study.best_params["learning_rate"],
            "max_depth": study.best_params["max_depth"],
            "min_child_weight": study.best_params["min_child_weight"],
            "subsample": study.best_params["subsample"],
            "colsample_bytree": study.best_params["colsample_bytree"],
        },
        value=study.best_value,
        n_trials=n_trials,
    )


def create_lgb_study(X_train, y_train, X_val, y_val, n_trials: int = 40):
    """Optimize LightGBM."""

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 200, 800)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        num_leaves = trial.suggest_int("num_leaves", 15, 127)
        min_child_samples = trial.suggest_int("min_child_samples", 10, 30)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        return rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return BestParams(
        model="lgb",
        params={
            "n_estimators": study.best_params["n_estimators"],
            "learning_rate": study.best_params["learning_rate"],
            "num_leaves": study.best_params["num_leaves"],
            "min_child_samples": study.best_params["min_child_samples"],
            "subsample": study.best_params["subsample"],
            "colsample_bytree": study.best_params["colsample_bytree"],
        },
        value=study.best_value,
        n_trials=n_trials,
    )


def optimize_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: Dict[str, int] = None,
) -> Dict[str, BestParams]:
    """
    Optimize all baseline models.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    n_trials : Dictionary of model -> n_trials

    Returns:
    --------
    Dictionary of model_name -> BestParams
    """
    if n_trials is None:
        n_trials = {
            "ridge": 30,
            "rf": 30,
            "xgb": 40,
            "lgb": 40,
        }

    results = {}

    print("\n[Bayesian Hyperparameter Optimization]")
    print(f"  Ridge ({n_trials.get('ridge', 30)} trials)...")
    results["ridge"] = create_ridge_study(
        X_train, y_train, X_val, y_val, n_trials.get("ridge", 30)
    )
    print(f"    Best RMSE: {results['ridge'].value:.4f}")

    print(f"  Random Forest ({n_trials.get('rf', 30)} trials)...")
    results["rf"] = create_rf_study(
        X_train, y_train, X_val, y_val, n_trials.get("rf", 30)
    )
    print(f"    Best RMSE: {results['rf'].value:.4f}")

    print(f"  XGBoost ({n_trials.get('xgb', 40)} trials)...")
    results["xgb"] = create_xgb_study(
        X_train, y_train, X_val, y_val, n_trials.get("xgb", 40)
    )
    print(f"    Best RMSE: {results['xgb'].value:.4f}")

    print(f"  LightGBM ({n_trials.get('lgb', 40)} trials)...")
    results["lgb"] = create_lgb_study(
        X_train, y_train, X_val, y_val, n_trials.get("lgb", 40)
    )
    print(f"    Best RMSE: {results['lgb'].value:.4f}")

    save_hyperparameters(results)
    return results


def save_hyperparameters(
    results: Dict[str, BestParams], path: str = "configs/hyperparameters.json"
):
    """Save optimized hyperparameters to JSON."""
    configs = Path("configs")
    configs.mkdir(exist_ok=True)

    data = {}
    for model_name, best_params in results.items():
        data[model_name] = {
            "params": best_params.params,
            "val_rmse": best_params.value,
            "n_trials": best_params.n_trials,
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved hyperparameters to {path}")


def load_hyperparameters(
    path: str = "configs/hyperparameters.json",
) -> Optional[Dict[str, Dict]]:
    """Load saved hyperparameters."""
    if not Path(path).exists():
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return data


def create_model_from_config(model_name: str, params: Dict[str, Any]):
    """Create a model from hyperparameters config."""
    if model_name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=params.get("alpha", 1.0))

    elif model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "xgb":
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=params.get("n_estimators", 800),
            learning_rate=params.get("learning_rate", 0.03),
            max_depth=params.get("max_depth", 8),
            min_child_weight=params.get("min_child_weight", 3),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    elif model_name == "lgb":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=params.get("n_estimators", 800),
            learning_rate=params.get("learning_rate", 0.03),
            num_leaves=params.get("num_leaves", 63),
            min_child_samples=params.get("min_child_samples", 20),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    print("Testing hyperopt...")
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)

    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    results = optimize_all_models(X_train, y_train, X_val, y_val)

    print("\n=== Best Parameters ===")
    for name, bp in results.items():
        print(f"{name}: {bp.params}")
