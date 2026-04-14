"""
Stacking Ensemble for Traditional ML Models

This module implements stacking ensemble with Ridge meta-learner.
Note: LSTM is kept separate from this ensemble due to different feature spaces.
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class StackingResult:
    """Result from stacking ensemble."""

    base_model_metrics: Dict[str, Dict[str, float]]
    meta_model: str
    meta_weights: Optional[np.ndarray]
    final_metrics: Dict[str, float]
    oof_predictions: np.ndarray


class StackingEnsemble:
    """
    Stacking ensemble using OOF predictions.

    Architecture:
    - Base models: Ridge, Random Forest, XGBoost, LightGBM
    - Meta-learner: Ridge regression
    - Note: LSTM NOT included (separate pipeline)
    """

    def __init__(
        self,
        base_models: Dict[str, Callable],
        meta_model: str = "ridge",
    ):
        """
        Initialize stacking ensemble.

        Parameters:
        -----------
        base_models : Dict[str, Callable]
            Dictionary of model_name -> model_factory
        meta_model : str
            Meta-learner type ("ridge" or "linear")
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.fitted_base_models = {}
        self.meta_learner = None
        self.n_models = len(base_models)

    def fit_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Generate OOF predictions for base models.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        train_indices : np.ndarray
            Indices for training
        test_indices : np.ndarray
            Indices for OOF prediction

        Returns:
        --------
        OOF predictions (n_test_samples, n_models)
        """
        n_test = len(test_indices)

        oof_predictions = np.zeros((n_test, self.n_models))

        for i, (name, model_factory) in enumerate(self.base_models.items()):
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]

            model = model_factory()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            oof_predictions[:, i] = predictions

            self.fitted_base_models[name] = model

        return oof_predictions

    def fit_meta(
        self,
        oof_predictions: np.ndarray,
        y_test: np.ndarray,
    ):
        """Fit meta-learner on OOF predictions."""
        from sklearn.linear_model import Ridge, LinearRegression

        if self.meta_model == "ridge":
            self.meta_learner = Ridge(alpha=1.0)
        else:
            self.meta_learner = LinearRegression()

        self.meta_learner.fit(oof_predictions, y_test)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate stacked predictions.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix

        Returns:
        --------
        Final predictions
        """
        base_predictions = np.zeros((len(X), self.n_models))

        for i, (name, model) in enumerate(self.fitted_base_models.items()):
            base_predictions[:, i] = model.predict(X)

        return self.meta_learner.predict(base_predictions)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> StackingResult:
        """
        Full stacking fit pipeline.
        """
        print("\n[Stacking Ensemble - Traditional ML]")

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        base_metrics = {}

        print("\n  Training base models...")
        oof_predictions = np.zeros((len(test_indices), self.n_models))

        for i, (name, model_factory) in enumerate(self.base_models.items()):
            model = model_factory()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            oof_predictions[:, i] = pred
            self.fitted_base_models[name] = model

            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)

            base_metrics[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
            print(f"    {name}: RMSE={rmse:.4f}, R2={r2:.4f}")

        print("\n  Training meta-learner...")
        self.fit_meta(oof_predictions, y_test)

        final_pred = self.meta_learner.predict(oof_predictions)

        final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
        final_mae = mean_absolute_error(y_test, final_pred)
        final_r2 = r2_score(y_test, final_pred)

        final_metrics = {
            "RMSE": final_rmse,
            "MAE": final_mae,
            "R2": final_r2,
        }

        meta_weights = None
        if hasattr(self.meta_learner, "coef_"):
            meta_weights = self.meta_learner.coef_

        print(f"\n  Ensemble: RMSE={final_rmse:.4f}, R2={final_r2:.4f}")

        return StackingResult(
            base_model_metrics=base_metrics,
            meta_model=self.meta_model,
            meta_weights=meta_weights,
            final_metrics=final_metrics,
            oof_predictions=oof_predictions,
        )


def create_default_baseline_models() -> Dict[str, Callable]:
    """Create default baseline models for stacking."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    import lightgbm as lgb

    models = {
        "ridge": lambda: Ridge(alpha=1.0),
        "rf": lambda: RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "xgb": lambda: xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "lgb": lambda: lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    return models


def create_tuned_baseline_models(hyperparams: Dict[str, Dict]) -> Dict[str, Callable]:
    """Create baseline models from optimized hyperparams."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    import lightgbm as lgb

    models = {}

    if "ridge" in hyperparams:
        p = hyperparams["ridge"]["params"]
        models["ridge"] = lambda p=p: Ridge(alpha=p.get("alpha", 1.0))

    if "rf" in hyperparams:
        p = hyperparams["rf"]["params"]
        models["rf"] = lambda p=p: RandomForestRegressor(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            min_samples_split=p.get("min_samples_split", 2),
            random_state=42,
            n_jobs=-1,
        )

    if "xgb" in hyperparams:
        p = hyperparams["xgb"]["params"]
        models["xgb"] = lambda p=p: xgb.XGBRegressor(
            n_estimators=p.get("n_estimators", 800),
            learning_rate=p.get("learning_rate", 0.03),
            max_depth=p.get("max_depth", 8),
            min_child_weight=p.get("min_child_weight", 3),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if "lgb" in hyperparams:
        p = hyperparams["lgb"]["params"]
        models["lgb"] = lambda p=p: lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 800),
            learning_rate=p.get("learning_rate", 0.03),
            num_leaves=p.get("num_leaves", 63),
            min_child_samples=p.get("min_child_samples", 20),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def run_stacking_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_tuned: bool = False,
    hyperparams: Optional[Dict[str, Dict]] = None,
) -> StackingResult:
    """
    Run complete stacking experiment.
    """
    n = len(X_train) + len(X_test)
    train_end = int(n * 0.70)

    train_indices = np.arange(0, train_end)
    test_indices = np.arange(train_end, n)

    if use_tuned and hyperparams:
        base_models = create_tuned_baseline_models(hyperparams)
    else:
        base_models = create_default_baseline_models()

    ensemble = StackingEnsemble(base_models, meta_model="ridge")

    result = ensemble.fit(X_train, y_train, train_indices, test_indices)
    return result


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    print("Testing stacking ensemble...")

    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)

    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    result = run_stacking_experiment(X_train, y_train, X_test, y_test)

    print(f"\nFinal Ensemble Metrics:")
    print(f"  RMSE: {result.final_metrics['RMSE']:.4f}")
    print(f"  R2: {result.final_metrics['R2']:.4f}")
