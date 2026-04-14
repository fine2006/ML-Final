"""
Walk-Forward Cross-Validation Framework for Time Series

This module implements the gold standard for time series validation:
- Expanding train window
- Fixed test window (simulates real deployment)
- Confidence intervals on metrics
"""

import numpy as np
from typing import Iterator, Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class Split:
    """Represents a single walk-forward split."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class WalkForwardSplitter:
    """
    Generates expanding window splits for time series cross-validation.

    Parameters:
    -----------
    n_splits : int
        Number of splits to generate (default: 10)
    test_size : float
        Fraction of data for test window (default: 0.15)
    min_train_size : int
        Minimum number of samples for initial training set
    """

    def __init__(
        self,
        n_splits: int = 10,
        test_size: float = 0.15,
        min_train_size: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Iterator[Split]:
        """
        Generate walk-forward splits.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray, optional
            Target vector

        Yields:
        -------
        Split
            Each split contains indices for train/test
        """
        n = len(X)

        if self.min_train_size is None:
            min_train = int(n * 0.5)
        else:
            min_train = self.min_train_size

        test_window = int(n * self.test_size)

        for i in range(self.n_splits):
            train_end = int(n * 0.70) + (i * test_window)

            if train_end + test_window > n:
                break
            if train_end < min_train:
                continue

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(train_end, train_end + test_window)

            if len(test_indices) == 0:
                continue

            yield Split(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=0,
                train_end=train_end,
                test_start=train_end,
                test_end=train_end + test_window,
            )

    def get_n_splits(self, X: np.ndarray) -> int:
        """Calculate actual number of splits."""
        count = 0
        for _ in self.split(X):
            count += 1
        return count


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    Dict with RMSE, MAE, R2, MAPE
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
    }


@dataclass
class CVResult:
    """Results from walk-forward CV."""

    metrics: Dict[str, float]
    std: Dict[str, float]
    ci_lower: Dict[str, float]
    ci_upper: Dict[str, float]
    n_splits: int
    split_results: List[Dict[str, float]]


class WalkForwardCV:
    """
    Complete walk-forward cross-validation evaluator.
    """

    def __init__(
        self,
        n_splits: int = 10,
        test_size: float = 0.15,
        min_train_size: Optional[int] = None,
    ):
        self.splitter = WalkForwardSplitter(
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=min_train_size,
        )

    def evaluate(
        self,
        model_factory: Callable,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> CVResult:
        """
        Evaluate a model using walk-forward CV.

        Parameters:
        -----------
        model_factory : Callable
            Function that returns a fresh model instance
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        verbose : bool
            Print progress

        Returns:
        --------
        CVResult with metrics and confidence intervals
        """
        all_results = []

        for i, split in enumerate(self.splitter.split(X, y)):
            X_train, X_test = X[split.train_indices], X[split.test_indices]
            y_train, y_test = y[split.train_indices], y[split.test_indices]

            model = model_factory()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            all_results.append(metrics)

            if verbose:
                print(
                    f"  Split {i + 1}: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}"
                )

        return self._aggregate_results(all_results)

    def _aggregate_results(self, results: List[Dict[str, float]]) -> CVResult:
        """Aggregate results with confidence intervals."""
        if not results:
            raise ValueError("No results to aggregate")

        metrics_names = list(results[0].keys())
        aggregated = {}
        std_dict = {}
        ci_lower = {}
        ci_upper = {}

        for metric in metrics_names:
            values = [r[metric] for r in results]
            mean_val = np.mean(values)
            std_val = np.std(values)

            aggregated[metric] = mean_val
            std_dict[metric] = std_val

            ci_lower[metric] = mean_val - 1.96 * std_val
            ci_upper[metric] = mean_val + 1.96 * std_val

        return CVResult(
            metrics=aggregated,
            std=std_dict,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_splits=len(results),
            split_results=results,
        )


def evaluate_base_models(
    models: Dict[str, Callable],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
) -> Dict[str, CVResult]:
    """
    Evaluate multiple base models with walk-forward CV.

    Parameters:
    -----------
    models : Dict[str, Callable]
        Dictionary of model_name -> model_factory
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_splits : int
        Number of splits

    Returns:
    --------
    Dictionary of model_name -> CVResult
    """
    results = {}

    print("\n[Walk-Forward Cross-Validation]")
    print(f"  Splits: {n_splits}, Test size: 15%")

    for name, model_factory in models.items():
        print(f"\n  Evaluating {name}...")
        cv = WalkForwardCV(n_splits=n_splits)
        results[name] = cv.evaluate(model_factory, X, y, verbose=True)

    return results


if __name__ == "__main__":
    from sklearn.linear_model import Ridge

    print("Testing WalkForwardCV...")

    X_dummy = np.random.randn(1000, 20)
    y_dummy = np.random.randn(1000)

    def ridge_factory():
        return Ridge(alpha=1.0)

    cv = WalkForwardCV(n_splits=3)
    result = cv.evaluate(ridge_factory, X_dummy, y_dummy)

    print(f"\nOverall RMSE: {result.metrics['RMSE']:.4f} ± {result.std['RMSE']:.4f}")
    print(f"95% CI: [{result.ci_lower['RMSE']:.4f}, {result.ci_upper['RMSE']:.4f}]")
