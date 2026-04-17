"""
Phase 5: Train XGB quantile baseline models.

Consumes Phase 3 artifacts from data/preprocessed_xgb_v1.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


ALL_HORIZONS = [1, 24, 672]
QUANTILES = [0.05, 0.50, 0.95, 0.99]


DATA_DIR = PROJECT_ROOT / "data" / "preprocessed_xgb_v1"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs" / "train_xgb"


def setup_logging() -> logging.Logger:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_xgb")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_xgb_{stamp}.log"

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


def load_split_tables(
    data_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    X_val = pd.read_csv(data_dir / "X_val.csv")
    y_val = pd.read_csv(data_dir / "y_val.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def metric_summary(y_true: np.ndarray, y_pred_p50: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_p50))),
        "mae": float(mean_absolute_error(y_true, y_pred_p50)),
        "r2": float(r2_score(y_true, y_pred_p50)),
    }


def add_region_metrics(
    y_true: np.ndarray,
    p05: np.ndarray,
    p50: np.ndarray,
    p95: np.ndarray,
    regions: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    unique_regions = sorted(set(regions.tolist()))
    rmse_values: list[float] = []

    for region in unique_regions:
        mask = regions == region
        if not mask.any():
            continue
        y_r = y_true[mask]
        p05_r = p05[mask]
        p50_r = p50[mask]
        p95_r = p95[mask]
        rmse_r = float(np.sqrt(mean_squared_error(y_r, p50_r)))
        rmse_values.append(rmse_r)
        out[str(region)] = {
            "rmse": rmse_r,
            "mae": float(mean_absolute_error(y_r, p50_r)),
            "coverage_p05_p95": float(np.mean((y_r >= p05_r) & (y_r <= p95_r))),
            "count": int(mask.sum()),
        }

    fairness = None
    if rmse_values:
        fairness = {
            "rmse_max_min_ratio": float(max(rmse_values) / max(min(rmse_values), 1e-8)),
            "rmse_max": float(max(rmse_values)),
            "rmse_min": float(min(rmse_values)),
        }

    return {"by_region": out, "fairness": fairness}


def train_quantile_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantile: float,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    device: str,
    logger: logging.Logger,
) -> xgb.XGBRegressor:
    n_jobs = 1 if device == "cuda" else -1
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
        tree_method="hist",
        device=device,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.debug("Trained quantile model q=%.2f", quantile)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGB quantile baseline models")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--pollutants",
        type=str,
        default="pm25,pm10,no2,o3",
        help="Comma-separated pollutant list",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=",".join(str(h) for h in ALL_HORIZONS),
        help="Comma-separated horizons in hours",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="XGBoost device backend",
    )
    return parser.parse_args()


def _xgb_cuda_available(logger: logging.Logger) -> bool:
    try:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1,
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            random_state=RANDOM_SEED,
        )
        X = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
        y = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        model.fit(X, y, verbose=False)

        config = json.loads(model.get_booster().save_config())
        device_actual = (
            config.get("learner", {})
            .get("generic_param", {})
            .get("device", "")
            .lower()
            .strip()
        )
        return device_actual == "cuda"
    except Exception as exc:  # noqa: BLE001
        logger.debug("XGBoost CUDA smoke test failed: %s", exc)
        return False


def resolve_xgb_device(requested: str, logger: logging.Logger) -> str:
    req = requested.lower().strip()
    if req not in {"auto", "cuda", "cpu"}:
        raise ValueError("--device must be one of: auto, cuda, cpu")

    if req == "cpu":
        logger.info("Using XGBoost device: cpu")
        return "cpu"

    cuda_ok = _xgb_cuda_available(logger)
    if req == "cuda":
        if not cuda_ok:
            raise RuntimeError(
                "--device cuda requested, but XGBoost CUDA backend is unavailable."
            )
        logger.info("Using XGBoost device: cuda")
        return "cuda"

    # auto
    resolved = "cuda" if cuda_ok else "cpu"
    logger.info("Using XGBoost device: %s", resolved)
    return resolved


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    data_dir = Path(args.data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_split_tables(
        data_dir
    )

    xgb_device = resolve_xgb_device(args.device, logger)

    feature_cols = [c for c in X_train.columns if c not in {"timestamp", "region"}]
    pollutants = [p.strip().lower() for p in args.pollutants.split(",") if p.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    invalid_horizons = [h for h in horizons if h not in ALL_HORIZONS]
    if invalid_horizons:
        raise ValueError(
            f"Unsupported horizons: {invalid_horizons}; allowed={ALL_HORIZONS}"
        )

    train_X = np.ascontiguousarray(X_train[feature_cols].to_numpy(dtype=np.float32))
    val_X = np.ascontiguousarray(X_val[feature_cols].to_numpy(dtype=np.float32))
    test_X = np.ascontiguousarray(X_test[feature_cols].to_numpy(dtype=np.float32))
    test_regions = X_test["region"].to_numpy()

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "horizons": horizons,
        "quantiles": QUANTILES,
        "feature_count": len(feature_cols),
        "xgb_device": xgb_device,
        "pollutants": {},
    }

    for pollutant in pollutants:
        logger.info("Training pollutant: %s", pollutant)
        pollutant_result: dict[str, Any] = {
            "models": {},
            "metrics": {},
        }

        for horizon in horizons:
            target_col = f"target_{pollutant}_h{horizon}"
            train_y = y_train[target_col].to_numpy(dtype=np.float32)
            val_y = y_val[target_col].to_numpy(dtype=np.float32)
            test_y = y_test[target_col].to_numpy(dtype=np.float32)

            preds_by_q: dict[float, np.ndarray] = {}
            for quantile in QUANTILES:
                model = train_quantile_model(
                    X_train=train_X,
                    y_train=train_y,
                    X_val=val_X,
                    y_val=val_y,
                    quantile=quantile,
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    subsample=args.subsample,
                    colsample_bytree=args.colsample_bytree,
                    device=xgb_device,
                    logger=logger,
                )

                model_path = (
                    MODELS_DIR
                    / f"xgb_quantile_{pollutant}_h{horizon}_q{int(quantile * 100):02d}.json"
                )
                model.save_model(str(model_path))

                preds = model.predict(test_X)
                preds_by_q[quantile] = preds

                pollutant_result["models"][f"h{horizon}_q{quantile}"] = str(model_path)

            pred_matrix = np.stack([preds_by_q[q] for q in QUANTILES], axis=1)
            p05 = pred_matrix[:, 0]
            p50 = pred_matrix[:, 1]
            p95 = pred_matrix[:, 2]

            horizon_metrics = metric_summary(test_y, p50)
            horizon_metrics["coverage_p05_p95"] = float(
                np.mean((test_y >= p05) & (test_y <= p95))
            )
            horizon_metrics["crps_approx"] = float(
                np.mean(np.abs(pred_matrix - test_y[:, None]))
            )
            region_eval = add_region_metrics(
                y_true=test_y,
                p05=p05,
                p50=p50,
                p95=p95,
                regions=test_regions,
            )
            horizon_metrics["by_region"] = region_eval["by_region"]
            horizon_metrics["fairness"] = region_eval["fairness"]

            pollutant_result["metrics"][f"h{horizon}"] = horizon_metrics
            logger.info(
                "[%s h%d] RMSE=%.6f MAE=%.6f R2=%.6f coverage=%.4f",
                pollutant,
                horizon,
                horizon_metrics["rmse"],
                horizon_metrics["mae"],
                horizon_metrics["r2"],
                horizon_metrics["coverage_p05_p95"],
            )

        summary["pollutants"][pollutant] = pollutant_result

    summary["source"] = {
        "data_dir": str(data_dir.resolve()),
        "merge_policy": metadata.get("source", {}).get("merge_policy"),
        "floor_policy": metadata.get("source", {}).get("floor_policy"),
        "source_contribution": metadata.get("source", {}).get(
            "source_contribution", {}
        ),
    }

    out_path = MODELS_DIR / "xgb_training_summary.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Saved XGB training summary: %s", out_path)


if __name__ == "__main__":
    main()
