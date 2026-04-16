"""
Phase 6: Evaluation and LSTM vs XGB comparison.

This script evaluates available trained quantile models, computes calibration and
fairness metrics, and generates Phase 6 visualizations.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import kstest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.train_lstm import (  # noqa: E402
    ALL_HORIZONS,
    HierarchicalQuantileLSTM,
    MAX_SEQ_LEN,
    QUANTILES,
)


MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs" / "evaluate"
VIS_DIR = PROJECT_ROOT / "visualizations" / "phase_6_evaluation"

DEFAULT_POLLUTANTS = ["pm25", "pm10", "no2", "o3"]


@dataclass
class QuantilePayload:
    pollutant: str
    horizons: list[int]
    quantiles: list[float]
    predictions: dict[int, np.ndarray]
    targets: dict[int, np.ndarray]
    regions: dict[int, np.ndarray]
    timestamps: dict[int, np.ndarray]
    metrics: dict[str, Any]


class LSTMInferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_columns: list[str],
        region_mapping: dict[str, int],
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.region_mapping = region_mapping
        self.max_seq_len = max_seq_len

        self.region_features: dict[str, np.ndarray] = {}
        self.region_targets: dict[str, np.ndarray] = {}
        self.records: list[tuple[str, int]] = []
        self.keys: list[tuple[str, pd.Timestamp]] = []

        for region_name, region_df in df.groupby("region"):
            region_df = region_df.sort_values("timestamp").reset_index(drop=True)
            feat = region_df[feature_columns].to_numpy(dtype=np.float32)
            targ = region_df[target_columns].to_numpy(dtype=np.float32)
            ts = pd.to_datetime(region_df["timestamp"], errors="coerce").dt.floor("h")
            self.region_features[region_name] = feat
            self.region_targets[region_name] = targ
            for end_idx in range(max_seq_len, len(region_df)):
                self.records.append((region_name, end_idx))
                self.keys.append((region_name, pd.Timestamp(ts.iloc[end_idx])))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        region_name, end_idx = self.records[index]
        feat = self.region_features[region_name]
        targ = self.region_targets[region_name]

        x = feat[end_idx - self.max_seq_len : end_idx]
        y = targ[end_idx]
        region_id = self.region_mapping.get(region_name, -1)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(region_id, dtype=torch.long),
        )


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"evaluate_{stamp}.log"

    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTM and XGB models")
    parser.add_argument(
        "--pollutants",
        type=str,
        default=",".join(DEFAULT_POLLUTANTS),
        help="Comma-separated pollutant list",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=",".join(str(h) for h in ALL_HORIZONS),
        help="Comma-separated horizon list",
    )
    parser.add_argument(
        "--lstm-data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "preprocessed_lstm_v1"),
    )
    parser.add_argument(
        "--xgb-data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "preprocessed_xgb_v1"),
    )
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--fair-intersection",
        action="store_true",
        help="Run fair comparison on common (region,timestamp) rows only",
    )
    return parser.parse_args()


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def pit_values_from_quantiles(
    pred_q: np.ndarray,
    y_true: np.ndarray,
    quantiles: list[float],
) -> np.ndarray:
    q_levels = np.asarray(quantiles, dtype=np.float32)
    pit = np.zeros(len(y_true), dtype=np.float32)

    for i in range(len(y_true)):
        xp = pred_q[i].astype(np.float32).copy()
        xp = np.maximum.accumulate(xp)
        if len(xp) > 1:
            # Ensure strict monotonicity for interpolation stability.
            xp += np.arange(len(xp), dtype=np.float32) * 1e-6

        pit[i] = float(np.interp(y_true[i], xp, q_levels, left=0.0, right=1.0))

    return pit


def ks_uniformity(pit_values: np.ndarray) -> tuple[float, float]:
    if len(pit_values) < 2:
        return float("nan"), float("nan")
    result = kstest(pit_values, "uniform")
    return float(result.statistic), float(result.pvalue)


def compute_region_metrics(
    y_true: np.ndarray,
    pred_q: np.ndarray,
    regions: np.ndarray,
    quantiles: list[float],
) -> tuple[dict[str, Any], dict[str, float] | None]:
    q_idx = {q: i for i, q in enumerate(quantiles)}
    p05 = pred_q[:, q_idx[0.05]]
    p50 = pred_q[:, q_idx[0.50]]
    p95 = pred_q[:, q_idx[0.95]]

    region_metrics: dict[str, Any] = {}
    rmse_values: list[float] = []

    for region_name in sorted(set(regions.tolist())):
        mask = regions == region_name
        if not mask.any():
            continue

        y_r = y_true[mask]
        p05_r = p05[mask]
        p50_r = p50[mask]
        p95_r = p95[mask]
        pred_q_r = pred_q[mask]

        rmse_r = float(np.sqrt(mean_squared_error(y_r, p50_r)))
        rmse_values.append(rmse_r)
        region_metrics[str(region_name)] = {
            "rmse": rmse_r,
            "mae": float(mean_absolute_error(y_r, p50_r)),
            "crps_approx": float(np.mean(np.abs(pred_q_r - y_r[:, None]))),
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

    return region_metrics, fairness


def compute_horizon_metrics(
    y_true: np.ndarray,
    pred_q: np.ndarray,
    regions: np.ndarray,
    quantiles: list[float],
) -> tuple[dict[str, Any], np.ndarray]:
    q_idx = {q: i for i, q in enumerate(quantiles)}
    p05 = pred_q[:, q_idx[0.05]]
    p50 = pred_q[:, q_idx[0.50]]
    p95 = pred_q[:, q_idx[0.95]]
    p99 = pred_q[:, q_idx[0.99]]

    quantile_crossing_rate = float(np.mean(np.any(np.diff(pred_q, axis=1) < 0, axis=1)))
    interval_width_p05_p95 = float(np.mean(p95 - p05))

    pit = pit_values_from_quantiles(pred_q, y_true, quantiles)
    pit_ks_stat, pit_ks_pvalue = ks_uniformity(pit)

    region_metrics, fairness = compute_region_metrics(
        y_true, pred_q, regions, quantiles
    )

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, p50))),
        "mae": float(mean_absolute_error(y_true, p50)),
        "r2": safe_r2(y_true, p50),
        "crps_approx": float(np.mean(np.abs(pred_q - y_true[:, None]))),
        "coverage_p05_p95": float(np.mean((y_true >= p05) & (y_true <= p95))),
        "tail_below_p05": float(np.mean(y_true < p05)),
        "tail_above_p95": float(np.mean(y_true > p95)),
        "tail_above_p99": float(np.mean(y_true > p99)),
        "interval_width_p05_p95": interval_width_p05_p95,
        "quantile_crossing_rate": quantile_crossing_rate,
        "pit_ks_stat": pit_ks_stat,
        "pit_pvalue": pit_ks_pvalue,
        "by_region": region_metrics,
        "fairness": fairness,
    }
    return metrics, pit


def evaluate_lstm_models(
    models_dir: Path,
    data_dir: Path,
    pollutants: list[str],
    requested_horizons: list[int],
    device: torch.device,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, QuantilePayload], pd.DataFrame]:
    test_df = pd.read_csv(data_dir / "test.csv", parse_dates=["timestamp"])
    test_df["timestamp"] = pd.to_datetime(
        test_df["timestamp"], errors="coerce"
    ).dt.floor("h")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    region_name_by_id = {int(v): str(k) for k, v in metadata["region_mapping"].items()}

    summary: dict[str, Any] = {"pollutants": {}, "source": str(data_dir.resolve())}
    payloads: dict[str, QuantilePayload] = {}

    key_records: list[pd.DataFrame] = []

    for pollutant in pollutants:
        model_path = models_dir / f"lstm_quantile_{pollutant}.pt"
        if not model_path.exists():
            logger.info("[LSTM %s] model not found: %s", pollutant, model_path.name)
            continue

        checkpoint = torch.load(model_path, map_location=device)
        checkpoint_horizons = [int(h) for h in checkpoint.get("horizons", ALL_HORIZONS)]
        horizons = [h for h in checkpoint_horizons if h in requested_horizons]
        if not horizons:
            logger.info("[LSTM %s] no requested horizons available", pollutant)
            continue

        quantiles = [float(q) for q in checkpoint.get("quantiles", QUANTILES)]
        feature_cols = checkpoint["feature_columns"]
        target_cols = [f"target_{pollutant}_h{h}" for h in horizons]
        missing_targets = [c for c in target_cols if c not in test_df.columns]
        if missing_targets:
            logger.warning(
                "[LSTM %s] missing target columns in test split: %s",
                pollutant,
                missing_targets,
            )
            continue

        horizon_idx_map = {h: i for i, h in enumerate(checkpoint_horizons)}

        infer_ds = LSTMInferenceDataset(
            df=test_df,
            feature_columns=feature_cols,
            target_columns=target_cols,
            region_mapping=metadata["region_mapping"],
            max_seq_len=MAX_SEQ_LEN,
        )
        if len(infer_ds) == 0:
            logger.warning("[LSTM %s] no test sequences available", pollutant)
            continue

        infer_loader = DataLoader(infer_ds, batch_size=batch_size, shuffle=False)

        model = HierarchicalQuantileLSTM(
            input_dim=len(feature_cols),
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            dropout=0.3,
            horizons=checkpoint_horizons,
            quantiles=quantiles,
            min_attn_window=int(checkpoint.get("min_attn_window", 2)),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        preds: list[np.ndarray] = []
        tgts: list[np.ndarray] = []
        region_ids: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb, region_id in infer_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                tgts.append(yb.numpy())
                region_ids.append(region_id.numpy())

        pred_all = np.concatenate(preds, axis=0)
        tgt_all = np.concatenate(tgts, axis=0)
        region_id_all = np.concatenate(region_ids, axis=0)
        key_df = pd.DataFrame(infer_ds.keys, columns=["region", "timestamp"])
        key_df["timestamp"] = pd.to_datetime(
            key_df["timestamp"], errors="coerce"
        ).dt.floor("h")
        region_name_all = np.array(
            [region_name_by_id.get(int(r), str(int(r))) for r in region_id_all],
            dtype=object,
        )

        by_horizon: dict[str, Any] = {}
        pred_by_h: dict[int, np.ndarray] = {}
        tgt_by_h: dict[int, np.ndarray] = {}
        region_by_h: dict[int, np.ndarray] = {}
        ts_by_h: dict[int, np.ndarray] = {}
        pit_by_h: dict[int, np.ndarray] = {}

        pred_eval = np.stack(
            [pred_all[:, horizon_idx_map[h], :] for h in horizons],
            axis=1,
        )

        q_idx = {q: i for i, q in enumerate(quantiles)}
        p50 = pred_eval[:, :, q_idx[0.50]]

        for i, horizon in enumerate(horizons):
            y_h = tgt_all[:, i]
            pred_h = pred_eval[:, i, :]
            metrics_h, pit_h = compute_horizon_metrics(
                y_true=y_h,
                pred_q=pred_h,
                regions=region_name_all,
                quantiles=quantiles,
            )
            by_horizon[f"h{horizon}"] = metrics_h
            pred_by_h[horizon] = pred_h
            tgt_by_h[horizon] = y_h
            region_by_h[horizon] = region_name_all
            ts_by_h[horizon] = key_df["timestamp"].to_numpy(copy=True)
            pit_by_h[horizon] = pit_h

            key_records.append(
                pd.DataFrame(
                    {
                        "pollutant": pollutant,
                        "horizon": horizon,
                        "region": region_name_all,
                        "timestamp": key_df["timestamp"].to_numpy(copy=True),
                        "lstm_target": y_h,
                    }
                )
            )

        summary["pollutants"][pollutant] = {
            "model_path": str(model_path),
            "sample_count": int(len(pred_all)),
            "horizons": horizons,
            "quantiles": quantiles,
            "overall_rmse_p50": float(
                np.sqrt(mean_squared_error(tgt_all.ravel(), p50.ravel()))
            ),
            "overall_mae_p50": float(mean_absolute_error(tgt_all.ravel(), p50.ravel())),
            "overall_crps_approx": float(
                np.mean(np.abs(pred_eval - tgt_all[:, :, None]))
            ),
            "by_horizon": by_horizon,
        }

        payloads[pollutant] = QuantilePayload(
            pollutant=pollutant,
            horizons=horizons,
            quantiles=quantiles,
            predictions=pred_by_h,
            targets=tgt_by_h,
            regions=region_by_h,
            timestamps=ts_by_h,
            metrics={"by_horizon": by_horizon, "pit": pit_by_h},
        )

        logger.info(
            "[LSTM %s] evaluated horizons=%s sample_count=%d",
            pollutant,
            horizons,
            len(pred_all),
        )

    keys_df = (
        pd.concat(key_records, ignore_index=True)
        if key_records
        else pd.DataFrame(
            columns=["pollutant", "horizon", "region", "timestamp", "lstm_target"]
        )
    )
    return summary, payloads, keys_df


def evaluate_xgb_models(
    models_dir: Path,
    data_dir: Path,
    pollutants: list[str],
    requested_horizons: list[int],
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, QuantilePayload], pd.DataFrame, pd.DataFrame]:
    X_test = pd.read_csv(data_dir / "X_test.csv", parse_dates=["timestamp"])
    X_test["timestamp"] = pd.to_datetime(X_test["timestamp"], errors="coerce").dt.floor(
        "h"
    )
    y_test = pd.read_csv(data_dir / "y_test.csv")
    feature_cols = [c for c in X_test.columns if c not in {"timestamp", "region"}]

    X_matrix = X_test[feature_cols]
    region_names = X_test["region"].to_numpy(dtype=object)

    summary: dict[str, Any] = {"pollutants": {}, "source": str(data_dir.resolve())}
    payloads: dict[str, QuantilePayload] = {}

    for pollutant in pollutants:
        by_horizon: dict[str, Any] = {}
        pred_by_h: dict[int, np.ndarray] = {}
        tgt_by_h: dict[int, np.ndarray] = {}
        region_by_h: dict[int, np.ndarray] = {}
        ts_by_h: dict[int, np.ndarray] = {}
        pit_by_h: dict[int, np.ndarray] = {}
        available_horizons: list[int] = []

        for horizon in requested_horizons:
            model_paths = {
                q: models_dir
                / f"xgb_quantile_{pollutant}_h{horizon}_q{int(q * 100):02d}.json"
                for q in QUANTILES
            }
            if any(not path.exists() for path in model_paths.values()):
                continue

            target_col = f"target_{pollutant}_h{horizon}"
            if target_col not in y_test.columns:
                continue

            pred_matrix = np.zeros((len(X_matrix), len(QUANTILES)), dtype=np.float32)
            for j, q in enumerate(QUANTILES):
                model = xgb.XGBRegressor()
                model.load_model(str(model_paths[q]))
                pred_matrix[:, j] = model.predict(X_matrix)

            y_h = y_test[target_col].to_numpy(dtype=np.float32)
            metrics_h, pit_h = compute_horizon_metrics(
                y_true=y_h,
                pred_q=pred_matrix,
                regions=region_names,
                quantiles=QUANTILES,
            )

            by_horizon[f"h{horizon}"] = metrics_h
            pred_by_h[horizon] = pred_matrix
            tgt_by_h[horizon] = y_h
            region_by_h[horizon] = region_names
            ts_by_h[horizon] = X_test["timestamp"].to_numpy(copy=True)
            pit_by_h[horizon] = pit_h
            available_horizons.append(horizon)

        if not available_horizons:
            logger.info("[XGB %s] no complete quantile model set found", pollutant)
            continue

        summary["pollutants"][pollutant] = {
            "horizons": available_horizons,
            "quantiles": QUANTILES,
            "by_horizon": by_horizon,
        }
        payloads[pollutant] = QuantilePayload(
            pollutant=pollutant,
            horizons=available_horizons,
            quantiles=QUANTILES,
            predictions=pred_by_h,
            targets=tgt_by_h,
            regions=region_by_h,
            timestamps=ts_by_h,
            metrics={"by_horizon": by_horizon, "pit": pit_by_h},
        )
        logger.info(
            "[XGB %s] evaluated horizons=%s sample_count=%d",
            pollutant,
            available_horizons,
            len(X_matrix),
        )

    return summary, payloads, X_test, y_test


def build_comparison_table(
    lstm_summary: dict[str, Any],
    xgb_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lstm_pollutants = lstm_summary.get("pollutants", {})
    xgb_pollutants = xgb_summary.get("pollutants", {})

    for pollutant in sorted(set(lstm_pollutants) & set(xgb_pollutants)):
        lstm_h = lstm_pollutants[pollutant].get("by_horizon", {})
        xgb_h = xgb_pollutants[pollutant].get("by_horizon", {})
        common = sorted(
            {
                int(k[1:])
                for k in set(lstm_h.keys()) & set(xgb_h.keys())
                if k.startswith("h")
            }
        )
        for horizon in common:
            l = lstm_h[f"h{horizon}"]
            x = xgb_h[f"h{horizon}"]

            def improve(l_val: float, x_val: float) -> float:
                if not np.isfinite(x_val) or abs(x_val) < 1e-8:
                    return float("nan")
                return float((x_val - l_val) / x_val * 100.0)

            rows.append(
                {
                    "pollutant": pollutant,
                    "horizon": horizon,
                    "lstm_rmse": float(l["rmse"]),
                    "xgb_rmse": float(x["rmse"]),
                    "rmse_improvement_pct": improve(float(l["rmse"]), float(x["rmse"])),
                    "lstm_crps": float(l["crps_approx"]),
                    "xgb_crps": float(x["crps_approx"]),
                    "crps_improvement_pct": improve(
                        float(l["crps_approx"]), float(x["crps_approx"])
                    ),
                    "lstm_coverage": float(l["coverage_p05_p95"]),
                    "xgb_coverage": float(x["coverage_p05_p95"]),
                    "lstm_pit_pvalue": float(l["pit_pvalue"]),
                    "xgb_pit_pvalue": float(x["pit_pvalue"]),
                }
            )
    return rows


def build_fair_intersection_comparison(
    lstm_payloads: dict[str, QuantilePayload],
    xgb_payloads: dict[str, QuantilePayload],
    x_test_df: pd.DataFrame,
    y_test_df: pd.DataFrame,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    x_lookup = (
        x_test_df[["region", "timestamp"]]
        .reset_index()
        .rename(columns={"index": "x_idx"})
        .drop_duplicates(subset=["region", "timestamp"], keep="first")
    )

    for pollutant in sorted(set(lstm_payloads) & set(xgb_payloads)):
        l_payload = lstm_payloads[pollutant]
        x_payload = xgb_payloads[pollutant]
        common_h = sorted(set(l_payload.horizons) & set(x_payload.horizons))

        for horizon in common_h:
            l_pred = l_payload.predictions[horizon]
            l_tgt = l_payload.targets[horizon]
            l_regions = l_payload.regions[horizon]
            l_ts = pd.to_datetime(
                l_payload.timestamps[horizon], errors="coerce"
            ).astype("datetime64[ns]")

            base_df = pd.DataFrame(
                {
                    "l_idx": np.arange(len(l_tgt), dtype=np.int64),
                    "region": l_regions,
                    "timestamp": l_ts,
                    "y_true": l_tgt,
                }
            )
            base_df = base_df.dropna(subset=["timestamp", "y_true"])

            merged = base_df.merge(x_lookup, on=["region", "timestamp"], how="inner")
            if merged.empty:
                logger.info(
                    "[FAIR %s h%d] no common rows between LSTM and XGB",
                    pollutant,
                    horizon,
                )
                continue

            x_target_col = f"target_{pollutant}_h{horizon}"
            if x_target_col not in y_test_df.columns:
                logger.info(
                    "[FAIR %s h%d] missing XGB target column %s",
                    pollutant,
                    horizon,
                    x_target_col,
                )
                continue

            l_idx = merged["l_idx"].to_numpy(dtype=np.int64)
            x_idx = merged["x_idx"].to_numpy(dtype=np.int64)
            y_true = merged["y_true"].to_numpy(dtype=np.float32)
            regions = merged["region"].to_numpy(dtype=object)

            l_pred_aligned = l_pred[l_idx]
            x_pred_aligned = x_payload.predictions[horizon][x_idx]

            l_metrics, _ = compute_horizon_metrics(
                y_true=y_true,
                pred_q=l_pred_aligned,
                regions=regions,
                quantiles=l_payload.quantiles,
            )
            x_metrics, _ = compute_horizon_metrics(
                y_true=y_true,
                pred_q=x_pred_aligned,
                regions=regions,
                quantiles=x_payload.quantiles,
            )

            def improve(l_val: float, x_val: float) -> float:
                if not np.isfinite(x_val) or abs(x_val) < 1e-8:
                    return float("nan")
                return float((x_val - l_val) / x_val * 100.0)

            rows.append(
                {
                    "pollutant": pollutant,
                    "horizon": int(horizon),
                    "n_common": int(len(merged)),
                    "lstm_rmse": float(l_metrics["rmse"]),
                    "xgb_rmse": float(x_metrics["rmse"]),
                    "rmse_improvement_pct": improve(
                        float(l_metrics["rmse"]), float(x_metrics["rmse"])
                    ),
                    "lstm_crps": float(l_metrics["crps_approx"]),
                    "xgb_crps": float(x_metrics["crps_approx"]),
                    "crps_improvement_pct": improve(
                        float(l_metrics["crps_approx"]), float(x_metrics["crps_approx"])
                    ),
                    "lstm_coverage": float(l_metrics["coverage_p05_p95"]),
                    "xgb_coverage": float(x_metrics["coverage_p05_p95"]),
                    "lstm_pit_ks": float(l_metrics["pit_ks_stat"]),
                    "xgb_pit_ks": float(x_metrics["pit_ks_stat"]),
                    "lstm_pit_pvalue": float(l_metrics["pit_pvalue"]),
                    "xgb_pit_pvalue": float(x_metrics["pit_pvalue"]),
                    "lstm_fairness_ratio": float(
                        (l_metrics.get("fairness") or {}).get(
                            "rmse_max_min_ratio", np.nan
                        )
                    ),
                    "xgb_fairness_ratio": float(
                        (x_metrics.get("fairness") or {}).get(
                            "rmse_max_min_ratio", np.nan
                        )
                    ),
                }
            )

    return sorted(rows, key=lambda r: (r["pollutant"], r["horizon"]))


def _first_common_payload(
    lstm_payloads: dict[str, QuantilePayload],
    xgb_payloads: dict[str, QuantilePayload],
) -> tuple[str | None, int | None]:
    common_pollutants = sorted(set(lstm_payloads) & set(xgb_payloads))
    if not common_pollutants:
        return None, None
    pollutant = common_pollutants[0]
    common_horizons = sorted(
        set(lstm_payloads[pollutant].horizons) & set(xgb_payloads[pollutant].horizons)
    )
    if not common_horizons:
        return pollutant, None
    preferred = 24 if 24 in common_horizons else common_horizons[0]
    return pollutant, preferred


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rmse_and_crps(rows: list[dict[str, Any]], pollutant: str) -> list[str]:
    created: list[str] = []
    subset = [r for r in rows if r["pollutant"] == pollutant]
    if not subset:
        return created

    subset = sorted(subset, key=lambda d: d["horizon"])
    labels = [f"t+{r['horizon']}h" for r in subset]
    x = np.arange(len(labels))
    width = 0.35

    # RMSE
    fig, ax = plt.subplots(figsize=(10, 5))
    lstm_vals = [r["lstm_rmse"] for r in subset]
    xgb_vals = [r["xgb_rmse"] for r in subset]
    ax.bar(x - width / 2, lstm_vals, width, label="LSTM", color="#1f77b4")
    ax.bar(x + width / 2, xgb_vals, width, label="XGB", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE")
    ax.set_title(f"LSTM vs XGB - RMSE by Horizon ({pollutant.upper()})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    rmse_path = VIS_DIR / "lstm_vs_xgb_rmse_by_horizon.png"
    _save_plot(rmse_path)
    created.append(str(rmse_path))

    # CRPS
    fig, ax = plt.subplots(figsize=(10, 5))
    lstm_vals = [r["lstm_crps"] for r in subset]
    xgb_vals = [r["xgb_crps"] for r in subset]
    ax.bar(x - width / 2, lstm_vals, width, label="LSTM", color="#1f77b4")
    ax.bar(x + width / 2, xgb_vals, width, label="XGB", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CRPS (approx)")
    ax.set_title(f"LSTM vs XGB - CRPS by Horizon ({pollutant.upper()})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    crps_path = VIS_DIR / "lstm_vs_xgb_crps_by_horizon.png"
    _save_plot(crps_path)
    created.append(str(crps_path))

    return created


def plot_calibration_curve(
    lstm_payload: QuantilePayload,
    xgb_payload: QuantilePayload,
    pollutant: str,
    horizon: int,
) -> str | None:
    if horizon not in lstm_payload.horizons or horizon not in xgb_payload.horizons:
        return None

    lstm_pred = lstm_payload.predictions[horizon]
    lstm_y = lstm_payload.targets[horizon]
    xgb_pred = xgb_payload.predictions[horizon]
    xgb_y = xgb_payload.targets[horizon]

    q = np.asarray(lstm_payload.quantiles, dtype=np.float32)
    lstm_emp = np.asarray([np.mean(lstm_y <= lstm_pred[:, i]) for i in range(len(q))])
    xgb_emp = np.asarray([np.mean(xgb_y <= xgb_pred[:, i]) for i in range(len(q))])

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(q, lstm_emp, marker="o", color="#1f77b4", label="LSTM")
    ax.plot(q, xgb_emp, marker="s", color="#ff7f0e", label="XGB")
    ax.fill_between([0, 1], [0, 0], [0.05, 1.05], alpha=0.06, color="#2ca02c")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Empirical quantile")
    ax.set_title(f"Quantile Calibration ({pollutant.upper()} h{horizon})")
    ax.grid(alpha=0.3)
    ax.legend()

    out_path = VIS_DIR / "quantile_calibration_curves.png"
    _save_plot(out_path)
    return str(out_path)


def plot_pit_histogram(
    payload: QuantilePayload,
    pollutant: str,
    horizon: int,
    model_name: str,
) -> str | None:
    if horizon not in payload.horizons:
        return None
    pit = payload.metrics.get("pit", {}).get(horizon)
    if pit is None or len(pit) < 2:
        return None

    ks_stat, pvalue = ks_uniformity(pit)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pit, bins=20, range=(0, 1), color="#1f77b4", alpha=0.85)
    ax.axhline(y=len(pit) / 20.0, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"PIT Histogram ({model_name} {pollutant.upper()} h{horizon})\n"
        f"KS={ks_stat:.4f}, p={pvalue:.4f}"
    )
    ax.grid(alpha=0.25)

    out_path = VIS_DIR / "pit_histogram_uniformity.png"
    _save_plot(out_path)
    return str(out_path)


def plot_coverage_bars(
    lstm_payload: QuantilePayload,
    xgb_payload: QuantilePayload,
    pollutant: str,
    horizon: int,
) -> str | None:
    if horizon not in lstm_payload.horizons or horizon not in xgb_payload.horizons:
        return None

    l = lstm_payload.metrics["by_horizon"][f"h{horizon}"]
    x = xgb_payload.metrics["by_horizon"][f"h{horizon}"]

    categories = ["Below p5", "p5-p95", "Above p95"]
    lstm_vals = [
        l["tail_below_p05"] * 100.0,
        l["coverage_p05_p95"] * 100.0,
        l["tail_above_p95"] * 100.0,
    ]
    xgb_vals = [
        x["tail_below_p05"] * 100.0,
        x["coverage_p05_p95"] * 100.0,
        x["tail_above_p95"] * 100.0,
    ]

    x_pos = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x_pos - width / 2, lstm_vals, width, label="LSTM", color="#1f77b4")
    ax.bar(x_pos + width / 2, xgb_vals, width, label="XGB", color="#ff7f0e")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Coverage Analysis ({pollutant.upper()} h{horizon})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    out_path = VIS_DIR / "coverage_analysis_by_quantile.png"
    _save_plot(out_path)
    return str(out_path)


def plot_region_fairness(
    lstm_payload: QuantilePayload,
    xgb_payload: QuantilePayload,
    pollutant: str,
    horizon: int,
) -> str | None:
    if horizon not in lstm_payload.horizons or horizon not in xgb_payload.horizons:
        return None

    l_regions = lstm_payload.metrics["by_horizon"][f"h{horizon}"]["by_region"]
    x_regions = xgb_payload.metrics["by_horizon"][f"h{horizon}"]["by_region"]
    common_regions = sorted(set(l_regions) & set(x_regions))
    if not common_regions:
        return None

    lstm_vals = [l_regions[r]["rmse"] for r in common_regions]
    xgb_vals = [x_regions[r]["rmse"] for r in common_regions]

    x_pos = np.arange(len(common_regions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width / 2, lstm_vals, width, label="LSTM", color="#1f77b4")
    ax.bar(x_pos + width / 2, xgb_vals, width, label="XGB", color="#ff7f0e")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(common_regions)
    ax.set_ylabel("RMSE")
    ax.set_title(f"Per-Region Fairness ({pollutant.upper()} h{horizon})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    out_path = VIS_DIR / "per_region_fairness_metrics.png"
    _save_plot(out_path)
    return str(out_path)


def plot_predictions_vs_actual(payload: QuantilePayload, pollutant: str) -> str | None:
    horizons = sorted(payload.horizons)
    if not horizons:
        return None

    q_idx = {q: i for i, q in enumerate(payload.quantiles)}
    p50_idx = q_idx[0.50]
    n = len(horizons)
    cols = 3
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if isinstance(axes, np.ndarray):
        ax_list = axes.flatten().tolist()
    else:
        ax_list = [axes]

    for i, horizon in enumerate(horizons):
        ax = ax_list[i]
        y = payload.targets[horizon]
        pred = payload.predictions[horizon][:, p50_idx]

        max_points = min(len(y), 5000)
        if len(y) > max_points:
            idx = np.linspace(0, len(y) - 1, num=max_points, dtype=int)
            y = y[idx]
            pred = pred[idx]

        ax.scatter(y, pred, s=8, alpha=0.3, color="#1f77b4")
        lo = float(min(np.min(y), np.min(pred)))
        hi = float(max(np.max(y), np.max(pred)))
        ax.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--", linewidth=1.5)

        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        ax.set_title(f"h{horizon} | RMSE={rmse:.2f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted p50")
        ax.grid(alpha=0.25)

    for j in range(n, len(ax_list)):
        ax_list[j].axis("off")

    fig.suptitle(f"Predictions vs Actual ({pollutant.upper()})", y=1.01)
    out_path = VIS_DIR / "predictions_vs_actual_by_horizon.png"
    _save_plot(out_path)
    return str(out_path)


def plot_final_summary_table(rows: list[dict[str, Any]], pollutant: str) -> str | None:
    subset = [r for r in rows if r["pollutant"] == pollutant]
    if not subset:
        return None
    subset = sorted(subset, key=lambda d: d["horizon"])

    col_labels = [f"h{r['horizon']}" for r in subset]
    row_labels = ["RMSE", "CRPS", "Coverage", "PIT p-value"]
    cell_text: list[list[str]] = []

    rmse_row = [f"{r['lstm_rmse']:.2f} | {r['xgb_rmse']:.2f}" for r in subset]
    crps_row = [f"{r['lstm_crps']:.2f} | {r['xgb_crps']:.2f}" for r in subset]
    cov_row = [
        f"{100 * r['lstm_coverage']:.1f}% | {100 * r['xgb_coverage']:.1f}%"
        for r in subset
    ]
    pit_row = [
        f"{r['lstm_pit_pvalue']:.3f} | {r['xgb_pit_pvalue']:.3f}" for r in subset
    ]
    cell_text.extend([rmse_row, crps_row, cov_row, pit_row])

    fig, ax = plt.subplots(figsize=(1.4 * max(len(col_labels), 2) + 5.5, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title(f"Final Comparison Summary ({pollutant.upper()})\nLSTM | XGB", pad=12)

    out_path = VIS_DIR / "final_comparison_summary_table.png"
    _save_plot(out_path)
    return str(out_path)


def generate_visualizations(
    comparison_rows: list[dict[str, Any]],
    lstm_payloads: dict[str, QuantilePayload],
    xgb_payloads: dict[str, QuantilePayload],
    logger: logging.Logger,
) -> list[str]:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    pollutant, horizon = _first_common_payload(lstm_payloads, xgb_payloads)
    if pollutant is None:
        logger.info("No common LSTM/XGB pollutant available for Phase 6 plots")
        return created

    created.extend(plot_rmse_and_crps(comparison_rows, pollutant))

    if horizon is not None:
        l_payload = lstm_payloads[pollutant]
        x_payload = xgb_payloads[pollutant]

        calibration = plot_calibration_curve(l_payload, x_payload, pollutant, horizon)
        if calibration:
            created.append(calibration)

        pit = plot_pit_histogram(l_payload, pollutant, horizon, "LSTM")
        if pit:
            created.append(pit)

        coverage = plot_coverage_bars(l_payload, x_payload, pollutant, horizon)
        if coverage:
            created.append(coverage)

        fairness = plot_region_fairness(l_payload, x_payload, pollutant, horizon)
        if fairness:
            created.append(fairness)

    preds = plot_predictions_vs_actual(lstm_payloads[pollutant], pollutant)
    if preds:
        created.append(preds)

    table = plot_final_summary_table(comparison_rows, pollutant)
    if table:
        created.append(table)

    logger.info("Generated %d visualization(s) in %s", len(created), VIS_DIR)
    return created


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    models_dir = Path(args.models_dir)
    lstm_data_dir = Path(args.lstm_data_dir)
    xgb_data_dir = Path(args.xgb_data_dir)

    pollutants = [p.strip().lower() for p in args.pollutants.split(",") if p.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    invalid_horizons = [h for h in horizons if h not in ALL_HORIZONS]
    if invalid_horizons:
        raise ValueError(
            f"Unsupported horizons: {invalid_horizons}; allowed={ALL_HORIZONS}"
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("Starting Phase 6 evaluation")
    logger.info("Using device: %s", device)

    lstm_summary, lstm_payloads, _lstm_keys_df = evaluate_lstm_models(
        models_dir=models_dir,
        data_dir=lstm_data_dir,
        pollutants=pollutants,
        requested_horizons=horizons,
        device=device,
        batch_size=args.batch_size,
        logger=logger,
    )

    xgb_summary, xgb_payloads, x_test_df, y_test_df = evaluate_xgb_models(
        models_dir=models_dir,
        data_dir=xgb_data_dir,
        pollutants=pollutants,
        requested_horizons=horizons,
        logger=logger,
    )

    comparison_rows = build_comparison_table(lstm_summary, xgb_summary)
    fair_rows: list[dict[str, Any]] = []
    if args.fair_intersection:
        fair_rows = build_fair_intersection_comparison(
            lstm_payloads=lstm_payloads,
            xgb_payloads=xgb_payloads,
            x_test_df=x_test_df,
            y_test_df=y_test_df,
            logger=logger,
        )
        fair_path = models_dir / "fair_benchmark_summary.json"
        with open(fair_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "created_at": datetime.now().isoformat(),
                    "pollutants_requested": pollutants,
                    "horizons_requested": horizons,
                    "rows": fair_rows,
                },
                handle,
                indent=2,
            )
        logger.info("Saved fair benchmark summary: %s", fair_path)

    visuals = generate_visualizations(
        comparison_rows, lstm_payloads, xgb_payloads, logger
    )

    out = {
        "created_at": datetime.now().isoformat(),
        "pollutants_requested": pollutants,
        "horizons_requested": horizons,
        "lstm": lstm_summary,
        "xgb": xgb_summary,
        "comparison": comparison_rows,
        "fair_intersection_enabled": bool(args.fair_intersection),
        "fair_comparison": fair_rows,
        "visualizations": visuals,
    }

    out_path = models_dir / "evaluation_summary.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    logger.info("Saved evaluation summary: %s", out_path)


if __name__ == "__main__":
    main()
