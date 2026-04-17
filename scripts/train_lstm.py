"""
Phase 5: Train hierarchical multi-horizon quantile LSTM models.

Consumes Phase 3 artifacts from data/preprocessed_lstm_v1.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


ALL_HORIZONS = [1, 24, 672]
QUANTILES = [0.05, 0.50, 0.95, 0.99]
MAX_SEQ_LEN = 8760
DEFAULT_SEQ_LEN_BY_HORIZON = {
    1: 168,
    24: 336,
    672: 2402,
}


DATA_DIR = PROJECT_ROOT / "data" / "preprocessed_lstm_v1"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs" / "train_lstm"


def setup_logging() -> logging.Logger:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_lstm")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_lstm_{stamp}.log"

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


@dataclass
class SplitTables:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any]


def load_split_tables(data_dir: Path) -> SplitTables:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return SplitTables(train=train, val=val, test=test, metadata=metadata)


class RegionSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_columns: list[str],
        region_weights: dict[str, float],
        region_mapping: dict[str, int],
        seq_len: int = MAX_SEQ_LEN,
        max_samples: int | None = None,
        seed: int = RANDOM_SEED,
    ):
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.seq_len = seq_len
        self.seed = seed

        self.region_arrays: dict[str, np.ndarray] = {}
        self.target_arrays: dict[str, np.ndarray] = {}
        self.region_id_by_name = region_mapping
        self.region_weight_by_name = region_weights
        self.record_counts_by_region: dict[str, int] = {
            str(region_name): 0 for region_name in region_mapping
        }

        records: list[tuple[str, int]] = []
        for region_name, region_df in df.groupby("region"):
            region_df = region_df.sort_values("timestamp").reset_index(drop=True)
            feat = region_df[feature_columns].to_numpy(dtype=np.float32)
            targ = region_df[target_columns].to_numpy(dtype=np.float32)
            self.region_arrays[region_name] = feat
            self.target_arrays[region_name] = targ

            for end_idx in range(seq_len, len(region_df)):
                records.append((region_name, end_idx))
                self.record_counts_by_region[str(region_name)] = (
                    self.record_counts_by_region.get(str(region_name), 0) + 1
                )

        if max_samples is not None and len(records) > max_samples:
            rng = random.Random(seed)
            records = rng.sample(records, max_samples)

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        region_name, end_idx = self.records[index]
        feat = self.region_arrays[region_name]
        targ = self.target_arrays[region_name]

        x = np.array(
            feat[end_idx - self.seq_len : end_idx], dtype=np.float32, copy=True
        )
        y = np.array(targ[end_idx], dtype=np.float32, copy=True)
        region_id = self.region_id_by_name.get(region_name, -1)
        region_weight = float(self.region_weight_by_name.get(region_name, 1.0))

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(region_id, dtype=torch.long),
            torch.tensor(region_weight, dtype=torch.float32),
        )

    def region_ids(self) -> list[int]:
        return [self.region_id_by_name.get(region, -1) for region, _ in self.records]

    def sample_counts_by_region(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.record_counts_by_region.items()}


class HierarchicalQuantileLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        horizons: list[int] | None = None,
        quantiles: list[float] | None = None,
        min_attn_window: int = 2,
    ):
        super().__init__()
        self.horizons = horizons or ALL_HORIZONS
        self.quantiles = quantiles or QUANTILES
        self.min_attn_window = max(1, int(min_attn_window))

        self.backbone = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        embed_dim = hidden_dim * 2
        self.attn_heads = nn.ModuleDict(
            {
                str(h): nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for h in self.horizons
            }
        )

        self.quantile_heads = nn.ModuleDict(
            {
                str(h): nn.Sequential(
                    nn.Linear(embed_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, len(self.quantiles)),
                )
                for h in self.horizons
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, feature]
        seq_out, _ = self.backbone(x)

        per_horizon = []
        separated_mode = len(self.horizons) == 1
        for horizon in self.horizons:
            if separated_mode:
                window = seq_out.size(1)
            else:
                window = min(max(2 * horizon, self.min_attn_window), seq_out.size(1))
            tail = seq_out[:, -window:, :]
            query = tail[:, -1:, :]

            context, _ = self.attn_heads[str(horizon)](
                query=query,
                key=tail,
                value=tail,
                need_weights=False,
            )
            pred = self.quantile_heads[str(horizon)](context.squeeze(1))
            per_horizon.append(pred.unsqueeze(1))

        return torch.cat(per_horizon, dim=1)


def pinball_tensor(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: list[float],
):
    q = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype)
    error = targets.unsqueeze(-1) - predictions
    return torch.maximum(q * error, (q - 1.0) * error)


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: list[float],
    region_weights: torch.Tensor,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = pinball_tensor(predictions, targets, quantiles)  # [batch, horizon, quantile]
    per_horizon = loss.mean(dim=2)  # [batch, horizon]

    if horizon_weights is not None:
        denom = torch.clamp(horizon_weights.sum(), min=1e-8)
        per_sample = (per_horizon * horizon_weights.unsqueeze(0)).sum(dim=1) / denom
    else:
        per_sample = per_horizon.mean(dim=1)

    weighted = per_sample * region_weights
    return weighted.sum() / torch.clamp(region_weights.sum(), min=1e-8)


def make_stratified_sampler(dataset: RegionSequenceDataset) -> WeightedRandomSampler:
    region_ids = dataset.region_ids()
    counts: dict[int, int] = {}
    for rid in region_ids:
        counts[rid] = counts.get(rid, 0) + 1

    weights = np.asarray(
        [1.0 / max(counts[rid], 1) for rid in region_ids], dtype=np.float64
    )
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    horizon_weights: torch.Tensor | None,
    quantiles: list[float],
    horizons: list[int],
) -> tuple[float, dict[str, float]]:
    model.train(mode=train_mode)
    total = 0.0
    steps = 0
    horizon_weighted_sum = np.zeros(len(horizons), dtype=np.float64)
    horizon_weight_norm = 0.0

    for xb, yb, _region_id, wb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        wb = wb.to(device)

        if train_mode:
            optimizer.zero_grad()

        pred = model(xb)
        loss = pinball_loss(pred, yb, quantiles, wb, horizon_weights)

        with torch.no_grad():
            per_h = pinball_tensor(pred, yb, quantiles).mean(dim=2)  # [batch, horizon]
            horizon_weighted_sum += (
                (per_h * wb.unsqueeze(1)).sum(dim=0).detach().cpu().numpy()
            )
            horizon_weight_norm += float(wb.sum().item())

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total += float(loss.item())
        steps += 1

    epoch_loss = total / max(steps, 1)
    by_h: dict[str, float] = {}
    if horizon_weight_norm > 0:
        horizon_avg = horizon_weighted_sum / horizon_weight_norm
        by_h = {
            f"h{horizon}": float(horizon_avg[i]) for i, horizon in enumerate(horizons)
        }
    return epoch_loss, by_h


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    horizons: list[int],
    quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    regions: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb, region_id, _wb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            tgts.append(yb.numpy())
            regions.append(region_id.numpy())

    if not preds:
        return (
            np.empty((0, len(horizons), len(quantiles))),
            np.empty((0, len(horizons))),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(tgts, axis=0),
        np.concatenate(regions, axis=0),
    )


def compute_quantile_metrics(
    pred: np.ndarray,
    tgt: np.ndarray,
    horizons: list[int],
    quantiles: list[float],
    region_ids: np.ndarray | None = None,
    region_name_by_id: dict[int, str] | None = None,
) -> dict[str, Any]:
    q_index = {q: i for i, q in enumerate(quantiles)}
    p50 = pred[:, :, q_index[0.50]]
    p05 = pred[:, :, q_index[0.05]]
    p95 = pred[:, :, q_index[0.95]]

    horizon_metrics: dict[str, Any] = {}
    crps_values = np.mean(np.abs(pred - tgt[:, :, None]), axis=(0, 2))

    for hi, horizon in enumerate(horizons):
        y = tgt[:, hi]
        yhat = p50[:, hi]
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        mae = float(mean_absolute_error(y, yhat))
        r2 = float(r2_score(y, yhat))
        coverage = float(np.mean((y >= p05[:, hi]) & (y <= p95[:, hi])))
        horizon_metrics[f"h{horizon}"] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "coverage_p05_p95": coverage,
            "crps_approx": float(crps_values[hi]),
        }

    metrics: dict[str, Any] = {
        "overall_rmse_p50": float(
            np.sqrt(mean_squared_error(tgt.ravel(), p50.ravel()))
        ),
        "overall_mae_p50": float(mean_absolute_error(tgt.ravel(), p50.ravel())),
        "overall_crps_approx": float(np.mean(np.abs(pred - tgt[:, :, None]))),
        "by_horizon": horizon_metrics,
    }

    if region_ids is not None and len(region_ids) == len(tgt):
        region_metrics: dict[str, Any] = {}
        fairness: dict[str, Any] = {}
        unique_regions = sorted(set(region_ids.tolist()))

        for rid in unique_regions:
            mask = region_ids == rid
            region_name = (
                region_name_by_id.get(int(rid), str(int(rid)))
                if region_name_by_id
                else str(int(rid))
            )
            region_metrics[region_name] = {}
            for hi, horizon in enumerate(horizons):
                y = tgt[mask, hi]
                yhat = p50[mask, hi]
                if len(y) == 0:
                    continue
                region_metrics[region_name][f"h{horizon}"] = {
                    "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
                    "mae": float(mean_absolute_error(y, yhat)),
                    "coverage_p05_p95": float(
                        np.mean((y >= p05[mask, hi]) & (y <= p95[mask, hi]))
                    ),
                    "count": int(len(y)),
                }

        for hi, horizon in enumerate(horizons):
            rmses = [
                region_metrics[name][f"h{horizon}"]["rmse"]
                for name in region_metrics
                if f"h{horizon}" in region_metrics[name]
            ]
            if rmses:
                fairness[f"h{horizon}"] = {
                    "rmse_max_min_ratio": float(max(rmses) / max(min(rmses), 1e-8)),
                    "rmse_max": float(max(rmses)),
                    "rmse_min": float(min(rmses)),
                }

        metrics["by_region"] = region_metrics
        metrics["fairness"] = fairness

    return metrics


def resolve_seq_len_for_horizon(horizon: int, override: int | None) -> int:
    if override is not None:
        seq_len = int(override)
    else:
        seq_len = int(
            DEFAULT_SEQ_LEN_BY_HORIZON.get(horizon, min(2 * horizon, MAX_SEQ_LEN))
        )
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if seq_len > MAX_SEQ_LEN:
        raise ValueError(f"seq_len={seq_len} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}")
    return seq_len


def parse_seq_len_map(raw: str, horizons: list[int]) -> dict[int, int]:
    parsed: dict[int, int] = {}
    text = raw.strip()
    if text:
        for part in text.split(","):
            chunk = part.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(
                    f"--seq-len-map must use horizon:seq_len pairs, got '{chunk}'"
                )
            h_str, s_str = chunk.split(":", 1)
            horizon = int(h_str.strip())
            seq_len = int(s_str.strip())
            parsed[horizon] = seq_len

    resolved: dict[int, int] = {}
    for horizon in horizons:
        resolved[horizon] = resolve_seq_len_for_horizon(
            horizon=horizon,
            override=parsed.get(horizon),
        )
    return resolved


def validate_seq_len_feasibility(
    split_tables: SplitTables,
    horizons: list[int],
    seq_len_by_horizon: dict[int, int],
    logger: logging.Logger,
) -> None:
    required = ["train", "val", "test"]
    split_frames = {
        "train": split_tables.train,
        "val": split_tables.val,
        "test": split_tables.test,
    }

    for horizon in horizons:
        seq_len = int(seq_len_by_horizon[horizon])
        for split_name in required:
            split_df = split_frames[split_name]
            counts_by_region: dict[str, int] = {}
            for region, region_df in split_df.groupby("region"):
                counts_by_region[str(region)] = int(max(0, len(region_df) - seq_len))

            if not counts_by_region:
                raise ValueError(f"Split {split_name} has no region rows")

            min_count = min(counts_by_region.values())
            total = sum(counts_by_region.values())
            logger.info(
                "[h%d seq_len=%d] feasibility split=%s total=%d min_region=%d by_region=%s",
                horizon,
                seq_len,
                split_name,
                total,
                min_count,
                counts_by_region,
            )

            if min_count == 0:
                raise ValueError(
                    f"Infeasible horizon configuration: h{horizon} seq_len={seq_len} "
                    f"produces zero samples for at least one region in split={split_name}. "
                    f"Counts={counts_by_region}"
                )


def train_for_pollutant(
    pollutant: str,
    split_tables: SplitTables,
    horizon: int,
    seq_len: int,
    min_attn_window: int,
    epochs: int,
    batch_size: int,
    patience: int,
    lr: float,
    device: torch.device,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    metadata = split_tables.metadata
    feature_cols = metadata["feature_columns"]
    target_cols = [f"target_{pollutant}_h{horizon}"]
    horizons = [horizon]
    horizon_loss_weights = [1.0]
    region_weights = {
        str(k): float(v) for k, v in metadata.get("region_weights", {}).items()
    }
    region_mapping = {
        str(k): int(v) for k, v in metadata.get("region_mapping", {}).items()
    }

    train_ds = RegionSequenceDataset(
        split_tables.train,
        feature_columns=feature_cols,
        target_columns=target_cols,
        region_weights=region_weights,
        region_mapping=region_mapping,
        seq_len=seq_len,
        max_samples=max_train_samples,
    )
    val_ds = RegionSequenceDataset(
        split_tables.val,
        feature_columns=feature_cols,
        target_columns=target_cols,
        region_weights=region_weights,
        region_mapping=region_mapping,
        seq_len=seq_len,
        max_samples=max_val_samples,
    )
    test_ds = RegionSequenceDataset(
        split_tables.test,
        feature_columns=feature_cols,
        target_columns=target_cols,
        region_weights=region_weights,
        region_mapping=region_mapping,
        seq_len=seq_len,
        max_samples=max_test_samples,
    )

    if len(train_ds) == 0:
        raise ValueError(
            f"No train sequences for pollutant={pollutant} h{horizon} seq_len={seq_len}"
        )
    if len(val_ds) == 0:
        raise ValueError(
            f"No val sequences for pollutant={pollutant} h{horizon} seq_len={seq_len}; "
            "reduce seq_len for this horizon"
        )
    if len(test_ds) == 0:
        raise ValueError(
            f"No test sequences for pollutant={pollutant} h{horizon} seq_len={seq_len}; "
            "reduce seq_len for this horizon"
        )

    logger.info(
        "[%s h%d] seq_len=%d sequence samples train=%d val=%d test=%d",
        pollutant,
        horizon,
        seq_len,
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    logger.info(
        "[%s h%d] per-region samples train=%s val=%s test=%s",
        pollutant,
        horizon,
        train_ds.sample_counts_by_region(),
        val_ds.sample_counts_by_region(),
        test_ds.sample_counts_by_region(),
    )

    logger.info(
        "[%s h%d] min_attn_window=%d horizon_loss_weights=%s",
        pollutant,
        horizon,
        min_attn_window,
        {f"h{h}": float(w) for h, w in zip(horizons, horizon_loss_weights)},
    )

    train_sampler = make_stratified_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    horizon_weights_tensor = torch.tensor(
        horizon_loss_weights,
        dtype=torch.float32,
        device=device,
    )

    model = HierarchicalQuantileLSTM(
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
        horizons=horizons,
        quantiles=QUANTILES,
        min_attn_window=min_attn_window,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    history: dict[str, list[float] | list[dict[str, float]]] = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_by_horizon": [],
        "val_loss_by_horizon": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    wait = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_h_losses = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_mode=True,
            horizon_weights=horizon_weights_tensor,
            quantiles=QUANTILES,
            horizons=horizons,
        )
        val_loss, val_h_losses = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train_mode=False,
            horizon_weights=horizon_weights_tensor,
            quantiles=QUANTILES,
            horizons=horizons,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_loss_by_horizon"].append(train_h_losses)
        history["val_loss_by_horizon"].append(val_h_losses)

        horizon_log = " ".join(
            [f"h{h}={val_h_losses.get(f'h{h}', float('nan')):.4f}" for h in horizons]
        )
        logger.info(
            "[%s h%d] epoch %d/%d train_loss=%.6f val_loss=%.6f val_by_h=[%s]",
            pollutant,
            horizon,
            epoch,
            epochs,
            train_loss,
            val_loss,
            horizon_log,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    "[%s h%d] early stopping at epoch %d", pollutant, horizon, epoch
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, val_tgt, val_regions = collect_predictions(
        model,
        val_loader,
        device,
        horizons=horizons,
        quantiles=QUANTILES,
    )
    test_pred, test_tgt, test_regions = collect_predictions(
        model,
        test_loader,
        device,
        horizons=horizons,
        quantiles=QUANTILES,
    )
    region_name_by_id = {int(v): str(k) for k, v in region_mapping.items()}

    val_metrics = compute_quantile_metrics(
        val_pred,
        val_tgt,
        horizons=horizons,
        quantiles=QUANTILES,
        region_ids=val_regions,
        region_name_by_id=region_name_by_id,
    )
    test_metrics = compute_quantile_metrics(
        test_pred,
        test_tgt,
        horizons=horizons,
        quantiles=QUANTILES,
        region_ids=test_regions,
        region_name_by_id=region_name_by_id,
    )

    prediction_path = MODELS_DIR / f"lstm_predictions_{pollutant}_h{horizon}.npz"
    np.savez_compressed(
        prediction_path,
        predictions=test_pred.astype(np.float32),
        targets=test_tgt.astype(np.float32),
        regions=test_regions.astype(np.int64),
        horizons=np.asarray(horizons, dtype=np.int64),
        quantiles=np.asarray(QUANTILES, dtype=np.float32),
    )

    model_path = MODELS_DIR / f"lstm_quantile_{pollutant}_h{horizon}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_cols,
            "target_columns": target_cols,
            "horizons": horizons,
            "seq_len": int(seq_len),
            "quantiles": QUANTILES,
            "min_attn_window": int(min_attn_window),
            "horizon_loss_weights": [float(x) for x in horizon_loss_weights],
            "region_weights": region_weights,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
        model_path,
    )

    logger.info(
        "[%s h%d] saved model to %s and predictions %s | test overall RMSE(p50)=%.6f CRPS=%.6f",
        pollutant,
        horizon,
        model_path,
        prediction_path,
        test_metrics["overall_rmse_p50"],
        test_metrics["overall_crps_approx"],
    )

    return {
        "pollutant": pollutant,
        "horizon": int(horizon),
        "seq_len": int(seq_len),
        "model_path": str(model_path),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "min_attn_window": int(min_attn_window),
        "horizon_loss_weights": [float(x) for x in horizon_loss_weights],
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "prediction_path": str(prediction_path),
        "sample_counts": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "sample_counts_by_region": {
            "train": train_ds.sample_counts_by_region(),
            "val": val_ds.sample_counts_by_region(),
            "test": test_ds.sample_counts_by_region(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hierarchical quantile LSTM models"
    )
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
    parser.add_argument(
        "--seq-len-map",
        type=str,
        default=",".join(f"{h}:{s}" for h, s in DEFAULT_SEQ_LEN_BY_HORIZON.items()),
        help=("Comma-separated horizon:seq_len map; default is 1:168,24:336,672:2402"),
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--min-attn-window",
        type=int,
        default=24,
        help="Minimum attention tail window in hours for every horizon head",
    )
    parser.add_argument(
        "--horizon-weighting",
        type=str,
        choices=["equal", "inverse_sqrt"],
        default="inverse_sqrt",
        help="Horizon loss weighting scheme",
    )
    parser.add_argument(
        "--horizon-loss-weights",
        type=str,
        default="",
        help="Ignored in separated-horizon training mode",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    data_dir = Path(args.data_dir)
    split_tables = load_split_tables(data_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    pollutants = [p.strip().lower() for p in args.pollutants.split(",") if p.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    invalid_horizons = [h for h in horizons if h not in ALL_HORIZONS]
    if invalid_horizons:
        raise ValueError(
            f"Unsupported horizons: {invalid_horizons}; allowed={ALL_HORIZONS}"
        )

    if args.horizon_weighting != "equal" or args.horizon_loss_weights.strip():
        logger.info(
            "Horizon weighting args are ignored in separated-horizon training mode"
        )

    seq_len_by_horizon = parse_seq_len_map(args.seq_len_map, horizons)
    validate_seq_len_feasibility(
        split_tables=split_tables,
        horizons=horizons,
        seq_len_by_horizon=seq_len_by_horizon,
        logger=logger,
    )
    results: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "training_mode": "separated_horizon_models",
        "horizons": horizons,
        "seq_len_by_horizon": {f"h{h}": int(seq_len_by_horizon[h]) for h in horizons},
        "quantiles": QUANTILES,
        "min_attn_window": int(args.min_attn_window),
        "pollutants": {},
    }

    for pollutant in pollutants:
        logger.info("Training pollutant: %s", pollutant)
        pollutant_result: dict[str, Any] = {}
        for horizon in horizons:
            seq_len = int(seq_len_by_horizon[horizon])
            logger.info(
                "Launching training for pollutant=%s horizon=h%d seq_len=%d",
                pollutant,
                horizon,
                seq_len,
            )
            horizon_result = train_for_pollutant(
                pollutant=pollutant,
                split_tables=split_tables,
                horizon=horizon,
                seq_len=seq_len,
                min_attn_window=args.min_attn_window,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                lr=args.lr,
                device=device,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                max_test_samples=args.max_test_samples,
                logger=logger,
            )
            pollutant_result[f"h{horizon}"] = horizon_result
        results["pollutants"][pollutant] = pollutant_result

    out_path = MODELS_DIR / "lstm_training_summary.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Saved LSTM training summary: %s", out_path)


if __name__ == "__main__":
    main()
