"""
Single-region, single-horizon multi-pollutant inference for LSTM quantile models.

This script returns quantile forecasts with a 95% CI focus (p05, p50, p95)
and optional p99 diagnostic for all requested pollutants for one region at one
selected horizon.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.train_lstm import (  # noqa: E402
    ALL_HORIZONS,
    HierarchicalQuantileLSTM,
)


DATA_DIR = PROJECT_ROOT / "data" / "preprocessed_lstm_v1"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs" / "predict"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"predict_{stamp}.log"

    logger = logging.getLogger("predict")
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
    parser = argparse.ArgumentParser(description="LSTM quantile inference")
    parser.add_argument(
        "--pollutants",
        type=str,
        default="pm25,pm10,no2,o3",
        help="Comma-separated pollutant list",
    )
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument(
        "--timestamp",
        type=str,
        default="",
        help="Optional timestamp cutoff; latest <= timestamp is used",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help="Optional preprocessed input CSV (must include timestamp, region, feature columns)",
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON output file path",
    )
    return parser.parse_args()


def load_default_inference_table(data_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["timestamp"])
    val = pd.read_csv(data_dir / "val.csv", parse_dates=["timestamp"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["timestamp"])
    df = pd.concat([train, val, test], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("h")
    df = df.dropna(subset=["timestamp", "region"]).copy()
    df = df.sort_values(["region", "timestamp"]).drop_duplicates(
        subset=["region", "timestamp"], keep="last"
    )
    return df.reset_index(drop=True)


def resolve_region_name(df: pd.DataFrame, requested: str) -> str:
    regions = sorted(df["region"].dropna().astype(str).unique().tolist())
    if requested in regions:
        return requested

    lower_map: dict[str, list[str]] = {}
    for region in regions:
        lower_map.setdefault(region.lower(), []).append(region)
    matches = lower_map.get(requested.lower(), [])
    if len(matches) == 1:
        return matches[0]

    raise ValueError(f"Region '{requested}' not found. Available regions: {regions}")


def resolve_anchor_row(
    region_df: pd.DataFrame,
    timestamp: str,
) -> tuple[int, pd.Timestamp, str | None]:
    if not timestamp.strip():
        pos = len(region_df) - 1
        ts = pd.Timestamp(region_df.iloc[pos]["timestamp"])
        return pos, ts, None

    ts_req = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(ts_req):
        raise ValueError(f"Invalid --timestamp value: {timestamp}")
    ts_req = pd.Timestamp(ts_req).floor("h")

    candidates = np.where(region_df["timestamp"].to_numpy() <= ts_req)[0]
    if len(candidates) == 0:
        raise ValueError(
            f"No rows for region before or at requested timestamp {ts_req.isoformat()}"
        )

    pos = int(candidates[-1])
    ts_used = pd.Timestamp(region_df.iloc[pos]["timestamp"])
    note = None
    if ts_used != ts_req:
        note = (
            f"No exact row at requested timestamp; used latest prior row at "
            f"{ts_used.isoformat()}"
        )
    return pos, ts_used, note


def label_quantile(q: float) -> str:
    return f"p{int(round(q * 100)):02d}"


def adapt_checkpoint_feature_columns(
    feature_cols: list[str],
    available_cols: set[str],
) -> tuple[list[str], dict[str, str]]:
    adapted = list(feature_cols)
    remapped: dict[str, str] = {}

    legacy_to_raw = {
        "temperature_lag_1": "temperature",
        "humidity_lag_1": "humidity",
        "wind_speed_lag_1": "wind_speed",
        "wind_direction_lag_1": "wind_direction",
    }

    for i, col in enumerate(adapted):
        if col in available_cols:
            continue
        mapped = legacy_to_raw.get(col)
        if mapped and mapped in available_cols:
            adapted[i] = mapped
            remapped[col] = mapped

    return adapted, remapped


def load_checkpoint_for_horizon(
    models_dir: Path,
    pollutant: str,
    horizon: int,
    device: torch.device,
) -> tuple[dict[str, Any], Path]:
    preferred = models_dir / f"lstm_quantile_{pollutant}_h{horizon}.pt"
    if preferred.exists():
        return torch.load(preferred, map_location=device), preferred

    legacy = models_dir / f"lstm_quantile_{pollutant}.pt"
    if legacy.exists():
        checkpoint = torch.load(legacy, map_location=device)
        model_horizons = [int(h) for h in checkpoint.get("horizons", ALL_HORIZONS)]
        if horizon in model_horizons:
            return checkpoint, legacy

    raise FileNotFoundError(
        f"Missing model for pollutant={pollutant} horizon=h{horizon}: "
        f"checked {preferred.name} and {legacy.name}"
    )


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    if args.horizon not in ALL_HORIZONS:
        raise ValueError(f"Unsupported horizon={args.horizon}; allowed={ALL_HORIZONS}")

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.input_csv.strip():
        df = pd.read_csv(args.input_csv, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("h")
    else:
        df = load_default_inference_table(data_dir)

    region = resolve_region_name(df, args.region)
    region_df = (
        df[df["region"].astype(str) == region]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if region_df.empty:
        raise ValueError(f"No rows available for region={region}")

    pollutants = [p.strip().lower() for p in args.pollutants.split(",") if p.strip()]
    checkpoint_cache: dict[str, tuple[dict[str, Any], Path]] = {}
    seq_lens: list[int] = []
    for pollutant in pollutants:
        checkpoint, model_path = load_checkpoint_for_horizon(
            models_dir=models_dir,
            pollutant=pollutant,
            horizon=args.horizon,
            device=device,
        )
        model_horizons = [int(h) for h in checkpoint.get("horizons", ALL_HORIZONS)]
        if args.horizon not in model_horizons:
            raise ValueError(
                f"Model {model_path.name} does not support horizon {args.horizon}; "
                f"available={model_horizons}"
            )
        seq_len = int(checkpoint.get("seq_len", 672))
        if seq_len <= 0:
            raise ValueError(f"Invalid seq_len={seq_len} in {model_path.name}")
        checkpoint_cache[pollutant] = (checkpoint, model_path)
        seq_lens.append(seq_len)

    max_required_seq_len = max(seq_lens) if seq_lens else 1

    pos, ts_anchor, ts_note = resolve_anchor_row(region_df, args.timestamp)
    if pos < max_required_seq_len:
        earliest_idx = min(max_required_seq_len, len(region_df) - 1)
        earliest = pd.Timestamp(region_df.iloc[earliest_idx]["timestamp"])
        raise ValueError(
            f"Not enough history for region={region} at {ts_anchor.isoformat()}. "
            f"Need at least {max_required_seq_len} prior rows; earliest eligible anchor is {earliest.isoformat()}"
        )

    forecast: dict[str, Any] = {}

    logger.info(
        "Running inference region=%s horizon=%dh anchor=%s device=%s pollutants=%s",
        region,
        args.horizon,
        ts_anchor,
        device,
        pollutants,
    )

    for pollutant in pollutants:
        checkpoint, model_path = checkpoint_cache[pollutant]
        feature_cols = checkpoint["feature_columns"]
        adapted_feature_cols, remapped = adapt_checkpoint_feature_columns(
            feature_cols=feature_cols,
            available_cols=set(region_df.columns),
        )
        if remapped:
            logger.info(
                "[%s h%d] adapted legacy feature columns: %s",
                pollutant,
                args.horizon,
                remapped,
            )
        model_horizons = [int(h) for h in checkpoint.get("horizons", ALL_HORIZONS)]
        seq_len = int(checkpoint.get("seq_len", 672))

        missing = [c for c in adapted_feature_cols if c not in region_df.columns]
        if missing:
            raise ValueError(
                f"Input table missing required feature columns for {pollutant}: {missing}"
            )

        window = region_df.iloc[pos - seq_len : pos][adapted_feature_cols]
        if window.isna().any().any():
            raise ValueError(
                f"NaN values present in inference window for {pollutant}; cannot run prediction"
            )

        xb_arr = np.ascontiguousarray(window.to_numpy(dtype=np.float32))
        xb = torch.from_numpy(xb_arr).unsqueeze(0).to(device)

        model_hparams = checkpoint.get("model_hparams", {})
        hidden_dim = int(model_hparams.get("hidden_dim", 128))
        num_layers = int(model_hparams.get("num_layers", 2))
        num_heads = int(model_hparams.get("num_heads", 4))
        dropout = float(model_hparams.get("dropout", 0.3))
        head_dropout = float(model_hparams.get("head_dropout", 0.2))

        embed_dim = hidden_dim * 2
        if num_heads <= 0 or (embed_dim % num_heads) != 0:
            raise ValueError(
                f"Invalid model_hparams in checkpoint for {pollutant}: "
                f"hidden_dim={hidden_dim}, num_heads={num_heads}"
            )

        model = HierarchicalQuantileLSTM(
            input_dim=len(adapted_feature_cols),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            head_dropout=head_dropout,
            horizons=model_horizons,
            quantiles=[
                float(q) for q in checkpoint.get("quantiles", [0.05, 0.50, 0.95, 0.99])
            ],
            min_attn_window=int(checkpoint.get("min_attn_window", 2)),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        with torch.no_grad():
            pred = model(xb).cpu().numpy()[0]

        h_idx = model_horizons.index(args.horizon)
        q_levels = [
            float(q) for q in checkpoint.get("quantiles", [0.05, 0.50, 0.95, 0.99])
        ]
        q_values = pred[h_idx]
        q_map = {
            label_quantile(q): float(v)
            for q, v in zip(q_levels, np.maximum.accumulate(q_values))
        }

        p05 = q_map.get("p05")
        p95 = q_map.get("p95")
        interval_width = (
            float(p95 - p05) if p05 is not None and p95 is not None else None
        )

        forecast[pollutant] = {
            "quantiles": q_map,
            "interval_width_p05_p95": interval_width,
            "model_path": str(model_path),
            "model_horizons": model_horizons,
            "seq_len": int(seq_len),
        }

    out = {
        "created_at": datetime.now().isoformat(),
        "device": str(device),
        "region": region,
        "horizon_hours": int(args.horizon),
        "anchor_timestamp": ts_anchor.isoformat(),
        "forecast_timestamp": (ts_anchor + timedelta(hours=args.horizon)).isoformat(),
        "history_window_hours": int(max_required_seq_len),
        "pollutants": forecast,
    }
    if args.timestamp.strip():
        out["requested_timestamp"] = (
            pd.Timestamp(pd.to_datetime(args.timestamp, errors="coerce"))
            .floor("h")
            .isoformat()
        )
    if ts_note:
        out["note"] = ts_note

    out_json = json.dumps(out, indent=2)
    print(out_json)

    if args.output.strip():
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(out_json + "\n", encoding="utf-8")
        logger.info("Saved prediction JSON: %s", output_path)


if __name__ == "__main__":
    main()
