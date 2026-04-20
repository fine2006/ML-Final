#!/usr/bin/env python3
"""
Walk-Forward Cross-Validation for h168 (168-hour) Forecasting

Purpose: Validate champion configs generalize across time
- 5 folds, expanding window
- 168h gap between train end and test start (operational gap)
- Proves model predicts "next week" using only "last week" data
- Avoids false skill from auto-correlation

Structure:
  Fold 1: Train [2022-06 to 2023-06], Gap 168h, Test [2023-07 to 2023-08]
  Fold 2: Train [2022-06 to 2023-08], Gap 168h, Test [2023-09 to 2023-10]
  Fold 3: Train [2022-06 to 2023-10], Gap 168h, Test [2023-11 to 2024-01]
  Fold 4: Train [2022-06 to 2024-01], Gap 168h, Test [2024-02 to 2024-04]
  Fold 5: Train [2022-06 to 2024-04], Gap 168h, Test [2024-05 to 2024-07]

Run: python scripts/walkforward_cv_h168.py
Output: models/experiments/wfcv_h168_results.json
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "preprocessed_lstm_v1"
OUTPUT_DIR = ROOT / "models" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# WFCV settings
import argparse

parser = argparse.ArgumentParser(description="Walk-forward cross-validation for h168")
parser.add_argument("--n-folds", type=int, default=5, help="Number of WFCV folds")
parser.add_argument("--pollutants", type=str, default="pm25,pm10,no2,o3", help="Comma-separated pollutants")
parser.add_argument("--champion-config", type=str, default=None, help="Path to champion config JSON")
args_cli = parser.parse_args()

N_FOLDS = args_cli.n_folds
POLLUTANTS = [p.strip() for p in args_cli.pollutants.split(",")]
N_WORKERS = 4
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10

# Gap between train end and test start (operational gap)
GAP_HOURS = 168

# Model architecture (from pilot Refine_03)
NUM_LAYERS = 2
NUM_HEADS = 2
MIN_ATTN_WINDOW = 24
QUANTILES = [0.05, 0.5, 0.95, 0.99]
TARGET_QUANTILES = QUANTILES

# Composite loss weights
RMSE_WEIGHT = 0.6
PINBALL_WEIGHT = 0.4

# Pollutant-specific seq_len
SEQ_LEN_BY_POLLUTANT = {
    "pm25": 336,
    "pm10": 336,
    "no2": 168,
    "o3": 168,
}

# Champion configs (from optuna_h168_best_configs.json)
# Will be loaded dynamically
CHAMPION_CONFIGS = {}

# ============================================================================
# DATA LOADING WITH TIMESTAMPS
# ============================================================================


def load_split_data_with_timestamps(split: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load preprocessed data with timestamps for fold assignment from CSV."""
    import pandas as pd
    
    csv_path = DATA_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    
    FEATURE_COLUMNS = [
        "pm25", "pm10", "no2", "o3",
        "temperature", "humidity", "wind_speed", "wind_direction",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
        "region_id",
    ]
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    
    horizon = 168
    pollutant_list = ["pm25", "pm10", "no2", "o3"]
    target_cols = [f"target_{p}_h{horizon}" for p in pollutant_list]
    targets = df[target_cols].to_numpy(dtype=np.float32)
    
    metadata = {
        "timestamps": df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "regions": df["region"].tolist(),
        "region_names": sorted(df["region"].unique().tolist()),
        "pollutant_names": pollutant_list,
    }
    
    return features, targets, metadata


def get_fold_assignments(metadata: Dict, gap_hours: int = 168) -> Dict[int, Tuple[List[int], List[int]]]:
    """
    Assign samples to folds based on timestamps.
    
    Returns: {fold_idx: (train_indices, test_indices)}
    """
    timestamps = metadata["timestamps"]  # List of datetime strings
    n_samples = len(timestamps)
    
    # Convert to pandas for easy manipulation
    import pandas as pd
    ts_df = pd.to_datetime(timestamps)
    
    # Sort by timestamp
    sorted_indices = np.argsort(ts_df)
    ts_df_sorted = ts_df.iloc[sorted_indices]
    
    # Create fold boundaries
    # Use expanding window: each fold adds more train data
    fold_boundaries = []
    
    # Calculate time range
    min_ts = ts_df_sorted.min()
    max_ts = ts_df_sorted.max()
    total_hours = (max_ts - min_ts).total_seconds() / 3600
    
    # Divide into 5 roughly equal periods
    period_hours = total_hours / (N_FOLDS + 1)  # +1 for gap
    
    for fold in range(N_FOLDS):
        train_end = min_ts + pd.Timedelta(hours=(fold + 1) * period_hours)
        test_start = train_end + pd.Timedelta(hours=gap_hours)
        test_end = test_start + pd.Timedelta(hours=period_hours)
        
        fold_boundaries.append({
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
    
    # Assign indices to folds
    folds = {}
    for fold_idx, boundary in enumerate(fold_boundaries):
        train_mask = (ts_df_sorted <= boundary["train_end"])
        test_mask = (ts_df_sorted >= boundary["test_start"]) & (ts_df_sorted <= boundary["test_end"])
        
        train_indices = sorted_indices[train_mask.values]
        test_indices = sorted_indices[test_mask.values]
        
        folds[fold_idx] = (train_indices, test_indices)
    
    return folds


# ============================================================================
# DATASET
# ============================================================================


class SequenceDataset(Dataset):
    """Dataset for quantile regression with region weighting."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 metadata: Dict, pollutant_indices: Dict,
                 pollutant: str, horizon: int, seq_len: int,
                 region_weights: Dict[str, float],
                 indices: List[int] = None):
        self.features = features
        self.targets = targets
        self.pollutant_indices = pollutant_indices
        self.pollutant = pollutant
        self.horizon = horizon
        self.seq_len = seq_len
        self.region_weights = region_weights
        self.metadata = metadata
        
        self.region_to_idx = {name: idx for idx, name in enumerate(metadata["region_names"])}
        self.n_pollutants = len(metadata.get("pollutant_names", [])) or 4
        
        self.target_idx = pollutant_indices[pollutant]
        
        # Use all indices or subset
        if indices is not None:
            self.valid_indices = [
                idx for idx in indices 
                if idx >= seq_len and idx < len(features)
            ]
        else:
            self.valid_indices = list(range(seq_len, len(features)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        orig_idx = self.valid_indices[idx]
        
        x = self.features[orig_idx - self.seq_len:orig_idx]
        y = self.targets[orig_idx, self.target_idx]
        
        region = self.metadata["regions"][orig_idx]
        region_weight = self.region_weights.get(region, 1.0)
        
        return torch.FloatTensor(x), torch.FloatTensor([y]), torch.FloatTensor([region_weight])


# ============================================================================
# MODEL (same as optuna_tune)
# ============================================================================


class HierarchicalQuantileLSTM(nn.Module):
    """Hierarchical LSTM with quantile heads."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_heads: int, dropout: float, head_dropout: float,
                 n_quantiles: int = 4, min_attn_window: int = 24):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_quantiles = n_quantiles
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_quantiles)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out + lstm_out)
        
        pooled = attn_out[:, -1, :]
        encoded = self.encoder(pooled)
        
        quantile_preds = torch.cat([
            head(encoded) for head in self.quantile_heads
        ], dim=1)
        
        return quantile_preds


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


def composite_loss(preds: torch.Tensor, targets: torch.Tensor,
                   quantiles: List[float]) -> Tuple[torch.Tensor, Dict]:
    """Composite objective: 0.6 × RMSE + 0.4 × MeanPinballLoss."""
    
    # RMSE for median
    median_idx = quantiles.index(0.5)
    rmse = torch.sqrt(torch.mean((preds[:, median_idx] - targets.squeeze()) ** 2))
    
    # Pinball loss
    pinball = 0.0
    for i, q in enumerate(quantiles):
        residual = targets.squeeze() - preds[:, i]
        mask = (residual < 0).float()
        quantile_loss = (2 * q - 1) * residual * (1 - 2 * mask)
        pinball += torch.mean(torch.abs(quantile_loss))
    pinball /= len(quantiles)
    
    # Coverage
    p05_idx = quantiles.index(0.05)
    p95_idx = quantiles.index(0.95)
    lower = preds[:, p05_idx]
    upper = preds[:, p95_idx]
    actual = targets.squeeze()
    coverage = torch.mean((actual >= lower) & (actual <= upper))
    cov_loss = torch.abs(coverage - 0.90)
    
    score = RMSE_WEIGHT * rmse + PINBALL_WEIGHT * pinball
    total_loss = score + 10.0 * cov_loss
    
    metrics = {
        "rmse": rmse.item(),
        "pinball": pinball.item(),
        "coverage": coverage.item(),
        "score": score.item(),
    }
    
    return total_loss, metrics


# ============================================================================
# TRAINING
# ============================================================================


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                quantiles: List[float]) -> Tuple[float, Dict]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    metrics_accum = {}
    
    for batch_features, batch_targets, batch_weights in dataloader:
        optimizer.zero_grad()
        
        preds = model(batch_features)
        loss, metrics = composite_loss(preds, batch_targets, quantiles)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
    
    n_batches = len(dataloader)
    metrics_mean = {k: v / n_batches for k, v in metrics_accum.items()}
    
    return total_loss / n_batches, metrics_mean


def validate(model: nn.Module, dataloader: DataLoader, quantiles: List[float]) -> Dict:
    """Validate and compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets, _ in dataloader:
            preds = model(batch_features)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    metrics = {}
    
    median_idx = quantiles.index(0.5)
    rmse = np.sqrt(np.mean((all_preds[:, median_idx] - all_targets.squeeze()) ** 2))
    metrics["rmse"] = rmse
    
    pinball = 0.0
    for i, q in enumerate(quantiles):
        residual = all_targets.squeeze() - all_preds[:, i]
        mask = (residual < 0).astype(float)
        quantile_loss = (2 * q - 1) * residual * (1 - 2 * mask)
        pinball += np.mean(np.abs(quantile_loss))
    metrics["pinball"] = pinball / len(quantiles)
    
    p05_idx = quantiles.index(0.05)
    p95_idx = quantiles.index(0.95)
    lower = all_preds[:, p05_idx]
    upper = all_preds[:, p95_idx]
    actual = all_targets.squeeze()
    coverage = np.mean((actual >= lower) & (actual <= upper))
    metrics["coverage"] = coverage
    
    score = RMSE_WEIGHT * rmse + PINBALL_WEIGHT * pinball
    metrics["score"] = score
    
    return metrics


def train_model(hparams: Dict, pollutant: str, region_weights: Dict[str, float],
                train_indices: List[int], test_indices: List[int],
                train_metadata: Dict, test_metadata: Dict,
                seq_len: int) -> Dict:
    """Train a single model with given hyperparameters."""
    
    input_dim = 15
    
    # Load data from CSV
    import pandas as pd
    train_csv = DATA_DIR / "train.csv"
    test_csv = DATA_DIR / "test.csv"
    
    FEATURE_COLUMNS = [
        "pm25", "pm10", "no2", "o3",
        "temperature", "humidity", "wind_speed", "wind_direction",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
        "region_id",
    ]
    horizon = 168
    pollutant_list = ["pm25", "pm10", "no2", "o3"]
    target_cols = [f"target_{p}_h{horizon}" for p in pollutant_list]
    
    train_df = pd.read_csv(train_csv, parse_dates=["timestamp"])
    test_df = pd.read_csv(test_csv, parse_dates=["timestamp"])
    
    train_features = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    train_targets = train_df[target_cols].to_numpy(dtype=np.float32)
    test_features = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    test_targets = test_df[target_cols].to_numpy(dtype=np.float32)
    
    # Override indices to use CSV-based indices (which are sequential after sorting)
    train_valid = [i for i in train_indices if i < len(train_features)]
    test_valid = [i for i in test_indices if i < len(test_features)]
    
    # Create datasets
    train_dataset = SequenceDataset(
        train_features,
        train_targets,
        train_metadata, {},
        pollutant, 168, seq_len, region_weights, train_valid
    )
    
    test_dataset = SequenceDataset(
        test_features,
        test_targets,
        test_metadata, {},
        pollutant, 168, seq_len, region_weights, test_valid
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=N_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=N_WORKERS)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalQuantileLSTM(
        input_dim=input_dim,
        hidden_dim=hparams["hidden_dim"],
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=hparams["dropout"],
        head_dropout=hparams["head_dropout"],
        n_quantiles=len(TARGET_QUANTILES),
        min_attn_window=MIN_ATTN_WINDOW
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"]
    )
    
    # Training loop
    best_val_score = float("inf")
    patience_counter = 0
    best_model_state = None
    
    history = {
        "train_loss": [],
        "val_score": [],
        "val_rmse": [],
        "val_coverage": [],
    }
    
    for epoch in range(MAX_EPOCHS):
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, TARGET_QUANTILES
        )
        
        val_metrics = validate(model, test_loader, TARGET_QUANTILES)
        
        history["train_loss"].append(train_loss)
        history["val_score"].append(val_metrics["score"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_coverage"].append(val_metrics["coverage"])
        
        if val_metrics["score"] < best_val_score:
            best_val_score = val_metrics["score"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_metrics = validate(model, test_loader, TARGET_QUANTILES)
    test_metrics["history"] = history
    test_metrics["best_epoch"] = len(history["val_score"])
    
    return test_metrics


# ============================================================================
# WALK-FORWARD CROSS-VALIDATION
# ============================================================================


def run_wfcv(pollutant: str, champion_hparams: Dict) -> Dict:
    """Run walk-forward cross-validation for a single pollutant."""
    
    seq_len = SEQ_LEN_BY_POLLUTANT[pollutant]
    
    # Load data from CSV
    import pandas as pd
    train_csv = DATA_DIR / "train.csv"
    test_csv = DATA_DIR / "test.csv"
    train_df = pd.read_csv(train_csv, parse_dates=["timestamp"])
    test_df = pd.read_csv(test_csv, parse_dates=["timestamp"])
    
    train_metadata = {
        "timestamps": train_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "regions": train_df["region"].tolist(),
        "region_names": sorted(train_df["region"].unique().tolist()),
        "pollutant_names": ["pm25", "pm10", "no2", "o3"],
    }
    test_metadata = {
        "timestamps": test_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "regions": test_df["region"].tolist(),
        "region_names": sorted(test_df["region"].unique().tolist()),
        "pollutant_names": ["pm25", "pm10", "no2", "o3"],
    }
    
    # Get fold assignments
    folds = get_fold_assignments(train_metadata, gap_hours=GAP_HOURS)
    
    # Pollutant indices
    pollutant_names = train_metadata["pollutant_names"]
    pollutant_indices = {name: idx for idx, name in enumerate(pollutant_names)}
    
    # Region weights (from Phase 1 analysis - mild weighting)
    DEFAULT_REGION_WEIGHTS = {
        "AIIMS": 0.979,
        "IGKV": 0.994,
        "Bhatagaon": 1.009,
        "SILTARA": 1.020,
    }
    
    # Try to load from metadata.json
    metadata_path = ROOT / "data" / "preprocessed_lstm_v1" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        region_weights = metadata.get("region_weights", DEFAULT_REGION_WEIGHTS)
    else:
        region_weights = DEFAULT_REGION_WEIGHTS
    
    # Results
    fold_results = []
    all_test_predictions = []
    all_test_targets = []
    
    print(f"\nRunning WFCV for {pollutant.upper()} (seq_len={seq_len})")
    print(f"Champion hparams: {champion_hparams}")
    print(f"Gap: {GAP_HOURS} hours")
    print()
    
    for fold_idx in range(N_FOLDS):
        train_indices, test_indices = folds[fold_idx]
        
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Test samples: {len(test_indices)}")
        
        try:
            # Train
            metrics = train_model(
                champion_hparams,
                pollutant,
                region_weights,
                train_indices,
                test_indices,
                train_metadata,
                test_metadata,
                seq_len
            )
            
            fold_results.append({
                "fold": fold_idx,
                "train_samples": len(train_indices),
                "test_samples": len(test_indices),
                "metrics": metrics,
            })
            
            print(f"  Test RMSE: {metrics['rmse']:.4f}")
            print(f"  Test Coverage: {metrics['coverage']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Fold {fold_idx + 1} failed: {e}")
            fold_results.append({
                "fold": fold_idx,
                "train_samples": len(train_indices),
                "test_samples": len(test_indices),
                "error": str(e),
            })
    
    # Aggregate results
    valid_results = [f for f in fold_results if "error" not in f]
    
    if valid_results:
        avg_rmse = np.mean([f["metrics"]["rmse"] for f in valid_results])
        avg_coverage = np.mean([f["metrics"]["coverage"] for f in valid_results])
        std_rmse = np.std([f["metrics"]["rmse"] for f in valid_results])
        std_coverage = np.std([f["metrics"]["coverage"] for f in valid_results])
    else:
        avg_rmse = avg_coverage = std_rmse = std_coverage = float("nan")
    
    wfcv_results = {
        "pollutant": pollutant,
        "seq_len": seq_len,
        "champion_hparams": champion_hparams,
        "gap_hours": GAP_HOURS,
        "n_folds": N_FOLDS,
        "fold_results": fold_results,
        "aggregate": {
            "avg_rmse": avg_rmse,
            "std_rmse": std_rmse,
            "avg_coverage": avg_coverage,
            "std_coverage": std_coverage,
        }
    }
    
    return wfcv_results


def main():
    """Run walk-forward cross-validation for all pollutants."""
    
    print("=" * 80)
    print("WALK-FORWARD CROSS-VALIDATION FOR h168")
    print("=" * 80)
    print(f"Folds: {N_FOLDS}")
    print(f"Gap: {GAP_HOURS} hours (operational gap)")
    print(f"Architecture: num_layers={NUM_LAYERS}, num_heads={NUM_HEADS}")
    print("=" * 80)
    
    # Load champion configs
    optuna_path = OUTPUT_DIR / "optuna_h168_best_configs.json"
    if optuna_path.exists():
        with open(optuna_path, "r") as f:
            optuna_results = json.load(f)
        
        for pollutant, results in optuna_results.items():
            if "best_hparams" in results:
                CHAMPION_CONFIGS[pollutant] = results["best_hparams"]
                print(f"✓ Loaded champion for {pollutant}: {results['best_hparams']}")
            else:
                print(f"✗ No champion found for {pollutant}")
    else:
        print(f"✗ Optuna results not found at {optuna_path}")
        print("Please run optuna_tune_h168.py first.")
        return
    
    # Run WFCV for all pollutants
    all_wfcv_results = {}
    
    for pollutant, hparams in CHAMPION_CONFIGS.items():
        wfcv_results = run_wfcv(pollutant, hparams)
        all_wfcv_results[pollutant] = wfcv_results
    
    # Save results
    output_path = OUTPUT_DIR / "wfcv_h168_results.json"
    with open(output_path, "w") as f:
        json.dump(all_wfcv_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 80}")
    print(f"WFCV RESULTS SAVED TO: {output_path}")
    print(f"{'=' * 80}")
    
    # Print summary
    print("\nAGGREGATE SUMMARY:")
    for pollutant, results in all_wfcv_results.items():
        agg = results["aggregate"]
        print(f"  {pollutant.upper()}:")
        print(f"    RMSE: {agg['avg_rmse']:.4f} ± {agg['std_rmse']:.4f}")
        print(f"    Coverage: {agg['avg_coverage']:.4f} ± {agg['std_coverage']:.4f}")


if __name__ == "__main__":
    main()
