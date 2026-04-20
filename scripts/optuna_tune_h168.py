#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for h168 (168-hour) Forecasting

Search Space (approved 2026-04-19):
- hidden_dim: [64, 96]
- dropout: [0.35, 0.42]
- head_dropout: [0.15, 0.20]
- lr: [2e-4, 5e-4]
- weight_decay: [1e-5, 1e-3]
- seq_len: Fixed (PM=336, Gases=168)
- num_layers: Fixed at 2 (Refine_03 showed leaner is better)
- num_heads: Fixed at 2

Objective: Minimize composite loss = 0.6 × RMSE + 0.4 × MeanPinballLoss
Pareto Gate: Penalize trials where Coverage < 0.70 even if RMSE low
Coverage Target: 0.90 for p05-p95 interval

Run: python scripts/optuna_tune_h168.py
Output: models/experiments/optuna_h168_best_configs.json
"""

import json
import math
import os
import random
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

OUTPUT_FILE = "optuna_h168_best_configs.json"

# Optuna settings
N_TRIALS = 50
N_WORKERS = 4
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10

# CLI argument parsing
import argparse
parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for h168")
parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="Number of Optuna trials per pollutant")
parser.add_argument("--pollutants", type=str, default="pm25,pm10,no2,o3", help="Comma-separated pollutants")
parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID to use (0 or 1)")
parser.add_argument("--output", type=str, default=None, help="Output filename (default: optuna_h168_best_configs.json)")
args_cli = parser.parse_args()

# Set GPU device if specified
if args_cli.gpu_id is not None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu_id)

# Custom output path for multi-GPU runs
if args_cli.output:
    OUTPUT_FILE = args_cli.output

# Override with CLI args if provided
if args_cli.n_trials != N_TRIALS:
    N_TRIALS = args_cli.n_trials

POLLUTANTS = [p.strip() for p in args_cli.pollutants.split(",")]

# Search space (approved)
SEARCH_SPACE = {
    "hidden_dim": [64, 96],
    "dropout": [0.35, 0.37, 0.389, 0.40, 0.42],
    "head_dropout": [0.15, 0.167, 0.183, 0.20],
    "lr": [2e-4, 2.5e-4, 3e-4, 3.78e-4, 4e-4, 5e-4],
    "weight_decay": [1e-5, 5e-5, 1e-4, 1.31e-4, 5e-4, 1e-3],
}

# Fixed settings
NUM_LAYERS = 2
NUM_HEADS = 2
MIN_ATTN_WINDOW = 24
QUANTILES = [0.05, 0.5, 0.95, 0.99]
TARGET_QUANTILES = QUANTILES

# Composite loss weights
RMSE_WEIGHT = 0.6
PINBALL_WEIGHT = 0.4

# Pareto gate threshold
COVERAGE_PENALTY_THRESHOLD = 0.70
COVERAGE_PENALTY_MULTIPLIER = 10.0

# Pollutant-specific seq_len
SEQ_LEN_BY_POLLUTANT = {
    "pm25": 336,
    "pm10": 336,
    "no2": 168,
    "o3": 168,
}

# ============================================================================
# DATA LOADING
# ============================================================================


def load_split_data(split: str) -> Tuple[np.ndarray, np.ndarray, Dict, Dict[str, int]]:
    """Load preprocessed CSV data for a given split.
    Returns:
        features (np.ndarray): shape (N, n_features)
        targets (np.ndarray): shape (N, n_pollutants) for the configured horizon (168h)
        metadata (dict): minimal metadata with region info
        pollutant_indices (dict): mapping pollutant name to column index in targets
    """
    import pandas as pd
    # Load CSV for the split
    csv_path = DATA_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Feature columns (must match preprocess_lstm FEATURE_COLUMNS)
    FEATURE_COLUMNS = [
        "pm25",
        "pm10",
        "no2",
        "o3",
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        "region_id",
    ]
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    # Build targets for the fixed horizon (168 hours)
    horizon = 168
    pollutant_list = ["pm25", "pm10", "no2", "o3"]
    target_cols = [f"target_{p}_h{horizon}" for p in pollutant_list]
    targets = df[target_cols].to_numpy(dtype=np.float32)
    # Metadata needed for the dataset
    metadata = {
        "region_names": sorted(df["region"].unique().tolist()),
        "regions": df["region"].tolist(),
        "pollutant_names": pollutant_list,
    }
    pollutant_indices = {name: idx for idx, name in enumerate(pollutant_list)}
    return features, targets, metadata, pollutant_indices


# ============================================================================
# DATASET
# ============================================================================


class SequenceDataset(Dataset):
    """Dataset for quantile regression with region weighting."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 metadata: Dict, pollutant_indices: Dict,
                 pollutant: str, horizon: int, seq_len: int,
                 region_weights: Dict[str, float]):
        self.features = features
        self.targets = targets
        self.pollutant_indices = pollutant_indices
        self.pollutant = pollutant
        self.horizon = horizon
        self.seq_len = seq_len
        self.region_weights = region_weights
        self.metadata = metadata
        
        self.region_to_idx = {name: idx for idx, name in enumerate(metadata["region_names"])}
        self.n_pollutants = len(pollutant_names) if (pollutant_names := metadata.get("pollutant_names")) else 4
        
        # Extract target column for this pollutant/horizon
        self.target_idx = pollutant_indices[pollutant]
        
        # Calculate valid indices (ensure we have seq_len history before target)
        self.valid_indices = list(range(seq_len, len(features)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        orig_idx = self.valid_indices[idx]
        
        # Extract sequence
        x = self.features[orig_idx - self.seq_len:orig_idx]
        y = self.targets[orig_idx, self.target_idx]
        
        # Get region for weighting
        region = self.metadata["regions"][orig_idx]
        region_weight = self.region_weights.get(region, 1.0)
        
        return torch.FloatTensor(x), torch.FloatTensor([y]), torch.FloatTensor([region_weight])


# ============================================================================
# MODEL
# ============================================================================


class HierarchicalQuantileLSTM(nn.Module):
    """Hierarchical LSTM with quantile heads (simplified for tuning)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_heads: int, dropout: float, head_dropout: float,
                 n_quantiles: int = 4, min_attn_window: int = 24):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_quantiles = n_quantiles
        
        # BiLSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Quantile heads
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_quantiles)
        ])
        
        # Initialize weights
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
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out + lstm_out)
        
        # Pool to sequence length 1 (last time step)
        pooled = attn_out[:, -1, :]
        
        # Shared encoder
        encoded = self.encoder(pooled)
        
        # Quantile predictions
        quantile_preds = torch.cat([
            head(encoded) for head in self.quantile_heads
        ], dim=1)
        
        return quantile_preds


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


def pinball_loss(preds: torch.Tensor, targets: torch.Tensor, 
                 quantiles: List[float]) -> torch.Tensor:
    """Calculate pinball loss for quantile regression."""
    batch_size = targets.shape[0]
    total_loss = 0.0
    
    for i, q in enumerate(quantiles):
        residual = targets.squeeze() - preds[:, i]
        mask = (residual < 0).float()
        quantile_loss = (2 * q - 1) * residual * (1 - 2 * mask)
        total_loss += torch.mean(torch.abs(quantile_loss))
    
    return total_loss / len(quantiles)


def rmse_loss(preds: torch.Tensor, targets: torch.Tensor,
              quantiles: List[float]) -> torch.Tensor:
    """Calculate RMSE for median (p50) prediction."""
    median_idx = quantiles.index(0.5)
    median_preds = preds[:, median_idx]
    return torch.sqrt(torch.mean((median_preds - targets.squeeze()) ** 2))


def coverage_loss(preds: torch.Tensor, targets: torch.Tensor,
                   quantiles: List[float]) -> torch.Tensor:
    """Calculate raw coverage for p05-p95 interval (target 0.90).
    Returns the proportion of targets within the predicted interval.
    """
    p05_idx = quantiles.index(0.05)
    p95_idx = quantiles.index(0.95)
    
    lower = preds[:, p05_idx]
    upper = preds[:, p95_idx]
    actual = targets.squeeze()
    
    in_interval = ((actual >= lower) & (actual <= upper)).float()
    coverage = torch.mean(in_interval)
    
    return coverage


def composite_loss(preds: torch.Tensor, targets: torch.Tensor,
                   quantiles: List[float]) -> Tuple[torch.Tensor, Dict]:
    """
    Composite objective: 0.6 × RMSE + 0.4 × MeanPinballLoss
    Includes coverage penalty for Pareto gate.
    """
    rmse = rmse_loss(preds, targets, quantiles)
    pinball = pinball_loss(preds, targets, quantiles)
    cov = coverage_loss(preds, targets, quantiles)
    
    # Composite score
    score = RMSE_WEIGHT * rmse + PINBALL_WEIGHT * pinball
    
    # Coverage penalty (Pareto gate)
    coverage_penalty = 0.0
    # Penalize if coverage falls below the threshold
    if cov.item() < COVERAGE_PENALTY_THRESHOLD:
        coverage_penalty = COVERAGE_PENALTY_MULTIPLIER * (COVERAGE_PENALTY_THRESHOLD - cov.item())
    
    total_loss = score + coverage_penalty
    
    metrics = {
        "rmse": rmse.item(),
        "pinball": pinball.item(),
        "coverage": cov.item(),
        "score": score.item(),
        "coverage_penalty": coverage_penalty,
    }
    
    return total_loss, metrics


# ============================================================================
# TRAINING
# ============================================================================


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                quantiles: List[float], device: torch.device) -> Tuple[float, Dict]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    metrics_accum = {}
    
    for batch_features, batch_targets, batch_weights in dataloader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        preds = model(batch_features)
        loss, metrics = composite_loss(preds, batch_targets, quantiles)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
    
    n_batches = len(dataloader)
    metrics_mean = {k: v / n_batches for k, v in metrics_accum.items()}
    
    return total_loss / n_batches, metrics_mean


def validate(model: nn.Module, dataloader: DataLoader, quantiles: List[float], device: torch.device) -> Dict:
    """Validate and compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets, _ in dataloader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            preds = model(batch_features)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Compute metrics
    metrics = {}
    
    # RMSE for median
    median_idx = quantiles.index(0.5)
    rmse = np.sqrt(np.mean((all_preds[:, median_idx] - all_targets.squeeze()) ** 2))
    metrics["rmse"] = rmse
    
    # Pinball loss
    pinball = 0.0
    for i, q in enumerate(quantiles):
        residual = all_targets.squeeze() - all_preds[:, i]
        mask = (residual < 0).astype(float)
        quantile_loss = (2 * q - 1) * residual * (1 - 2 * mask)
        pinball += np.mean(np.abs(quantile_loss))
    metrics["pinball"] = pinball / len(quantiles)
    
    # Coverage
    p05_idx = quantiles.index(0.05)
    p95_idx = quantiles.index(0.95)
    lower = all_preds[:, p05_idx]
    upper = all_preds[:, p95_idx]
    actual = all_targets.squeeze()
    coverage = np.mean((actual >= lower) & (actual <= upper))
    metrics["coverage"] = coverage
    
    # Composite score
    score = RMSE_WEIGHT * rmse + PINBALL_WEIGHT * pinball
    metrics["score"] = score
    
    return metrics


def train_model(hparams: Dict, pollutant: str, region_weights: Dict[str, float]) -> Dict:
    """Train a single model with given hyperparameters."""
    
    seq_len = SEQ_LEN_BY_POLLUTANT[pollutant]
    input_dim = 15  # From preprocess_lstm.py FEATURE_COLUMNS
    
    # Load data for validation and test splits
    try:
        val_features, val_targets, val_metadata, pollutant_indices = load_split_data("val")
        test_features, test_targets, test_metadata, _ = load_split_data("test")
    except Exception as e:
        return {"error": f"Data loading failed: {e}"}
    
    # Create datasets using loaded arrays
    val_dataset = SequenceDataset(
        val_features,
        val_targets,
        val_metadata, pollutant_indices,
        pollutant, 168, seq_len, region_weights
    )
    
    test_dataset = SequenceDataset(
        test_features,
        test_targets,
        test_metadata, pollutant_indices,
        pollutant, 168, seq_len, region_weights
    )
    
    # Create dataloaders
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
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
        "val_pinball": [],
        "val_coverage": [],
    }
    
    for epoch in range(MAX_EPOCHS):
        # Train
        train_loss, train_metrics = train_epoch(
            model, 
            DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=N_WORKERS),
            optimizer,
            TARGET_QUANTILES,
            device
        )
        
        # Validate
        val_metrics = validate(model, val_loader, TARGET_QUANTILES, device)
        
        history["train_loss"].append(train_loss)
        history["val_score"].append(val_metrics["score"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_pinball"].append(val_metrics["pinball"])
        history["val_coverage"].append(val_metrics["coverage"])
        
        # Early stopping
        if val_metrics["score"] < best_val_score:
            best_val_score = val_metrics["score"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    test_metrics = validate(model, test_loader, TARGET_QUANTILES, device)
    test_metrics["history"] = history
    test_metrics["best_epoch"] = len(history["val_score"])
    
    return test_metrics


# ============================================================================
# OPTUNA INTERFACE
# ============================================================================


def objective(trial, pollutant: str, region_weights: Dict[str, float]) -> float:
    """Optuna objective function."""
    
    # Suggest hyperparameters
    hparams = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", SEARCH_SPACE["hidden_dim"]),
        "dropout": trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"]),
        "head_dropout": trial.suggest_categorical("head_dropout", SEARCH_SPACE["head_dropout"]),
        "lr": trial.suggest_float("lr", SEARCH_SPACE["lr"][0], SEARCH_SPACE["lr"][-1], log=True),
        "weight_decay": trial.suggest_float("weight_decay", 
                                                   SEARCH_SPACE["weight_decay"][0],
                                                   SEARCH_SPACE["weight_decay"][-1], log=True),
    }
    
    try:
        # Train and evaluate
        metrics = train_model(hparams, pollutant, region_weights)
        
        if "error" in metrics:
            return float("inf")
        
        # Track user attrs for Pareto analysis
        trial.set_user_attr("rmse", metrics.get("rmse", 0.0))
        trial.set_user_attr("coverage", metrics.get("coverage", 0.0))
        trial.set_user_attr("pinball", metrics.get("pinball", 0.0))
        
        # Return composite score for minimization
        return metrics["score"]
    
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run Optuna hyperparameter optimization."""
    
    print("=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION FOR h168")
    print("=" * 80)
    print(f"Search space: {json.dumps(SEARCH_SPACE, indent=2)}")
    print(f"Fixed: num_layers={NUM_LAYERS}, num_heads={NUM_HEADS}")
    print(f"Objective: {RMSE_WEIGHT}×RMSE + {PINBALL_WEIGHT}×PinballLoss")
    print(f"Pareto gate: Coverage < {COVERAGE_PENALTY_THRESHOLD} → penalty")
    print(f"Target coverage: 0.90 for p05-p95")
    print(f"Max trials: {N_TRIALS}")
    print("=" * 80)
    
    # Region weights (from Phase 1 analysis - mild weighting)
    # Calculated as: weight_r = (1/4) / fraction_r
    DEFAULT_REGION_WEIGHTS = {
        "AIIMS": 0.979,
        "IGKV": 0.994,
        "Bhatagaon": 1.009,
        "SILTARA": 1.020,
    }
    
    # Try to load from metadata.json (where preprocess saves them)
    metadata_path = ROOT / "data" / "preprocessed_lstm_v1" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        region_weights = metadata.get("region_weights", DEFAULT_REGION_WEIGHTS)
    else:
        region_weights = DEFAULT_REGION_WEIGHTS
    
    print(f"Region weights: {region_weights}")
    print()
    
    # Pollutants to tune
    pollutants = POLLUTANTS
    
    # Results storage
    all_results = {}
    
    # Try to import optuna
    try:
        import optuna
        USE_OPTUNA = True
        print("✓ Optuna detected - using Optuna tuner")
    except ImportError:
        USE_OPTUNA = False
        print("✗ Optuna not detected - using grid search fallback")
    
    for pollutant in pollutants:
        print(f"\n{'=' * 80}")
        print(f"POLLLUTANT: {pollutant.upper()}")
        print(f"{'=' * 80}")
        
        pollutant_results = {
            "pollutant": pollutant,
            "seq_len": SEQ_LEN_BY_POLLUTANT[pollutant],
            "trials": [],
            "best_hparams": None,
            "best_metrics": None,
        }
        
        if USE_OPTUNA:
            # Optuna tuning
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
                study_name=f"h168_{pollutant}"
            )
            
            study.optimize(
                lambda trial: objective(trial, pollutant, region_weights),
                n_trials=N_TRIALS,
                show_progress_bar=True
            )
            
            best_trial = study.best_trial
            best_hparams = {
                "hidden_dim": best_trial.params["hidden_dim"],
                "dropout": best_trial.params["dropout"],
                "head_dropout": best_trial.params["head_dropout"],
                "lr": best_trial.params["lr"],
                "weight_decay": best_trial.params["weight_decay"],
            }
            
            # Retrain with best hparams for full metrics
            print(f"\nRetraining with best hparams: {best_hparams}")
            best_metrics = train_model(best_hparams, pollutant, region_weights)
            
            pollutant_results["best_hparams"] = best_hparams
            pollutant_results["best_metrics"] = best_metrics
            pollutant_results["n_trials"] = len(study.trials)
            pollutant_results["best_score"] = best_trial.value
            
        else:
            # Grid search fallback
            print("Running grid search...")
            from itertools import product
            
            param_grid = list(product(
                SEARCH_SPACE["hidden_dim"],
                SEARCH_SPACE["dropout"],
                SEARCH_SPACE["head_dropout"],
                SEARCH_SPACE["lr"],
                SEARCH_SPACE["weight_decay"],
            ))
            
            best_score = float("inf")
            best_hparams = None
            
            for i, (hd, drop, hd_drop, lr, wd) in enumerate(param_grid):
                print(f"  Grid trial {i+1}/{len(param_grid)}: hidden_dim={hd}, dropout={drop:.3f}")
                
                hparams = {
                    "hidden_dim": hd,
                    "dropout": drop,
                    "head_dropout": hd_drop,
                    "lr": lr,
                    "weight_decay": wd,
                }
                
                metrics = train_model(hparams, pollutant, region_weights)
                
                if metrics["score"] < best_score:
                    best_score = metrics["score"]
                    best_hparams = hparams.copy()
                
                pollutant_results["trials"].append({
                    "hparams": hparams,
                    "score": metrics["score"],
                    "rmse": metrics["rmse"],
                    "coverage": metrics["coverage"],
                })
            
            # Retrain with best hparams
            print(f"\nRetraining with best hparams: {best_hparams}")
            best_metrics = train_model(best_hparams, pollutant, region_weights)
            
            pollutant_results["best_hparams"] = best_hparams
            pollutant_results["best_metrics"] = best_metrics
            pollutant_results["n_trials"] = len(param_grid)
            pollutant_results["best_score"] = best_score
        
        all_results[pollutant] = pollutant_results
        
        # Print summary
        print(f"\n✓ {pollutant.upper()} Complete")
        print(f"  Best score: {pollutant_results['best_score']:.4f}")
        print(f"  Best hparams: {pollutant_results['best_hparams']}")
        if "best_metrics" in pollutant_results:
            bm = pollutant_results["best_metrics"]
            # Safely format numeric metrics; use N/A when missing
            rmse_val = bm.get('rmse')
            cov_val = bm.get('coverage')
            print(f"  Test RMSE: {rmse_val:.4f}" if isinstance(rmse_val, (int,float)) else "  Test RMSE: N/A")
            print(f"  Test Coverage: {cov_val:.4f}" if isinstance(cov_val, (int,float)) else "  Test Coverage: N/A")
    
    # Save results
    output_path = OUTPUT_DIR / OUTPUT_FILE
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS SAVED TO: {output_path}")
    print(f"{'=' * 80}")
    
    return all_results


def merge_results():
    """Merge results from multiple GPU runs."""
    import argparse
    parser = argparse.ArgumentParser(description="Merge Optuna results")
    parser.add_argument("--files", type=str, nargs="+", help="Files to merge")
    parser.add_argument("--output", type=str, default="optuna_h168_best_configs_merged.json", help="Output file")
    args = parser.parse_args()
    
    merged = {}
    for filepath in args.files:
        p = Path(filepath)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            merged.update(data)
            print(f"Merged {len(data)} entries from {p.name}")
    
    out_path = OUTPUT_DIR / args.output
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged total: {len(merged)} pollutants -> {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_results()
    else:
        main()
    main()
