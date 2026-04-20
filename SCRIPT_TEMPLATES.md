# SCRIPT_TEMPLATES.md - Boilerplate Structure for All Pipeline Scripts

**CRITICAL**: This is documentation for FRESH IMPLEMENTATION. Do NOT run any git commands (git add, git commit, git push, etc.). Work directly on files.

**Source of Truth**: All decisions in this file align with DECISIONS.md section 4.X. If code needs to deviate, update DECISIONS.md FIRST.

---

## Overview

This document provides boilerplate templates for all 6 pipeline scripts. Every script must follow the same structure for consistency, reproducibility, and maintainability:

1. **Imports & Setup** (fixed random seeds, logging)
2. **Configuration** (hyperparameters from config files)
3. **Main Logic** (implementation per phase)
4. **Outputs** (versioned data/models/logs)
5. **Error Handling** (fail gracefully with clear messages)

---

## Critical: Fixed Random Seeds

**Every script must set RANDOM_SEED = 42 at the top**

```python
import random
import numpy as np
import torch

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
```

**Why**: Reproducibility requirement. All analyses must produce identical results on re-runs. See REPRODUCIBILITY.md section 1.

---

## Universal Template: All Scripts

```python
"""
Script Name: [phase_name.py]
Purpose: [Phase X: Description]
Inputs: [data source]
Outputs: [data/model/log files]
Configuration: [config file path]

CRITICAL: 
- NO future data leakage (weather t-24 to t-1 only)
- Region weighting applied
- Time-based splits (no shuffling)
- Fixed random seed for reproducibility
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import json

import pandas as pd
import numpy as np
import torch
import yaml

# ==================== RANDOM SEED ====================
# MUST be first after imports. Reproducibility.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==================== SETUP ====================

# Directories (create if not exist)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
VIZ_DIR = PROJECT_ROOT / "visualizations"

for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, VIZ_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== LOGGING ====================

def setup_logging(script_name: str) -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Args:
        script_name: Name of script (e.g., "data_investigation")
    
    Returns:
        Logger object
    """
    log_dir = LOGS_DIR / script_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ==================== CONFIGURATION ====================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix == ".json":
        with open(config_path) as f:
            config = json.load(f)
    elif config_path.suffix in [".yaml", ".yml"]:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config

# ==================== MAIN FUNCTION ====================

def main():
    """
    Main entry point for script.
    
    Procedure:
    1. Setup logging
    2. Load configuration
    3. Execute phase logic
    4. Save outputs
    5. Log summary
    """
    
    script_name = Path(__file__).stem
    logger = setup_logging(script_name)
    
    logger.info(f"{'='*60}")
    logger.info(f"Starting {script_name.upper()}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"{'='*60}")
    
    try:
        # Load configuration (if needed for this phase)
        config = load_config(f"{SCRIPTS_DIR.parent}/configs/[config_name].json")
        logger.info(f"Config loaded: {json.dumps(config, indent=2)}")
        
        # ==================== PHASE LOGIC ====================
        # Insert main logic here
        # See specific template for each phase below
        
        logger.info(f"{'='*60}")
        logger.info(f"Completed {script_name.upper()} successfully")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Phase-Specific Templates

### Phase 1: Data Investigation (`scripts/data_investigation.py`)

```python
"""
Script: data_investigation.py
Purpose: Phase 1 - Analyze extreme data quality issues
Inputs: Raw Excel files from Pollution Data Raipur/
Outputs: 
  - data/raw/pollution_data_raw.csv (loaded raw data)
  - DATA_INVESTIGATION.md (filled with findings)
  - logs/data_investigation/data_investigation_*.log
Configuration: None (exploratory)

CRITICAL:
- Document all findings in DATA_INVESTIGATION.md sections 1-3
- Classify Bhatagaon Sept 2025 spike: real event or sensor error?
- Quantify canonical pipeline data loss: where does attrition come from?
- Calculate region weights for Phase 5 training
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== SETUP (See Universal Template) ====================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
VIZ_DIR = PROJECT_ROOT / "visualizations"

def setup_logging(script_name: str) -> logging.Logger:
    """[Use function from universal template]"""
    # [Copy from universal template]
    pass

# ==================== PHASE 1 LOGIC ====================

def load_raw_data() -> pd.DataFrame:
    """
    Load all raw pollution data from Excel files.
    
    Returns:
        DataFrame with columns: [region, pollutant, timestamp, value]
    """
    from data_loading import load_all_raw_data
    
    logger.info("Loading raw Excel data...")
    raw_data = load_all_raw_data()
    logger.info(f"Loaded {len(raw_data):,} records")
    
    return raw_data

def analyze_bhatagaon_spike(data: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze Bhatagaon September 2025 PM2.5 spike >500 µg/m³.
    
    Determine: Real pollution event (gradual ramp-up/decay) or sensor error (sudden spike)?
    
    Returns:
        Classification and analysis results
    
    Procedure (from DATA_INVESTIGATION.md section 1.2):
    1. Filter: region="Bhatagaon", pollutant="PM2.5", timestamp in Sept 2025
    2. Plot: Time series with 24h moving average
    3. Measure: Rise time (hours from baseline to peak), decay time (peak to baseline)
    4. Check: Weather data correlation (rainfall, wind, temp)
    5. Classify:
       - Gradual rise + decay + weather support → KEEP ("real_event")
       - Single-point spike + no weather support → REMOVE ("sensor_error")
       - Uncertain → MANUAL_REVIEW ("uncertain")
    """
    logger.info("Analyzing Bhatagaon September 2025 spike...")
    
    # Implementation details provided in DATA_INVESTIGATION.md section 2
    # Placeholder for now
    
    results = {
        "spike_classification": "pending",  # real_event, sensor_error, uncertain
        "peak_value": None,
        "rise_time_hours": None,
        "decay_time_hours": None,
        "weather_correlation": None,
        "recommendation": None
    }
    
    return results

def analyze_data_loss(data: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """
    Root cause analysis for canonical hourly data attrition.
    
    Returns:
        Breakdown of loss sources
    
    Procedure (from DATA_INVESTIGATION.md section 2.2):
    1. Baseline canonical rows: count_before = 125,017
    2. After sanitization/interpolation/sequence checks: quantify retained rows
    3. Analyze: Which regions/pollutants lost most?
    4. Identify: Missing values, outliers removed, temporal gaps
    5. Quantify: % loss per region, per pollutant
    """
    logger.info("Analyzing data loss sources...")
    
    results = {
        "total_records_before": None,
        "total_records_after": None,
        "total_loss_percent": None,
        "loss_by_region": {},
        "loss_by_pollutant": {},
        "loss_by_reason": {}  # missing, outliers, gaps, etc.
    }
    
    return results

def analyze_region_imbalance(data: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """
    Quantify post-canonical region imbalance and compute weights.
    
    Returns:
        Region distribution and calculated training weights
    
    Procedure (from DATA_INVESTIGATION.md section 3.2):
    1. Count records per region
    2. Calculate fraction: fraction_r = count_r / total
    3. Calculate weights: weight_r = (1/4) / fraction_r
    4. Verify: sum(weights) ≈ 4 (4 regions)
    5. Document: Use these weights in Phase 5 training loss function
    """
    logger.info("Analyzing region imbalance...")
    
    region_counts = data.groupby("region").size()
    total = region_counts.sum()
    fractions = region_counts / total
    weights = (1.0 / 4.0) / fractions
    
    results = {
        "region_counts": region_counts.to_dict(),
        "region_fractions": fractions.to_dict(),
        "region_weights": weights.to_dict(),
        "imbalance_ratio": fractions.max() / fractions.min(),
    }
    
    logger.info(f"Region distribution:")
    for region, count in region_counts.items():
        logger.info(f"  {region}: {count:,} ({fractions[region]:.1%}) → weight {weights[region]:.2f}×")
    
    return results

def save_findings(
    spike_analysis: Dict,
    loss_analysis: Dict,
    imbalance_analysis: Dict,
    logger: logging.Logger
) -> None:
    """
    Save Phase 1 findings to DATA_INVESTIGATION.md and DECISIONS.md.
    
    Updates:
    - DATA_INVESTIGATION.md sections 1-3 (findings)
    - DECISIONS.md section 7 (results)
    """
    logger.info("Saving Phase 1 findings...")
    
    findings = {
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "spike_analysis": spike_analysis,
        "loss_analysis": loss_analysis,
        "imbalance_analysis": imbalance_analysis,
    }
    
    # Save to logs for reference
    findings_file = LOGS_DIR / "data_investigation" / f"findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(findings_file, "w") as f:
        json.dump(findings, f, indent=2)
    
    logger.info(f"Findings saved to {findings_file}")
    
    # TODO: Update DATA_INVESTIGATION.md sections 1-3 with findings
    # TODO: Update DECISIONS.md section 7 with results

def main():
    """Phase 1 entry point."""
    script_name = "data_investigation"
    logger = setup_logging(script_name)
    
    logger.info(f"{'='*60}")
    logger.info(f"Starting PHASE 1: DATA INVESTIGATION")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"{'='*60}")
    
    try:
        # Load raw data
        raw_data = load_raw_data()
        
        # Analyze extreme values
        spike_analysis = analyze_bhatagaon_spike(raw_data, logger)
        logger.info(f"Spike classification: {spike_analysis['spike_classification']}")
        
        # Analyze data loss
        loss_analysis = analyze_data_loss(raw_data, logger)
        logger.info(f"Data loss: {loss_analysis['total_loss_percent']:.1f}%")
        
        # Analyze region imbalance
        imbalance_analysis = analyze_region_imbalance(raw_data, logger)
        logger.info(f"Imbalance ratio: {imbalance_analysis['imbalance_ratio']:.1f}×")
        
        # Save findings
        save_findings(spike_analysis, loss_analysis, imbalance_analysis, logger)
        
        logger.info(f"{'='*60}")
        logger.info(f"Completed PHASE 1 successfully")
        logger.info(f"Next: Fill DATA_INVESTIGATION.md sections 1-3 with findings")
        logger.info(f"Then: Read PREPROCESSING_STRATEGY.md for Phase 3")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Phase 3: Preprocessing LSTM (`scripts/preprocess_lstm.py`)

```python
"""
Script: preprocess_lstm.py
Purpose: Phase 3 - Prepare LSTM-ready data (sequence-focused, keeps outliers)
Inputs: 
  - data/raw/pollution_data_raw.csv (from Phase 1)
Outputs:
  - data/preprocessed_lstm_v1/ (versioned preprocessed data)
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── metadata.json (scaler, feature names, horizon info)

CRITICAL:
- NO future data leakage (weather t-24 to t-1 only)
- Time-based 70/15/15 split (2022-Q1 2024 | Q1-Q2 2024 | Q2 2024-present)
- Keep ALL outliers (PM2.5 >500 OK if pattern real)
- Light imputation (interpolate <6h gaps, break >6h)
- RobustScaler (median/IQR, preserves outlier info)
- Sequence length adaptive: 2×horizon
- Region weighting calculated
- Seed = 42 for reproducibility
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import torch

# ==================== SETUP ====================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"

# ==================== LSTM CONFIG ====================

LSTM_CONFIG = {
    "horizons": [1, 24, 168],  # active scope
    "quantiles": [0.05, 0.50, 0.95, 0.99],
    "seq_len_map": {1: 168, 24: 336, 168: 720},
    "max_seq_len": 8760,
    "imputation_threshold": 6,  # hours (interpolate <6h, break >6h gaps)
    "scaler": "robust",  # RobustScaler (median/IQR)
    "feature_count": 15,  # Minimal features for LSTM
}

# ==================== PHASE 3 LOGIC ====================

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw data from Phase 1."""
    logger.info(f"Loading raw data from {filepath}...")
    data = pd.read_csv(filepath, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(data):,} records")
    return data

def create_sequences(
    data: pd.DataFrame,
    horizon: int,
    seq_len: int,
    quantiles: list,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM.
    
    Input shape: (T, features)
    Output shape: X (N, seq_len, features), y (N, quantiles)
    
    NO FUTURE DATA LEAKAGE:
    - Features use indices [t-seq_len, t-1]
    - Target uses t+horizon (never future)
    """
    logger.info(f"Creating sequences for horizon {horizon}h...")
    # Implementation per PREPROCESSING_STRATEGY.md section 5
    pass

def preprocess_lstm(
    raw_data: pd.DataFrame,
    config: Dict,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Main preprocessing for LSTM.
    
    Steps:
    1. Filter: Keep data 2022-present (all available data)
    2. Outliers: Keep ALL (PM2.5 >500 OK if pattern real)
    3. Imputation: Light (interpolate <6h gaps, break >6h)
    4. Features: 15 minimal (pollutant, weather, time, region)
    5. Scaling: RobustScaler (median/IQR)
    6. Sequences: Create (N, seq_len, features) per horizon
    7. Split: 70/15/15 time-based (2022-Q1 2024 | Q1-Q2 2024 | Q2 2024-present)
    8. Weights: Region weights for loss function
    """
    logger.info("Preprocessing data for LSTM...")
    
    # Placeholder: actual implementation in PREPROCESSING_STRATEGY.md section 5
    
    return {
        "X_train": None,
        "y_train": None,
        "X_val": None,
        "y_val": None,
        "X_test": None,
        "y_test": None,
        "metadata": {
            "feature_names": [],
            "horizons": config["horizons"],
            "quantiles": config["quantiles"],
            "scaler_params": None,
            "region_weights": {},
            "data_retention": None,
        }
    }

def main():
    """Phase 3 LSTM preprocessing entry point."""
    script_name = "preprocess_lstm"
    logger = setup_logging(script_name)
    
    logger.info(f"{'='*60}")
    logger.info(f"Starting PHASE 3: LSTM PREPROCESSING")
    logger.info(f"{'='*60}")
    
    try:
        # Load raw data
        raw_data = load_raw_data(str(DATA_DIR / "raw" / "pollution_data_raw.csv"))
        
        # Preprocess
        result = preprocess_lstm(raw_data, LSTM_CONFIG, logger)
        
        # Save preprocessed data
        output_dir = DATA_DIR / "preprocessed_lstm_v1"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save split tables and metadata
        result["train"].to_csv(output_dir / "train.csv", index=False)
        result["val"].to_csv(output_dir / "val.csv", index=False)
        result["test"].to_csv(output_dir / "test.csv", index=False)
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(result["metadata"], f, indent=2)
        
        logger.info(f"Saved preprocessed data to {output_dir}")
        logger.info(f"{'='*60}")
        logger.info(f"Completed PHASE 3 (LSTM) successfully")
        logger.info(f"Next: Implement preprocess_xgb.py for XGB pipeline")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Phase 5: Training LSTM (`scripts/train_lstm.py`)

```python
"""
Script: train_lstm.py
Purpose: Phase 5 - Train hierarchical quantile regression LSTM
Inputs:
  - data/preprocessed_lstm_v1/ (from Phase 3)
Outputs:
  - models/lstm_quantile_{pollutant}_h{horizon}.pt (trained models)
  - models/lstm_predictions_{pollutant}_h{horizon}.npz (test prediction bundles)
  - logs/train_lstm/ (convergence curves, epoch logs)
  - TRAINING_LOG.md (to create: epoch logs, debug notes)

CRITICAL:
- Region weighting applied in loss (weight_r = (1/4) / fraction_r)
- Multi-quantile pinball loss (slopes: q - I(y<ŷ))
- Early stopping on val CRPS (patience=10)
- Batch stratification: ~8 samples per region per batch
- NO test set used during training/validation
- Seed = 42 for reproducibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class HierarchicalQuantileLSTM(nn.Module):
    """
    Hierarchical multi-horizon quantile regression LSTM.
    
    Architecture (from ARCHITECTURE.md section 2):
    - Input: (batch_size, seq_len, features)
    - Backbone: BiLSTM (2 layers, 128 hidden, dropout 0.3)
    - Head: horizon-specific attention + quantile MLP
    - Output (separated horizon): (batch_size, 1, 4)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        # Implementation per ARCHITECTURE.md section 2
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implementation per ARCHITECTURE.md section 2
        pass

def multi_quantile_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: list,
    region_weights: Dict,
    region_indices: torch.Tensor
) -> torch.Tensor:
    """
    Multi-quantile pinball loss with region weighting.
    
    Loss = sum over quantiles and regions:
        weight_r × mean((q - I(y<ŷ)) × (y - ŷ))
    
    See ARCHITECTURE.md section 3.1-3.2 for full formula.
    """
    loss = 0.0
    for q_idx, q in enumerate(quantiles):
        q_pred = y_pred[:, q_idx]
        q_true = y_true[:, q_idx]
        
        # Pinball loss: (q - I(y<ŷ)) × (y - ŷ)
        residuals = q_true - q_pred
        pinball = (q - (residuals < 0).float()) * residuals
        
        # Apply region weights
        weighted_loss = 0.0
        for region, weight in region_weights.items():
            mask = (region_indices == region).float()
            weighted_loss += weight * (pinball * mask).mean()
        
        loss += weighted_loss / len(quantiles)
    
    return loss

def main():
    """Phase 5 LSTM training entry point."""
    script_name = "train_lstm"
    logger = setup_logging(script_name)
    
    logger.info(f"{'='*60}")
    logger.info(f"Starting PHASE 5: LSTM TRAINING")
    logger.info(f"{'='*60}")
    
    try:
        # Load preprocessed data
        # Train loop with early stopping
        # Save model and logs
        logger.info(f"Completed PHASE 5 (LSTM) successfully")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Output Organization

Every script should save outputs with this pattern:

```
[OUTPUT_TYPE]/
├── [script_name]_v[VERSION]/
│   ├── [DATE_TIME]_[DESCRIPTION]/
│   │   ├── main_output (data, model, results)
│   │   ├── metadata.json (parameters, hyperparameters)
│   │   ├── summary.txt (human-readable summary)
│   │   └── debug/ (detailed logs, intermediate outputs)
```

### Example: Phase 5 Training

```
models/
├── lstm_quantile_pm25_h1.pt
├── lstm_quantile_pm25_h24.pt
├── lstm_quantile_pm25_h168.pt
├── lstm_predictions_pm25_h168.npz
└── ... (same pattern for pm10/no2/o3)

logs/
├── train_lstm/
│   └── train_lstm_20260416_143022.log (epoch logs, debug)

visualizations/
├── phase_5_training/
│   ├── convergence_curves.png (train/val loss)
│   ├── attention_weights_sample.png (debug)
│   └── quantile_calibration.png (predictions vs actuals)
```

---

## Error Handling & Debugging

### Required: Graceful Failures

```python
def main():
    try:
        # Main logic
        pass
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Did you run the previous phase?")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid data or configuration: {e}")
        logger.error("Check DECISIONS.md for expected ranges")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory - reduce batch size in config")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
```

### Required: Data Validation Checks

Before training, always verify:

```python
def validate_data(X_train, y_train, X_val, y_val, X_test, y_test, logger):
    """Validate preprocessed data before training."""
    
    # Check shapes
    assert X_train.shape[0] > 0, "Train set is empty"
    assert X_val.shape[0] > 0, "Val set is empty"
    assert X_test.shape[0] > 0, "Test set is empty"
    
    # Check no future data leakage
    # (specific checks per phase)
    
    # Check time-based split
    # (verify train.max() < val.min() < test.min())
    
    # Check no NaNs in training data
    assert not np.isnan(X_train).any(), "NaNs in X_train"
    assert not np.isnan(y_train).any(), "NaNs in y_train"
    
    logger.info("Data validation passed")
```

---

## Dependencies

All scripts require (in `pyproject.toml`):

```toml
[dependencies]
pandas = ">=2.0.0"
numpy = ">=1.24.0"
torch = ">=2.0.0"
scikit-learn = ">=1.3.0"
xgboost = ">=1.7.0"
matplotlib = ">=3.7.0"
seaborn = ">=0.12.0"
pyyaml = ">=6.0"
```

---

## Next Steps

1. **For implementation**: Copy relevant template for your phase
2. **Fill in phase-specific logic** (see per-phase docs: DATA_INVESTIGATION.md, PREPROCESSING_STRATEGY.md, ARCHITECTURE.md)
3. **Add validation checks** (prevent future data leakage, check splits)
4. **Test locally** before running on full data
5. **Update DECISIONS.md** with any deviations from specifications

See AGENTS.md for phase details and IMPLEMENTATION_CONTRACT.md for binding requirements.
