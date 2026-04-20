# ARCHITECTURE.md - Hierarchical Quantile LSTM (Current Scope)

## ⚠️ CRITICAL: GIT INSTRUCTIONS
**DO NOT RUN ANY GIT COMMANDS** (git add, git commit, git push, or any git modification commands).
See DECISIONS.md for full git warning. Work directly on files; user handles git operations.

## Overview
This document describes the active model architecture and evaluation contract implemented in the current codebase.

Current active scope:
- Pollutants: `pm25`, `pm10`, `no2`, `o3`
- Horizons: `h1`, `h24`, `h168`
- Quantiles: `p05`, `p50`, `p95`, `p99`

The pipeline trains one LSTM checkpoint per `(pollutant, horizon)` and compares against XGB quantile baselines with fair row intersection.

---

## 1. Problem Formulation

### 1.1 Quantile Forecasting Target
For each `(pollutant, horizon)` pair, predict a calibrated distribution summary using four quantiles:

- `q=0.05`
- `q=0.50`
- `q=0.95`
- `q=0.99`

Output shape per model inference call (separated-horizon mode):
- `(batch, 1, 4)`

Why quantiles (not point-only):
- support interval risk decisions,
- support calibration diagnostics (coverage + PIT),
- preserve tail behavior that RMSE alone cannot describe.

### 1.2 Current Training Topology
The code now uses **horizon-separated training**:

- one checkpoint per `(pollutant, horizon)` in `scripts/train_lstm.py`,
- shared architecture pattern across horizons,
- horizon-specific sequence length via `--seq-len-map`.

Artifacts:
- `models/lstm_quantile_{pollutant}_h{horizon}.pt`
- `models/lstm_predictions_{pollutant}_h{horizon}.npz`

---

## 2. LSTM Architecture (Implemented)

### 2.1 Inputs
Input tensor:
- shape `(batch_size, seq_len, n_features)`

Current sequence-length defaults:
- `h1 -> 168`
- `h24 -> 336`
- `h168 -> 720`

Features follow the preprocessed LSTM metadata contract and include multi-pollutant context plus weather/time/region channels.

### 2.2 Backbone
Backbone in `train_lstm.py`:
- BiLSTM, `num_layers=2` (default)
- hidden dim per direction configurable (`hidden_dim`, default 128)
- dropout configurable (`dropout`, default 0.3)

### 2.3 Attention + Quantile Head
Per model (single horizon active in separated mode):
- multi-head attention over BiLSTM outputs,
- MLP quantile head projecting to four quantile values.

Head output per sample:
- `[p05, p50, p95, p99]`

Monotonicity at evaluation/inference:
- quantiles are post-ordered via cumulative max where required.

### 2.4 Delta Target Option for Gas Rescue
`train_lstm.py` supports:
- `--target-mode level`
- `--target-mode delta_ma3`

For `delta_ma3`:
- training target is `y(t+h) - MA_k(y_t)` (default `k=3`),
- evaluation and prediction reconstruct level quantiles before scoring/output.

Checkpoint metadata stores:
- `target_mode`
- `delta_baseline_window`
- `training_target_columns`

---

## 3. Loss, Optimization, and Fairness

### 3.1 Quantile Loss
Primary objective:
- multi-quantile pinball loss over `q in {0.05, 0.50, 0.95, 0.99}`.

### 3.2 Region Weighting
Region weighting is applied in training loss:
- `weight_r = (1/4) / fraction_r`

With canonical current data, weights are mild and near uniform (example from metadata):
- AIIMS ~0.979
- IGKV ~0.994
- Bhatagaon ~1.009
- SILTARA ~1.020

### 3.3 Optimization Defaults
Implemented knobs include:
- Adam optimizer
- LR scheduler (`ReduceLROnPlateau`)
- gradient clipping (`--max-grad-norm`)
- early stopping (`--patience`)

---

## 4. Leakage and Split Contract

### 4.1 Time Safety
Required invariants:
- features are past-only relative to target anchor,
- train/val/test remain time-ordered,
- no test data used in training/calibration fitting.

### 4.2 Split Scope
Active evaluation/training scope in code:
- horizons `1,24,168` only.

---

## 5. XGB Baseline Contract

### 5.1 XGB Model Family
Quantile XGB models are trained per:
- pollutant,
- horizon in `{1,24,168}`,
- quantile in `{05,50,95,99}`.

Active total per pollutant: `3 x 4 = 12` models.

### 5.2 Outlier Caps (XGB Pipeline)
Fixed pollutant-specific caps in `preprocess_xgb.py`:
- `pm25 <= 300`
- `pm10 <= 600`
- `no2 <= 250`
- `o3 <= 150`

LSTM and XGB intentionally use different outlier philosophies.

---

## 6. Evaluation and Operational Gating

### 6.1 Core Metrics
`evaluate.py` reports, per pollutant/horizon:
- RMSE, MAE, R2
- CRPS (approx)
- coverage for `p05-p95`
- PIT KS statistic and p-value
- region fairness ratio

### 6.2 Fair Intersection Benchmark
When `--fair-intersection` is enabled:
- LSTM and XGB are compared on the exact same `(region,timestamp)` rows.

Additional fair-table fields include:
- `mean_concentration`
- `lstm_rmse_over_mean`, `xgb_rmse_over_mean`
- `persistence_rmse_raw_yt`
- `climatology_rmse_month_hour`
- `beats_persistence`, `beats_climatology`

### 6.3 Tiered Operational Gates
Current gate definitions:

- PM (`pm25`,`pm10`):
  - `rmse_over_mean < 0.5`
  - `r2 > 0.3`

- Gas (`no2`,`o3`):
  - `rmse_over_mean < 0.8`
  - must beat raw persistence baseline

Gate outputs are written to:
- `models/fair_benchmark_summary.json` (`operational_gates`)
- `models/evaluation_summary.json` (`operational_gate_summary`)

---

## 7. Reference Files
- `scripts/train_lstm.py`
- `scripts/train_xgb.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `NOTEBOOK_RUN_CLEAN.py`
- `DECISIONS.md`

This architecture doc is intentionally aligned to current implemented behavior and active experimental scope.
