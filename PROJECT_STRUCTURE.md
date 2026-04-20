# PROJECT_STRUCTURE.md - Current Pipeline Structure

## ⚠️ CRITICAL: GIT INSTRUCTIONS
**DO NOT RUN ANY GIT COMMANDS** (git add, git commit, git push, etc.).

Work directly on files. The user handles git operations.

---

## Overview

This document reflects the **actual implemented pipeline** and file layout.

Current active pipeline scripts are:
- `scripts/data_investigation.py`
- `scripts/preprocess_common.py`
- `scripts/preprocess_lstm.py`
- `scripts/preprocess_xgb.py`
- `scripts/train_lstm.py`
- `scripts/train_xgb.py`
- `scripts/evaluate.py`
- `scripts/predict.py`

Orchestration entrypoint:
- `main.py` (uses `uv run python ...` internally)

---

## Current Directory Tree

```
ML-Final/
├── AGENTS.md
├── ARCHITECTURE.md
├── PREPROCESSING_STRATEGY.md
├── DATA_INVESTIGATION.md
├── DECISIONS.md
├── REPRODUCIBILITY.md
├── VISUALIZATIONS.md
├── IMPLEMENTATION_CONTRACT.md
├── GETTING_STARTED.md
├── DATA_LOADING.md
├── SCRIPT_TEMPLATES.md
├── PROJECT_STRUCTURE.md
├── TRAINING_LOG.md
├── EVALUATION_RESULTS.md
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── main.py
│
├── scripts/
│   ├── data_investigation.py
│   ├── preprocess_common.py
│   ├── preprocess_lstm.py
│   ├── preprocess_xgb.py
│   ├── train_lstm.py
│   ├── train_xgb.py
│   ├── evaluate.py
│   └── predict.py
│
├── data/
│   ├── raw/
│   │   ├── pollution_data_raw.csv
│   │   ├── pollution_data_hourly_unique.csv
│   │   └── phase1_investigation_results.json
│   │
│   ├── preprocessed_lstm_v1/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── metadata.json
│   │
│   └── preprocessed_xgb_v1/
│       ├── X_train.csv
│       ├── X_val.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_val.csv
│       ├── y_test.csv
│       └── metadata.json
│
├── models/
│   ├── lstm_quantile_{pollutant}_h{horizon}.pt
│   ├── lstm_predictions_{pollutant}_h{horizon}.npz
│   ├── xgb_quantile_{pollutant}_h{horizon}_q{quantile}.json
│   ├── lstm_training_summary.json
│   ├── xgb_training_summary.json
│   ├── evaluation_summary.json
│   ├── fair_benchmark_summary.json
│   ├── experiments/
│   └── prediction_output.json
│
├── logs/
│   ├── data_investigation/
│   ├── preprocess_lstm/
│   ├── preprocess_xgb/
│   ├── train_lstm/
│   ├── train_xgb/
│   └── evaluate/
│
├── visualizations/
│   ├── phase_1_data_investigation/
│   └── phase_6_evaluation/
│
└── Pollution Data Raipur/
```

---

## Script Responsibilities

| Script | Phase | Input | Output |
|---|---|---|---|
| `scripts/data_investigation.py` | 1 | `Pollution Data Raipur/` | `data/raw/*.csv`, `data/raw/phase1_investigation_results.json`, `visualizations/phase_1_data_investigation/*` |
| `scripts/preprocess_common.py` | 3 | `data/raw/*` | Shared canonical merge + sanitization utilities |
| `scripts/preprocess_lstm.py` | 3 | `data/raw/pollution_data_hourly_unique.csv` + Phase 1 JSON | `data/preprocessed_lstm_v1/{train,val,test}.csv`, metadata |
| `scripts/preprocess_xgb.py` | 3 | `data/raw/pollution_data_hourly_unique.csv` + Phase 1 JSON | `data/preprocessed_xgb_v1/X_*.csv`, `y_*.csv`, metadata |
| `scripts/train_lstm.py` | 5 | `data/preprocessed_lstm_v1/*` | `models/lstm_quantile_*.pt`, `models/lstm_predictions_*.npz`, `models/lstm_training_summary.json` |
| `scripts/train_xgb.py` | 5 | `data/preprocessed_xgb_v1/*` | `models/xgb_quantile_*_h*_q*.json`, `models/xgb_training_summary.json` |
| `scripts/evaluate.py` | 6 | preprocessed data + model artifacts | `models/evaluation_summary.json`, `visualizations/phase_6_evaluation/*` |
| `scripts/predict.py` | inference | preprocessed tables + LSTM models | JSON forecast for one region + one horizon + all pollutants |

---

## Runtime Commands

Use `uv` to run scripts in the managed environment:

```bash
uv run python scripts/data_investigation.py
uv run python scripts/preprocess_lstm.py
uv run python scripts/preprocess_xgb.py
uv run python scripts/train_lstm.py --pollutants pm25,pm10,no2,o3 --horizons 1,24,168
uv run python scripts/train_xgb.py --pollutants pm25,pm10,no2,o3 --horizons 1,24,168 --device cuda
uv run python scripts/evaluate.py --pollutants pm25,pm10,no2,o3 --horizons 1,24,168
uv run python scripts/evaluate.py --pollutants pm25,pm10,no2,o3 --horizons 1,24,168 --fair-intersection
uv run python scripts/predict.py --region AIIMS --horizon 24 --pollutants pm25,pm10,no2,o3
```

Orchestrated run:

```bash
uv run python main.py
```

---

## Notes on Removed Legacy Files

Legacy scripts from the previous implementation (e.g., `*_v2.py`, stacking/baseline extras) are intentionally removed from the active pipeline and should not be referenced for current phases.

---

## Next Steps

1. Run Phase 5 training to completion for active horizons (`h1`,`h24`,`h168`) across all pollutants (LSTM + XGB).
2. Run Phase 6 evaluation with fair intersection and quantile calibration where required.
3. Snapshot key outputs under `models/experiments/` for run-to-run comparability.
