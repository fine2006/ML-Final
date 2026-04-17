# Training Log

## Run Context
- Date (latest checked): 2026-04-17
- Environment: `uv run python ...`
- Active scope: all pollutants (`pm25`, `pm10`, `no2`, `o3`) and horizons `h1`, `h24`, `h168`
- Training mode: separated-horizon LSTM models + quantile XGB baseline

## Artifact Naming (Current)

### LSTM
- Checkpoints: `models/lstm_quantile_{pollutant}_h{horizon}.pt`
- Prediction bundles: `models/lstm_predictions_{pollutant}_h{horizon}.npz`
- Summary: `models/lstm_training_summary.json`

### XGB
- Quantile models: `models/xgb_quantile_{pollutant}_h{horizon}_q{quantile}.json`
- Summary: `models/xgb_training_summary.json`

## Active Hyperparameter Pattern

### LSTM (`scripts/train_lstm.py`)
- Pollutants: `pm25,pm10,no2,o3`
- Horizons: `1,24,168`
- Default seq map: `1:168,24:336,168:720`
- Typical stable settings on T4:
  - `batch-size`: `96-128`
  - `lr`: `1e-4` (h1), `8e-5` (h24), `6e-5` (h168)
  - `epochs`: `120-170`
  - `patience`: `16-28`

### XGB (`scripts/train_xgb.py`)
- Pollutants: `pm25,pm10,no2,o3`
- Horizons: `1,24,168`
- Quantiles: `q05,q50,q95,q99`
- Typical settings:
  - `n_estimators=500`
  - `learning_rate=0.1`
  - `max_depth=7`
  - `subsample=0.8`
  - `colsample_bytree=0.8`

## Data Quality Context Included in Pipeline
- Phase 1 now investigates all pollutants and writes:
  - `data/raw/phase1_investigation_results.json`
    - `all_pollutants_outlier_analysis`
    - `all_pollutants_loss`
- Preprocessing metadata now carries that context into:
  - `data/preprocessed_lstm_v1/metadata.json`
  - `data/preprocessed_xgb_v1/metadata.json`
- Evaluation summary further embeds context in:
  - `models/evaluation_summary.json` (`phase1_data_quality_context`)

## Recent Quality Snapshot (From Prior Runs)
- Strong LSTM short-horizon examples observed:
  - `pm25 h1` RMSE around `8.2`
  - `no2 h1` RMSE around `11.9-12.2`
- Mid-horizon (`h24`) quality is model/pollutant dependent and needs continued tuning.
- `h168` is now the long-horizon representative for runtime/quality balance.

## Operational Notes
- `--horizon-weighting` arguments are ignored in separated-horizon mode by design.
- Evaluation supports validation-based quantile calibration for both families via:
  - `--calibrate-quantiles`
- 95% CI (`p05-p95`) is the primary uncertainty KPI; `p99` remains diagnostic.
