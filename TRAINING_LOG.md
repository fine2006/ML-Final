# Training Log

## Run Context
- Date (latest checked): 2026-04-16
- Environment: `uv run python ...` (project-managed environment)
- Phase: Phase 5 (LSTM + XGB training)
- Notes: Current completed **LSTM** artifacts are for `pm25` only; **XGB** now has all 4 pollutants and 5 horizons.

## LSTM Training (`scripts/train_lstm.py`)
- Log file: `logs/train_lstm/train_lstm_20260416_143717.log`
- Model artifact: `models/lstm_quantile_pm25.pt`
- Summary artifact: `models/lstm_training_summary.json`

### Configuration snapshot
- Pollutant: `pm25`
- Horizons trained: `h1`, `h12`
- Quantiles: `p05`, `p50`, `p95`, `p99`
- Device: `cpu`
- Batch size: `32`
- Epochs configured: `100`
- Early stopping patience: `10`
- Best epoch: `34`
- Best validation loss: `2.2895`

### Sample counts (sequence dataset)
- Train: `256`
- Validation: `64`
- Test: `64`

### Test metrics (from summary)
- Overall RMSE (p50): `13.4606`
- Overall CRPS (approx): `14.9629`
- h1: RMSE `11.5315`, CRPS `15.5227`, coverage p05-p95 `0.8281`
- h12: RMSE `15.1459`, CRPS `14.4031`, coverage p05-p95 `0.7812`

### Convergence notes
- Validation loss decreases strongly in early epochs, then plateaus.
- Early stopping triggers at epoch `44` after best epoch `34`.
- Test prediction artifact is now saved per pollutant:
  - `models/lstm_predictions_pm25.npz`
  - Arrays include: predictions, targets, regions, horizons, quantiles.

## XGB Training (`scripts/train_xgb.py`)
- Log file (latest full run): `logs/train_xgb/train_xgb_20260416_174113.log`
- Model artifacts:
  - `models/xgb_quantile_{pollutant}_h{horizon}_q{quantile}.json`
  - Full matrix present for `pm25`, `pm10`, `no2`, `o3` across `h1`, `h12`, `h24`, `h168`, `h672` and quantiles `q05`, `q50`, `q95`, `q99`.
- Summary artifact: `models/xgb_training_summary.json`

### Configuration snapshot
- Pollutants: `pm25`, `pm10`, `no2`, `o3`
- Horizons trained: `h1`, `h12`, `h24`, `h168`, `h672`
- Quantiles: `p05`, `p50`, `p95`, `p99`
- Estimators: `500`
- Learning rate: `0.1`
- Max depth: `7`
- Subsample / Colsample: `0.8 / 0.8`
- Device backend: configurable via `--device` (`auto`, `cuda`, `cpu`), with CUDA smoke-test auto-detection

### Test metrics (latest full run, from summary)
- `pm25`: h1 RMSE `6.7111`, h12 `11.6442`, h24 `12.3045`, h168 `16.6285`, h672 `21.2194`
- `pm10`: h1 RMSE `16.1433`, h12 `29.1070`, h24 `29.5848`, h168 `39.4748`, h672 `47.2998`
- `no2`: h1 RMSE `10.5456`, h12 `11.5205`, h24 `11.5894`, h168 `13.2264`, h672 `17.9088`
- `o3`: h1 RMSE `13.6416`, h12 `15.5105`, h24 `15.3825`, h168 `17.2571`, h672 `17.0852`

## Operational Notes
- The orchestrator (`main.py`) now launches pipeline scripts through `uv`:
  - `uv run python scripts/preprocess_lstm.py`
  - `uv run python scripts/preprocess_xgb.py`
  - `uv run python scripts/train_lstm.py ...`
  - `uv run python scripts/train_xgb.py ...`
  - `uv run python scripts/evaluate.py ...`

## LSTM Pipeline Improvements (2026-04-17)
- Added short-horizon context floor in `scripts/train_lstm.py`:
  - attention window now uses `min(max(2*horizon, min_attn_window), seq_len)`
  - default `min_attn_window=24` via CLI `--min-attn-window`
- Added horizon-loss balancing in `scripts/train_lstm.py`:
  - CLI `--horizon-weighting` (`equal` or `inverse_sqrt`, default `inverse_sqrt`)
  - CLI `--horizon-loss-weights` for manual overrides
- Added per-horizon epoch logging (`val_by_h=[h1=..., h12=..., ...]`) to expose task-level drift during training.

## Pending Phase 5 Work
- Train remaining pollutants: `pm10`, `no2`, `o3`.
- Train/evaluate longer horizons for LSTM (`h24`, `h168`, `h672`) and align XGB horizon coverage.
- Re-run with production-scale sample counts (current LSTM run uses very small sequence sample counts).

Update:
- XGB pollutant/horizon coverage is now complete for this pipeline version.
- Remaining Phase 5 gap is full-coverage LSTM training on a CUDA-capable device.
