# Evaluation Results

## Run Context
- Date (latest checked): 2026-04-17
- Environment: `uv run python ...`
- Evaluation script: `scripts/evaluate.py`
- Primary output: `models/evaluation_summary.json`
- Active scope: all pollutants (`pm25, pm10, no2, o3`) on horizons `h1, h24, h168`

## Artifacts Produced
- Summary JSON: `models/evaluation_summary.json`
- Fair benchmark JSON: `models/fair_benchmark_summary.json` (enabled with `--fair-intersection`)
- Plots: `visualizations/phase_6_evaluation/`
  - `lstm_vs_xgb_rmse_by_horizon.png`
  - `lstm_vs_xgb_crps_by_horizon.png`
  - `quantile_calibration_curves.png`
  - `pit_histogram_uniformity.png`
  - `coverage_analysis_by_quantile.png`
  - `per_region_fairness_metrics.png`
  - `predictions_vs_actual_by_horizon.png`
  - `final_comparison_summary_table.png`

## What Changed in This Phase
- The evaluation pipeline is now all-pollutant aware end-to-end.
- Quantile calibration can be applied to **both** model families with:
  - `--calibrate-quantiles`
- Benchmarking now prioritizes 95% interval quality (`p05-p95`) while keeping `p99` diagnostics.
- Evaluation summary now includes richer context blocks:
  - `split_representativeness`
  - `phase1_data_quality_context`
  - `calibration_benchmark`

## 95% CI-Centric Evaluation Protocol
- Run command (all pollutants, all active horizons):

```bash
MPLBACKEND=Agg uv run python scripts/evaluate.py \
  --pollutants pm25,pm10,no2,o3 \
  --horizons 1,24,168 \
  --device auto \
  --fair-intersection \
  --calibrate-quantiles
```

- Core uncertainty metrics reported per pollutant/horizon:
  - `coverage_p05_p95`
  - `tail_below_p05`
  - `tail_above_p95`
  - `interval_width_p05_p95`
  - `quantile_crossing_rate`
  - PIT KS / p-value

## Data Quality Context (All Pollutants)

Phase 1 now contributes all-pollutant loss and outlier-tail diagnostics into
evaluation outputs. This gives model differences clearer causal context.

Latest Phase 1 totals (canonical hourly baseline `125,017` rows):
- `pm25`: sequence-ready `122,487` (`loss_impossible=4,469`, `loss_outliers=156`)
- `pm10`: sequence-ready `121,539` (`loss_impossible=5,490`, `loss_outliers=0`)
- `no2`: sequence-ready `122,706` (`loss_impossible=3,853`, `loss_outliers=0`)
- `o3`: sequence-ready `122,646` (`loss_impossible=3,568`, `loss_outliers=0`)

Interpretation:
- Pollutant-specific quality profiles differ materially, especially in impossible-value frequency.
- PM2.5 is currently the only pollutant with a fixed XGB clip policy (`<=300`) in this pipeline.
- These differences are now directly embedded in `evaluation_summary.json` for downstream interpretation.

## Representativeness Checks
- `split_representativeness` now reports:
  - region mix shifts (train vs val/test)
  - pollutant-level distribution shifts for LSTM rows
  - KS checks on XGB target distributions (`train-vs-val`, `train-vs-test`, `val-vs-test`)
- This helps verify val/test are reasonably representative before trusting benchmark deltas.

## Calibration Benchmark Table
- `calibration_benchmark` now compares LSTM and XGB per pollutant/horizon on:
  - `cov95` absolute error from `0.90`
  - `p05` / `p95` tail absolute errors from `0.05`
  - `p99` tail absolute error from `0.01`
  - interval width and crossing rate

Recommended acceptance thresholds for production-style confidence intervals:
- `|coverage_p05_p95 - 0.90| <= 0.03`
- `|tail_below_p05 - 0.05| <= 0.02`
- `|tail_above_p95 - 0.05| <= 0.02`
- `quantile_crossing_rate <= 0.005`

## Notes
- Metrics vary by run depending on which model artifacts are present in `models/`.
- For final reporting, always regenerate `evaluation_summary.json` after the latest full training/import.
