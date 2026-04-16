# Evaluation Results

## Run Context
- Date (latest checked): 2026-04-16
- Environment: `uv run python ...`
- Evaluation script: `scripts/evaluate.py`
- Primary output: `models/evaluation_summary.json`
- Scope of this checkpoint: `pm25`; shared LSTM/XGB horizons = `h1`, `h12`.

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

## LSTM vs XGB (PM2.5, overlap horizons)

### h1 (t+1h)
- RMSE: LSTM `10.1547` vs XGB `6.7111` (LSTM `-51.31%` worse)
- CRPS: LSTM `15.2498` vs XGB `7.0665` (LSTM `-115.80%` worse)
- Coverage p05-p95: LSTM `84.54%`, XGB `82.31%`

### h12 (t+12h)
- RMSE: LSTM `12.9237` vs XGB `11.6442` (LSTM `-10.99%` worse)
- CRPS: LSTM `21.4210` vs XGB `17.2977` (LSTM `-23.84%` worse)
- Coverage p05-p95: LSTM `79.59%`, XGB `75.50%`

## Calibration (PIT + coverage tails)

### PIT KS (uniformity test)
- LSTM h1: KS `0.1338`, p-value `0.0000`
- LSTM h12: KS `0.1939`, p-value `0.0000`
- XGB h1: KS `0.1617`, p-value `0.0000`
- XGB h12: KS `0.1843`, p-value `0.0000`
- XGB h24: KS `0.2041`, p-value `0.0000`

Interpretation:
- All tested models fail strict PIT uniformity at this checkpoint (`p < 0.05`).
- LSTM has lower PIT KS than XGB at shared horizons, indicating relatively better calibration shape.

## Fair-Benchmark Mode

To avoid row-misalignment bias between sequence and tabular evaluations, the pipeline now supports:

```bash
uv run python scripts/evaluate.py --pollutants pm25,pm10,no2,o3 --horizons 1,12,24,168,672 --fair-intersection
```

This computes LSTM vs XGB only on common `(region, timestamp)` rows and writes:
- `models/fair_benchmark_summary.json`

The same mode is exposed in the orchestrator:

```bash
uv run python main.py --evaluate-only --fair-bench --pollutants pm25,pm10,no2,o3 --horizons 1,12,24,168,672
```

### Tail behavior
- LSTM h1: below p5 `9.03%`, above p95 `6.43%`, above p99 `1.75%`
- LSTM h12: below p5 `7.92%`, above p95 `12.49%`, above p99 `6.65%`
- XGB h1: below p5 `10.66%`, above p95 `7.03%`, above p99 `3.42%`
- XGB h12: below p5 `18.11%`, above p95 `6.39%`, above p99 `1.45%`

## Per-region Fairness Snapshot

### h12 fairness ratio (max/min RMSE)
- LSTM: `1.353x` (meets target)
- XGB: `1.543x` (above target)

### h12 RMSE by region
- LSTM: AIIMS `16.1927`, Bhatagaon `12.3836`, IGKV `11.9663`, SILTARA `14.3525`
- XGB: AIIMS `11.0295`, Bhatagaon `13.0990`, IGKV `8.4873`, SILTARA `13.1204`

## Important Scope Limitations
- LSTM is not yet trained for `h24`, `h168`, `h672` in this checkpoint.
- Only `pm25` is trained/evaluated currently.
- Current LSTM training summary shows smoke-scale sequence sample counts (`train=256`, `val=64`, `test=64`), so these are not final production-quality results.

## Recommended Next Evaluation Pass
- Complete full training for all pollutants (`pm25`, `pm10`, `no2`, `o3`) and all horizons (`1,12,24,168,672`).
- Re-run `scripts/evaluate.py` on full artifacts.
- Re-check:
  - long-horizon LSTM advantage (`h168`, `h672`),
  - fairness target (`max/min RMSE < 1.5x`),
  - PIT calibration (p-value threshold > 0.05 where feasible).
