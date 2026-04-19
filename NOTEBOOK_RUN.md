# Notebook Run Script (h168 Rescue Sweep, Tiered Gates, 8-Run Pilot -> 24 Total)

This runbook implements the supervisor-approved execution policy:

- Persistence baseline for gas gates: **raw `y_t`**
- Shadow baseline: **climatology by (`month`, `hour_of_day`) on train split**
- Delta target for gas rescue: **`y(t+168) - MA3(y_t)`**
- Tiered acceptance gates:
  - PM (`pm25`,`pm10`): `RMSE/mean < 0.5` and `R2 > 0.3`
  - Gas (`no2`,`o3`): `RMSE/mean < 0.8` and beat raw persistence
- Search budget split: **16 gas rescue + 8 PM refinement**
- Execution: **8-run pilot first**, then continue remaining 16 only if gas signal appears

Use `NOTEBOOK_RUN_CLEAN.py` directly as the runnable notebook script source.

## What `NOTEBOOK_RUN_CLEAN.py` now does

1. Clone + dependency setup with Kaggle fallback for `uv sync`
2. Run Phase 1 and preprocessing
3. Verify XGB artifacts (no retrain by default)
4. Generate 24 h168 configs (randomized ranges, pollutant-specific spaces)
5. Run 8 pilot configs on dual GPUs (one model per GPU)
6. Evaluate each pilot run with fair intersection + calibration
7. Apply tiered gates from `fair_benchmark_summary.json > operational_gates.rows`
8. Auto-continue remaining 16 only if pilot gas signal exists
9. Print operational leaderboard with gate pass/fail fields
10. Save final h168 operational snapshot

## Metric fields to track in outputs

From `models/experiments/*_fair_benchmark_summary.json`:

- `mean_concentration`
- `lstm_rmse`, `xgb_rmse`
- `lstm_r2`, `xgb_r2`
- `lstm_rmse_over_mean`, `xgb_rmse_over_mean`
- `lstm_coverage`, `xgb_coverage`
- `lstm_pit_ks`, `xgb_pit_ks`
- `persistence_rmse_raw_yt`
- `climatology_rmse_month_hour`
- `beats_persistence`, `beats_climatology`
- `gate_label`, `gate_pass`
- `lstm_pit_shape`, `xgb_pit_shape`

## Notes

- Backend is forced via `MPLBACKEND=agg` for script stability.
- Legacy feature fallback remains enabled in eval/predict.
- For delta-mode checkpoints, eval/predict reconstruct level quantiles before computing metrics.
