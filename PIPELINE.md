# PIPELINE.md

## Purpose
This document describes the end-to-end ML pipeline for building robust single-target LSTM forecasters for air-pollution time series across multiple pollutants and regions.

It is intentionally model-pipeline focused (not developer tooling). The goal is reproducibility and transferability: if a new pollutant (for example SOx/NOx) is added, the same steps can be followed and audited.

---

## Core Design

- One LSTM model is trained per `(pollutant, horizon)` target.
- Each model receives multi-pollutant context in its lookback window (`pm25`, `pm10`, `no2`, `o3` + weather/time/region features).
- Each model outputs quantiles for one target pollutant only (`p05`, `p50`, `p95`, `p99`).
- Training/evaluation remains time-ordered (no shuffle split) and leakage-safe.

This design separates target specialization (output) from shared environmental context (input).

---

## Stage 1: Canonical Data Assembly

1. Parse all available source workbooks by region and period.
2. Normalize timestamps to hourly resolution.
3. Merge hourly and quarter-hourly feeds with deterministic precedence.
4. Deduplicate on `(region, timestamp)`.
5. Persist canonical hourly table for reproducibility.

Expected outcome:
- A single hourly canonical dataset used by all downstream pipelines.

---

## Stage 2: Data Quality Investigation

Run investigation before training changes. For each pollutant:

1. Quantify impossible values and sensor-error signatures.
2. Quantify tail behavior (quantiles, tail exceedance, extreme maxima).
3. Quantify retention/loss by pipeline step (sanitize, impute, sequence constraints).
4. Quantify regional imbalance and missingness concentration.

Why this matters:
- Hyperparameters and sequence strategy should be justified by observed data behavior.
- New pollutants should enter through the same quality gate before modeling.

---

## Stage 3: LSTM Preprocessing (Sequence-Focused)

1. Remove only impossible/sensor-error values.
2. Interpolate short gaps (bounded), keep unresolved long gaps excluded.
3. Add weather/time/region context features under a single-latency contract.
4. Add region id feature.
5. Generate targets `target_<pollutant>_h<horizon>`.
6. Apply robust scaling using train statistics only.
7. Split into train/val/test by time blocks.

Leakage policy:
- Feature rows must be strictly before anchor time (past-only windowing).
- Apply latency exactly once (source delay OR explicit lag transforms, not both).
- Validation/test statistics are never used for scaling or calibration fitting.

---

## Stage 4: Model Construction

Per `(pollutant, horizon)`:

Input:
- Sequence of feature vectors containing all pollutant context channels.

Backbone:
- BiLSTM with configurable hidden size/layers/dropout.

Head:
- Horizon-specific attention + quantile MLP head.

Output:
- Quantiles for a single target pollutant at one horizon.

Contract:
- Inputs include all pollutant channels.
- Output target is exactly one pollutant-horizon target.

---

## Stage 5: Training Strategy

1. Train separate models for each `(pollutant, horizon)`.
2. Use region-aware sample balancing and region-weighted loss.
3. Optimize multi-quantile pinball loss.
4. Use early stopping + LR scheduling.
5. Save model, predictions, and full run metadata.

Long-horizon emphasis (`h168` and beyond):
- Prefer horizon-focused sweeps first, then expand to shorter horizons.
- Tune sequence length and regularization jointly; monitor both RMSE and CRPS.

---

## Stage 6: Calibration and Evaluation

1. Calibrate quantiles on validation set (no test leakage).
2. Evaluate on test with fair row intersection across models.
3. Report per pollutant and per horizon:
   - RMSE (p50 accuracy)
   - CRPS (distribution quality)
   - Coverage (p05-p95)
   - PIT KS diagnostics
   - Region fairness diagnostics

Interpretation rule:
- A model can improve RMSE/CRPS while still under-covering.
- Coverage must be treated as a first-class acceptance criterion.

---

## Stage 7: Pollutant-Specific Selection

Because pollutants behave differently, evaluate candidates both:

1. Global score (all pollutants) for one-config simplicity.
2. Per-pollutant score for targeted deployment/tuning.

For practical deployment:
- Use one shared pipeline contract.
- Allow pollutant-specific hyperparameters when justified by metrics.

---

## Stage 8: Extending to New Pollutants (SOx/NOx)

When adding a new pollutant:

1. Add raw channel to canonical merge + quality investigation.
2. Add target generation for all active horizons.
3. Keep all existing pollutant channels as input context.
4. Train new single-target models `(new_pollutant, horizon)`.
5. Run same calibration/evaluation protocol and compare against baseline model family.

No pipeline redesign should be required if this contract is respected.

---

## Stage 9: Reproducibility Checklist

- Fixed random seeds.
- Versioned preprocessing outputs.
- Serialized model and optimizer hyperparameters.
- Logged sequence settings per horizon.
- Logged calibration settings and fair-comparison metrics.

This ensures runs are traceable and comparable across pollutants and horizons.
