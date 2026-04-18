# DECISIONS.md - Living Final Report
## Air Pollution Forecasting: Hierarchical Multi-Horizon Quantile Regression LSTM

## ⚠️ CRITICAL: GIT INSTRUCTIONS
**DO NOT RUN ANY GIT COMMANDS** (git add, git commit, git push, or any git modification commands).

This repository contains a messy previous implementation that will be completely replaced. Treat all existing code in `scripts/`, `main.py`, and `models/` as **irrelevant**. You will delete and rewrite them all.

**Safe to use**: Read-only commands like `git status`, `git log`, `git diff` are fine for understanding history.

**Never run**:
```bash
git add .                    # ✗ NO
git commit -m "..."         # ✗ NO
git push origin main         # ✗ NO
git checkout -b new_branch  # ✗ NO
git rebase, git merge, etc  # ✗ NO
```

Work directly on files. When this build is complete, the user will handle all git operations themselves.

---

This document is the **living final report** tracking all architectural decisions, rationales, evidence, and performance results. Update continuously as decisions are made and tested.

---

## Document Structure
- **Section 1-3**: Problem & Dataset (reference)
- **Section 4+**: Architectural Decisions (UPDATE AS BUILD PROGRESSES)
- **Format per decision**: Title | Options | Choice | Rationale | Evidence | Performance

---

## 1. Problem Statement

### 1.1 Objective
Build a production-ready multi-horizon quantile regression forecasting model for air pollution (PM2.5, PM10, NO2, O3) across 4 Raipur regions using hierarchical LSTM with XGB baseline comparison.

### 1.2 Target Variables
- **Pollutants**: PM2.5, PM10, NO2, O3 (independent models per pollutant)
- **Quantiles**: p5, p50, p95, p99 (4 quantile levels per horizon)
- **Horizons**: t+1h, t+12h, t+24h, t+7d, t+28d (5 time horizons)
- **Total outputs per model**: 5 horizons × 4 quantiles = 20 predictions

Current experimental scope (active implementation):
- Horizons: `h1`, `h24`, `h168`
- LSTM training mode: separated per `(pollutant, horizon)`

### 1.3 Research Questions
1. Can hierarchical LSTM with quantile regression provide calibrated uncertainty estimates?
2. Does LSTM outperform XGB at long horizons (7d/28d)?
3. Can region weighting ensure fair treatment of rare regions (Bhatagaon 8.8%)?
4. Can dual preprocessing pipelines optimize each model's strengths?

---

## 2. Dataset Description

### 2.1 Data Source
```
Location: ./Pollution Data Raipur/[region]/[year]/[month]/*.xlsx
Regions: Bhatagaon DCR, DCR AIIMS, IGKV DCR, SILTARA DCR
Time range (Phase 1 canonical): 2022-06-01 to 2025-12-31
Frequency: Mixed hourly + quarter-hourly source files, canonicalized to hourly
```

### 2.2 Data Characteristics
```
Phase 1 canonical counts (fresh implementation, hourly + quarterly):
  - Parsed rows from source workbooks: 750,540
    * Hourly source rows: 169,611
    * Quarterly source rows: 580,929
  - Canonical hourly unique rows (region + floored hour): 125,017

Quarterly contribution to canonical dataset:
  - Quarterly-only hourly timestamps added: 3,690
  - PM2.5 values filled from quarterly where hourly missing: 5,689

Canonical region distribution (hourly unique):
  - Bhatagaon: 31,440 (25.15%)
  - AIIMS: 31,415 (25.13%)
  - SILTARA: 31,154 (24.92%)
  - IGKV: 31,008 (24.80%)
  - Imbalance ratio: 1.01×

Phase 1 data-quality findings:
  [x] Bhatagaon Sept 2025 extreme PM2.5 spikes are sensor errors (3 clusters, 7 points)
  [x] Major hard loss source is impossible PM2.5 values (<0 or >5000)
  [x] SILTARA 2023 is NOT fully missing (8,737 records present)
  [x] Outlier clipping (>300) contributes minimal additional loss in canonical hourly baseline

Legacy discrepancy note:
  - Historical numbers in earlier drafts (170,591 with 6.5× imbalance) came from legacy loading behavior.
  - Canonical Phase 1 loader now merges hourly and quarterly sources, floors to hour, then de-duplicates.
```

### 2.3 Data Retention Timeline (TO BE UPDATED AFTER PHASE 1)
```
Stage                                            Records     Retention
Parsed rows (hourly + quarterly)                 750,540     100.00%
Canonical hourly unique rows                     125,017      16.66% (post-canonicalization)
After impossible values removed                  120,548      96.43% (of canonical)
After interpolation (<=6h gaps)                 122,808      98.23% (of canonical)
After outlier removal (XGB style, PM2.5 <=300)  122,652      98.11% (of canonical)
After sequence boundary handling                 122,487      97.98% (of canonical)

Gap-threshold sensitivity (sequence-ready):
  - 6h baseline: 122,487
  - 12h threshold: 123,271 (+0.64%)
  - 24h threshold: 123,916 (+1.17%)
```

---

## 3. Extreme Data Issues (Phase 1: Under Investigation)

**See DATA_INVESTIGATION.md for details**

### 3.1 Bhatagaon Sept 2025 Spike
```
Issue: PM2.5 >500 µg/m³ in Bhatagaon Sept 2025
Status: RESOLVED (Phase 1 complete)
Finding: 7 extreme points grouped into 3 abrupt clusters
Decision: SENSOR ERROR clusters -> REMOVE these 7 points
Evidence:
  - Abrupt jumps from normal baseline (e.g., 9.32 -> 2963.90) and immediate reversion
  - No regional corroboration at spike timestamps (other regions remain low)
  - Even with stagnant weather, event remains sensor-localized rather than city-wide
Impact: Keep asymmetric policy but with explicit exception:
  - Remove confirmed sensor-error clusters
  - Keep plausible non-impossible outliers for LSTM modeling context
```

### 3.2 Data Loss Root Cause
```
Issue: Root-cause attribution of data attrition in canonical pipeline
Status: RESOLVED (Phase 1 complete)
Canonical baseline: 125,017 hourly unique rows
Key attribution:
  - Impossible values removed: 4,469
  - Interpolation recovery (<=6h): +2,260
  - Unresolved missing after interpolation: 2,948 timeline hours
  - Outlier removal (>300): 156
  - Sequence boundary loss: 165
Decision: Keep conservative 6h interpolation for robustness; test 12h/24h as ablations
Evidence: +0.64% recovery at 12h, +1.17% at 24h (sequence-ready counts)
Impact: Primary attention should be on impossible values and long-gap handling, not outlier clipping
```

### 3.3 Region Imbalance Mitigation
```
Issue: Regional imbalance severity in canonical hourly dataset after integrating quarterly files
Status: RESOLVED (Phase 1 complete)
Finding: Imbalance is mild after canonical deduplication
  - Raw hourly unique ratio: 1.01×
  - Post-sequence ratio: 1.04×
Region weights (post-sequence fractions):
  - Bhatagaon: 1.009×
  - IGKV: 0.994×
  - AIIMS: 0.979×
  - SILTARA: 1.020×
Decision: Retain weighted loss + stratified sampling, but with mild weights
Impact: Fairness controls remain required, but expected distortion from imbalance is low
```

---

## 4. Architectural Decisions

### Decision 4.1: Model Type - Hierarchical LSTM vs Separate Models

| Aspect | Details |
|--------|---------|
| **Options** | (A) Single hierarchical model (5 horizons, 1 backbone) |
| | (B) Separate models per horizon (5 independent models) |
| | (C) Single model all horizons (no attention specialization) |
| **Chosen** | (A) Single hierarchical model |
| **Rationale** | - Shared BiLSTM backbone learns global patterns (e.g., "PM2.5↑ at night") |
| | - Horizon-specific attention heads specialize for each timescale |
| | - (B) would create 5×4 pollutants=20 models, maintenance nightmare |
| | - (C) removes horizon-awareness, likely worse performance |
| **Evidence** | None yet (pre-implementation). Implementation will compare. |
| **Status** | ✓ APPROVED (historical baseline; superseded by 4.13 for current experimental scope) |

### Decision 4.2: Output Type - Quantile Regression vs Point Predictions

| Aspect | Details |
|--------|---------|
| **Options** | (A) Quantile regression (p5, p50, p95, p99 per horizon) |
| | (B) Point predictions (RMSE/MAE loss only) |
| | (C) Quantile regression with external uncertainty model |
| **Chosen** | (A) Quantile regression |
| **Rationale** | - Prediction intervals (not point estimates) enable risk-aware decisions |
| | - "90% confident pollution ≤ 150 µg/m³" more useful than "predict 120 µg/m³" |
| | - Calibrated quantiles = honest uncertainty (PIT test validates) |
| | - (B) ignores uncertainty, poor for alerting systems |
| | - (C) adds complexity; direct quantile regression simpler |
| **Evidence** | None yet (standard practice in uncertainty quantification) |
| **Status** | ✓ APPROVED |

### Decision 4.3: Dual Preprocessing vs Unified Pipeline

| Aspect | Details |
|--------|---------|
| **Options** | (A) Separate LSTM and XGB pipelines (opposite philosophies) |
| | (B) Single unified preprocessing (compromise features) |
| | (C) Two preprocessing, feed same features to both (redundant/suboptimal) |
| **Chosen** | (A) Separate pipelines |
| **Rationale** | **LSTM philosophy**: Sequences ARE information |
| | - Keep outliers (PM2.5 >500 OK if pattern real) - attention learns to weight |
| | - Light imputation (break sequences on >6h gaps) - temporal continuity sacred |
| | - RobustScaler (preserve outlier info via z-score) |
| | - Minimal features (15) - implicit lags via sequence memory |
| | **XGB philosophy**: Features ARE information |
| | - Remove outliers with pollutant-specific caps (PM2.5/PM10/NO2/O3) - tree splits distort on extremes |
| | - Aggressive imputation + 6 missingness features - encode gaps as predictors |
| | - No scaling (scale-invariant) |
| | - Rich features (55-60) - explicit lags, rolling stats, interactions |
| | - (B) neither model at full potential; (C) duplicates features unnecessarily |
| **Evidence** | Theoretical (opposite trees vs sequences). Will validate Phase 5-6. |
| **Status** | ✓ APPROVED (see PREPROCESSING_STRATEGY.md) |

### Decision 4.4: Loss Function - Region-Weighted Multi-Quantile

| Aspect | Details |
|--------|---------|
| **Options** | (A) Region-weighted multi-quantile pinball loss |
| | (B) Standard pinball loss (no region weighting) |
| | (C) Separate models per region (no sharing) |
| **Chosen** | (A) Region-weighted multi-quantile pinball loss |
| **Rationale** | - Fairness still requires explicit control even with mild imbalance |
| | - (B) can still drift toward cleaner/easier regions during optimization |
| | - (C) no shared learning and higher variance per-region models |
| | - (A) preserves shared backbone while keeping regional parity pressure |
| | - Formula: weight_r = (1/4) / (fraction_r) |
| | - Phase 1 canonical weights (hourly+quarterly) are mild: BH 1.009×, IGKV 0.994×, AIIMS 0.979×, SILTARA 1.020× |
| **Implementation** | loss_weighted = sum(weight_r × loss_r for all regions) |
| **Evidence** | Phase 1 shows post-sequence imbalance ratio 1.04×; weighting retained as guardrail |
| **Status** | ✓ APPROVED (see ARCHITECTURE.md section 3.2) |

### Decision 4.5: Sequence Length Strategy - Adaptive vs Fixed

| Aspect | Details |
|--------|---------|
| **Options** | (A) Pure `2×horizon` adaptive lengths |
| | (B) Fixed length for all horizons |
| | (C) Horizon-specific calibrated map for reduced scope |
| **Chosen** | (C) Horizon-specific calibrated map |
| **Rationale** | - Reduced scope (`h1`,`h24`,`h168`) allows direct sequence-budget tuning per horizon |
| | - `h168` provides meaningful medium-horizon difficulty while keeping runtime and convergence behavior practical |
| | - Calibrated map preserves short-horizon efficiency and long-horizon context without collapsing validation samples |
| | **Sequence lengths (active)**: `h1=168`, `h24=336`, `h168=720` |
| **Implementation** | `train_lstm.py` accepts `--seq-len-map`; default map is `1:168,24:336,168:720` |
| **Evidence** | `h168` provides strong sequence context while preserving robust sample counts across regions and faster end-to-end experimentation |
| **Status** | ✓ APPROVED |

### Decision 4.6: Weather Features - Lagged vs Current vs Future

| Aspect | Details |
|--------|---------|
| **Options** | (A) Lagged weather only (t-24 to t-1, never t or future) |
| | (B) Current weather (t) - realistic for nowcasting, not forecasting |
| | (C) Future weather predictions (t+1 to t+28) |
| **Chosen** | (A) Lagged weather only |
| **Rationale** | - **Goal**: Operational forecast (don't know future weather) |
| | - (A) uses yesterday's weather to predict tomorrow's pollution (realistic) |
| | - (B) current weather not available at forecast time (can't use) |
| | - (C) introduces weather model error; complicates comparison |
| | - **Assumption**: "Tomorrow's pollution depends on today's weather patterns" |
| | - **Limitation**: Long horizons (28d) limited by weather dependency |
| **Implementation** | Features: temperature[t-24:t-1], humidity[t-24:t-1], wind[t-24:t-1] |
| **Validation** | Explicit check: assert all features have lag ≥ 0 (no future) |
| **Evidence** | None yet (design choice). Will validate no leakage Phase 5. |
| **Status** | ✓ APPROVED (critical for production use) |

### Decision 4.7: Baseline Model - XGB vs RF vs Ridge vs None

| Aspect | Details |
|--------|---------|
| **Options** | (A) XGBoost (gradient boosting, feature-rich) |
| | (B) Random Forest (ensemble, simpler) |
| | (C) Ridge Regression (linear baseline) |
| | (D) No baseline (LSTM only) |
| **Chosen** | (A) XGBoost |
| **Rationale** | - **LSTM's claim**: "Excels at long horizons (7d/28d)" |
| | - Need fair baseline to validate this claim |
| | - (A) XGB is state-of-the-art for structured data (strong baseline) |
| | - (B) RF likely weaker than XGB (less gradient boosting advantage) |
| | - (C) Ridge too simple (linear can't capture pollution patterns) |
| | - (D) no baseline = can't quantify LSTM advantage |
| | - **Fair comparison**: XGB gets rich features (55-60), LSTM gets minimal (15) |
| | - **Expected result**: LSTM 40-65% better on t+7d/t+28d |
| **Implementation** | Separate XGB models per quantile per horizon (20 models total per pollutant) |
| | Same 70/15/15 split as LSTM for direct comparison |
| **Evidence** | None yet (will measure Phase 5-6) |
| **Status** | ✓ APPROVED |

### Decision 4.8: Train/Val/Test Split - Time-Based 70/15/15 vs Walk-Forward CV

| Aspect | Details |
|--------|---------|
| **Options** | (A) Simple time-based 70/15/15 (Phase 4) |
| | (B) 5-fold walk-forward CV (Phase 4 extended) |
| | (C) Random shuffle (INVALID - data leakage) |
| **Chosen** | (A) Phase 4 (simple 70/15/15), then (B) Phase 5+ (walk-forward) |
| **Rationale** | - Time series MUST use time-based split (no random shuffle) |
| | - (A) faster to iterate, good for architecture validation |
| | - (B) needed for final robustness assessment |
| | **Split dates**: |
| | - Train: 2022-01-01 to 2024-03-31 (27 months, 70%) |
| | - Val: 2024-04-01 to 2024-06-30 (3 months, 15%) |
| | - Test: 2024-07-01 to 2025-04-16 (9 months, 15%) |
| **Validation** | assert train.max_date < val.min_date < test.min_date |
| **Evidence** | None yet (timing setup only) |
| **Status** | ✓ APPROVED Phase 4 / ◯ TODO Phase 5+ (walk-forward) |

### Decision 4.9: Outlier Handling - LSTM vs XGB Asymmetry

| Aspect | Details |
|--------|---------|
| **Options** | (A) Asymmetric: LSTM keeps outliers, XGB removes |
| | (B) Symmetric: Both keep outliers |
| | (C) Symmetric: Both remove outliers |
| **Chosen** | (A) Asymmetric |
| **Rationale** | **LSTM**: Sequences encode information |
| | - Outlier pattern matters (rise/fall/spike/decay) |
| | - Attention learns to downweight extreme z-scores |
| | - Example: [100, 200, 500, 600, 400, 100] = real spike → KEEP |
| | **XGB**: Tree splits distort on extremes |
| | - Single extreme value splits tree for ALL 1000 samples |
| | - Can't learn individual outlier patterns (binary split) |
| | - Example: Single PM2.5=2000 forces tree split, hurts majority |
| | **Trade-off**: |
| | - LSTM keeps 73% data (less loss) |
| | - XGB keeps 70% data (more loss) |
| | - Each model gets optimized handling |
| **Implementation** | LSTM: keep PM2.5 if pattern real (post-Phase 1 analysis) |
| | XGB: remove by pollutant caps (PM2.5 >300, PM10 >600, NO2 >250, O3 >150) |
| **Evidence** | Will validate Phase 5-6 (compare per-region CRPS) |
| **Status** | ✓ APPROVED (see PREPROCESSING_STRATEGY.md) |

### Decision 4.10: Batch Sampling - Stratified vs Random

| Aspect | Details |
|--------|---------|
| **Options** | (A) Stratified sampling (~8 samples per region per batch) |
| | (B) Random sampling (ignore region imbalance) |
| | (C) Weighted sampling per region |
| **Chosen** | (A) Stratified sampling |
| **Rationale** | - Combine two fairness strategies: |
| | 1. Loss weighting (upweight rare regions) |
| | 2. Batch sampling (ensure rare regions in every batch) |
| | - (B) leads to batches with 25 IGKV + 3 Bhatagaon (imbalance preserved) |
| | - (A) enforces ~8 per region = 4×8 = 32 batch size |
| | - (C) probabilistic, unpredictable per-batch composition |
| **Implementation** | Batch composition: [Bhatagaon_8, IGKV_8, AIIMS_8, SILTARA_8] |
| | If region <8 samples in epoch: resample with replacement |
| **Validation** | Monitor per-region val loss (should be similar, not 5× different) |
| **Evidence** | None yet (will validate Phase 5) |
| **Status** | ✓ APPROVED |

### Decision 4.11: LSTM Short-Horizon Context Floor

| Aspect | Details |
|--------|---------|
| **Options** | (A) Pure adaptive window `2×horizon` |
| | (B) Adaptive with minimum context floor `max(2×h, min_attn_window)` |
| **Chosen** | (B) Adaptive + minimum context floor |
| **Rationale** | - Pure `2×h` gives h1 only 2h tail attention, while XGB uses richer short-horizon lag context |
| | - This causes unfair short-horizon information budget and weak h1/h24 LSTM robustness |
| | - Minimum window preserves horizon-adaptive behavior for long horizons while avoiding h1 starvation |
| **Implementation** | In `train_lstm.py`, attention window now uses `min(max(2*h, min_attn_window), seq_len)` |
| | New CLI: `--min-attn-window` (default: 24h) |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.12: Horizon Loss Balancing for Multi-Horizon LSTM

| Aspect | Details |
|--------|---------|
| **Options** | (A) Equal weighting across horizons |
| | (B) Inverse-sqrt horizon weighting |
| | (C) Manual user-specified weights |
| **Chosen** | (B) default + (C) optional override |
| **Rationale** | - Long-horizon targets are inherently harder/noisier and can dominate total loss |
| | - Weighted objective reduces gradient domination and stabilizes short/mid horizon fit |
| | - Manual override retained for targeted ablations |
| **Implementation** | `pinball_loss` now supports horizon weights; CLI adds `--horizon-weighting` and `--horizon-loss-weights` |
| **Status** | ✓ APPROVED (historical for joint-horizon mode; ignored in separated-horizon mode) |

### Decision 4.13: Horizon-Separated LSTM Training (Scope Reduction)

| Aspect | Details |
|--------|---------|
| **Options** | (A) Keep single multi-horizon model per pollutant |
| | (B) Train separated models per `(pollutant, horizon)` |
| **Chosen** | (B) Separated models |
| **Rationale** | - Joint optimization across distant horizons caused interference and unstable quality |
| | - Scope reduction to `h1`, `h24`, `h168` enables cleaner diagnosis and faster iteration |
| | - Explicit horizon-specialized training gives each horizon dedicated early stopping and checkpointing |
| **Implementation** | `train_lstm.py` now trains 12 models (`4 pollutants × 3 horizons`) and saves per-horizon checkpoints (`lstm_quantile_<pollutant>_h<h>.pt`) and predictions (`lstm_predictions_<pollutant>_h<h>.npz`) |
| | `evaluate.py` and `predict.py` load horizon-specific checkpoints first, with legacy fallback |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.14: Horizon Scope Shift from h672 to h168

| Aspect | Details |
|--------|---------|
| **Options** | (A) Keep `h672` experimental horizon |
| | (B) Replace `h672` with `h168` |
| **Chosen** | (B) Replace with `h168` |
| **Rationale** | - `h672` training is expensive and less aligned with immediate quality target; `h168` offers a better quality/runtime trade-off |
| | - `h168` keeps medium-horizon forecasting challenge without excessive memory/time burden |
| | - Enables tighter iteration cycles and clearer calibration benchmarking in one session |
| **Implementation** | Active horizon scope is now `h1`,`h24`,`h168`; default seq map `1:168,24:336,168:720` |
| **Evidence** | Observed run behavior and user priority shifted toward high-quality, calibrated 95% CI with practical runtime constraints |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.15: Quantile Calibration Policy (Both LSTM and XGB)

| Aspect | Details |
|--------|---------|
| **Options** | (A) No calibration |
| | (B) Calibrate only LSTM |
| | (C) Calibrate both LSTM and XGB |
| **Chosen** | (C) Calibrate both |
| **Rationale** | - Fair probabilistic benchmarking requires both model families to be calibrated the same way |
| | - Validation-set calibration improves interval reliability without using test labels |
| | - Product focus is reliable 95% CI behavior (`p05-p95`) rather than only point RMSE |
| **Implementation** | `evaluate.py --calibrate-quantiles` fits per-quantile additive bias on validation predictions and applies it to test predictions for both model families |
| | Quantile monotonicity is enforced post-calibration with cumulative max |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.16: 95% CI Priority in Benchmarking

| Aspect | Details |
|--------|---------|
| **Options** | (A) Treat all quantile tails equally |
| | (B) Prioritize p05-p95 interval quality |
| **Chosen** | (B) Prioritize 95% CI |
| **Rationale** | - Operationally, stable 95% confidence bands are the primary uncertainty product |
| | - Still retain p99 diagnostics, but optimize and report primarily on p05/p95 coverage and tails |
| **Implementation** | Evaluation summary includes explicit 95% CI-focused calibration benchmark table: coverage error, p05/p95 tail errors, interval width, crossing rate |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.17: Extend Phase 1 Investigation to All Pollutants

| Aspect | Details |
|--------|---------|
| **Options** | (A) Keep Phase 1 analysis PM2.5-centric only |
| | (B) Extend investigation to PM10/NO2/O3 as well |
| **Chosen** | (B) Extend to all pollutants |
| **Rationale** | - PM2.5-only diagnostics hide pollutant-specific tail behavior and potential preprocessing opportunities |
| | - We need explicit evidence before introducing any pollutant-specific clipping/handling changes |
| | - Better transparency for future alternative processing approaches |
| **Implementation** | `scripts/data_investigation.py` now computes all-pollutant outlier-tail summaries + per-pollutant loss attribution, stores them in `phase1_investigation_results.json`, and emits dedicated visualizations |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.18: XGB Outlier Policy Consistency Across Pollutants

| Aspect | Details |
|--------|---------|
| **Options** | (A) Cap PM2.5 only |
| | (B) Drop caps entirely |
| | (C) Apply explicit caps to all pollutants |
| **Chosen** | (C) Apply caps to all pollutants |
| **Rationale** | - Mixed policy (cap one pollutant only) creates inconsistent feature hygiene for tree models |
| | - User requested either all-pollutant caps or no caps; all-pollutant caps preserves robust tree split behavior |
| | - Caps align with Phase 1 all-pollutant tail diagnostics and remain XGB-only (LSTM still preserves plausible tails) |
| **Implementation** | `preprocess_xgb.py` now uses caps: PM2.5<=300, PM10<=600, NO2<=250, O3<=150 |
| | Metadata now stores per-pollutant cap policy and removed-row stats |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.19: Configurable LSTM Regularization + Capacity Sweep Controls

| Aspect | Details |
|--------|---------|
| **Options** | (A) Keep hardcoded LSTM architecture/regularization in code |
| | (B) Expose architecture + optimizer/scheduler regularization knobs via CLI |
| **Chosen** | (B) Expose knobs via CLI |
| **Rationale** | - Current long-horizon runs (`h24`,`h168`) peak early and then drift, requiring rapid controlled sweeps |
| | - Hardcoded `hidden_dim/layers/heads/dropout/weight_decay` slows experimentation and increases edit risk between runs |
| | - Run-level hyperparameter control improves reproducibility and clear audit trail in saved checkpoints/summaries |
| **Implementation** | `train_lstm.py` now supports CLI args: `--hidden-dim`, `--num-layers`, `--num-heads`, `--dropout`, `--head-dropout`, `--weight-decay`, `--scheduler-factor`, `--scheduler-patience`, `--max-grad-norm` |
| | Model + optimizer hyperparameters are persisted in checkpoint (`model_hparams`, `optimizer_hparams`) and training summary |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.20: Checkpoint-Driven LSTM Architecture Loading in Evaluation/Inference

| Aspect | Details |
|--------|---------|
| **Options** | (A) Assume fixed model shape (`hidden=128`, `layers=2`, `heads=4`) in evaluate/predict |
| | (B) Rebuild architecture from checkpoint metadata |
| **Chosen** | (B) Load architecture from checkpoint metadata |
| **Rationale** | - Capacity sweeps are invalid unless evaluation/inference can reconstruct the exact trained model shape |
| | - Fixed-shape loaders fail on any architecture change and block fair comparisons |
| **Implementation** | `evaluate.py` and `predict.py` now read `model_hparams` from checkpoint with backward-compatible defaults for legacy models |
| | Added divisibility guard (`2*hidden_dim % num_heads == 0`) before model instantiation |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.21: Aggressive Recovery Run Profile for Horizon-Separated LSTM (Current Sweep)

| Aspect | Details |
|--------|---------|
| **Options** | (A) Keep prior long-context profile (`h1=168,h24=336,h168=720`) |
| | (B) Increase sample density with `seq_len=max(24,h)` and tighter regularization/capacity |
| **Chosen** | (B) for current recovery sweep |
| **Rationale** | - Long-horizon checkpoints underperform XGB and show early-epoch peaks, suggesting overfitting/optimization instability |
| | - `seq_len=max(24,h)` materially increases val/test sequences for `h24/h168` while preserving minimum temporal context |
| | - Reduced capacity + stronger regularization enables faster iterative search near observed effective epoch range (~<=25) |
| **Implementation** | Notebook run profile uses: `--seq-len-map 1:24,24:24,168:168`, `--hidden-dim 64`, `--num-layers 2`, `--num-heads 2`, `--dropout 0.35`, `--head-dropout 0.30`, `--weight-decay 5e-4`, `--lr 3e-4`, `--epochs 40`, `--patience 6`, `--scheduler-patience 2` |
| **Status** | ✓ APPROVED (run-profile; non-default) |

### Decision 4.22: Single-Target LSTM with Multi-Pollutant Context Contract

| Aspect | Details |
|--------|---------|
| **Options** | (A) Single-target models using only self-pollutant inputs |
| | (B) Single-target models using all pollutant context inputs |
| | (C) Multi-output all-pollutant model per horizon |
| **Chosen** | (B) Single-target output with all-pollutant input context |
| **Rationale** | - Pollutants have distinct target dynamics and benefit from target-specific optimization/calibration |
| | - Cross-pollutant co-movement still contains predictive signal (e.g., PM2.5/PM10 coupling), so context channels should remain available to each model |
| | - Supports future onboarding of new pollutants (SOx/NOx) without changing core training/evaluation contracts |
| **Implementation** | `train_lstm.py` now enforces input-target contract at runtime: required context channels include `pm25,pm10,no2,o3`; output target remains exactly `target_<pollutant>_h<horizon>` |
| | Added `PIPELINE.md` as pipeline architecture source for reproducible single-target + multi-pollutant-context workflow and extension path |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.23: Pollutant-Specific h168-First Sweep Workflow

| Aspect | Details |
|--------|---------|
| **Options** | (A) Keep one shared sweep config across all pollutants and horizons |
| | (B) Use pollutant-specific sweep configs, starting with h168 focus |
| **Chosen** | (B) Pollutant-specific, h168-first |
| **Rationale** | - Empirical h168 behavior differs materially by pollutant (pm25/pm10 vs no2/o3) |
| | - Long-horizon quality is the current bottleneck and should be optimized first for fastest signal |
| | - Pollutant-specific seeds improve extensibility to future pollutant onboarding without changing core pipeline contract |
| **Implementation** | `NOTEBOOK_RUN.md` now defines a pollutant-specific h168 workflow: per-pollutant training calls, evaluation snapshots, and rank tables sourced from `models/experiments` |
| | Uses `--summary-path` in `train_lstm.py` for clean per-run experiment tracking |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.24: Single-Latency Feature Contract (No Double Weather Lag)

| Aspect | Details |
|--------|---------|
| **Options** | (A) Keep explicit weather lag transforms in addition to past-only window slicing |
| | (B) Remove explicit global weather lag if source rows are already delayed/finalized by >=1h |
| **Chosen** | (B) Remove extra weather lag; apply latency exactly once |
| **Rationale** | - Existing setup combined delayed feature rows and additional weather lag, effectively pushing weather signal too far back |
| | - This can hurt short/mid-horizon quality (notably h1/h24) without improving leakage safety |
| | - Leakage is still prevented by strict past-only sequence construction (no anchor/future rows) |
| **Implementation** | `preprocess_lstm.py` now uses raw weather columns (`temperature`,`humidity`,`wind_speed`,`wind_direction`) in feature set; explicit `*_lag_1` weather columns removed |
| | `preprocess_xgb.py` no longer applies global weather `shift(1)` before lag feature generation |
| | Metadata leakage policy now documents single-latency contract (source delay OR feature lag, not both) |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.25: Runtime Compatibility Hardening (Backend + Legacy Feature Fallback)

| Aspect | Details |
|--------|---------|
| **Options** | (A) Keep strict runtime assumptions (interactive matplotlib backend, exact legacy feature names only) |
| | (B) Enforce non-interactive plotting backend and add backward-compatible feature mapping for legacy checkpoints |
| **Chosen** | (B) Runtime hardening + legacy compatibility |
| **Rationale** | - Notebook/script execution can inherit an inline matplotlib backend that fails in non-interactive script runs |
| | - New single-latency preprocessing uses raw weather columns, while older checkpoints may still reference `*_lag_1` weather feature names |
| | - Evaluation/inference should fail only on truly missing signal, not on recoverable naming/back-end incompatibilities |
| **Implementation** | `data_investigation.py` and `evaluate.py` now force `Agg` backend via env + `matplotlib.use(..., force=True)` |
| | `evaluate.py` and `predict.py` now adapt legacy weather feature names (`temperature_lag_1`,`humidity_lag_1`,`wind_speed_lag_1`,`wind_direction_lag_1`) to raw columns when available |
| | Notebook run scripts now use `--device auto` for train/eval calls to remain portable across CPU-only and GPU environments |
| **Evidence** | `uv run` smoke tests passed for `scripts/evaluate.py` and `scripts/predict.py` with legacy-to-raw remap logging enabled |
| **Status** | ✓ APPROVED (implemented) |

---

## 5. Evaluation Framework

### Decision 5.1: Metrics - CRPS, PIT, Coverage vs RMSE/MAE

| Aspect | Details |
|--------|---------|
| **Options** | (A) CRPS + PIT + Coverage (quantile-focused) |
| | (B) RMSE + MAE (point-estimate-focused) |
| | (C) Both |
| **Chosen** | (C) Both |
| **Rationale** | - (A) validates calibration (main goal) |
| | - (B) enables fair comparison to XGB baseline |
| | - Combined: show LSTM calibrated + accurate |
| | **Metrics per horizon**: |
| | - RMSE (vs XGB for comparison) |
| | - CRPS (quantile quality) |
| | - Coverage (% actuals in p5-p95 interval) |
| | - PIT uniformity (KS test for calibration) |
| **Implementation** | See ARCHITECTURE.md section 7 for detailed formulas |
| **Evidence** | None yet (will measure Phase 6) |
| **Status** | ✓ APPROVED |

### Decision 5.2: Per-Region Fairness - Separate Metrics vs Weighted

| Aspect | Details |
|--------|---------|
| **Options** | (A) Report metrics separately per region |
| | (B) Report only weighted average |
| | (C) Both |
| **Chosen** | (C) Both |
| **Rationale** | - (A) ensures rare regions not ignored |
| | - (B) shows overall performance |
| | - Combined: validate weighted loss works (fairness verified) |
| | **Target**: Max RMSE ratio = 1.5× (no region 50% worse) |
| **Implementation** | RMSE_per_region, max_ratio = max_rmse / min_rmse |
| **Evidence** | None yet (will check Phase 6) |
| **Status** | ✓ APPROVED |

---

## 6. Performance Results (PARTIAL - PM2.5 CHECKPOINT)

Source artifacts:
- `models/lstm_training_summary.json`
- `models/xgb_training_summary.json`
- `models/evaluation_summary.json`

Scope note:
- Current checkpoint includes `pm25` only.
- LSTM currently has trained/evaluated horizons `h1`, `h24`, `h168`.
- XGB currently has trained/evaluated horizons `h1`, `h24`, `h168`.

### 6.1 LSTM Quantile Regression (Partial)

```
Horizon    RMSE (vs actual)    CRPS      PIT KS stat    Coverage p5-p95
------     ---------------     ----      ----------     ----------------
t+1h       10.1547             15.2498   0.1338         84.54%
t+12h      12.9237             21.4210   0.1939         79.59%
t+24h      [Not trained]       [N/A]     [N/A]          [N/A]
t+7d       [Not trained]       [N/A]     [N/A]          [N/A]
t+28d      [Not trained]       [N/A]     [N/A]          [N/A]

Note: RMSE computed vs p50 (median prediction)
```

### 6.2 XGB Baseline (Partial)

```
Horizon    RMSE (vs actual)    CRPS      Coverage p5-p95
------     ---------------     ----      ----------------
t+1h       6.7111              7.0665    82.31%
t+12h      11.6442             17.2977   75.50%
t+24h      12.3045             18.8345   76.40%
t+7d       [Not trained]       [N/A]     [N/A]
t+28d      [Not trained]       [N/A]     [N/A]
```

### 6.3 LSTM vs XGB Comparison (Partial)

```
Horizon    LSTM CRPS    XGB CRPS    CRPS Improvement (LSTM vs XGB)
------     ----------   ---------   ------------------------------
t+1h       15.2498      7.0665      -115.80% (LSTM worse)
t+12h      21.4210      17.2977     -23.84% (LSTM worse)
t+24h      [N/A]        18.8345     [No overlap yet]

RMSE comparison on overlap:
  - t+1h: 10.1547 vs 6.7111  => -51.31% (LSTM worse)
  - t+12h: 12.9237 vs 11.6442 => -10.99% (LSTM worse)
```

Checkpoint interpretation:
- Current metrics in this section are stale and retained for historical context.
- Active pipeline now uses horizon-separated models on reduced scope (`h1`,`h24`,`h168`),
  with per-horizon checkpoints and validation-based quantile calibration support.

### 6.4 Per-Region Fairness (Partial)

```
Region      LSTM RMSE (t+12h)    XGB RMSE (t+12h)    Weight
------      ------------------   -----------------   ------
Bhatagaon   12.3836              13.0990             1.009×
IGKV        11.9663              8.4873              0.994×
AIIMS       16.1927              11.0295             0.979×
SILTARA     14.3525              13.1204             1.020×

Fairness ratio (max/min RMSE, target <1.5×):
  - LSTM t+12h: 1.353× (meets target)
  - XGB t+12h:  1.543× (above target)
```

Phase 1 note:
```
Using canonical hourly+quarterly distribution, fairness weights are near-uniform.
The original example weights (2.84× / 0.44× / 1.26× / 1.27×) are superseded by Section 7.4.
```

### 6.5 Quantile Calibration Quality (Partial)

```
Horizon/model     Below p5    p5-p95 interval    Above p95    Above p99
-------------     --------    ---------------    ---------    ---------
LSTM t+1h         9.03%       84.54%             6.43%        1.75%
LSTM t+12h        7.92%       79.59%             12.49%       6.65%
XGB t+1h          10.66%      82.31%             7.03%        3.42%
XGB t+12h         18.11%      75.50%             6.39%        1.45%
```

PIT uniformity (KS test):
```
LSTM t+1h:  KS=0.1338, p=0.0000
LSTM t+12h: KS=0.1939, p=0.0000
XGB t+1h:   KS=0.1617, p=0.0000
XGB t+12h:  KS=0.1843, p=0.0000
```

Calibration conclusion (checkpoint):
- PIT p-values are below 0.05 for all tested cases, so this checkpoint is not
  yet calibration-complete.
- LSTM PIT KS is lower than XGB at both shared horizons, suggesting relatively
  better calibration shape despite mixed accuracy outcomes.

---

## 7. Data Loss Investigation Results (Phase 1)

**See DATA_INVESTIGATION.md for analysis procedures**

### 7.1 Bhatagaon Sept 2025 Spike Classification (COMPLETED)

```
Finding: SENSOR ERROR (3 abrupt clusters, 7 points >500)

Clusters:
  1) 2025-09-01 23:00 to 2025-09-02 00:00, peak 3248.78
  2) 2025-09-10 08:00 to 2025-09-10 09:00, peak 2963.90
  3) 2025-09-11 03:00 to 2025-09-11 05:00, peak 5904.78

Evidence:
  - Abrupt spike shape (normal -> extreme -> normal in <=3h windows)
  - Other regions remain low at spike times (no cross-region corroboration)
  - Weather is stagnant but does not explain a city-wide pollution event

Decision: REMOVE these 7 points during preprocessing
Impact: Preserve asymmetric outlier philosophy but explicitly drop confirmed sensor-error clusters
```

### 7.2 Data Loss Root Causes (COMPLETED)

```
Canonical baseline:
  - Parsed rows (hourly + quarterly): 750,540
  - Canonical hourly unique rows: 125,017
  - Sequence-ready rows (baseline 6h interpolation): 122,487

Quarterly contribution:
  - Quarterly-only hourly timestamps added: 3,690
  - PM2.5 values filled from quarterly: 5,689

Loss / recovery breakdown (vs canonical hourly unique):
  - Impossible values removed: 4,469 (3.57%)
  - Interpolation recovery (<=6h): +2,260 (+1.81%)
  - Unresolved missing after interpolation: 2,948 timeline hours (2.36%)
  - Outlier removal (>300): 156 (0.12%)
  - Sequence boundary loss: 165 (0.13%)
  
Per-region retention:
  - Bhatagaon: 30,354 / 31,440 = 96.55%
  - IGKV: 30,818 / 31,008 = 99.39%
  - AIIMS: 31,289 / 31,415 = 99.60%
  - SILTARA: 30,026 / 31,154 = 96.38% (largest unresolved gap burden)

SILTARA 2023 check:
  - NOT fully missing; 8,737 rows present in 2023

Recovery opportunities:
  - 6h -> 12h interpolation threshold: +784 rows (+0.64%)
  - 6h -> 24h interpolation threshold: +1,429 rows (+1.17%)
  - Looser outlier bounds have minimal effect (outlier loss already tiny)
  
Recommendation:
  - Keep 6h interpolation as robust default for production baseline
  - Evaluate 12h/24h interpolation as ablations (especially SILTARA)
  - Prioritize impossible-value handling and spike removal over outlier-threshold tuning
```

### 7.3 Phase 1 Implementation Notes (COMPLETED)

```
Implemented script: scripts/data_investigation.py

Key implementation choices:
  - Fresh canonical loader reads .xls/.xlsx/.xlsb hourly and quarterly files across all 4 regions
  - Frequency harmonization: floor timestamps to hour; prefer hourly values; fill gaps from quarterly
  - Daily sheets are parsed from numeric tabs only (01..31)
  - Timestamp normalization floors to hour before region+timestamp deduplication
  - Parsed rows and canonical hourly unique rows are both saved for auditability

Quarterly merge diagnostics:
  - Unique-hour overlap (hourly ∩ quarterly): 112,094
  - Quarterly-only added hours: 3,690
  - PM2.5 filled from quarterly (hourly missing): 5,689

Known data issue:
  - 1 quarterly workbook failed to open (Bhatagaon Jan 2024 malformed .xls)
    and was skipped with warning; pipeline continues safely.

Generated artifacts:
  - data/raw/pollution_data_raw.csv
  - data/raw/pollution_data_hourly_unique.csv
  - data/raw/phase1_investigation_results.json
  - visualizations/phase_1_data_investigation/*.png (6 figures)

Phase 3 consistency:
  - preprocess_lstm.py and preprocess_xgb.py reuse the same canonical merge contract
  - Quarterly timestamps (:15/:30/:45) are floored to hourly before split/feature generation
  - All Phase 3 split outputs now have minute == 00 only
```

### 7.4 Region Imbalance Quantification (COMPLETED)

```
Raw distribution (canonical hourly unique):
  - Bhatagaon: 25.15%
  - AIIMS: 25.13%
  - SILTARA: 24.92%
  - IGKV: 24.80%
  Imbalance ratio: 1.01×

Post-sequence distribution:
  - AIIMS: 25.54%
  - IGKV: 25.16%
  - Bhatagaon: 24.78%
  - SILTARA: 24.51%
  Post-sequence imbalance ratio: 1.04×

Conclusion:
  - Severe imbalance claim (6.5×) is not observed in canonical hourly data.
  - Integrating quarterly data further improves parity and coverage.
  - Fairness controls remain active, but weighting intensity is very mild.

Region weights for training (calculated):
  - Bhatagaon: 1.009×
  - IGKV: 0.994×
  - AIIMS: 0.979×
  - SILTARA: 1.020×
```

---

## 8. Key Decisions Summary Table

| # | Decision | Choice | Status | Phase | Notes |
|---|----------|--------|--------|-------|-------|
| 4.1 | Model architecture | Hierarchical LSTM | ✓ APPROVED | 2 | See ARCHITECTURE.md |
| 4.2 | Output type | Quantile regression | ✓ APPROVED | 2 | 4 quantiles per horizon |
| 4.3 | Preprocessing | Dual pipelines | ✓ APPROVED | 3 | LSTM vs XGB optimized |
| 4.4 | Loss function | Region-weighted multi-quantile | ✓ APPROVED | 5 | Pinball loss per quantile |
| 4.5 | Sequence length | Horizon-specific calibrated map | ✓ APPROVED | 5 | h1=168, h24=336, h168=720 |
| 4.6 | Weather features | Lagged (t-24 to t-1) | ✓ APPROVED | 3 | No future leakage |
| 4.7 | Baseline model | XGBoost | ✓ APPROVED | 5 | Fair LSTM comparison |
| 4.8 | Train/val/test split | 70/15/15 time-based | ✓ APPROVED | 4 | Walk-forward after Phase 4 |
| 4.9 | Outlier handling | Asymmetric LSTM/XGB | ✓ APPROVED | 3 | LSTM keeps, XGB removes |
| 4.10 | Batch sampling | Stratified per region | ✓ APPROVED | 5 | Fairness enforcement |
| 4.13 | LSTM training mode | Separated per horizon | ✓ APPROVED | 5 | 12 models (`4×3`) |
| 4.14 | Horizon scope | h1/h24/h168 | ✓ APPROVED | 5 | Replaced h672 |
| 4.15 | Quantile calibration | Both LSTM + XGB | ✓ APPROVED | 6 | Validation bias calibration |
| 4.16 | Interval priority | 95% CI focus | ✓ APPROVED | 6 | p05-p95 primary KPI |
| 5.1 | Evaluation metrics | CRPS + PIT + RMSE | ✓ APPROVED | 6 | Quantile + accuracy validation |
| 5.2 | Region fairness | Per-region + weighted | ✓ APPROVED | 6 | Verify imbalance mitigation |

---

## 9. Lessons Learned & Evolution

(Section to be updated as project progresses)

### 9.1 From Phase 1 (Data Investigation)
- **Finding 1**: Bhatagaon Sept 2025 PM2.5 extremes (>500, peak 5904.78) are sensor-error clusters, not regional events.
- **Finding 2**: Integrating quarterly files recovers additional coverage (3,690 extra hourly timestamps; 5,689 PM2.5 fills), with post-sequence imbalance ~1.04×.
- **Impact**: Phase 3 must explicitly remove confirmed sensor-error clusters, keep hourly+quarterly canonical merge, and use updated mild region weights.

### 9.2 From Phase 4 (Preprocessing)
- **Finding 1**: Phase 3 preprocessing now enforces one canonical source contract (`hourly_preferred_quarterly_fill`) with explicit hourly flooring for both hourly and quarterly data.
- **Finding 2**: Leakage controls are explicit in metadata: weather inputs are lagged-only for LSTM and shifted/lagged for XGB.
- **Impact**: Split outputs are minute-00 aligned and leakage policy is auditable/reproducible across both model families.

### 9.3 From Phase 5 (Training)
- **Finding 1**: Phase 5 scripts (`train_lstm.py`, `train_xgb.py`) are operational and produce model + summary artifacts.
- **Finding 2**: LSTM now trains separated models per `(pollutant,horizon)` on scope `h1,h24,h168`, with configurable `--seq-len-map`.
- **Impact**: Better horizon isolation, per-horizon early stopping/checkpointing, and faster diagnosis for quality tuning.

### 9.4 From Phase 6 (Evaluation)
- **Finding 1**: `scripts/evaluate.py` now computes RMSE/MAE/R2, CRPS-approx, coverage tails, PIT KS test, and per-region fairness, then saves `models/evaluation_summary.json`.
- **Finding 2**: Phase 6 visual outputs are generated in `visualizations/phase_6_evaluation/` (8 plots), but metrics indicate calibration/fairness gaps in this checkpoint.
- **Impact**: Evaluation pipeline is complete and reproducible; model quality work now shifts to full-horizon training and calibration/fairness tuning.

### 9.5 Pipeline Evolution (Inference + Fair Benchmarking)
- **Finding 1**: Product inference requirement is horizon-first and region-specific: one region + one horizon should return all pollutants with prediction intervals.
- **Finding 2**: Unfair row alignment can over/understate LSTM-vs-XGB gaps; strict fair benchmarking must use common `(region, timestamp)` rows.
- **Impact**: Added `scripts/predict.py` (single-region, selected-horizon multi-pollutant quantile inference) and `evaluate.py --fair-intersection` output (`models/fair_benchmark_summary.json`).

### 9.6 Compute Placement Optimization
- **Finding 1**: Remote environment cost profile favors GPU-heavy execution, and CPU jobs should be minimized.
- **Finding 2**: XGBoost quantile training can run on GPU (`device='cuda'`) with histogram tree method in the current stack.
- **Impact**: `train_xgb.py` now supports explicit device selection (`--device auto|cuda|cpu`) with CUDA availability smoke-test; orchestrator forwards `--device` to both LSTM and XGB training/evaluation.

---

## 10. Future Work & Open Questions

1. **Walk-forward cross-validation** (Phase 5+): Implement 5-fold rolling windows for robustness
2. **Deployment API** (Phase 7): FastAPI server for real-time predictions with monitoring
3. **Retraining strategy** (Phase 7): How often to retrain? Drift detection?
4. **Alternative architectures** (Phase 8): Transformer-based models? Mixture of experts per region?
5. **Extreme event handling** (Phase 8): Better detection of real pollution spikes vs sensor errors?

---

## Document Maintenance Guidelines

### When to Update This File
- **Decision made**: Add section under 4.X with full rationale
- **Phase completed**: Update corresponding results section
- **Finding discovered**: Add to "Lessons Learned" section
- **Performance number obtained**: Update results table immediately

### Format for New Decision
```markdown
### Decision 4.X: [Title]

| Aspect | Details |
|--------|---------|
| **Options** | (A) Option A |
| | (B) Option B |
| **Chosen** | (A) |
| **Rationale** | - Reason 1 |
| | - Reason 2 |
| **Implementation** | [Code/config details] |
| **Evidence** | [Data supporting choice or "none yet"] |
| **Status** | ✓ APPROVED / ◯ TODO / ✗ REJECTED |
```

### Approval Status Codes
- ✓ **APPROVED**: Decision made, ready for implementation
- ◯ **TODO**: Decision pending, waiting for investigation/test
- ◯ **IN PROGRESS**: Currently being implemented/tested
- ✗ **REJECTED**: Alternative chosen instead
- ⚠ **BLOCKED**: Needs external input or resolution

---

## 11. Sign-Off & Accountability

**Document Owner**: [To be assigned]
**Last Updated**: [Auto-fill on save]
**Review Frequency**: After each phase completion
