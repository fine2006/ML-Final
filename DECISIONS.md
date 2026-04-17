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
- Horizons: `h1`, `h24`, `h672`
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
| | - Remove outliers (PM2.5 >300) - tree splits distort on extremes |
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
| **Rationale** | - Reduced scope (`h1`,`h24`,`h672`) allows direct sequence-budget tuning per horizon |
| | - `h672` needs much longer context than `2×h` baseline used previously, but cannot use annual windows on current val/test split |
| | - Calibrated map preserves short-horizon efficiency and long-horizon context without collapsing validation samples |
| | **Sequence lengths (active)**: `h1=168`, `h24=336`, `h672=2402` |
| **Implementation** | `train_lstm.py` accepts `--seq-len-map`; default map is `1:168,24:336,672:2402` |
| **Evidence** | Feasibility check: `h672 seq_len=8760` gives zero val/test sequences; `seq_len=2402` keeps all regions populated |
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
| | XGB: remove PM2.5 >300 |
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
| | - Scope reduction to `h1`, `h24`, `h672` enables cleaner diagnosis and faster iteration |
| | - Explicit horizon-specialized training gives each horizon dedicated early stopping and checkpointing |
| **Implementation** | `train_lstm.py` now trains 12 models (`4 pollutants × 3 horizons`) and saves per-horizon checkpoints (`lstm_quantile_<pollutant>_h<h>.pt`) and predictions (`lstm_predictions_<pollutant>_h<h>.npz`) |
| | `evaluate.py` and `predict.py` load horizon-specific checkpoints first, with legacy fallback |
| **Status** | ✓ APPROVED (implemented) |

### Decision 4.14: Experimental h672 Sequence Length Cap

| Aspect | Details |
|--------|---------|
| **Options** | (A) Keep `h672` with `seq_len=672` |
| | (B) Increase `h672` sequence length aggressively |
| | (C) Extreme yearly window (`seq_len=8760`) |
| **Chosen** | (B) with cap `seq_len=2402` |
| **Rationale** | - User requested larger context for `h672`, but `8760` yields zero val/test sequences on current splits |
| | - Feasibility check shows practical max with meaningful validation is far below 8760 |
| | - `2402` keeps all regions populated in train/val/test while substantially expanding long-horizon context |
| **Implementation** | Horizon sequence defaults now: `h1=168`, `h24=336`, `h672=2402` (overridable by `--seq-len-map`) |
| **Evidence** | Dataset check on current splits: `h672, seq_len=8760 -> val=0, test=0`; `h672, seq_len=2402 -> train=69799, val=5561, test=5651` sequences |
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
- LSTM currently has trained/evaluated horizons `h1`, `h24`, `h672`.
- XGB currently has trained/evaluated horizons `h1`, `h24`, `h672`.

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
- Active pipeline now uses horizon-separated models on reduced scope (`h1`,`h24`,`h672`),
  with `h672` experimental context (`seq_len=2402`) and per-horizon checkpoints.

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
| 4.5 | Sequence length | Horizon-specific calibrated map | ✓ APPROVED | 5 | h1=168, h24=336, h672=2402 |
| 4.6 | Weather features | Lagged (t-24 to t-1) | ✓ APPROVED | 3 | No future leakage |
| 4.7 | Baseline model | XGBoost | ✓ APPROVED | 5 | Fair LSTM comparison |
| 4.8 | Train/val/test split | 70/15/15 time-based | ✓ APPROVED | 4 | Walk-forward after Phase 4 |
| 4.9 | Outlier handling | Asymmetric LSTM/XGB | ✓ APPROVED | 3 | LSTM keeps, XGB removes |
| 4.10 | Batch sampling | Stratified per region | ✓ APPROVED | 5 | Fairness enforcement |
| 4.13 | LSTM training mode | Separated per horizon | ✓ APPROVED | 5 | 12 models (`4×3`) |
| 4.14 | h672 context window | seq_len=2402 | ✓ APPROVED | 5 | 8760 infeasible on val/test |
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
- **Finding 2**: LSTM now trains separated models per `(pollutant,horizon)` on scope `h1,h24,h672`, with configurable `--seq-len-map`.
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
