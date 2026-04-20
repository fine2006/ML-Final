# AGENTS.md - Build Phase: Complete Documentation Index & Instructions

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

## Project Overview

**Goal**: Build a production-ready multi-horizon quantile regression forecasting system for air pollution (PM2.5, PM10, NO2, O3) across 4 Raipur regions using hierarchical LSTM with XGB baseline comparison.

**Current State**: Previous implementation discarded. Starting fresh from Phase 1.

**Status**: ✓ Documentation complete (3,231 lines) | ⏳ Code implementation pending

---

## 🎯 Core Principles: Robustness, Reproducibility, Transparency

### Robustness Goal
Build a **production-ready, robust pipeline** where:
- Every decision is justified and traceable (not ad-hoc)
- All hyperparameters are data-driven or explicitly documented
- Code handles edge cases and fails gracefully with clear error messages
- Performance is validated across multiple horizons and regions

### Reproducibility Goal
Pipeline must be **fully reproducible**:
- Fixed random seeds (Python, NumPy, PyTorch)
- Exact library versions pinned (see REPRODUCIBILITY.md)
- Preprocessing outputs versioned and saved
- Training logs capture all hyperparameters and convergence info
- Any analysis can be re-run with identical results

### Transparency Requirement
**Agent must be explicit about ANY deviations**:
- If an implementation is too complex → document why, propose simplification, update DECISIONS.md
- If a shortcut is taken → explain trade-off, update DECISIONS.md
- If a design assumption fails → document failure, try alternative, update DECISIONS.md
- **Every code change must have corresponding entry in DECISIONS.md**

### Source of Truth: DECISIONS.md
**BINDING CONTRACT**: The actual implementation **MUST NOT differ** from DECISIONS.md
- Every architectural decision in DECISIONS.md is law (section 4.X)
- If code needs to deviate, DECISIONS.md must be updated FIRST
- If DECISIONS.md is incomplete, add missing detail BEFORE implementing
- At end of project, DECISIONS.md = accurate description of what was built

---

## Documentation Structure

This is a **unified index** combining all build instructions. Navigation by section:

- **Section 1**: This file (commands, phases, critical notes)
- **Section 2**: Linked docs (ARCHITECTURE.md, PREPROCESSING_STRATEGY.md, etc.)

### Core Documentation Files

| File | Purpose | Lines |
|------|---------|-------|
| **AGENTS.md** (this file) | Build instructions + navigation index | 520 |
| **ARCHITECTURE.md** | LSTM technical blueprint | 575 |
| **PREPROCESSING_STRATEGY.md** | Dual data pipelines (LSTM vs XGB) | 766 |
| **DATA_INVESTIGATION.md** | Phase 1 analysis framework | 489 |
| **DECISIONS.md** | Living final report (decisions + results) | 625 |
| **REPRODUCIBILITY.md** | Environment setup, seeding, versioning | 392 |
| **VISUALIZATIONS.md** | All plot specifications & outputs | 635 |
| **IMPLEMENTATION_CONTRACT.md** | Binding agreement: code must match DECISIONS.md | 325 |
| **GETTING_STARTED.md** | First 30 minutes startup checklist | 420 |
| **DATA_LOADING.md** | Excel file parsing guide + code examples | 520 |
| **SCRIPT_TEMPLATES.md** | Boilerplate for all pipeline scripts | 680 |
| **PROJECT_STRUCTURE.md** | Directory layout with file purposes | 610 |
| **PIPELINE.md** | ML pipeline architecture (single-target, multi-pollutant context, reproducible extension path) | NEW |

**Total**: 5,953 lines of detailed specifications (95% complete)

---

## Entry Point & Commands

```bash
# Install dependencies (from pyproject.toml)
uv sync

# Run full pipeline (once implementation complete)
python main.py

# Individual components
python scripts/data_investigation.py      # Phase 1: Analyze extreme outliers
python scripts/preprocess_lstm.py         # Phase 3: LSTM preprocessing
python scripts/preprocess_xgb.py          # Phase 3: XGB preprocessing
python scripts/train_lstm.py              # Phase 5: Train hierarchical LSTM
python scripts/train_xgb.py               # Phase 5: Train XGB baseline
python scripts/evaluate.py                # Phase 6: Compare models, metrics
python scripts/predict.py                 # Inference: one region + one horizon -> all pollutants
```

---

## 6 Build Phases (Execution Order)

### Phase 1: Data Investigation (IMMEDIATE)

**Objective**: Understand extreme data quality issues before building models

**Location**: `scripts/data_investigation.py`

**Outputs**: `DATA_INVESTIGATION.md` (filled with findings)

**Tasks**:

#### 1.1 Bhatagaon September 2025 Spike Analysis
- **Problem**: PM2.5 values >500 µg/m³ in Sept 2025
- **Determine**: Real pollution event (gradual ramp-up/decay) vs sensor error (sudden spike)?
- **Procedure**: See DATA_INVESTIGATION.md section 1.2
- **Decision Framework**: 
  - Gradual rise + decay + weather correlation → KEEP
  - Single-point spike + no weather support → REMOVE
  - Uncertain → Document for manual review

#### 1.2 Data Loss Root Cause Analysis
- **Problem**: Quantify root-cause attrition in canonical hourly pipeline (125,017 baseline rows)
- **Determine**: Where is loss coming from?
  - Missing data gaps?
  - Outlier removal?
  - Temporal gaps breaking sequences?
- **Procedure**: See DATA_INVESTIGATION.md section 2.2
- **Outcome**: Per-region loss breakdown + recovery opportunities

#### 1.3 Region Imbalance Quantification
- **Problem**: Validate region imbalance after canonical merge + dedup
- **Determine**: Is this inherent or fixable?
- **Procedure**: See DATA_INVESTIGATION.md section 3.2
- **Outcome**: Imbalance characterization + training weights calculation

**Update After Phase 1**:
- Fill DATA_INVESTIGATION.md sections 1-3 with findings
- Update DECISIONS.md section 7 with results
- Set region weights for Phase 5 training

---

### Phase 2: Architecture & Design (COMPLETE - DOCUMENTED)

✓ Already designed. See ARCHITECTURE.md for full specifications.

**Key Design Points**:
- Hierarchical quantile LSTM per `(pollutant, horizon)` (shared architecture pattern, horizon-separated training)
- Quantile regression (p5, p50, p95, p99 per horizon)
- Active horizons: `h1`, `h24`, `h168`
- BiLSTM (2 layers, 128 hidden, bidirectional, dropout 0.3)
- Region-weighted multi-quantile pinball loss
- XGB baseline for fair comparison

---

### Phase 3: Preprocessing Strategy (COMPLETE - DOCUMENTED)

✓ Already designed. See PREPROCESSING_STRATEGY.md for full specifications.

**Key Strategy**: Dual pipelines (opposite philosophies)

#### LSTM Pipeline (sequence-focused)
- Keep outliers (PM2.5 >500 OK if pattern real)
- Light imputation (interpolate <6h gaps, break >6h)
- RobustScaler (median/IQR, preserves outlier info)
- Minimal features (15)
- Data retention: ~73%

#### XGB Pipeline (feature-focused)
- Remove outliers with pollutant caps (PM2.5 >300, PM10 >600, NO2 >250, O3 >150)
- Aggressive imputation (all gaps + 6 missingness features)
- Rich features (55-60)
- No scaling (scale-invariant)
- Data retention: ~70%

**Both use**: 70/15/15 time-based split (train 2022-Q1 2024 | val Q1-Q2 2024 | test Q2 2024-present)

---

### Phase 4: Implementation Workflow (CODE ORGANIZATION)

```
scripts/
├── data_investigation.py        # Phase 1: Analyze outliers, data loss, imbalance
├── preprocess_common.py         # Phase 3: Shared canonical merge + sanitization
├── preprocess_lstm.py           # Phase 3: LSTM-specific preprocessing
├── preprocess_xgb.py            # Phase 3: XGB-specific preprocessing
├── train_lstm.py                # Phase 5: Hierarchical quantile LSTM training
├── train_xgb.py                 # Phase 5: XGB baseline training
├── evaluate.py                  # Phase 6: Compare models, visualizations
└── predict.py                   # Inference interface (single region, selected horizon)

main.py                          # Entry point: orchestrate full pipeline

data/                            # Preprocessed data storage
models/                          # Saved trained models
visualizations/                  # Output plots & results
```

---

### Phase 5: Training (LSTM + XGB)

**Objective**: Train hierarchical LSTM with quantile regression + XGB baseline

**Location**: `scripts/train_lstm.py` and `scripts/train_xgb.py`

**LSTM Training** (`train_lstm.py`):
- Input: LSTM-ready data from Phase 3
- Output: Trained model → `models/lstm_quantile_[pollutant]_h[horizon].pt`
- Output: Test prediction bundle → `models/lstm_predictions_[pollutant]_h[horizon].npz`
- Architecture: See ARCHITECTURE.md section 2
- Loss: Multi-quantile pinball with region weights (see ARCHITECTURE.md 3.1-3.2)
- Optimizer: Adam (lr=0.001)
- Epochs: 100 with early stopping (patience=10, monitor=val CRPS)
- Batch size: 32 (stratified: ~8 per region)
- For all 4 pollutants: PM2.5, PM10, NO2, O3 (independent per-horizon models)

**XGB Training** (`train_xgb.py`):
- Input: XGB-ready data from Phase 3
- Output: Trained models (12 per pollutant in active scope: 3 horizons × 4 quantiles)
- Models: Separate per quantile (p5, p50, p95, p99)
- Hyperparameters: See ARCHITECTURE.md section 6.2
- For all 4 pollutants: PM2.5, PM10, NO2, O3

**Create After Phase 5**:
- TRAINING_LOG.md (convergence curves, epoch logs, debug notes)
- Update DECISIONS.md section 6 with training results

---

### Phase 6: Evaluation & Comparison

**Objective**: Validate model quality, compare LSTM vs XGB, measure fairness

**Location**: `scripts/evaluate.py`

**Metrics per Horizon**:
- RMSE (accuracy)
- CRPS (calibration + sharpness)
- Coverage (% actuals in p5-p95, target 90%)
- PIT KS stat (calibration test)

---

### h168 Forensic Refinement Pipeline

**Phase h168-1**: Optuna Hyperparameter Tuning (dual-GPU)
- GPU 0: PM25 (15 trials)
- GPU 1: PM10 (15 trials)
- Output: optuna_h168_best_configs.json

**Phase h168-2**: Walk-Forward Cross-Validation (dual-GPU)
- GPU 0: PM25 (5-fold WFCV)
- GPU 1: PM10 (5-fold WFCV)
- Output: wfcv_h168_results.json

**Phase h168-3**: Aggregation + NO2/O3 Holdout Eval
- Trains NO2 (2 configs) + O3 pilot models
- Holdout eval on test set
- Output: aggregated_h168_results.json

**Per-Region Fairness**:
- RMSE per region (target: max ratio <1.5×)
- Verify region weighting worked

**Expected Results**:
- h1/h24/h168 quality varies by pollutant and uncertainty criterion (RMSE, CRPS, coverage, PIT)
- PM family tends to be near-utility at h168 under RMSE/mean gates; gas family requires rescue tuning
- Use tiered operational gates in `evaluate.py` outputs for pass/fail decisions

**Create After Phase 6**:
- EVALUATION_RESULTS.md (final metrics, plots, fairness analysis)
- Update DECISIONS.md section 6 with results & section 9 with lessons learned

---

## Quick Navigation by Task

### "I'm starting Phase 1 (Data Investigation)"
1. Read this file (sections above)
2. Read **DATA_INVESTIGATION.md** (investigation procedures, section 2-4)
3. Implement `scripts/data_investigation.py` following procedures
4. Fill in DATA_INVESTIGATION.md sections 1-3 with findings
5. Update DECISIONS.md section 7 with results

### "I'm implementing Phase 3 (Preprocessing)"
1. Read **PREPROCESSING_STRATEGY.md** (full dual pipeline design)
2. Read section "Phase 3" above
3. Implement `scripts/preprocess_lstm.py` (LSTM pipeline)
4. Implement `scripts/preprocess_xgb.py` (XGB pipeline)
5. Verify QA checks in PREPROCESSING_STRATEGY.md section 8

### "I'm implementing Phase 5 (Training)"
1. Read **ARCHITECTURE.md** sections 1-6 (architecture + loss + training)
2. Read section "Phase 5" above
3. Implement `scripts/train_lstm.py` (hierarchical LSTM + quantile regression)
4. Implement `scripts/train_xgb.py` (XGB baseline)
5. Create TRAINING_LOG.md with logs
6. Update DECISIONS.md section 6 with results

### "I'm implementing Phase 6 (Evaluation)"
1. Read **ARCHITECTURE.md** section 7 (evaluation metrics)
2. Read section "Phase 6" above
3. Implement `scripts/evaluate.py` (CRPS, PIT, coverage, per-region metrics)
4. Create EVALUATION_RESULTS.md with results
5. Update DECISIONS.md sections 6 & 9

### "I need to understand a design decision"
1. Go to **DECISIONS.md** section 4.X (decision table)
2. Read Options/Chosen/Rationale/Evidence
3. Follow references to detailed docs (ARCHITECTURE.md, PREPROCESSING_STRATEGY.md)

### "I need pipeline architecture guidance (not developer docs)"
1. Read **PIPELINE.md** first (single-target LSTM pipeline contract)
2. Follow the stage order in PIPELINE.md (canonical data -> quality gate -> preprocessing -> training -> calibration -> evaluation)
3. For new pollutant onboarding (e.g., SOx/NOx), use PIPELINE.md section "Extending to New Pollutants"

---

## Critical Implementation Notes

### ⚠️ NO FUTURE DATA LEAKAGE
This is the most critical requirement. Verify explicitly:

**Check before training**:
```python
# 1. All features use indices ≤ t (current time)
assert all(feature.lag >= 0 for feature in features), "Found negative lag (future data)!"

# 2. Feature availability contract (lag applied once)
# If source table is already delayed/finalized by >=1h, do NOT add extra lag transforms.
# Features must still be past-only by window construction (indices <= anchor-1).
assert max(feature_timestamps) <= anchor_time - pd.Timedelta(hours=1), "Feature row leaks anchor/future time!"

# 3. Target uses t+horizon (never t+horizon+1 or later)
assert target_time == t + horizon, "Target is in the future!"

# 4. Manual spot check: inspect 5 random samples
for i in [0, 100, 500, 1000, 5000]:
    sample = data[i]
    assert all(f.timestamp <= sample.target_time for f in sample.features), f"Sample {i} has future features!"
```

**Test set is held out**:
```python
assert train_data.timestamp.max() < val_data.timestamp.min()
assert val_data.timestamp.max() < test_data.timestamp.min()
# Never use test data during training or validation
```

### ⚠️ OUTLIER HANDLING MUST MATCH PHILOSOPHY
- **LSTM pipeline**: Keep ALL outliers (PM2.5 >500 OK if pattern real)
  - Only remove physically impossible (negatives, -2B)
  - Reason: Attention learns to weight extreme z-scores
- **XGB pipeline**: Remove pollutant-specific extremes via fixed caps
  - PM2.5 >300, PM10 >600, NO2 >250, O3 >150
  - Reason: Extreme values distort tree splits for entire dataset

**Don't mix**: If LSTM removes outliers, it won't learn patterns. If XGB keeps outliers, trees split on noise.

### ⚠️ REGION WEIGHTING MUST BE APPLIED
```python
# Loss function must weight by region
weight_r = (1/4) / fraction_r

# Example:
# Canonical pipeline weights are mild and close to uniform.
# Use values from Phase 1 outputs / metadata (example order):
# AIIMS ~0.979×, IGKV ~0.994×, Bhatagaon ~1.009×, SILTARA ~1.020×

# Validation: per-region val loss should be similar (not 5× different)
```

### ⚠️ SEQUENCE LENGTH MUST RESPECT HORIZON
```python
# NOT fixed: Don't use same seq_len for all horizons
seq_len = 24  # ✗ WRONG for mixed horizons (under-context for h168)

# YES horizon-calibrated map for active scope
seq_len_map = {1: 168, 24: 336, 168: 720}  # ✓ CURRENT DEFAULT
```

### ⚠️ TIME-BASED SPLIT (NO SHUFFLING)
```python
# NEVER shuffle time series data
X_train, X_test = train_test_split(data, test_size=0.2, shuffle=True)  # ✗ WRONG

# YES use explicit date ranges
train = data[data.timestamp < '2024-04-01']  # ✓ CORRECT
val = data[(data.timestamp >= '2024-04-01') & (data.timestamp < '2024-07-01')]
test = data[data.timestamp >= '2024-07-01']
```

---

## Troubleshooting & Common Pitfalls

### Pitfall 1: Overfitting on Majority Region (IGKV)
**Symptom**: IGKV metrics great, Bhatagaon metrics terrible

**Root**: Region weighting not applied or stratified sampling broken

**Fix**:
1. Check loss function includes region weights
2. Verify DataLoader uses stratified sampling (~8 per region per batch)
3. Monitor per-region val loss: should be similar, not 5× different

### Pitfall 2: LSTM Predicting Mean (No Variation)
**Symptom**: LSTM predictions = constant mean value

**Root**: Outlier handling too aggressive; sequences broken; attention not learning

**Fix**:
1. Verify LSTM pipeline keeps PM2.5 >500 (if pattern real)
2. Check sequences don't cross gaps >6h (breaks continuity)
3. Add debug: print attention weights (should vary per sample)

### Pitfall 3: Quantile Predictions Not Calibrated
**Symptom**: p5 captures 20% of actuals (should be 5%); p95 captures 60% (should be 5%)

**Root**: Loss function not converged; wrong quantile level; model not learning asymmetry

**Fix**:
1. Check pinball loss has correct slopes: (q - I(y<ŷ))
2. Verify quantile parameter matches quantile level (0.05 for p5, etc.)
3. Extend training epochs; may need more data

### Pitfall 4: Data Leakage via Weather Features
**Symptom**: h168 performs unrealistically well

**Root**: Weather features include t or future (today's weather predicts tomorrow)

**Fix**:
1. Audit feature windowing; ensure features never include anchor/future rows
2. Apply data latency exactly once (source delay OR feature lag, not both)
3. Never use: any feature at t+1 or later
3. Spot-check 5 samples manually

### Pitfall 5: Train/Val/Test Mixed
**Symptom**: Test metrics better than val (impossible if proper split)

**Root**: Shuffled timestamps; future data in training

**Fix**:
1. Use explicit date ranges; never shuffle
2. Verify: train.max() < val.min() < test.min()
3. Check: test data never seen during training loop

---

## Success Criteria

✅ **Phase 1 Complete**:
- Bhatagaon spike classified (real event vs error)
- Data loss root causes identified
- Region imbalance quantified + weights calculated
- DATA_INVESTIGATION.md filled with findings

✅ **Phase 3 Complete**:
- preprocess_lstm.py implemented & tested
- preprocess_xgb.py implemented & tested
- 70/15/15 splits created
- QA checks pass (no data leakage, proper ordering)

✅ **Phase 5 Complete**:
- train_lstm.py trained (all 4 pollutants)
- train_xgb.py trained (all 4 pollutants, 12 models each in active scope)
- Models saved to `models/`
- TRAINING_LOG.md created

✅ **Phase 6 Complete**:
- LSTM produces calibrated quantiles (PIT test passes)
- XGB baseline trained (RMSE baseline)
- Per-horizon metrics reported for active scope (`h1`,`h24`,`h168`) with gate-based interpretation
- Per-region fairness verified (max ratio <1.5×)
- EVALUATION_RESULTS.md created
- DECISIONS.md updated with all results

---

## Dependencies

**Required** (add to pyproject.toml):
- **Data**: pandas, numpy, scipy, openpyxl
- **LSTM**: torch, pytorch-lightning (or manual torch training)
- **XGB**: xgboost
- **Evaluation**: scikit-learn, optuna (optional for hyperopt)
- **Viz**: matplotlib, seaborn
- **Utils**: tqdm, pyyaml, joblib

```bash
uv sync  # Install all dependencies
```

---

## Related Documentation

### Detailed Design Specs
- **ARCHITECTURE.md**: Full LSTM architecture, loss function, XGB design, evaluation metrics
- **PREPROCESSING_STRATEGY.md**: Dual pipelines, feature engineering, data retention analysis
- **DATA_INVESTIGATION.md**: Phase 1 analysis procedures & templates

### Results & Tracking
- **DECISIONS.md**: Living report of all architectural decisions + performance results
- **TRAINING_LOG.md** (to create): Epoch logs, convergence curves, debug notes
- **EVALUATION_RESULTS.md** (to create): Final metrics, per-horizon comparison, per-region fairness

---

## Implementation Checklist

Before declaring a phase complete:

- [ ] Code implemented per specifications in relevant doc (ARCHITECTURE.md, PREPROCESSING_STRATEGY.md, etc.)
- [ ] **NO FUTURE DATA LEAKAGE** verified (all features lagged, explicit spot-check 5 samples)
- [ ] Region weighting applied & validated (per-region metrics similar)
- [ ] Test set held out (never used during training)
- [ ] DECISIONS.md updated with results
- [ ] Output documentation created (DATA_INVESTIGATION.md, TRAINING_LOG.md, EVALUATION_RESULTS.md)

---

## Starting Point

1. **First**: Read this file (you're reading it!)
2. **Next**: Read relevant detailed doc (ARCHITECTURE.md or PREPROCESSING_STRATEGY.md or DATA_INVESTIGATION.md)
3. **Then**: Implement the Phase
4. **Finally**: Update DECISIONS.md with results

**Ready to start Phase 1? → Read DATA_INVESTIGATION.md section 2-4 (procedures)**
