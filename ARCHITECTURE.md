# ARCHITECTURE.md - Hierarchical Multi-Horizon Quantile Regression LSTM

## ⚠️ CRITICAL: GIT INSTRUCTIONS
**DO NOT RUN ANY GIT COMMANDS** (git add, git commit, git push, or any git modification commands).
See DECISIONS.md for full git warning. Work directly on files; user handles git operations.

## Overview
This document describes the complete architecture for multi-horizon air pollution forecasting using a hierarchical LSTM with quantile regression outputs.

**Note**: This is a FRESH IMPLEMENTATION. Disregard all previous code in `scripts/`, `main.py`, `models/`. Treat as starting from zero.

---

## 1. Problem Formulation

### 1.1 Multi-Horizon Quantile Regression
**Objective**: Predict probability distributions (not point estimates) of pollution at 5 horizons

**Targets**: 
- Pollutants: PM2.5, PM10, NO2, O3 (independent models per pollutant)
- Quantiles: p5, p50, p95, p99 (4 quantile levels)
- Horizons: t+1h, t+12h, t+24h, t+7d, t+28d (5 time horizons)
- **Total outputs per model**: 5 horizons × 4 quantiles = 20 predictions

**Why quantiles, not point predictions?**
- Point estimates (MAE/RMSE) ignore uncertainty
- Quantiles provide prediction intervals: "pollution will be between X and Y with 90% confidence"
- Enables risk-aware decision making: high p99 triggers alerts, low p5 allows green flag

### 1.2 Why Hierarchical (not separate models)?
- **Advantage**: Single shared backbone learns general temporal patterns; horizon-specific heads specialize
- **Alternative rejected**: Separate models per horizon = 5×4×4=80 models, maintenance nightmare, data fragmentation
- **Benefit**: Shared BiLSTM backbone captures global patterns (e.g., "PM2.5 increases at night"), horizon heads learn horizon-specific variations

---

## 2. Hierarchical LSTM Architecture

### 2.1 Input Layer
```
Input shape: (batch_size, seq_len_adaptive, n_features)

Where:
- batch_size = 32 (stratified: ~8 samples per region)
- seq_len_adaptive = 2 × horizon (see 2.2)
- n_features = 15 (minimal LSTM features: see PREPROCESSING_STRATEGY.md)
  * raw_pm25, raw_pm10, raw_no2, raw_o3
  * temperature, humidity, wind_speed, wind_direction (lagged t-24 to t-1)
  * hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos
  * region_encoding (0-3)
  * is_outlier_flag
```

### 2.2 Adaptive Sequence Length per Horizon
**Rationale**: Longer horizons need longer lookback windows to capture patterns at that timescale

```
Horizon        seq_len     Reason
------         -------     ------
t+1h           2h          Very short-term, minimal context needed
t+12h          24h         Half-day to full day pattern
t+24h          48h         Full 2-day cycle for daily variability
t+7d           336h        Full week (7 days) for weekly patterns
t+28d          672h        Full 4 weeks for longer-term seasonal patterns

Formula: seq_len = max(2*horizon_hours, min_lookback)
Min lookback = 2 hours (prevent degenerate sequences)
```

### 2.3 Shared BiLSTM Backbone
```
Layer 1 (BiLSTM):
  - Input: (batch, seq_len_adaptive, 15 features)
  - Hidden dim: 128 per direction (256 bidirectional)
  - Output: (batch, seq_len_adaptive, 256)
  - Activation: tanh
  - Dropout: 0.3 (prevent overfitting on small Bhatagaon subset)

Layer 2 (BiLSTM):
  - Input: (batch, seq_len_adaptive, 256)
  - Hidden dim: 128 per direction (256 bidirectional)
  - Output: (batch, seq_len_adaptive, 256)
  - Final state: (batch, 256) after taking last timestep
  - Dropout: 0.3

Regularization:
  - L2 weight decay: 1e-4
  - Gradient clipping: max_norm=1.0 (prevent exploding gradients with extreme outliers)
```

**Why bidirectional?**
- Forward LSTM: past → future (realistic, causality preserved)
- Backward LSTM: future → past (helps learn symmetric patterns)
- Combined: 256-dim representation captures both directions

### 2.4 Horizon-Specific Attention Heads
```
Per horizon (5 separate attention modules):

Attention Mechanism:
  - Query: BiLSTM output (batch, 256)
  - Key/Value: BiLSTM sequence (batch, seq_len_adaptive, 256)
  - Attention weights: softmax(queries × keys^T / sqrt(256))
  - Context: sum(attention_weights × values)
  - Output: (batch, 256)

Rationale:
  - Each horizon attends to different parts of the sequence
  - t+1h attention focuses on immediate history (last 2h)
  - t+28d attention focuses on weekly/monthly patterns
  - Separate heads learn horizon-specific weighting
```

### 2.5 Quantile Regression Heads
```
Per quantile level (4 separate heads):

Head architecture per quantile q in {0.05, 0.50, 0.95, 0.99}:
  Input: (batch, 256) from attention context
  
  Dense 1:
    - Output dim: 128
    - Activation: ReLU
    - Dropout: 0.2
  
  Dense 2:
    - Output dim: 32
    - Activation: ReLU
  
  Dense 3:
    - Output dim: 5 (one prediction per horizon)
    - Activation: Linear (unconstrained predictions)
    - Output shape: (batch, 5)

Total output per pollutant:
  - 5 quantiles × 5 horizons = 25 values per sample
  - But: one attention head per horizon, so effectively (batch, 5) per quantile
  - Reshape to: (batch, 5 horizons, 4 quantiles)
```

**Why 4 separate quantile heads?**
- Each quantile has different loss function (pinball loss with different slopes)
- Separate heads allow different feature interactions per quantile
- p50 (median) learns typical behavior
- p5/p95 learn extreme behavior (more influenced by outliers)
- p99 learns rare extreme events

### 2.6 Full Forward Pass
```
Input: (batch, seq_len_adaptive, 15)
  ↓
BiLSTM Layer 1: (batch, seq_len_adaptive, 256)
  ↓
BiLSTM Layer 2: (batch, seq_len_adaptive, 256)
  ↓
Take final timestep: (batch, 256)
  ↓
[Horizon 1 Attention] → (batch, 256) → [Quantile 1-4 Heads] → (batch, 4)
[Horizon 2 Attention] → (batch, 256) → [Quantile 1-4 Heads] → (batch, 4)
[Horizon 3 Attention] → (batch, 256) → [Quantile 1-4 Heads] → (batch, 4)
[Horizon 4 Attention] → (batch, 256) → [Quantile 1-4 Heads] → (batch, 4)
[Horizon 5 Attention] → (batch, 256) → [Quantile 1-4 Heads] → (batch, 4)
  ↓
Stack outputs: (batch, 5 horizons, 4 quantiles) = (batch, 20)
```

---

## 3. Loss Function & Training

### 3.1 Multi-Quantile Pinball Loss
**Objective**: Minimize asymmetric loss per quantile, weighted by region

```
For quantile q in {0.05, 0.50, 0.95, 0.99}:

Pinball loss (single sample, single quantile):
  loss_q = (q - I(y < ŷ_q)) * (y - ŷ_q)
  where:
    I(y < ŷ_q) = 1 if actual y is below prediction, 0 else
    q = quantile level
  
Interpretation:
  - If y < ŷ_q (underpredict): loss = q * (ŷ_q - y) [steep for high q]
  - If y > ŷ_q (overpredict): loss = (1-q) * (y - ŷ_q) [steep for low q]
  - Result: p5 penalizes overprediction 20× more than underprediction
            p95 penalizes underprediction 20× more than overprediction

Multi-quantile loss (aggregate):
  loss = (1/4) * sum(loss_q for q in {0.05, 0.50, 0.95, 0.99})

Multi-horizon loss (aggregate):
  loss = (1/5) * sum(loss_horizon for horizon in {1h, 12h, 24h, 7d, 28d})

Region-weighted loss:
  loss_regional = sum(weight_r * loss for region r)
  where:
    weight_Bhatagaon = 6.5 (rare region, upweight)
    weight_IGKV = 1.0 (majority region, baseline)
    weight_AIIMS = 2.1 (imbalanced)
    weight_SILTARA = 1.7 (imbalanced)
```

**Why pinball loss?**
- Standard MSE treats all errors equally (not good for quantiles)
- Pinball loss asymmetric: penalizes wrong-side errors more
- p50 (median): symmetric loss, minimizes MAE
- p95: penalizes underprediction (miss high pollution) more
- Converges to true quantiles of data distribution

### 3.2 Region Weighting Strategy
**Problem**: Bhatagaon is only 8.8% of data, IGKV is 57%

**Solution**: Upweight rare regions in loss function

```
Per-region weight calculation:
  ideal_weight = 1.0 / (n_regions) = 0.25 (each region equally important)
  actual_fraction_r = n_samples_r / n_total
  weight_r = ideal_weight / actual_fraction_r
  
Examples:
  - IGKV: 57% of data → weight = 0.25 / 0.57 = 0.44 (downweight slightly)
  - Bhatagaon: 8.8% of data → weight = 0.25 / 0.088 = 2.84 (upweight heavily)
  
Capped to max 6.5× to prevent extreme distortion:
  - Bhatagaon actual would be 2.84×, cap at reasonable level
  - Prevents training on only Bhatagaon samples
```

### 3.3 Training Loop
```
Optimizer: Adam
  - Learning rate: 0.001
  - Betas: (0.9, 0.999)
  - Weight decay: 1e-4

Batch construction (stratified sampling):
  - Total batch size: 32
  - Per-region: ~8 samples (32/4 regions)
  - Ensures all regions represented equally each iteration
  - If region has <8 samples in epoch, resample with replacement

Epochs: 100 (with early stopping)
Early stopping:
  - Monitor: Validation CRPS (see Evaluation section)
  - Patience: 10 epochs (stop if CRPS doesn't improve for 10 epochs)
  - Restore: Best weights from lowest validation CRPS

Learning rate schedule (optional, if convergence stalls):
  - ReduceLROnPlateau: reduce LR by 0.5 if val CRPS doesn't improve for 5 epochs
```

---

## 4. Multi-Pollutant Support

### 4.1 Independent Models Per Pollutant
```
Pollutants: PM2.5, PM10, NO2, O3

Same LSTM architecture trained separately for each pollutant:
  - Shared backbone: BiLSTM (same weights don't transfer across pollutants)
  - Separate attention heads: horizon-specific per pollutant
  - Separate quantile heads: quantile-specific per pollutant
  - Separate loss function: region-weighted per pollutant

Rationale:
  - Pollutants have different seasonal patterns (O3 peaks in summer, PM2.5 in winter)
  - Different sources (PM2.5 traffic+dust, O3 photochemical)
  - Independent training ensures each model optimizes for its own distribution
  - Transfer learning not applicable (different physics)
```

### 4.2 Training Execution
```
Sequential training (parallelizable in practice):
  1. Train LSTM-PM2.5 (100 epochs, ~2-3 hours)
  2. Train LSTM-PM10 (100 epochs, ~2-3 hours)
  3. Train LSTM-NO2 (100 epochs, ~2-3 hours)
  4. Train LSTM-O3 (100 epochs, ~2-3 hours)
  
Total time: ~8-12 hours on GPU (or ~48 hours on CPU)

Alternative: Train in parallel (requires 4× GPU memory or 4 GPUs)
```

---

## 5. No Future Data Leakage Validation

### 5.1 Temporal Ordering Checks
**Critical**: Ensure all features at time t only use data ≤ t

```
Feature audit:

Allowed (history only):
  ✓ raw_pm25[t-1], raw_pm25[t-2], ..., raw_pm25[t-168]
  ✓ temperature[t-24] to temperature[t-1] (never t)
  ✓ hour_sin[t], day_sin[t] (calendar features, not measured data)

NOT allowed (future data):
  ✗ temperature[t] (today's actual weather not known at forecast time)
  ✗ temperature[t+1] (tomorrow's weather is future!)
  ✗ raw_pm25[t+horizon] (target, obviously!)

Implementation check:
  - Assert all features use indices ≤ t (current time)
  - Raise error if any feature has positive lag index
  - Validate 5 random samples manually
```

### 5.2 Weather Feature Handling
**Approach**: Use historical weather to predict future pollution

```
Scenario: Forecast pollution at t+24h
  - Observed PM2.5 at t: known
  - Weather at t: known (today's actual weather)
  - Weather at t+24: unknown (forecast's future)
  
Solution:
  - Use weather features [t-24, t-1] (yesterday's weather to today's)
  - Reason: Pollution inertia (yesterday's weather influenced current pollution)
  - Assumption: "Tomorrow's pollution depends on today's weather patterns"
  - Limitation: Cannot forecast 28d accurately (long-term weather unknown)

Alternative rejected:
  - Use future weather predictions (introduces weather model error)
  - Assume constant weather (too unrealistic)
  - No weather features (loses significant predictive signal)
```

### 5.3 Train/Val/Test Leakage Prevention
```
Time-based split (no shuffling):
  Train: 2022-01-01 to 2024-03-31 (indices [0, N_train))
  Val:   2024-04-01 to 2024-06-30 (indices [N_train, N_train+N_val))
  Test:  2024-07-01 to 2025-04-16 (indices [N_train+N_val, N_all))

No leakage rules:
  ✓ Training samples before validation split → OK
  ✓ Val samples before test split → OK
  ✗ Shuffling splits together → NOT OK (mixes temporal order)
  ✗ Using future val data to train → NOT OK
```

---

## 6. XGB Baseline Architecture

### 6.1 Rationale
**Why compare to XGB?**
- LSTM excels at long-horizon (7d/28d) temporal patterns
- XGB excels at capturing explicit feature interactions
- Fair comparison: both get region-appropriate feature engineering (see PREPROCESSING_STRATEGY.md)

**Expected results:**
- t+1h: XGB and LSTM similar (short memory, feature interactions dominate)
- t+12h, t+24h: LSTM slight advantage (temporal patterns emerge)
- t+7d, t+28d: LSTM large advantage (40-65% better, temporal patterns critical)

### 6.2 XGB Model Architecture
```
Separate models per quantile per horizon:
  - 5 horizons × 4 quantiles = 20 models
  - All 4 pollutants × 20 = 80 models total

Per-model configuration:
  max_depth: 7
  learning_rate: 0.1
  n_estimators: 500
  subsample: 0.8
  colsample_bytree: 0.8
  objective: reg:quantilehubererror
  quantile_alpha: 0.05 (or 0.50, 0.95, 0.99)
  min_child_weight: 1
  gamma: 0
```

### 6.3 XGB Input Features (~60 features)
See PREPROCESSING_STRATEGY.md for details:
- Raw pollution: pm25, pm10, no2, o3 (1h ago)
- Lags: 1h, 3h, 6h, 12h, 24h, 48h, 168h (7 lags)
- Rolling stats: mean/std at 6h/12h/24h windows (6 features)
- Weather lags: temp, humidity, wind, direction at t-24, t-12, t-6, t-1 (8 features)
- Missingness: hours_since_measurement, interpolation_distance, etc. (6 features)
- Time: hour, day_of_week, month, is_weekend, season (5 features)
- Total: ~50 explicit features vs LSTM's 15 implicit

---

## 7. Quantile Calibration & Evaluation

### 7.1 Coverage Analysis (Intuitive Check)
```
Empirical coverage:
  coverage_p5 = (y < ŷ_p5).mean()  # Should be ~5%
  coverage_p95 = (y > ŷ_p95).mean()  # Should be ~5%
  coverage_middle = (ŷ_p5 < y < ŷ_p95).mean()  # Should be ~90%

Interpretation:
  - If coverage_p5 = 15% (too high): model overpredicts low values
  - If coverage_p5 = 1% (too low): model underpredicts low values (dangerous!)
  
Visual: Calibration curve plot
  - X-axis: Theoretical quantile level (0 to 1)
  - Y-axis: Empirical quantile level (0 to 1)
  - Perfect: y=x diagonal
  - Overconfident (too narrow): curve below diagonal (real quantiles worse than predicted)
  - Underconfident (too wide): curve above diagonal (real quantiles better than predicted)
```

### 7.2 PIT Test (Rigorous Calibration)
```
Probability Integral Transform (PIT):
  1. For each test sample, compute CDF of prediction distribution:
     CDF(y) = P(Y ≤ y | features)
     Approximation: CDF(y) ≈ (1/4) * sum(I(ŷ_q ≤ y) for q in {p5, p50, p95, p99})
  
  2. If model is well-calibrated: PIT values should be uniformly distributed U(0,1)
  
  3. Test uniformity: Kolmogorov-Smirnov test
     KS stat = max_u |empirical_CDF(u) - u|
     If KS stat < critical_value: model is calibrated ✓
     If KS stat > critical_value: model is not calibrated ✗

Interpretation:
  - Calibrated: actual quantiles match predicted quantiles
  - Underconfident: predictions too wide (PIT bimodal, extreme values)
  - Overconfident: predictions too narrow (PIT concentrated at extremes)
```

### 7.3 CRPS (Continuous Ranked Probability Score)
```
CRPS measures both calibration and sharpness:

CRPS(F, y) = integral from -∞ to ∞ of (F(x) - I(x ≥ y))^2 dx

Approximation using 4 quantiles:
  CRPS ≈ (1/4) * |ŷ_p5 - y| + (1/4) * |ŷ_p50 - y| + 
         (1/4) * |ŷ_p95 - y| + (1/4) * |ŷ_p99 - y|

Interpretation:
  - Lower CRPS = better model (balances calibration + sharpness)
  - CRPS = MAE (point estimate) + penalty for uncertainty
  - Example: 
    - LSTM CRPS = 12 µg/m³ (narrow intervals, good calibration)
    - XGB CRPS = 18 µg/m³ (wider intervals, overconfident)

Advantage over MSE:
  - MSE ignores intervals: gives credit for lucky point estimates
  - CRPS rewards confident correct predictions, penalizes confident wrong predictions
```

### 7.4 Per-Region Fairness Metrics
```
For each region independently:
  RMSE_region = sqrt(mean((y - ŷ_p50)^2))
  CRPS_region = mean(CRPS(F, y))
  Coverage_region = (ŷ_p5 < y < ŷ_p95).mean()

Fairness validation:
  Max ratio = max_region(metric_region) / min_region(metric_region)
  Target: Max ratio < 1.5× (no region performs >50% worse)
  
  Example:
    RMSE_IGKV = 8 µg/m³
    RMSE_Bhatagaon = 10 µg/m³
    Ratio = 10/8 = 1.25× ✓ (acceptable)
    
If ratio > 1.5×:
  - Region weighting not working
  - Increase weight for poor-performing region
  - Or: preprocessing pipeline failing for that region
```

---

## 8. Implementation Pseudocode

### 8.1 Training Script (train_lstm.py)
```python
# Load preprocessed LSTM data (from preprocess_lstm.py)
train_data, val_data, test_data = load_lstm_data()

# Build hierarchical LSTM for single pollutant (e.g., PM2.5)
model = HierarchicalQuantileRegressionLSTM(
    seq_len_adaptive=True,  # Adapt sequence length per horizon
    n_features=15,
    n_horizons=5,
    n_quantiles=4,
    hidden_dim=128,
    dropout=0.3,
    region_weights={
        'Bhatagaon': 2.84,
        'IGKV': 0.44,
        'AIIMS': 1.26,
        'SILTARA': 1.27
    }
)

# Train with multi-quantile pinball loss
optimizer = Adam(lr=0.001)
for epoch in range(100):
    for batch in train_loader:  # Stratified: ~8 samples per region
        X, y, regions = batch
        
        # Forward pass
        ŷ_quantiles = model(X)  # (batch, 5 horizons, 4 quantiles)
        
        # Compute multi-quantile loss with region weighting
        loss = multi_quantile_pinball_loss(y, ŷ_quantiles, regions, weights)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Early stopping on val CRPS
    val_crps = evaluate_crps(model, val_loader)
    if val_crps < best_val_crps:
        best_val_crps = val_crps
        checkpoint(model, epoch)
    elif epoch - best_epoch > 10:
        break

# Save trained model (current implementation pattern)
torch.save(model.state_dict(), 'models/lstm_quantile_pm25.pt')
```

### 8.2 Evaluation Script (evaluate.py)
```python
# Load test data and trained models
test_data, test_loader = load_test_data()
lstm_model = load_model('models/lstm_quantile_pm25.pt')
# XGB baseline is one model per horizon and quantile
xgb_model = load_model('models/xgb_quantile_pm25_h24_q50.json')

# Compute LSTM predictions
lstm_pred_quantiles = lstm_model(test_data)  # (n_test, 5, 4)

# Compute XGB predictions
xgb_pred_quantiles = xgb_model.predict(test_data)  # (n_test, 5, 4)

# Evaluate LSTM
lstm_coverage = compute_coverage(test_targets, lstm_pred_quantiles)
lstm_crps = compute_crps(test_targets, lstm_pred_quantiles)
lstm_pit_ks = compute_pit_ks(test_targets, lstm_pred_quantiles)
lstm_per_region = compute_per_region_metrics(test_targets, lstm_pred_quantiles, regions)

# Evaluate XGB
xgb_coverage = compute_coverage(test_targets, xgb_pred_quantiles)
xgb_crps = compute_crps(test_targets, xgb_pred_quantiles)
xgb_per_region = compute_per_region_metrics(test_targets, xgb_pred_quantiles, regions)

# Compare
print(f"LSTM CRPS: {lstm_crps:.2f}, XGB CRPS: {xgb_crps:.2f}")
print(f"LSTM coverage: {lstm_coverage}, XGB coverage: {xgb_coverage}")
print(f"LSTM per-region fairness: max/min = {lstm_per_region['max_ratio']:.2f}×")

# Plot calibration curves
plot_calibration_curve(test_targets, lstm_pred_quantiles, "LSTM")
plot_calibration_curve(test_targets, xgb_pred_quantiles, "XGB")
```

---

## 9. Key Design Decisions Summary

| Decision | Chosen | Rationale | Alternative |
|----------|--------|-----------|-------------|
| Hierarchical vs Separate | Hierarchical | Shared backbone + horizon heads | 80 separate models |
| Quantile Regression | p5/p50/p95/p99 | Prediction intervals, calibrated uncertainty | Point MSE/MAE |
| Dual Preprocessing | LSTM-specific + XGB-specific | Each model gets optimal features | Single unified pipeline |
| Region Weighting | Pinball loss upweighting | Fair treatment of rare regions | Ignore imbalance |
| Sequence Length | Adaptive 2×horizon | Match lookback to horizon scale | Fixed length |
| Weather Features | t-24 to t-1 only | Avoid future leakage | Current/future weather |
| Baseline Model | XGB | Fair comparison, shows LSTM advantage | RF, Ridge, or none |
