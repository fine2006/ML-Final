# DECISIONS.md - Air Pollution Forecasting Project

## ML Research Project Documentation

---

## 1. Problem Statement

### Objective
Build a machine learning pipeline to forecast PM2.5 air pollution concentrations across four monitoring regions in Raipur, India.

### Target Variables
- Primary: PM2.5 (µg/m³)
- Prediction Horizons: t+1 (1 hour), t+6 (6 hours), t+24 (24 hours)

### Research Questions
1. Which ML model best captures temporal patterns in air pollution data?
2. How does forecast accuracy vary with prediction horizon?
3. Can deep learning improve over traditional gradient boosting methods?

---

## 2. Dataset Description

### Data Source
| Region | Location | Data Quality |
|--------|----------|------------|
| Bhatagaon DCR | Raipur | Good |
| DCR AIIMS | Raipur | Good |
| IGKV DCR | Raipur | Excellent |
| SILTARA DCR | Raipur | Good |

### Temporal Coverage
- Years: 2022, 2023, 2024, 2025
- Frequency: Hourly (aggregated from 15-minute native data)

### Data Retention
| Stage | Records | Retention |
|-------|---------|------------|
| Raw Data | 170,591 | 100% |
| After Missing Handling | 119,882 | 70.2% |
| After Outlier Removal | 116,425 | 68.3% |
| Final | 116,257 | 68.2% |

---

## 3. Data Preprocessing Decisions

### 3.1 Missing Value Handling

#### Decision: Per-Region Multiple Imputation Strategy

| Step | Method | Rationale |
|------|--------|----------|
| 1 | Linear interpolation (gaps < 6 hours) | Preserve local temporal trends |
| 2 | Forward fill | Handle remaining short gaps |
| 3 | Backward fill | Handle leading/trailing gaps |

**Evidence**: Different regions have different data quality patterns; per-region handling preserves maximum data (68% vs 18% with global threshold).

### 3.2 Outlier Detection

#### Decision: Per-Region IQR-Based Thresholds

| Region | Lower Bound | Upper Bound |
|--------|------------|-------------|
| SILTARA DCR | 0 | 104.3 |
| Bhatagaon DCR | 0 | 95.3 |
| DCR AIIMS | 0 | 81.5 |
| IGKV DCR | 0 | 63.0 |

**Rationale**: Each station has different PM2.5 baselines; per-region prevents legitimate high values at polluted stations from being flagged.

### 3.3 Feature Engineering

#### Lag Features
| Feature | Lag | Rationale |
|---------|-----|-----------|
| pm25_lag_1 | 1 hour | Short-term dependency |
| pm25_lag_3 | 3 hours | Medium-term trend |
| pm25_lag_6 | 6 hours | Half-day pattern |
| pm25_lag_12 | 12 hours | Day/night transition |
| pm25_lag_24 | 24 hours | Full daily cycle |
| pm25_lag_48 | 48 hours | Two-day pattern |
| pm25_lag_168 | 168 hours | Weekly pattern |

#### Rolling Statistics
| Feature | Window | Purpose |
|---------|--------|---------|
| rolling_6h_mean | 6 hours | Short-term smoothing |
| rolling_12h_mean | 12 hours | Half-day trend |
| rolling_24h_mean | 24 hours | Daily average |
| rolling_6h_std | 6 hours | Volatility measure |
| rolling_24h_std | 24 hours | Daily volatility |

#### Cyclical Time Encoding
- Hour of day: sin/cos(2π × hour / 24)
- Day of week: sin/cos(2π × dow / 7)
- Month: sin/cos(2π × month / 12)
- Day of year: sin/cos(2π × doy / 365)

**Total Features**: 28

---

## 4. Train/Test Split

### Decision: Chronological Split (70/15/15)

| Split | Ratio | Coverage |
|-------|-------|----------|
| Train | 70% | 2022 - mid 2024 |
| Validation | 15% | mid 2024 - end 2024 |
| Test | 15% | 2025 |

**Rationale**: 
- Time series cannot use random splits (causes data leakage)
- Chronological preserves temporal order
- Simulates real deployment scenario

---

## 5. Model Selection Decisions

### 5.1 Traditional ML Models

| Model | Type | Rationale |
|-------|------|----------|
| Ridge Regression | Linear | Interpretable baseline |
| Random Forest | Tree Ensemble | Non-linear, robust |
| XGBoost | Gradient Boosting | State-of-the-art |
| LightGBM | Gradient Boosting | Fast, efficient |

### 5.2 Deep Learning

| Model | Architecture | Rationale |
|-------|-------------|-----------|
| BiLSTM + Attention | Bidirectional LSTM (2 layers, 128 hidden) + Multi-head Attention (4 heads) | Captures long-range temporal dependencies |

#### LSTM Configuration
- Sequence length: 24 hours
- Bidirectional: Yes
- Attention: 4 heads
- Dropout: 0.3
- Optimizer: Adam (lr = 0.001)
- Early stopping: patience = 10

### 5.3 Why These Models?

1. **Ridge**: Simple baseline, L2 regularization prevents overfitting
2. **Random Forest**: Ensemble of decision trees, handles non-linearity
3. **XGBoost/LightGBM**: Gradient boosting typically best-in-class for tabular data
4. **BiLSTM**: Sequence modeling, attention mechanism for temporal patterns

---

## 6. Ensemble Strategy

### Decision: Separate Pipelines

| Pipeline | Models | Rationale |
|----------|--------|----------|
| Traditional ML | Ridge, RF, XGBoost, LightGBM | Vector-based inputs |
| Deep Learning | BiLSTM + Attention | Sequence-based inputs |

**Rationale**: LSTM uses different feature representation (sequences of 24 timesteps); stacking would cause feature mismatch.

---

## 7. Evaluation Metrics

### Metrics Used
| Metric | Formula | Purpose |
|--------|---------|---------|
| RMSE | √(mean((y - ŷ)²)) | Primary, penalizes large errors |
| MAE | mean(|y - ŷ|) | Robust to outliers |
| R² | 1 - SS_res/SS_tot | Explained variance |

---

## 8. Results

### 8.1 Horizon t+1 (Next Hour)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Ridge | 11.54 | 8.33 | 0.527 |
| Random Forest | 11.59 | 8.47 | 0.522 |
| XGBoost | 11.38 | 8.21 | 0.539 |
| LightGBM | 11.32 | 8.13 | 0.544 |
| **LSTM** | **11.02** | **7.72** | **0.568** |

### 8.2 Horizon t+6 (6 Hours Ahead)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Ridge | 11.64 | 8.43 | 0.518 |
| Random Forest | 11.84 | 8.69 | 0.501 |
| XGBoost | 11.57 | 8.38 | 0.524 |
| LightGBM | 11.51 | 8.30 | 0.529 |
| **LSTM** | **11.51** | **8.13** | **0.529** |

### 8.3 Horizon t+24 (24 Hours Ahead)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Ridge | 12.82 | 9.67 | 0.415 |
| Random Forest | 13.28 | 10.14 | 0.372 |
| XGBoost | 12.88 | 9.78 | 0.410 |
| LightGBM | 12.76 | 9.65 | 0.421 |
| **LSTM** | **12.61** | **9.21** | **0.434** |

### 8.4 Key Findings

1. **LSTM performs best** across all horizons (RMSE 11.02-12.61)
2. **Performance degrades with horizon**: t+1 easiest, t+24 hardest (~15% RMSE increase)
3. **Gradient boosting competitive**: LightGBM close to LSTM at t+6, t+24
4. **Traditional models struggle at longer horizons**: RF shows highest degradation

---

## 9. Conclusions

### What We Learned
1. Deep learning (LSTM) captures temporal patterns better for pollution forecasting
2. Extended lag features (up to 168 hours) important for weekly patterns
3. Per-region preprocessing critical due to varying pollution baselines

### Limitations
1. CPU-only training (no GPU optimization)
2. Single target variable (PM2.5 only)
3. No exogenous variables (meteorological data)

---

## 10. Future Work

1. **Feature Selection**: Use SHAP/permutation importance to identify key features
2. **Multi-step Direct Forecast**: Predict t+24 directly rather than iterating t+1
3. **Exogenous Variables**: Add temperature, humidity, wind speed
4. **Transformer/TFT**: More advanced sequence models
5. **Multi-target**: Predict PM10, NO2, O3 simultaneously