# Air Pollution Forecasting - Project Plan

## Executive Summary

This project implements an ML pipeline to forecast PM2.5 air pollution levels across 4 regions in Raipur, India. We compare traditional ML models (Linear Regression, Random Forest) with a deep learning model (LSTM) to predict hourly PM2.5 concentrations.

---

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Target Variable** | PM2.5 | Most health-relevant pollutant; commonly used benchmark; available in all datasets |
| **Regions** | All 4 (Bhatagaon, DCR AIIMS, IGKV, SILTARA) | Ensures model generalizes across locations; robust predictions |
| **Initial Subset** | 2024 data | Most recent complete year; 4 regions available; manageable size for iteration |
| **Prediction Horizon** | t+1 (next hour) | Simplest to implement first; can extend to t+12/t+24 later if needed |
| **ML Framework** | PyTorch | Explicit preference; excellent for research prototypes; GPU acceleration |
| **Train/Val/Test Split** | Chronological (2022-2024 train, 2025 val, 2026 test) | Prevents data leakage; realistic for production deployment |
| **Baseline Models** | Linear Regression + Random Forest | Industry standards; interpretable; good baselines for comparison |
| **Feature Engineering** | Lag features (1,6,12,24h) + Rolling mean (24h) | Captures short-term and daily patterns; proven effective for time-series |
| **Normalization** | StandardScaler | Standard practice; preserves outliers; fitted on train only |
| **Metrics** | RMSE + MAE | Standard regression metrics; complementary (RMSE penalizes large errors) |
| **LSTM Architecture** | 2-layer, hidden_size=64, seq_len=24 | Balanced complexity; captures multi-day patterns |
| **Optimization** | Adam optimizer, MSE loss, early stopping | Standard for regression; prevents overfitting |

---

## Data Structure

```
Pollution Data Raipur/
├── Bhatagaon DCR/
│   └── DCR bhatagaon/DCR {year}/{month}/*.xlsx
├── DCR AIIMS/
│   └── {year}/{month}/*.xls
├── IGKV DCR/
│   └── Year {year}/{month}/*.xlsb
└── SILTARA DCR/
    └── [similar structure]
```

**File Formats**: xlsx, xls, xlsb (Excel formats)  
**Sheet Structure**: Each file contains multiple sheets (one per date)  
**Data Frequency**: Hourly and quarter-hourly readings

---

## Stages

### Stage 1: Explore & Understand Data

**Objective**: Analyze data structure, quality, and sparsity to inform pipeline decisions.

**Tasks**:
1. Load sample files from each region (one month each)
2. Identify column names for PM2.5 (may vary by region)
3. Determine data frequency (hourly vs quarter-hourly)
4. Calculate missing value percentage per region
5. Document date range coverage per region
6. Identify data quality issues (gaps, outliers, invalid values)

**Deliverable**: Data exploration report with findings to guide Stage 2.

---

### Stage 2: Iterate & Validate Pipeline

**Objective**: Build minimal end-to-end pipeline on subset to verify approach works.

**Tasks**:

#### 2.1 Setup
- Add dependencies to pyproject.toml
- Install with `uv sync`
- Create project structure (data/, models/, scripts/)

#### 2.2 Data Loader
- Load 2024 data from one region (e.g., Bhatagaon)
- Parse Excel sheets, extract PM2.5 column
- Aggregate quarter-hourly to hourly (mean)
- Handle date parsing and time index creation

#### 2.3 Preprocessing
- Handle missing values (interpolate + forward fill)
- Create lag features: PM2.5 at t-1, t-6, t-12, t-24
- Add rolling mean (24h window)
- Chronological train/val/test split
- StandardScaler normalization (fit on train only)

#### 2.4 Baseline Models
- Linear Regression: sklearn implementation
- Random Forest: sklearn with default params
- Train both on train set, evaluate on val set

#### 2.5 LSTM Model
- PyTorch implementation
- DataLoader with sequence length 24
- Training loop with MSE loss
- Early stopping on validation loss

#### 2.6 Evaluation
- Calculate RMSE and MAE for all models
- Compare results
- Visualize predictions vs actual (sample plot)

**Deliverable**: Working pipeline with all components; verified on single region, single year.

---

### Stage 3: Implement & Scale

**Objective**: Scale to full dataset; train final models; produce results.

**Tasks**:

#### 3.1 Full Data Loading
- Load all 4 regions
- Load all available years (2022-2026)
- Merge into single DataFrame with region indicator
- Handle region-specific column naming

#### 3.2 Enhanced Preprocessing
- Region-based feature engineering (optional)
- Handle region-specific data quality issues
- Finalize train/val/test split (chronological)

#### 3.3 Model Training
- Train Linear Regression (full data)
- Train Random Forest (full data, tune if needed)
- Train LSTM (full data, more epochs, GPU if available)

#### 3.4 Final Evaluation
- Calculate final RMSE, MAE on test set
- Create comparison table across all models
- Generate prediction plots for each model
- Save model artifacts

#### 3.5 Documentation
- Document final results
- Note any limitations or areas for improvement

**Deliverable**: Complete pipeline; trained models; evaluation results; visualizations.

---

## File Structure

```
ML-Final/
├── PLAN.md                     # This file
├── pyproject.toml              # Dependencies
├── main.py                     # Entry point
├── data/
│   └── processed/              # Cleaned data output
├── models/
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── random_forest.py
│   └── lstm.py
├── scripts/
│   ├── load_data.py           # Data loading
│   ├── preprocess.py          # Preprocessing
│   ├── train.py               # Training loop
│   └── evaluate.py            # Evaluation
├── visualizations/             # Plots output
└── Pollution Data Raipur/      # Raw data (input)
```

---

## Dependencies

```toml
# pyproject.toml
dependencies = [
    "pandas",       # Data manipulation
    "numpy",        # Numerical computing
    "openpyxl",     # Excel file reading
    "scikit-learn", # ML models
    "torch",        # Deep learning
    "matplotlib",  # Visualization
]
```

---

## Expected Output

1. **Trained Models**: Linear Regression, Random Forest, LSTM
2. **Metrics Table**:
   | Model | RMSE | MAE |
   |-------|------|-----|
   | Linear Regression | X.X | X.X |
   | Random Forest | X.X | X.X |
   | LSTM | X.X | X.X |
3. **Visualizations**: Prediction plots for each model
4. **Insights**: Which model performs best; key findings

---

## Timeline (Suggested)

- **Stage 1**: 1-2 hours (data exploration)
- **Stage 2**: 2-4 hours (iterate on subset)
- **Stage 3**: 2-4 hours (scale to full)

Total: ~5-10 hours

---

## Future Extensions

If time permits:
- Add t+12 and t+24 prediction models
- Feature: day-of-week, month, seasonality
- Hyperparameter tuning for Random Forest/LSTM
- Ensemble of models
- More regions (if more data becomes available)