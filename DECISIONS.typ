= DECISIONS.md - Air Pollution Forecasting Project
= ML Research Project Documentation

*Project*: PM2.5 Air Pollution Forecasting, Raipur India
*Date*: April 2026
*Purpose*: Research Project Presentation & Viva Voce

---

= 1. Problem Statement

== Objective
Build a machine learning pipeline to forecast PM2.5 air pollution concentrations across four monitoring regions in Raipur, India.

== Target Variables
- Primary: PM2.5 (µg/m³)
- Prediction Horizons: t+1 (1 hour), t+6 (6 hours), t+24 (24 hours)

== Research Questions
#list(
  [Which ML model best captures temporal patterns in air pollution data?],
  [How does forecast accuracy vary with prediction horizon?],
  [Can deep learning improve over traditional gradient boosting methods?],
)

---

= 2. Dataset Description

== Data Source
#table(
  columns: (auto, auto, auto),
  table.header([Region, Location, Data Quality]),
  [Bhatagaon DCR], [Raipur], [Good],
  [DCR AIIMS], [Raipur], [Good],
  [IGKV DCR], [Raipur], [Excellent],
  [SILTARA DCR], [Raipur], [Good],
)

== Temporal Coverage
- Years: 2022, 2023, 2024, 2025
- Frequency: Hourly

== Data Retention
#table(
  columns: (auto, auto, auto),
  table.header([Stage, Records, Retention]),
  [Raw Data], [170,591], [100%],
  [After Missing Handling], [119,882], [70.2%],
  [After Outlier Removal], [116,425], [68.3%],
  [Final], [116,257], [68.2%],
)

---

= 3. Data Preprocessing

== Missing Value Handling
*Decision*: Per-Region Multiple Imputation Strategy

#table(
  columns: (auto, auto, auto),
  table.header([Step, Method, Rationale]),
  [1], [Linear interpolation (gaps < 6 hours)], [Preserve local temporal trends],
  [2], [Forward fill], [Handle remaining short gaps],
  [3], [Backward fill], [Handle leading/trailing gaps],
)

== Outlier Detection
*Decision*: Per-Region IQR-Based Thresholds

#table(
  columns: (auto, auto, auto),
  table.header([Region, Lower Bound, Upper Bound]),
  [SILTARA DCR], [0], [104.3],
  [Bhatagaon DCR], [0], [95.3],
  [DCR AIIMS], [0], [81.5],
  [IGKV DCR], [0], [63.0],
)

== Feature Engineering
*Lag Features*: 1, 3, 6, 12, 24, 48, 168 hours
*Rolling Statistics*: 6h, 12h, 24h mean and std
*Cyclical Time*: Hour, day of week, month, day of year
*Total Features*: 28

---

= 4. Model Selection

== Traditional ML Models
#table(
  columns: (auto, auto, auto),
  table.header([Model, Type, Rationale]),
  [Ridge Regression], [Linear], [Interpretable baseline],
  [Random Forest], [Tree Ensemble], [Non-linear, robust],
  [XGBoost], [Gradient Boosting], [State-of-the-art],
  [LightGBM], [Gradient Boosting], [Fast, efficient],
)

== Deep Learning
*Architecture*: Bidirectional LSTM (2 layers, 128 hidden) + Multi-head Attention (4 heads)
*Sequence Length*: 24 hours
*Dropout*: 0.3

---

= 5. Train/Test Split

*Decision*: Chronological Split (70/15/15)

#table(
  columns: (auto, auto, auto),
  table.header([Split, Ratio, Coverage]),
  [Train], [70%], [2022 - mid 2024],
  [Validation], [15%], [mid 2024 - end 2024],
  [Test], [15%], [2025],
)

---

= 6. Results - Horizon t+1

#table(
  columns: (auto, auto, auto, auto),
  table.header([Model, RMSE, MAE, R²]),
  [Ridge], [11.54], [8.33], [0.527],
  [Random Forest], [11.59], [8.47], [0.522],
  [XGBoost], [11.38], [8.21], [0.539],
  [LightGBM], [11.32], [8.13], [0.544],
  [LSTM], [11.02], [7.72], [0.568],
)

== Horizon t+6
#table(
  columns: (auto, auto, auto, auto),
  table.header([Model, RMSE, MAE, R²]),
  [Ridge], [11.64], [8.43], [0.518],
  [Random Forest], [11.84], [8.69], [0.501],
  [XGBoost], [11.57], [8.38], [0.524],
  [LightGBM], [11.51], [8.30], [0.529],
  [LSTM], [11.51], [8.13], [0.529],
)

== Horizon t+24
#table(
  columns: (auto, auto, auto, auto),
  table.header([Model, RMSE, MAE, R²]),
  [Ridge], [12.82], [9.67], [0.415],
  [Random Forest], [13.28], [10.14], [0.372],
  [XGBoost], [12.88], [9.78], [0.410],
  [LightGBM], [12.76], [9.65], [0.421],
  [LSTM], [12.61], [9.21], [0.434],
)

---

= 7. Key Findings

#list(
  [LSTM performs best across all horizons (RMSE 11.02-12.61)],
  [Performance degrades with horizon: t+1 easiest, t+24 hardest (~15% RMSE increase)],
  [Gradient boosting competitive: LightGBM close to LSTM],
  [Traditional models struggle at longer horizons],
)

---

= 8. Conclusions & Future Work

== Limitations
- CPU-only training
- Single target variable (PM2.5 only)
- No exogenous variables

== Future Work
#list(
  [Feature Selection: SHAP/permutation importance],
  [Multi-step Direct Forecast: Predict t+24 directly],
  [Exogenous Variables: Temperature, humidity, wind],
  [Transformer/TFT: Advanced sequence models],
)

---

*End of DECISIONS.md*