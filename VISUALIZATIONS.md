# VISUALIZATIONS.md - All Plot Specifications & Outputs

## ⚠️ CRITICAL: GIT INSTRUCTIONS
**DO NOT RUN ANY GIT COMMANDS** (git add, git commit, git push, or any git modification commands).
See DECISIONS.md for full git warning. Work directly on files; user handles git operations.

---

## Output Location

All visualizations saved to `visualizations/` directory with descriptive names:

```
visualizations/
├── phase_1_data_investigation/
│   ├── bhatagaon_spike_temporal_pattern.png
│   ├── bhatagaon_spike_regional_correlation.png
│   ├── bhatagaon_spike_weather_context.png
│   ├── data_loss_breakdown_per_region.png
│   ├── data_loss_sources_stacked_bar.png
│   └── region_imbalance_distribution.png
│
├── phase_3_preprocessing/
│   ├── lstm_preprocessing_retention_flow.png
│   ├── xgb_preprocessing_retention_flow.png
│   ├── feature_distributions_lstm.png
│   ├── feature_distributions_xgb.png
│   ├── missing_value_patterns_heatmap.png
│   ├── outlier_detection_comparison.png
│   └── time_split_visualization.png
│
├── phase_5_training/
│   ├── lstm_pm25_training_curves.png
│   ├── lstm_pm25_attention_weights.png
│   ├── lstm_pm25_loss_per_quantile.png
│   ├── xgb_pm25_feature_importance.png
│   ├── lstm_all_pollutants_convergence.png
│   └── region_weights_impact_on_loss.png
│
└── phase_6_evaluation/
    ├── lstm_vs_xgb_rmse_by_horizon.png
    ├── lstm_vs_xgb_crps_by_horizon.png
    ├── quantile_calibration_curves.png
    ├── pit_histogram_uniformity.png
    ├── coverage_analysis_by_quantile.png
    ├── per_region_fairness_metrics.png
    ├── predictions_vs_actual_by_horizon.png
    └── final_comparison_summary_table.png
```

---

## Phase 1: Data Investigation Visualizations

### 1.1 Bhatagaon September 2025 Spike - Temporal Pattern

**File**: `bhatagaon_spike_temporal_pattern.png`

**Objective**: Visualize spike pattern to classify as real event vs sensor error

**Plot type**: Line plot with annotations

```
Layout: Subplots for each spike instance (if multiple)

X-axis: Time (hours, ±12 around spike)
Y-axis: PM2.5 (µg/m³)

Data shown:
  - Blue line: PM2.5 time series
  - Red vertical line: Spike timestamp
  - Green shaded area: Spike bounds (if real, gradual slope)
  - Annotations: Min, max, slope rate

Title: "Bhatagaon September 2025: PM2.5 Spike Pattern Analysis"

Decision marker:
  - If gradual rise/decay: "REAL EVENT - KEEP"
  - If sudden spike: "SENSOR ERROR - REMOVE"
```

### 1.2 Bhatagaon Spike - Regional Correlation

**File**: `bhatagaon_spike_regional_correlation.png`

**Objective**: Check if spike is localized (sensor error) or regional (real event)

**Plot type**: Multi-line plot

```
X-axis: Time (±6 hours around spike)
Y-axis: PM2.5 (µg/m³)

Lines (one per region):
  - Bhatagaon: Red (spiked)
  - IGKV: Blue
  - AIIMS: Green
  - SILTARA: Orange

Title: "Regional PM2.5 at Bhatagaon Spike Time"

Interpretation:
  - All regions elevated → REAL EVENT
  - Only Bhatagaon spike → SENSOR ERROR
```

### 1.3 Bhatagaon Spike - Weather Context

**File**: `bhatagaon_spike_weather_context.png`

**Objective**: Verify weather conditions support (or contradict) spike classification

**Plot type**: Multi-axis subplot

```
Subplots (aligned X-axis):
  1. PM2.5 (µg/m³) - line plot, red spike marker
  2. Temperature (°C) - line plot
  3. Wind speed (m/s) - line plot
  4. Humidity (%) - line plot

X-axis: Time (±24 hours around spike)

Title: "Weather Context During Bhatagaon Spike"

Annotations:
  - Inversion indicator: Temperature drop at spike
  - Stagnation indicator: Low wind speed at spike
  - Moisture indicator: High humidity at spike
```

### 1.4 Data Loss Breakdown - Per Region

**File**: `data_loss_breakdown_per_region.png`

**Objective**: Show retention percentage for each region

**Plot type**: Stacked bar chart

```
X-axis: Regions (Bhatagaon, IGKV, AIIMS, SILTARA)
Y-axis: Record count (0 to raw data max)

Stacked bars per region:
  - Green: Retained data
  - Red: Lost to gaps
  - Orange: Lost to outliers
  - Yellow: Lost to sequence breaking

Height = total raw records per region
Color proportion = loss source

Title: "Data Retention by Region (Raw → Final)"

Data labels on bars:
  - Total: XXX records
  - Retention: XX.X%
```

### 1.5 Data Loss Sources - Stacked Breakdown

**File**: `data_loss_sources_stacked_bar.png`

**Objective**: Show where data is lost (gaps vs outliers vs sequences)

**Plot type**: Horizontal stacked bar

```
X-axis: Percentage (0% to 100%)
Y-axis: Loss sources
  - Impossible values (negatives, -2B)
  - Missing value gaps
  - Outlier removal
  - Sequence breaking

Title: "Data Loss Attribution (Canonical Pipeline)"

Data labels: Percentage and absolute count for each source
```

### 1.6 Region Imbalance Distribution

**File**: `region_imbalance_distribution.png`

**Objective**: Visualize canonical region distribution and post-sequence imbalance

**Plot type**: Pie chart + bar chart

```
Pie chart:
  - Slice per region
  - Label: Region name + percentage
  - Highlight: near-balanced canonical distribution (~25% each)
  - Title: "Data Distribution Across Regions (Canonical Hourly)"

Bar chart (below):
  - X-axis: Regions
  - Y-axis: Record count
  - Color: One color per region
  - Label on top: post-sequence max/min ratio (~1.04×)
  - Title: "Post-Canonical Region Imbalance Ratio"
```

---

## Phase 3: Preprocessing Visualizations

### 3.1 LSTM Preprocessing Retention Flow

**File**: `lstm_preprocessing_retention_flow.png`

**Objective**: Show data loss at each preprocessing step (LSTM pipeline)

**Plot type**: Sankey diagram (or stacked bar)

```
Flow (left to right):
  1. Canonical hourly baseline: 125,017
  2. After impossible values removed: →
  3. After gap interpolation: →
  4. After gap breaking: →
  5. After feature engineering: →
  Final: ~125,000

Width of flow = record count
Labels = count + retention percentage

Title: "LSTM Preprocessing Data Retention Flow"

Color coding:
  - Green: Retained
  - Red: Lost
```

### 3.2 XGB Preprocessing Retention Flow

**File**: `xgb_preprocessing_retention_flow.png`

**Objective**: Show data loss at each preprocessing step (XGB pipeline)

**Plot type**: Sankey diagram (or stacked bar)

```
Flow (left to right):
  1. Canonical hourly baseline: 125,017
  2. After impossible values removed: →
  3. After aggressive gap interpolation: →
  4. After outlier removal: →
  5. After feature engineering: →
  Final: ~120,000

Title: "XGB Preprocessing Data Retention Flow"

Compare side-by-side with LSTM flow to show different strategies
```

### 3.3 Feature Distributions - LSTM

**File**: `feature_distributions_lstm.png`

**Objective**: Show 15 LSTM features after scaling

**Plot type**: 4×4 subplot grid (histograms)

```
Subplots: 15 histograms (one per feature)
  - pm25, pm10, no2, o3 (raw pollution)
  - temperature, humidity, wind_speed, wind_direction (weather)
  - hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos (time)
  - region_id (region encoding)

Each histogram:
  - X-axis: Feature value
  - Y-axis: Frequency
  - Color: Blue
  - Title: Feature name
  - Stats: Mean, std, min, max

Overall title: "LSTM Feature Distributions (Post-RobustScaler)"

Note: Outliers visible (not removed for LSTM)
```

### 3.4 Feature Distributions - XGB

**File**: `feature_distributions_xgb.png`

**Objective**: Show 55-60 XGB features (no scaling)

**Plot type**: Multi-page PDF or subplots

```
Subplots: 55-60 histograms organized by category:
  - Raw pollution (4)
  - Lag features (28)
  - Rolling statistics (24)
  - Weather lags (20)
  - Missingness features (6)
  - Time features (6)

Each histogram:
  - X-axis: Feature value
  - Y-axis: Frequency
  - Color: Category-specific color

Title: "XGB Feature Distributions (Pre-Training)"

Note: No outliers (removed by XGB pipeline)
```

### 3.5 Missing Value Patterns - Heatmap

**File**: `missing_value_patterns_heatmap.png`

**Objective**: Show where data is missing

**Plot type**: Heatmap

```
X-axis: Dates (time progression)
Y-axis: Regions (Bhatagaon, IGKV, AIIMS, SILTARA)

Heatmap cells:
  - Green: Data present
  - Red: Data missing (>10% of pollutants)
  - Yellow: Sparse data (1-10% missing)

Title: "Missing Data Patterns by Region Over Time"

Highlight: SILTARA 2023 gap (full year red)
```

### 3.6 Outlier Detection Comparison

**File**: `outlier_detection_comparison.png`

**Objective**: Show LSTM vs XGB outlier strategies

**Plot type**: Scatter plot with overlays

```
X-axis: Sample index
Y-axis: PM2.5 value

Scatter points:
  - Blue: Normal range
  - Orange: Statistical outliers (IQR+1.5)
  - Red: Extreme outliers (PM2.5 >300)

Lines:
  - Green horizontal: LSTM threshold (keep all)
  - Red horizontal: XGB threshold (remove >300)

Title: "Outlier Handling: LSTM (Keep) vs XGB (Remove)"

Annotations: Example real spike (kept by LSTM) vs sensor error (removed by XGB)
```

### 3.7 Time Split Visualization

**File**: `time_split_visualization.png`

**Objective**: Show train/val/test split

**Plot type**: Timeline with shaded regions

```
Timeline (horizontal):
  2022-01-01 ← → 2025-04-16

Shaded regions:
  - Green: Training (2022-2024 Q1) - 70%
  - Yellow: Validation (2024 Q1-Q2) - 15%
  - Red: Test (2024 Q2-present) - 15%

Title: "Train/Val/Test Split (Time-Based 70/15/15)"

Tick marks: Quarter boundaries
Labels: Start/end dates and record counts per split
```

---

## Phase 5: Training Visualizations

### 5.1 LSTM PM2.5 - Training Curves

**File**: `lstm_pm25_training_curves.png`

**Objective**: Show convergence during training

**Plot type**: Multi-line plot

```
X-axis: Epoch (0 to 100 or early stopping)
Y-axis: Loss/CRPS

Lines:
  - Blue: Training loss
  - Orange: Validation loss
  - Green: Validation CRPS (right Y-axis)

Markers:
  - Red dot: Best validation CRPS (early stopping point)
  - Dashed line: Epoch of best validation

Title: "LSTM PM2.5 - Training Convergence"

Annotations:
  - Early stopping at epoch XX
  - Best val CRPS: X.XX
  - Final train loss: X.XX
```

### 5.2 LSTM PM2.5 - Attention Weights

**File**: `lstm_pm25_attention_weights.png`

**Objective**: Show what attention heads focus on

**Plot type**: Heatmap (3 active horizons × sequence length)

```
X-axis: Sequence position (time steps back from prediction)
Y-axis: Horizons (`h1`, `h24`, `h168`)

Heatmap cells:
  - Color intensity: Attention weight (0 to 1)
  - Dark: High attention
  - Light: Low attention

Title: "LSTM PM2.5 - Attention Weight Distribution"

Pattern expected:
  - h1 attention: short-horizon recency emphasis
  - h168 attention: broader long-context emphasis
  - Each horizon has distinct pattern
```

### 5.3 LSTM PM2.5 - Loss Per Quantile

**File**: `lstm_pm25_loss_per_quantile.png`

**Objective**: Show convergence for each quantile

**Plot type**: Multi-line plot

```
X-axis: Epoch
Y-axis: Quantile-specific loss

Lines (one per quantile):
  - Blue: p5 loss
  - Orange: p50 loss (median)
  - Green: p95 loss
  - Red: p99 loss

Title: "LSTM PM2.5 - Per-Quantile Loss Convergence"

Observation: All quantiles should converge (not diverge)
```

### 5.4 XGB PM2.5 - Feature Importance

**File**: `xgb_pm25_feature_importance.png`

**Objective**: Show which features XGB relies on

**Plot type**: Horizontal bar chart (top 20 features)

```
X-axis: Importance score
Y-axis: Features (sorted by importance)

Bars colored by feature category:
  - Blue: Lag features
  - Green: Rolling statistics
  - Orange: Weather features
  - Red: Missingness features
  - Purple: Time features

Title: "XGB PM2.5 - Top 20 Most Important Features"

Labels: Importance value on bar
```

### 5.5 All Pollutants - Convergence Overview

**File**: `lstm_all_pollutants_convergence.png`

**Objective**: Compare convergence across PM2.5, PM10, NO2, O3

**Plot type**: 2×2 subplot grid

```
Subplots (one per pollutant):
  - Top-left: PM2.5
  - Top-right: PM10
  - Bottom-left: NO2
  - Bottom-right: O3

Each subplot:
  - X-axis: Epoch
  - Y-axis: Validation CRPS
  - Line: Val CRPS curve
  - Marker: Best epoch

Title: "LSTM - Convergence Comparison Across Pollutants"

Insight: All should converge smoothly (not diverge)
```

### 5.6 Region Weights - Impact on Loss

**File**: `region_weights_impact_on_loss.png`

**Objective**: Show how region weighting balances training

**Plot type**: Line plot (4 lines per region)

```
X-axis: Epoch
Y-axis: Validation loss

Lines (one per region):
  - Blue: Bhatagaon (~1.009×)
  - Orange: IGKV (~0.994×)
  - Green: AIIMS (~0.979×)
  - Red: SILTARA (~1.020×)

Title: "Per-Region Validation Loss (with Region Weighting)"

Expected:
  - All lines roughly parallel
  - No line significantly worse than others
  - Max ratio between best/worst <1.5×

Interpretation: If one region diverges, weighting not working
```

---

## Phase 6: Evaluation Visualizations

### 6.1 LSTM vs XGB - RMSE by Horizon

**File**: `lstm_vs_xgb_rmse_by_horizon.png`

**Objective**: Compare point prediction accuracy across horizons

**Plot type**: Grouped bar chart

```
X-axis: Horizons (`h1`, `h24`, `h168`)
Y-axis: RMSE (µg/m³)

Bars per horizon:
  - Blue: LSTM (computed vs p50)
  - Orange: XGB

Title: "LSTM vs XGB - RMSE by Prediction Horizon"

Labels: RMSE value on bar + percentage difference

Expected pattern:
  - performance differs by pollutant and horizon
  - PM may lead on RMSE while still under-covering
  - gas often requires rescue tuning at `h168`
```

### 6.2 LSTM vs XGB - CRPS by Horizon

**File**: `lstm_vs_xgb_crps_by_horizon.png`

**Objective**: Compare full distribution prediction quality

**Plot type**: Grouped bar chart

```
X-axis: Horizons (`h1`, `h24`, `h168`)
Y-axis: CRPS (µg/m³)

Bars per horizon:
  - Blue: LSTM
  - Orange: XGB

Title: "LSTM vs XGB - CRPS by Prediction Horizon"

Labels: CRPS value + percentage improvement

Key insight: CRPS combines calibration + sharpness
```

### 6.3 Quantile Calibration Curves

**File**: `quantile_calibration_curves.png`

**Objective**: Check if quantile predictions match theoretical levels

**Plot type**: 2×2 subplot grid (one per quantile)

```
Subplots:
  1. p5 calibration
  2. p50 (median) calibration
  3. p95 calibration
  4. p99 calibration

Each subplot:
  - X-axis: Theoretical quantile level (0 to 1)
  - Y-axis: Empirical quantile level (0 to 1)
  - Blue line: Diagonal (y=x, perfect calibration)
  - Orange line: Actual calibration curve
  - Shaded region: ±5% band (acceptable)

Title: "Quantile Calibration Curves (Test Set)"

Interpretation:
  - Line on diagonal → Well calibrated ✓
  - Line below diagonal → Overconfident (too narrow)
  - Line above diagonal → Underconfident (too wide)

Expected: All lines close to diagonal
```

### 6.4 PIT Histogram - Uniformity Test

**File**: `pit_histogram_uniformity.png`

**Objective**: Check if PIT values are uniformly distributed

**Plot type**: Histogram with KS test result

```
X-axis: PIT value (0 to 1)
Y-axis: Frequency

Histogram:
  - Blue bars: PIT value distribution
  - Red line: Uniform reference (flat line at mean)

Title: "PIT Histogram - Uniformity Test (Kolmogorov-Smirnov)"

Statistics (on plot):
  - KS statistic: X.XXX
  - p-value: X.XXX
  - Conclusion: Calibrated ✓ / Not calibrated ✗

Expected: Roughly flat histogram, p-value >0.05
```

### 6.5 Coverage Analysis - By Quantile

**File**: `coverage_analysis_by_quantile.png`

**Objective**: Check if interval coverage matches target

**Plot type**: Bar chart with error bars

```
X-axis: Quantile intervals
  - Below p5 (target 5%)
  - p5 to p95 (target 90%)
  - Above p95 (target 5%)

Y-axis: Empirical coverage (%)

Bars:
  - Green: Coverage within target band (±5%)
  - Red: Coverage outside target band

Title: "Prediction Interval Coverage (Test Set)"

Labels: Percentage value on bar + target

Expected:
  - Below p5: 5% ± 5% = [0%, 10%] ✓
  - p5-p95: 90% ± 5% = [85%, 95%] ✓
  - Above p95: 5% ± 5% = [0%, 10%] ✓
```

### 6.6 Per-Region Fairness Metrics

**File**: `per_region_fairness_metrics.png`

**Objective**: Check if rare regions not ignored

**Plot type**: Grouped bar chart (4 regions × metrics)

```
X-axis: Regions (Bhatagaon, IGKV, AIIMS, SILTARA)
Y-axis: Metric value

Metrics (separate bars per region):
  - RMSE (µg/m³)
  - CRPS (µg/m³)
  - Coverage (%)

Title: "Per-Region Fairness: LSTM Metric Comparison"

Annotations:
  - Max/min ratio on top (target <1.5×)
  - Weighted average line (reference)

Expected:
  - All regions roughly similar
  - Max RMSE ratio <1.5× ✓
  - No region significantly worse
```

### 6.7 Predictions vs Actual - By Horizon

**File**: `predictions_vs_actual_by_horizon.png`

**Objective**: Visually inspect prediction quality

**Plot type**: 3 scatter plots (one per active horizon)

```
Subplots (2×2 grid, one empty):
  1. h1
  2. h24
  3. h168
  4. (unused)

Each subplot:
  - X-axis: Actual PM2.5 (µg/m³)
  - Y-axis: Predicted PM2.5 (p50 median)
  - Points: Blue scatter (one per test sample)
  - Line: Red diagonal (perfect prediction, y=x)
  - Shaded region: ±10% band around diagonal

Title: "Predictions vs Actual (Test Set)"

Metric on subplot: R², RMSE, slope

Expected: Points clustered along diagonal (no systematic bias)
```

### 6.8 Final Comparison - Summary Table

**File**: `final_comparison_summary_table.png`

**Objective**: Summary metrics in table format

**Plot type**: Formatted table

```
Table structure:
  Rows: Metrics (RMSE, CRPS, Coverage, PIT p-value)
  Columns: Horizons (`h1`, `h24`, `h168`)

Cells: LSTM value | XGB value | LSTM advantage %

Colors:
  - Green cell: LSTM significantly better (>10%)
  - Yellow cell: Similar performance
  - Red cell: XGB better

Title: "Summary Comparison: LSTM vs XGB Across Active Horizons"

Bottom section: Key findings
  - Gate pass/fail by pollutant/horizon (from operational gates)
  - Per-region fairness: Max ratio XX.X×
  - Quantile calibration: PIT p-value X.XXX
```

---

## Visualization Output Standards

### 6.9 Common Requirements for All Plots

**Font & Labels:**
- Font size: 12pt for labels, 14pt for titles
- Always include units (µg/m³, hours, %, etc.)
- Axis labels clear and descriptive

**Colors:**
- Use colorblind-friendly palette (blue, orange, green, red)
- Consistent colors across all plots (e.g., LSTM always blue)

**Resolution:**
- Save as PNG (300 DPI for publication quality)
- Save as PDF for vector format (scalable)

**Annotations:**
- Include value labels on bars/points where helpful
- Add legends to distinguish lines/colors
- Include data source and date on plot

**File naming:**
- Lowercase with underscores
- Descriptive: `lstm_pm25_training_curves.png`
- Include phase prefix: `phase_5_` for Phase 5 plots

### 6.10 Saving Plot Code Template

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot data
ax.plot(x_data, y_data, label='LSTM', color='blue', linewidth=2)
ax.plot(x_data, y_baseline, label='XGB', color='orange', linewidth=2)

# Labels & title
ax.set_xlabel('Prediction Horizon', fontsize=12)
ax.set_ylabel('RMSE (µg/m³)', fontsize=12)
ax.set_title('LSTM vs XGB - RMSE by Horizon', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Save
output_dir = 'visualizations/phase_6_evaluation/'
os.makedirs(output_dir, exist_ok=True)
fig.savefig(f'{output_dir}/lstm_vs_xgb_rmse_by_horizon.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{output_dir}/lstm_vs_xgb_rmse_by_horizon.pdf', bbox_inches='tight')
plt.close()

print(f"✓ Saved: {output_dir}/lstm_vs_xgb_rmse_by_horizon.png")
```
