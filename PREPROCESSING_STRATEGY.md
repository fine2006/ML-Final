# PREPROCESSING_STRATEGY.md - Dual Pipelines for LSTM vs XGB

## ⚠️ CRITICAL: GIT INSTRUCTIONS
**DO NOT RUN ANY GIT COMMANDS** (git add, git commit, git push, or any git modification commands).
See DECISIONS.md for full git warning. Work directly on files; user handles git operations.

## Overview
This document describes the two fundamentally different preprocessing approaches: one optimized for LSTM (sequence-focused, outlier-preserving) and one for XGB (feature-focused, outlier-removing).

**Note**: This is a FRESH IMPLEMENTATION. Disregard all previous code. Treat as starting from zero.

---

## 1. Philosophy: Why Dual Pipelines?

### 1.1 LSTM Philosophy
**LSTM learns sequences, not features.**

```
LSTM insight:
  - Temporal sequence itself is information
  - PM2.5[t-5], PM2.5[t-4], ..., PM2.5[t] tells a story: "rising trend"
  - Outliers in sequence teach patterns: "spike then decay = real event"
  - Missing data breaks sequences, corrupting temporal continuity
  
LSTM preprocessing strategy:
  - Minimize data loss: keep anything that preserves temporal order
  - Keep outliers: extreme but real values teach temporal patterns
  - Light feature engineering: minimal explicit lags (implicit in sequence)
  - Scaling only: RobustScaler to preserve outlier info while stabilizing gradients
```

### 1.2 XGB Philosophy
**XGB learns feature interactions, not sequences.**

```
XGB insight:
  - Each row is independent (no temporal memory)
  - PM2.5_lag_1, PM2.5_lag_3, ..., PM2.5_lag_168 are separate features
  - Outliers distort tree splits: single extreme value splits tree for all data
  - Missing data handled via features (not continuity): "hours_since_measurement" tells story
  
XGB preprocessing strategy:
  - Remove outliers: extreme values distort tree splits for entire dataset
  - Aggressive imputation: fill gaps with features explaining missingness
  - Rich feature engineering: explicit lags, rolling stats, weather interactions
  - No scaling: trees are scale-invariant, scaling adds no value
```

---

## 2. Shared Data Loading (Both Pipelines)

### 2.1 Raw Data Source
```
Location: ./Pollution Data Raipur/[region]/[year]/[month]/*.xlsx

Regions:
  - Bhatagaon DCR
  - DCR AIIMS
  - IGKV DCR
  - SILTARA DCR

Time range:
  - 2022-01-01 to 2025-04-16
  - Hourly aggregated from 15-minute native data

Pollutants extracted:
  - PM2.5 (primary target)
  - PM10
  - NO2
  - O3

Weather features extracted:
  - Temperature (°C)
  - Humidity (%)
  - Wind speed (m/s)
  - Wind direction (degrees)
```

### 2.2 Raw Data Validation
```
Step 1: Drop physically impossible values
  - PM2.5 < 0: impossible (concentration can't be negative)
  - PM2.5 > 5000: likely sensor error (sensor saturates)
  - Humidity > 100% or < 0%: impossible
  
  Action: Remove rows with these values
  Data retention: ~99% (removes only clear errors)

Step 2: Extract by region & hourly aggregate
  - Group by region, date, hour
  - Average 15-minute data into hourly samples
  - Reason: Data natively at 15-minute; hourly aggregation smooths noise
  
  Output: (n_samples, 4 pollutants + 4 weather features + timestamp + region)
```

### 2.3 Initial Data Shape
```
After raw loading:
  - Total records: 170,591 (4 regions × 4 years × ~52 weeks × 168 hours/week)
  - By region:
    * Bhatagaon: 15,000 records (8.8%)
    * IGKV: 97,200 records (57.0%)
    * AIIMS: 28,200 records (16.5%)
    * SILTARA: 30,200 records (17.7%)
  - Missing: 31.9% (reasons: sensor downtime, data collection gaps, transmission errors)
```

---

## 3. LSTM Preprocessing Pipeline (preprocess_lstm.py)

### 3.1 Overview
```
Raw data
  ↓
Light missing value handling (interpolate gaps <6h, break on >6h)
  ↓
RobustScaler (scale by median/IQR)
  ↓
Create sequences (adaptive length per horizon)
  ↓
Add lag features & time encodings
  ↓
Create train/val/test split (70/15/15 time-based)
  ↓
LSTM-ready data
```

### 3.2 Missing Value Handling (LSTM-Centric)

**Strategy**: Preserve sequence continuity; lose data only for long gaps

```
Step 1: Identify gap lengths
  For each region:
    timestamps = sorted(data.timestamp)
    gaps = timestamps[1:] - timestamps[:-1]

Step 2: Handle gaps
  For gap < 6 hours:
    Action: Linear interpolation
    Reason: Short gaps likely measurement errors, interpolate to preserve continuity
    Example: [100, NaN, NaN, 120] (3-hour gap)
             → [100, 106.67, 113.33, 120] (interpolated)
  
  For gap = 6-24 hours:
    Action: Forward fill followed by linear interpolation
    Reason: Slight continuity break; fill with immediate past, then interpolate
    Example: [100, NaN, NaN, NaN, NaN, NaN, NaN, 140] (7-hour gap)
             → [100, 100, 105.71, 111.43, 117.14, 122.86, 128.57, 140]
  
  For gap > 24 hours:
    Action: Break sequence, discard data before gap
    Reason: Long gaps fundamentally break temporal continuity; LSTM can't learn across breaks
    Example: [100, 105, 110, NaN (48h gap), 120, 125]
             → Keep [100, 105, 110] as separate sequence
             → Start new sequence at 120

Step 3: Data retention
  Input: 170,591 records
  After interpolation: ~125,000 records (~73%)
  
  Loss sources:
    - 6-hour gaps: ~5% loss
    - 24-hour gaps: ~15% loss
    - Extremely long gaps (SILTARA 2023): ~7% loss
```

**Critical insight**: Losing data on >24h gaps is acceptable because LSTM's 672h lookback for t+28d doesn't cross those breaks anyway. Each sequence is independent.

### 3.3 RobustScaler (LSTM-Specific)

**Why RobustScaler, not StandardScaler?**

```
StandardScaler (mean/std):
  z = (x - mean) / std
  Problem: Outliers shift mean and inflate std
  Example: [100, 105, 110, 115, 1000]
           mean = 286, std = 381
           z_outlier = (1000 - 286) / 381 = 1.88 (only 1.88 std away)
           ↑ Outlier not clearly distinguished

RobustScaler (median/IQR):
  z = (x - median) / IQR
  Advantage: Median/IQR robust to outliers
  Example: [100, 105, 110, 115, 1000]
           median = 110, IQR = 15 (from q1=105 to q3=115)
           z_outlier = (1000 - 110) / 15 = 59.3 (clearly extreme!)
           ↑ Outlier clearly distinguished

For LSTM:
  - Attention learns to weight extreme z-scores down
  - RobustScaler preserves this distinction
  - Gradient flow remains stable (not saturated)
```

### 3.4 Sequence Creation (Adaptive per Horizon)

**Objective**: Create sequences of length = 2 × horizon

```python
def create_lstm_sequences(data, horizons=[1, 12, 24, 168, 672]):
    """
    horizons in hours: [1h, 12h, 24h, 7d, 28d]
    """
    sequences = []
    
    for horizon in horizons:
        seq_len = max(2 * horizon, 2)  # min 2 hours
        
        for t in range(seq_len, len(data) - horizon):
            # Input sequence: [t - seq_len, ..., t - 1]
            X = data[t - seq_len:t, :]  # shape: (seq_len, n_features)
            
            # Target: [t + horizon]
            y = data[t + horizon, :]  # shape: (n_features,)
            
            sequences.append({
                'X': X,
                'y': y,
                'horizon': horizon,
                'timestamp': data.timestamp[t]
            })
    
    return sequences
```

**Sequence length examples:**
```
Horizon    Seq_len    Reason
-------    -------    ------
t+1h       2h         Minimal context (yesterday's hour + this hour)
t+12h      24h        Full day cycle
t+24h      48h        2-day cycle (weekday patterns)
t+7d       336h       Full week (Monday effect, etc.)
t+28d      672h       Full 4 weeks (seasonal patterns within month)
```

### 3.5 Lag Features & Time Encodings

**Features to add (per sequence timestep):**

```
1. Lag features (redundant for LSTM, but help attention):
   - pm25_lag_1, pm25_lag_3, pm25_lag_6, pm25_lag_12, pm25_lag_24, pm25_lag_48, pm25_lag_168
   - Reason: Make long-range dependencies explicit (attention can focus on few lags)

2. Cyclical time encodings (help model learn periodic patterns):
   - hour_sin = sin(2π × hour / 24)
   - hour_cos = cos(2π × hour / 24)
   - day_sin = sin(2π × day_of_week / 7)
   - day_cos = cos(2π × day_of_week / 7)
   - month_sin = sin(2π × month / 12)
   - month_cos = cos(2π × month / 12)
   - Reason: Periodic patterns repeat; sin/cos encode periodicity naturally

3. Region encoding (tell LSTM which region):
   - region_id: 0=Bhatagaon, 1=IGKV, 2=AIIMS, 3=SILTARA
   - Reason: Different regions have different baselines (embed in model input)

4. Outlier flag (mark suspicious values):
   - is_outlier: 1 if |z_score| > 3, else 0
   - Reason: Attention can learn to downweight outliers, or use as feature
```

**Final feature vector (15 features):**
```
Raw pollution (4):
  - pm25, pm10, no2, o3

Weather (4):
  - temperature, humidity, wind_speed, wind_direction

Time encodings (6):
  - hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos

Region & outlier (2):
  - region_id, is_outlier
  
Total: 15 features per timestep
```

### 3.6 Train/Val/Test Split (Time-Based)

```
Data timeline: 2022-01-01 to 2025-04-16 (total ~39 months)

Split strategy:
  - NO SHUFFLING (preserve temporal order)
  - Use explicit date cutoffs

Train (70%):
  Start: 2022-01-01
  End:   2024-03-31
  Duration: 27 months
  Samples: ~0.70 × 125,000 = ~87,500 sequences

Validation (15%):
  Start: 2024-04-01
  End:   2024-06-30
  Duration: 3 months
  Samples: ~0.15 × 125,000 = ~18,750 sequences

Test (15%):
  Start: 2024-07-01
  End:   2025-04-16
  Duration: 9 months
  Samples: ~0.15 × 125,000 = ~18,750 sequences

Total: 125,000 sequences ✓
```

**Why time-based, not random split?**
- Random split mixes future data into training (data leakage)
- Time-based split respects temporal dependency
- Validation/test are "future" relative to training (realistic)

### 3.7 LSTM Data Output Format

```python
{
    'X_train': (87500, seq_len_adaptive, 15),  # Sequences varying length per horizon
    'y_train': (87500, 4),  # Target pollution (PM2.5, PM10, NO2, O3)
    'region_train': (87500,),  # Region ID for weighting
    'horizon_train': (87500,),  # Horizon for each sequence
    
    'X_val': (18750, seq_len_adaptive, 15),
    'y_val': (18750, 4),
    'region_val': (18750,),
    'horizon_val': (18750,),
    
    'X_test': (18750, seq_len_adaptive, 15),
    'y_test': (18750, 4),
    'region_test': (18750,),
    'horizon_test': (18750,)
}
```

---

## 4. XGB Preprocessing Pipeline (preprocess_xgb.py)

### 4.1 Overview
```
Raw data
  ↓
Aggressive missing value handling (interpolate ALL gaps + 6 engineer features)
  ↓
Remove extreme outliers (PM2.5 > 300 or < 0)
  ↓
Create lag features (1, 3, 6, 12, 24, 48, 168h)
  ↓
Create rolling statistics (mean/std at multiple windows)
  ↓
Create lagged weather features (shifted t-1, t-6, t-12, t-24)
  ↓
Create time encodings (hour, day, month, is_weekend, season)
  ↓
Window into single-row observations (one observation = one hour)
  ↓
Create train/val/test split (70/15/15 time-based)
  ↓
XGB-ready data
```

### 4.2 Missing Value Handling (XGB-Centric)

**Strategy**: Fill ALL gaps, encode missingness as features

```
Step 1: Identify gaps
  gaps = {timestamp_gaps, sizes, locations}

Step 2: Aggressive interpolation
  For ALL gaps (regardless of size):
    Action: Linear interpolation + forward fill + backward fill
    Reason: XGB can learn from features; gaps don't break tree structure
    Example: [100, NaN, NaN, NaN, 140] (4-hour gap)
             → [100, 113.3, 126.7, 140] (interpolated)

Step 3: Encode missingness as features (6 features)
  For each observation, compute:
    - hours_since_measurement: hours since last actual (non-interpolated) value
      Example: If last actual measurement 3h ago, value = 3
    
    - interpolation_distance: gap size where observation falls
      Example: If measurement gap was 6h, value = 6
    
    - consecutive_gaps: how many consecutive gaps is this observation in?
      Example: 2nd gap in a row, value = 2
    
    - gap_ratio: (hours_since_measurement / max_gap_in_region) × 100%
      Example: If 3h since measurement and region max gap 12h, value = 25%
    
    - measurement_frequency: (actual_measurements / total_hours) × 100%
      Example: If region has 90% data, value = 90%
    
    - interpolation_flag: binary 1 if interpolated, 0 if actual measurement
      Example: 1 if this value was filled, 0 if measured

Interpretation:
  - Trees learn: "high hours_since_measurement → prediction should shift down (stale data)"
  - Trees learn: "measurement_frequency high → trust this observation more"
  - Missingness captured as features, not gaps
```

### 4.3 Outlier Removal (XGB-Centric)

**Strategy**: Remove extreme values to prevent tree split distortion

```
Step 1: Identify outliers
  For each region independently:
    Q1 = 25th percentile (e.g., 45 µg/m³)
    Q3 = 75th percentile (e.g., 120 µg/m³)
    IQR = Q3 - Q1 = 75 µg/m³
    
    Lower bound = Q1 - 1.5 × IQR = 45 - 112.5 = SKIP (allow negatives to be caught by threshold)
    Upper bound = Q3 + 1.5 × IQR = 120 + 112.5 = 232.5 µg/m³

Step 2: Remove outliers
  For each region:
    Remove if PM2.5 < 0 (impossible)
    Remove if PM2.5 > 300 (extreme outlier, likely error)
    
  Alternative: Region-specific thresholds
    Bhatagaon (higher baseline): remove if > 400
    IGKV (lower baseline): remove if > 250
    
  Reason: Different regions have different natural variability

Step 3: Data retention
  Input: 125,000 records (after LSTM's light gap handling)
  After outlier removal: ~120,000 records (~96%)
  Loss: ~4% (extreme values)
```

**Why >300 threshold for XGB?**
```
Explanation:
  - PM2.5 > 300 is rare (< 1% of data)
  - For XGB: 1000-sample tree split on 10 extreme values damages all 1000 predictions
  - XGB trees can't learn from individual extreme values (they split at splits)
  - Better to lose rare data than corrupt model for all data
  
For LSTM:
  - LSTM learns sequences; one extreme value is okay if pattern makes sense
  - Attention learns to downweight outliers
  - Example: PM2.5 [100, 200, 500, 600, 400, 100] = real spike event
            LSTM keeps for learning, XGB removes for robustness
```

### 4.4 Lag Features (XGB-Centric)

**Strategy**: Create explicit lag features at multiple scales

```
Lags (hours):
  - pm25_lag_1: PM2.5 1 hour ago
  - pm25_lag_3: PM2.5 3 hours ago
  - pm25_lag_6: PM2.5 6 hours ago
  - pm25_lag_12: PM2.5 12 hours ago
  - pm25_lag_24: PM2.5 24 hours ago
  - pm25_lag_48: PM2.5 48 hours ago
  - pm25_lag_168: PM2.5 168 hours ago (1 week)

Repeated per pollutant (PM10, NO2, O3):
  - pm10_lag_1, pm10_lag_3, ..., pm10_lag_168
  - no2_lag_1, no2_lag_3, ..., no2_lag_168
  - o3_lag_1, o3_lag_3, ..., o3_lag_168

Total lag features: 4 pollutants × 7 lags = 28 features

Rationale:
  - Trees learn feature interactions: "if (pm25_lag_1 > 100 AND hour == 8) then predict high"
  - XGB can't learn sequences implicitly; needs explicit features
  - Multiple lags help tree discover important timescales
```

### 4.5 Rolling Statistics (XGB-Centric)

```
Windows:
  - 6-hour rolling: mean, std
  - 12-hour rolling: mean, std
  - 24-hour rolling: mean, std

Per pollutant:
  - rolling_6h_mean, rolling_6h_std
  - rolling_12h_mean, rolling_12h_std
  - rolling_24h_mean, rolling_24h_std

Total: 4 pollutants × 6 rolling features = 24 features

Rationale:
  - Mean captures trend: "if 6h average high, next hour likely high"
  - Std captures volatility: "if variance high, next hour likely volatile"
  - Trees learn: "high rolling_24h_std → less predictable → make interval wider"
```

### 4.6 Lagged Weather Features

```
Original weather (current hour t):
  - temperature[t], humidity[t], wind_speed[t], wind_direction[t]

Lagged weather (NEVER future, ALWAYS historical):
  - temperature[t-1], humidity[t-1], wind_speed[t-1], wind_direction[t-1]
  - temperature[t-6], humidity[t-6], wind_speed[t-6], wind_direction[t-6]
  - temperature[t-12], humidity[t-12], wind_speed[t-12], wind_direction[t-12]
  - temperature[t-24], humidity[t-24], wind_speed[t-24], wind_direction[t-24]

Total: 4 weather × 5 time shifts = 20 features

Rationale:
  - Weather drives pollution: temperature inversion traps pollution, wind disperses it
  - Historical weather influenced current pollution
  - Example: "If temperature[t-24] high AND wind_speed[t-24] low, pollution accumulated"
  - NEVER use current weather (not known at forecast time for operational use)
```

### 4.7 Time Features

```
Raw time features:
  - hour: 0-23
  - day_of_week: 0-6 (0=Monday)
  - day_of_month: 1-31
  - month: 1-12
  - is_weekend: binary (1 if Saturday/Sunday)
  - season: 0-3 (0=Winter, 1=Spring, 2=Summer, 3=Fall)

Why multiple time features (not just hour)?
  - Trees learn patterns: "if (hour == 8 AND day == 'Monday') then morning rush"
  - Season matters: PM2.5 peaks in winter (inversion) vs summer (ozone)

Total: 6 features
```

### 4.8 Final Feature Vector (~55-60 features)

```
Raw pollution (4):
  - pm25, pm10, no2, o3 (current hour)

Lag features (28):
  - pm25_lag_{1,3,6,12,24,48,168}
  - pm10_lag_{1,3,6,12,24,48,168}
  - no2_lag_{1,3,6,12,24,48,168}
  - o3_lag_{1,3,6,12,24,48,168}

Rolling statistics (24):
  - rolling_{6h,12h,24h}_mean for {pm25, pm10, no2, o3}
  - rolling_{6h,12h,24h}_std for {pm25, pm10, no2, o3}

Weather lags (20):
  - temperature[t], temperature[t-1,6,12,24]
  - humidity[t], humidity[t-1,6,12,24]
  - wind_speed[t], wind_speed[t-1,6,12,24]
  - wind_direction[t], wind_direction[t-1,6,12,24]

Missingness features (6):
  - hours_since_measurement
  - interpolation_distance
  - consecutive_gaps
  - gap_ratio
  - measurement_frequency
  - interpolation_flag

Time features (6):
  - hour, day_of_week, day_of_month, month, is_weekend, season

Total: 4 + 28 + 24 + 20 + 6 + 6 = 88 features (max)
Practical: ~55-60 after feature selection
```

### 4.9 Window into Single Rows

**Transform sequences into flat observation rows:**

```python
def window_to_rows(data_with_features):
    """
    Input: Time series with all lagged features computed
    Output: One row per hour, each row has 55-60 features
    """
    rows = []
    
    for t in range(lag_max, len(data)):  # Skip first N rows (no lags available)
        row = {
            'timestamp': data['timestamp'][t],
            'region': data['region'][t],
            
            # Raw features
            'pm25': data['pm25'][t],
            'pm10': data['pm10'][t],
            'no2': data['no2'][t],
            'o3': data['o3'][t],
            
            # Lag features (already computed)
            'pm25_lag_1': data['pm25_lag_1'][t],
            'pm25_lag_3': data['pm25_lag_3'][t],
            ...,
            
            # Rolling stats (already computed)
            'rolling_6h_mean': data['rolling_6h_mean'][t],
            ...,
            
            # Time features (already computed)
            'hour': data['hour'][t],
            ...,
        }
        rows.append(row)
    
    return pd.DataFrame(rows)  # (n_samples, 55-60 columns)
```

### 4.10 Train/Val/Test Split (Same as LSTM)

```
Train (70%): 2022-01-01 to 2024-03-31 (~87,500 rows)
Val (15%): 2024-04-01 to 2024-06-30 (~18,750 rows)
Test (15%): 2024-07-01 to 2025-04-16 (~18,750 rows)

Total: ~125,000 rows ✓
```

### 4.11 XGB Data Output Format

```python
{
    'X_train': (87500, 55),  # Flat rows with 55 features
    'y_train': (87500,),  # Target (single pollutant, e.g., PM2.5)
    'region_train': (87500,),  # Region for fairness evaluation
    
    'X_val': (18750, 55),
    'y_val': (18750,),
    'region_val': (18750,),
    
    'X_test': (18750, 55),
    'y_test': (18750,),
    'region_test': (18750,)
}

Note: Separate models for each quantile (p5, p50, p95, p99)
      So actually 4 sets of outputs per pollutant
```

---

## 5. Data Loss Breakdown & Recovery

### 5.1 Expected Data Retention

```
Stage                          Records      Retention    Loss
-----                          -------      ---------    ----
Raw data                        170,591      100%         0%
  ↓ Drop impossible values      168,500      98.8%        1.2%
  ↓ Interpolate gaps <6h        165,000      96.7%        3.1%
  ↓ Break on gaps >6h           125,000      73.3%        23.4%
  ↓ (LSTM pipeline end)

  ↓ Aggressive interpolation    160,000      93.8%        6.2%
  ↓ Remove outliers (>300)      120,000      70.4%        29.6%
  ↓ (XGB pipeline end)
```

**Note**: Different retention for LSTM (99% → 73%) vs XGB (99% → 70%) is expected:
- LSTM breaks sequences on long gaps → more loss
- XGB interpolates all gaps → less loss from gaps, more from outlier removal

### 5.2 Recovery Opportunities

```
Opportunity 1: Adjust gap threshold
  Current LSTM: break on gaps > 6h
  Test alternative: break on gaps > 12h
  Expected impact: +3-5% more data
  
Opportunity 2: Region-specific thresholds
  Current: Same threshold for all regions
  Alternative: Bhatagaon (natural high) → break on gaps > 12h
              IGKV (cleaner) → break on gaps > 6h
  Expected impact: +2-3% for Bhatagaon, more fairness

Opportunity 3: SILTARA 2023 cyclical interpolation
  Current: SILTARA missing entire 2023 (365-day gap → sequence breaks)
  Alternative: Average same day-of-year from 2022 & 2024
  Expected impact: +7% more data for SILTARA

Opportunity 4: Looser outlier bounds for XGB
  Current: Remove PM2.5 > 300
  Alternative: Remove PM2.5 > 400 (3-sigma for normal baseline)
  Trade-off: +2% more data, slightly worse tree calibration
```

---

## 6. Comparison: LSTM vs XGB Preprocessing

| Aspect | LSTM | XGB |
|--------|------|-----|
| **Gap handling** | Break sequences (>6h) | Interpolate all |
| **Outlier handling** | Keep all | Remove >300 |
| **Scaling** | RobustScaler | None |
| **Features** | 15 minimal | 55 rich |
| **Feature timing** | Sequences then lags | All features then rows |
| **Data retention** | ~73% | ~70% |
| **Rationale** | Preserve temporal patterns | Prevent tree splits on noise |
| **Strength** | Long horizons (7d/28d) | Short horizons (1h/12h) |

---

## 7. Implementation Checklist

- [ ] Load raw data from Excel files (shared)
- [ ] Validate data (drop impossible values)
- [ ] Split into two pipelines (LSTM & XGB branches)

**LSTM Pipeline:**
- [ ] Interpolate gaps <6h
- [ ] Break sequences on gaps >6h
- [ ] Apply RobustScaler
- [ ] Create adaptive-length sequences (2×horizon)
- [ ] Add lag features & time encodings
- [ ] Create 70/15/15 time-based split
- [ ] Save LSTM-ready data

**XGB Pipeline:**
- [ ] Interpolate all gaps
- [ ] Create missingness features (6)
- [ ] Remove outliers (PM2.5 >300)
- [ ] Create lag features (7 lags)
- [ ] Create rolling statistics
- [ ] Create lagged weather features
- [ ] Create time encodings
- [ ] Window into flat rows
- [ ] Create 70/15/15 time-based split
- [ ] Save XGB-ready data

---

## 8. QA/Testing

```python
# Verify no data leakage
assert train_data.timestamp.max() < val_data.timestamp.min()
assert val_data.timestamp.max() < test_data.timestamp.min()

# Verify sequence order
for seq in lstm_sequences:
    assert seq['X'].timestamp[-1] < seq['y'].timestamp
    assert seq['y'].timestamp + horizon == seq['target'].timestamp

# Verify features don't include future
for feature in xgb_features:
    assert feature.lag >= 0  # No negative (future) lags

# Verify region representation
assert len(train_data[train_data.region == 'Bhatagaon']) > 0  # All regions present
assert len(train_data[train_data.region == 'IGKV']) > 0

# Verify data shapes
assert lstm_X_train.shape == (87500, seq_len_var, 15)
assert xgb_X_train.shape == (87500, 55)
```

