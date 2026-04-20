# DATA_INVESTIGATION.md - Phase 1: Extreme Data Analysis

## ⚠️ CRITICAL: GIT INSTRUCTIONS
**DO NOT RUN ANY GIT COMMANDS** (git add, git commit, git push, or any git modification commands).
See DECISIONS.md for full git warning. Work directly on files; user handles git operations.

## Purpose
Document findings from Phase 1 data investigation, identifying and resolving extreme data quality issues before model training.

**Note**: This is a FRESH IMPLEMENTATION. Disregard all previous code. Treat as starting from zero.

---

## 1. Bhatagaon September 2025 Spike Analysis

### 1.1 Investigation Objective
Classify PM2.5 spike >500 µg/m³ as either:
- **Real pollution event** (gradual ramp-up + decay + weather correlation) → KEEP
- **Sensor error** (sudden spike + immediate return) → REMOVE

### 1.2 Analysis Procedure

**Step 1: Locate extreme values**
```python
# Find all PM2.5 > 500 in Bhatagaon Sept 2025
bhat_sept = data[
    (data['region'] == 'Bhatagaon') & 
    (data['timestamp'].dt.month == 9) & 
    (data['timestamp'].dt.year == 2025)
]
extremes = bhat_sept[bhat_sept['pm25'] > 500]
print(f"Found {len(extremes)} extreme values")
print(extremes[['timestamp', 'pm25', 'temperature', 'humidity', 'wind_speed']])
```

**Step 2: Analyze temporal pattern**
```python
# For each extreme spike, check ±12 hours
for idx, spike in extremes.iterrows():
    t = spike['timestamp']
    window = data[
        (data['region'] == 'Bhatagaon') & 
        (data['timestamp'] >= t - pd.Timedelta(hours=12)) &
        (data['timestamp'] <= t + pd.Timedelta(hours=12))
    ].sort_values('timestamp')
    
    # Plot temporal pattern
    plt.plot(window['timestamp'], window['pm25'], marker='o')
    plt.axvline(t, color='red', label='Spike')
    plt.title(f"Temporal pattern: {t}")
    plt.xlabel('Time')
    plt.ylabel('PM2.5 µg/m³')
    plt.legend()
    plt.show()
    
    # Check pattern type
    pm25_values = window['pm25'].values
    print(f"Pattern: {pm25_values}")
    print(f"Min: {pm25_values.min()}, Max: {pm25_values.max()}, Std: {pm25_values.std()}")
```

**Step 3: Assess regional correlation**
```python
# Did other regions show similar spike at same time?
t_spike = spike['timestamp']
other_regions = data[
    (data['region'] != 'Bhatagaon') & 
    (data['timestamp'] == t_spike)
]
print(f"Other regions at spike time: {other_regions[['region', 'pm25']]}")

# If all regions spiked simultaneously → likely real pollution event
# If only Bhatagaon spiked → likely sensor error
```

**Step 4: Check weather context**
```python
# Were conditions conducive to extreme pollution?
weather = data[
    (data['region'] == 'Bhatagaon') & 
    (data['timestamp'] >= t_spike - pd.Timedelta(hours=24)) &
    (data['timestamp'] <= t_spike)
][['timestamp', 'temperature', 'humidity', 'wind_speed', 'wind_direction']]

print(f"Weather before spike: {weather}")

# Real pollution event indicators:
#   - Temperature inversion (temp drops overnight)
#   - Low wind speed (stagnation)
#   - High humidity (moisture traps particles)
# Sensor error indicators:
#   - Normal weather patterns
#   - High wind (should disperse pollution)
```

### 1.3 Decision Framework

**Decision Rule:**

```
IF (
  temporal_pattern == 'ramp_up_and_decay' AND  # Gradual increase/decrease over hours
  other_regions_correlated == True AND  # Other regions also elevated
  weather_supports == True  # Inversion, low wind, high humidity
):
    classification = "REAL EVENT"
    action = "KEEP values"
    
ELIF (
  temporal_pattern == 'single_point_spike' AND  # t-1: normal, t: extreme, t+1: normal
  other_regions_correlated == False AND  # Other regions normal
  weather_neutral == True  # Normal weather, high wind
):
    classification = "SENSOR ERROR"
    action = "REMOVE values"
    
ELSE:
    classification = "UNCERTAIN"
    action = "INSPECT MANUALLY with domain expert"
```

### 1.4 Expected Output (to fill in after analysis)

```
Bhatagaon Sept 2025 Analysis Results:
====================================

Extreme values found: 7 instances of PM2.5 > 500

Cluster 1 (2 points):
  Time window: 2025-09-01 23:00 to 2025-09-02 00:00
  Peak PM2.5: 3248.78 µg/m³
  Temporal pattern: abrupt multi-hour spike
  Other regions: NOT correlated (max at peak time = 14.49)
  Weather: supports stagnation (humidity ~98%, wind ~0.07), but not regional event
  Classification: SENSOR ERROR
  Action: REMOVE
  Evidence: pre-spike 36.04 -> spike 3248.78 -> post-spike 1.60

Cluster 2 (2 points):
  Time window: 2025-09-10 08:00 to 2025-09-10 09:00
  Peak PM2.5: 2963.90 µg/m³
  Temporal pattern: abrupt multi-hour spike
  Other regions: NOT correlated (max at peak time = 58.74)
  Weather: low wind + high humidity, but no cross-region pollution confirmation
  Classification: SENSOR ERROR
  Action: REMOVE
  Evidence: pre-spike 9.32 -> spike 2963.90 -> post-spike 14.71

Cluster 3 (3 points):
  Time window: 2025-09-11 03:00 to 2025-09-11 05:00
  Peak PM2.5: 5904.78 µg/m³
  Temporal pattern: abrupt multi-hour spike
  Other regions: NOT correlated (max at peak time = 19.78)
  Weather: low wind + high humidity, but still localized to Bhatagaon sensor
  Classification: SENSOR ERROR
  Action: REMOVE
  Evidence: pre-spike 8.36 -> spike 5904.78 -> post-spike 54.60

Summary:
  - Total extreme values: 7
  - Real events: 0 (action: keep)
  - Sensor errors: 7 (3 clusters; action: remove)
  - Uncertain: 0
  
Recommendation:
  - Remove only the identified Bhatagaon Sept 2025 sensor-error spikes (>500 in 3 abrupt clusters)
  - Keep non-impossible high values outside these clusters for LSTM if temporal pattern is plausible
  - Use uniform XGB outlier-cap policy across pollutants (PM2.5/PM10/NO2/O3)
```

---

## 2. Data Loss Root Cause Analysis

### 2.1 Investigation Objective
Identify root-cause attrition in canonical hourly pipeline; find recovery opportunities

### 2.2 Analysis Procedure

**Step 1: Per-region loss tracking**
```python
# Measure retention at each preprocessing stage
regions = ['Bhatagaon', 'IGKV', 'AIIMS', 'SILTARA']

for region in regions:
    raw = len(data[data['region'] == region])
    after_missing = len(preprocess_step_missing[region])
    after_outliers = len(preprocess_step_outliers[region])
    after_sequences = len(preprocess_step_sequences[region])
    
    print(f"{region}:")
    print(f"  Raw: {raw} (100%)")
    print(f"  After missing handling: {after_missing} ({100*after_missing/raw:.1f}%)")
    print(f"  After outlier removal: {after_outliers} ({100*after_outliers/raw:.1f}%)")
    print(f"  After sequence creation: {after_sequences} ({100*after_sequences/raw:.1f}%)")
    print(f"  Total retention: {100*after_sequences/raw:.1f}%")
    
    # Loss sources
    missing_loss = raw - after_missing
    outlier_loss = after_missing - after_outliers
    sequence_loss = after_outliers - after_sequences
    
    print(f"  Loss from missing: {missing_loss} ({100*missing_loss/raw:.1f}%)")
    print(f"  Loss from outliers: {outlier_loss} ({100*outlier_loss/raw:.1f}%)")
    print(f"  Loss from sequences: {sequence_loss} ({100*sequence_loss/raw:.1f}%)")
    print()
```

**Step 2: Per-step loss attribution**
```python
# Track what happens at each preprocessing stage

stage_tracking = {
    'raw_data': 170591,
    'after_impossible_values': None,  # Drop negatives, sensor errors
    'after_gap_interpolation': None,  # Fill <6h gaps
    'after_gap_breaking': None,  # Break on >6h gaps
    'after_outlier_removal': None,  # Remove >IQR
    'after_sequence_creation': None,  # Window into sequences
}

for stage, count in stage_tracking.items():
    if count is not None:
        prev_count = list(stage_tracking.values())[list(stage_tracking.keys()).index(stage) - 1]
        loss = prev_count - count if prev_count else None
        pct = 100 * loss / prev_count if loss and prev_count else None
        print(f"{stage}: {count} ({pct:.1f}% loss)")
```

**Step 3: Region-specific gap analysis**
```python
# Which regions have more gaps?

for region in regions:
    data_region = data[data['region'] == region].sort_values('timestamp')
    timestamps = data_region['timestamp'].values
    gaps = pd.Series(timestamps[1:]) - pd.Series(timestamps[:-1])
    
    gap_counts = {
        '<6h': (gaps < pd.Timedelta(hours=6)).sum(),
        '6-24h': ((gaps >= pd.Timedelta(hours=6)) & (gaps < pd.Timedelta(hours=24))).sum(),
        '24h-7d': ((gaps >= pd.Timedelta(hours=24)) & (gaps < pd.Timedelta(days=7))).sum(),
        '>7d': (gaps >= pd.Timedelta(days=7)).sum(),
    }
    
    print(f"{region} gaps:")
    for size, count in gap_counts.items():
        print(f"  {size}: {count} instances")
```

**Step 4: SILTARA 2023 missing year analysis**
```python
# Is entire 2023 missing for SILTARA?

siltara_2023 = data[(data['region'] == 'SILTARA') & (data['timestamp'].dt.year == 2023)]
print(f"SILTARA 2023 records: {len(siltara_2023)}")

if len(siltara_2023) == 0:
    print("SILTARA 2023 is COMPLETELY MISSING - 365-day gap")
    print("This creates long sequence breaks for LSTM")
    
    # Check before & after
    siltara_2022 = data[(data['region'] == 'SILTARA') & (data['timestamp'].dt.year == 2022)]
    siltara_2024 = data[(data['region'] == 'SILTARA') & (data['timestamp'].dt.year == 2024)]
    print(f"SILTARA 2022: {len(siltara_2022)} records")
    print(f"SILTARA 2024: {len(siltara_2024)} records")
```

### 2.3 Expected Output (to fill in after analysis)

```
Data Loss Root Cause Analysis:
==============================

Note on baseline:
  - The legacy document baseline (170,591 -> 116,257) comes from older pipeline behavior.
  - Fresh Phase 1 implementation now combines hourly + quarterly sources and canonicalizes to hourly:
    750,540 parsed rows (169,611 hourly + 580,929 quarterly) -> 125,017 canonical hourly-unique rows.
  - Quarterly contribution is material:
    3,690 hourly timestamps are quarterly-only, and 5,689 PM2.5 values are filled from quarterly data.

Per-region retention:
  Bhatagaon: 96.55% retention (31,440 -> 30,354)
  IGKV: 99.39% retention (31,008 -> 30,818)
  AIIMS: 99.60% retention (31,415 -> 31,289)
  SILTARA: 96.38% retention (31,154 -> 30,026)

Per-step loss attribution:
  1. Canonical hourly-unique baseline: 125,017 records
  2. Drop impossible PM2.5 (<0 or >5000): -4,469 records (3.57%)
  3. Interpolate gaps <6h: +2,260 records recovered (1.81%)
  4. Remaining unresolved missing after interpolation: 2,948 hour-slots (2.36% of baseline timeline)
  5. XGB-style outlier removal (>300): -156 records (0.12%)
  6. Sequence boundary loss (contiguous-block edges): -165 records (0.13%)
  Final sequence-ready: 122,487 records (97.98% retention from canonical baseline)

Key findings:
  - Quarterly ingestion improves effective coverage and smooths regional continuity.
  - The largest true quality loss source remains impossible values (4,469), not outlier trimming.
  - Gap interpolation still recovers substantial data (+2,260).
  - SILTARA has the highest unresolved gaps and missing burden; AIIMS/IGKV are comparatively clean.
  
  - SILTARA missing 2023: NO
    SILTARA 2023 records found: 8,737 (so no full-year data hole)

Recovery opportunities:
  1. Relax interpolation threshold 6h -> 12h: +784 sequence samples (+0.64%)
  2. Relax interpolation threshold 6h -> 24h: +1,429 sequence samples (+1.17%)
  3. Region-specific handling is most useful for SILTARA (more 6h+ and 24h+ gaps)
  4. Looser outlier bounds provide limited gain relative to impossible-value filtering quality controls

Recommendation:
  - Keep 6h interpolation as conservative default for production robustness
  - Evaluate 12h interpolation as a low-risk ablation (especially SILTARA)
  - Prioritize impossible-value filtering + sensor-error removal over aggressive outlier clipping
  - Keep canonical merge contract in downstream preprocessing (hourly preferred, quarterly gap fill, floor-to-hour)
```

---

## 3. Region Imbalance Quantification

### 3.1 Investigation Objective
Characterize post-canonical region imbalance and plan mitigation

### 3.2 Analysis Procedure

**Step 1: Raw and preprocessed record counts**
```python
# Count records at each stage

regions = ['Bhatagaon', 'IGKV', 'AIIMS', 'SILTARA']

print("Raw data distribution:")
for region in regions:
    count = len(data[data['region'] == region])
    pct = 100 * count / len(data)
    print(f"  {region}: {count} ({pct:.1f}%)")

print("\nAfter preprocessing:")
for region in regions:
    count = len(preprocessed[preprocessed['region'] == region])
    pct = 100 * count / len(preprocessed)
    print(f"  {region}: {count} ({pct:.1f}%)")

# Calculate imbalance ratio
region_counts = [len(preprocessed[preprocessed['region'] == r]) for r in regions]
imbalance_ratio = max(region_counts) / min(region_counts)
print(f"\nImbalance ratio: {imbalance_ratio:.1f}×")
```

**Step 2: Seasonal patterns**
```python
# Does imbalance vary by season?

for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    season_mask = get_season_mask(data, season)
    print(f"\n{season}:")
    for region in regions:
        count = len(data[season_mask & (data['region'] == region)])
        pct = 100 * count / season_mask.sum()
        print(f"  {region}: {pct:.1f}%")
```

**Step 3: Data quality by region**
```python
# Is imbalance due to poor data quality or fewer stations?

for region in regions:
    data_region = data[data['region'] == region]
    
    # Measure data quality
    missing_pct = 100 * data_region['pm25'].isna().sum() / len(data_region)
    outlier_pct = 100 * (
        (data_region['pm25'] > 300) | 
        (data_region['pm25'] < 0)
    ).sum() / len(data_region)
    
    print(f"\n{region}:")
    print(f"  Missing data: {missing_pct:.1f}%")
    print(f"  Outliers (>300 or <0): {outlier_pct:.1f}%")
    print(f"  Effective data: {100 - missing_pct - outlier_pct:.1f}%")
    
    # If imbalance is due to poor data, that's OK (skip region)
    # If imbalance is just fewer stations, we need to upweight (don't skip)
```

**Step 4: Historical trend**
```python
# Is imbalance increasing or decreasing over time?

for year in [2022, 2023, 2024, 2025]:
    year_mask = data['timestamp'].dt.year == year
    year_data = data[year_mask]
    
    print(f"\n{year}:")
    for region in regions:
        count = len(year_data[year_data['region'] == region])
        pct = 100 * count / len(year_data)
        print(f"  {region}: {pct:.1f}%")
```

### 3.3 Expected Output (to fill in after analysis)

```
Region Imbalance Quantification:
================================

Raw data distribution:
  Bhatagaon: 31,440 (25.15%)
  IGKV: 31,008 (24.80%)
  AIIMS: 31,415 (25.13%)
  SILTARA: 31,154 (24.92%)
  Imbalance ratio: 1.01×

After preprocessing:
  Bhatagaon: 30,354 (24.78%)
  IGKV: 30,818 (25.16%)
  AIIMS: 31,289 (25.54%)
  SILTARA: 30,026 (24.51%)
  Imbalance ratio: 1.04×

Seasonal patterns:
  Year-wise variation is low:
    2022 ratio: ~1.00×
    2023 ratio: ~1.00×
    2024 ratio: ~1.05×
    2025 ratio: ~1.03×

Data quality by region:
  Bhatagaon: 4.82% missing, 0.33% outliers -> 94.85% effective
  IGKV: 1.33% missing, 0.01% outliers -> 98.66% effective
  AIIMS: 1.34% missing, 0.08% outliers -> 98.58% effective
  SILTARA: 6.68% missing, 0.13% outliers -> 93.20% effective

Historical trend:
  [2022] Imbalance ratio: ~1.00×
  [2023] Imbalance ratio: ~1.00×
  [2024] Imbalance ratio: ~1.05×
  [2025] Imbalance ratio: ~1.03×
  Trend: near-uniform across years after quarterly integration

Interpretation:
  - Imbalance is NOT severe after canonical hourly de-duplication with quarterly integration.
  - Quarterly data helps close coverage gaps and improves per-region parity.
  
Mitigation strategy (for training phase):
  1. Weighted loss: use mild weights from post-distribution
     - Bhatagaon: 1.009×, IGKV: 0.994×, AIIMS: 0.979×, SILTARA: 1.020×
  2. Stratified sampling: keep approximately equal regional composition per batch
  3. Per-region evaluation: still required for fairness auditing
  4. If fairness drifts in training, increase weight spread incrementally
```

---

## 4. Summary & Recommendations

### 4.1 Action Items (to fill in after investigations)

```
Bhatagaon Spike Analysis:
  [x] Spike classification: SENSOR ERROR (all 3 spike clusters)
  [x] If ERROR: Remove identified extreme outlier instances (7 points)

Data Loss Analysis:
  [x] Identify primary loss source: impossible values + unresolved long gaps
  [x] Evaluate recovery opportunities:
      [x] Adjust gap threshold tested (6h -> 12h/24h)
      [x] Region-specific threshold opportunity identified (SILTARA)
      [x] SILTARA 2023 interpolation not needed (year is present)
  [x] Estimate data recovery potential: +0.64% (12h), +1.17% (24h)
  [x] Baseline decision: keep conservative 6h default; evaluate relaxed threshold as ablation

Region Imbalance:
  [x] Confirm imbalance is mild after canonical hourly+quarterly deduplication
  [x] Set region weights for training:
      - Bhatagaon: 1.009×
      - IGKV: 0.994×
      - AIIMS: 0.979×
      - SILTARA: 1.020×
  [x] Plan stratified batch sampling in data loader
  [x] Keep per-region evaluation in Phase 6
```

### 4.2 Preprocessing Parameters (to fill in)

```
Based on Phase 1 analysis, recommend:

LSTM preprocessing (preprocess_lstm.py):
  - Gap interpolation threshold: 6h (default; conservative)
  - Gap breaking threshold: >6h breaks sequences
  - Outlier handling: keep plausible outliers, remove impossible values and confirmed sensor-error spikes
  - RobustScaler: Use median/IQR
  - Data retention target from canonical hourly baseline: ~98.0%

XGB preprocessing (preprocess_xgb.py):
  - Gap interpolation: ALL gaps (complete)
  - Outlier removal thresholds (uniform policy across pollutants):
    * PM2.5 >300
    * PM10 >600
    * NO2 >250
    * O3 >150
  - Missingness features: Add 6 engineered features
  - Feature engineering: as specified in PREPROCESSING_STRATEGY.md
  - Data retention target from canonical hourly baseline: ~98.0% after baseline filtering,
    with optional +0.64% to +1.17% via relaxed interpolation

Region weighting (training):
  - Loss weight formula: weight_r = (1/4) / fraction_r
  - Calculated weights:
    * Bhatagaon: 1.009×
    * IGKV: 0.994×
    * AIIMS: 0.979×
    * SILTARA: 1.020×

```

### 4.3 All-Pollutant Investigation Extension (Completed)

To broaden Phase 1 beyond PM2.5-only diagnostics, `scripts/data_investigation.py`
now computes outlier-tail and loss-attribution summaries for **all four pollutants**
(`pm25`, `pm10`, `no2`, `o3`) and writes them to:

- `data/raw/phase1_investigation_results.json`
  - `all_pollutants_outlier_analysis`
  - `all_pollutants_loss`
- visualizations:
  - `visualizations/phase_1_data_investigation/all_pollutants_distribution_tails.png`
  - `visualizations/phase_1_data_investigation/all_pollutants_threshold_exceedance.png`
  - `visualizations/phase_1_data_investigation/all_pollutants_tail_quantiles.png`

Latest totals from the extended run:

```
PM2.5  : raw 125,017 -> after_sequence 122,487
         loss_impossible=4,469  interpolated_gain=2,260  loss_outliers=156

PM10   : raw 125,017 -> after_sequence 121,539
         loss_impossible=5,490  interpolated_gain=2,109  loss_outliers=0

NO2    : raw 125,017 -> after_sequence 122,706
         loss_impossible=3,853  interpolated_gain=1,645  loss_outliers=0

O3     : raw 125,017 -> after_sequence 122,646
         loss_impossible=3,568  interpolated_gain=1,281  loss_outliers=0
```

Interpretation:
- Impossible/sentinel handling is the dominant quality filter across all pollutants.
- PM2.5 remains the only pollutant with an explicit fixed XGB outlier cap policy in this project.
- Other pollutants currently keep high-tail values after impossible-value filtering; this is now explicit and auditable.

Potential next-step ablations enabled by this extension:
1. Add optional XGB pollutant-specific clipping thresholds for `pm10/no2/o3` (off by default).
2. Keep LSTM permissive tails while tightening XGB-only caps for tree robustness.
3. Compare calibration and CRPS impact per pollutant under each clipping policy.

---

## 5. To Be Filled In After Analysis

- [x] Run `scripts/data_investigation.py` to analyze all extreme values
- [x] Document findings in this file (sections 1-3)
- [x] Data-driven thresholds and recovery options documented for Phase 3 implementation
- [x] Region weight update requirement identified (near-uniform weights after hourly+quarterly canonical dedup)
- [x] Recovery/robustness trade-off documented (6h default vs 12h/24h ablation)
