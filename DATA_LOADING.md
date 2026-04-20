# DATA_LOADING.md - Loading Raw Air Pollution Excel Files

**CRITICAL**: This is documentation for FRESH IMPLEMENTATION. Do NOT run any git commands (git add, git commit, git push, etc.). Work directly on files.

**Source of Truth**: All decisions in this file align with DECISIONS.md section 4.X. If code needs to deviate, update DECISIONS.md FIRST.

---

## Overview

This document specifies how to load raw air pollution data from Excel files into pandas DataFrames for preprocessing (Phase 3) and analysis (Phase 1). The raw data is organized by region and time, with inconsistent formatting that requires careful parsing.

---

## Raw Data Structure

### Directory Layout

```
Pollution Data Raipur/
├── Bhatagaon DCR/
│   ├── 2022/
│   │   ├── 01/
│   │   │   └── *.xlsx (multiple files per month)
│   │   ├── 02/
│   │   └── ... (12 months)
│   ├── 2023/
│   │   └── (same monthly structure)
│   └── 2024/
│       └── (same monthly structure)
├── DCR AIIMS/
│   └── (same structure)
├── IGKV DCR/
│   └── (same structure)
└── SILTARA DCR/
    └── (same structure)
```

### Region Name Mapping

| Directory | Standard Name | Code |
|-----------|---------------|------|
| Bhatagaon DCR | Bhatagaon | BH |
| DCR AIIMS | AIIMS | AI |
| IGKV DCR | IGKV | IG |
| SILTARA DCR | Siltara | SI |

---

## Expected Excel Schema

### Sheet Names & Structure

Each Excel file contains **pollutant data** in sheets named after pollutants:
- **PM2.5** (or "PM2_5", "PM2.5 µg/m³")
- **PM10** (or "PM10 µg/m³")
- **NO2** (or "NO2 µg/m³")
- **O3** (or "O3 µg/m³")

**Note**: Sheet names may vary. See "Handling Sheet Name Variations" below.

### Column Structure (Inside Each Sheet)

| Column Position | Expected Name | Data Type | Example | Notes |
|---|---|---|---|---|
| A | Date / Timestamp | str or datetime | "2022-01-15" or "01/15/2022" | Inconsistent formats |
| B | Time | str or int | "00:00" or "0" | May be missing (assume 00:00) |
| C | Value | float | 45.2, 89.5, 502.1 | The actual pollutant reading |
| D+ | Unknown / Metadata | str | (varies) | May contain comments or irrelevant data |

### Data Characteristics

**Valid range by pollutant**:
- **PM2.5**: 0-1000 µg/m³ (typically 0-500)
- **PM10**: 0-1500 µg/m³ (typically 0-800)
- **NO2**: 0-500 µg/m³ (typically 0-300)
- **O3**: 0-400 µg/m³ (typically 0-150)

**Values outside these ranges**: Documented in DATA_INVESTIGATION.md as potential sensor errors.

**Missing data indicators**:
- Empty cells (NaN)
- Text: "N/A", "–", "-", "NA", "null", "Missing"
- Zero: May be actual zero OR missing (context determines)

---

## Loading Code Example

### Basic Function: Load Single Excel File

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_pollutant_file(filepath: str, region_name: str) -> pd.DataFrame:
    """
    Load a single Excel file containing one or more pollutant sheets.
    
    Args:
        filepath: Path to Excel file
        region_name: Name of region (e.g., "Bhatagaon", "IGKV")
    
    Returns:
        DataFrame with columns: [date, time, pollutant, value, region, filepath_source]
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no valid pollutant sheets found
    """
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read Excel file to find sheet names
    try:
        xls = pd.ExcelFile(filepath)
    except Exception as e:
        raise ValueError(f"Cannot read Excel file {filepath}: {e}")
    
    # Pollutant sheet name patterns (case-insensitive)
    pollutant_patterns = {
        "PM2.5": ["PM2.5", "PM2_5", "pm2.5", "pm2_5"],
        "PM10": ["PM10", "pm10"],
        "NO2": ["NO2", "no2"],
        "O3": ["O3", "o3"]
    }
    
    # Find which sheets are pollutants
    pollutant_sheets = {}
    for sheet in xls.sheet_names:
        for pollutant, patterns in pollutant_patterns.items():
            if any(pattern in sheet for pattern in patterns):
                pollutant_sheets[pollutant] = sheet
                break
    
    if not pollutant_sheets:
        raise ValueError(f"No pollutant sheets found in {filepath}. Sheets: {xls.sheet_names}")
    
    # Load each pollutant sheet
    all_data = []
    
    for pollutant, sheet_name in pollutant_sheets.items():
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
        except Exception as e:
            print(f"Warning: Could not read {pollutant} from {sheet_name} in {filepath}: {e}")
            continue
        
        # Skip empty sheets
        if df.empty:
            continue
        
        # Extract columns: Date (A), Time (B), Value (C)
        # Assume first 3 columns are: date, time, value
        if len(df.columns) < 3:
            print(f"Warning: {pollutant} sheet has fewer than 3 columns in {filepath}")
            continue
        
        date_col = df.iloc[:, 0]
        time_col = df.iloc[:, 1] if len(df.columns) > 1 else "00:00"
        value_col = df.iloc[:, 2]
        
        # Parse dates
        parsed_dates = []
        for d in date_col:
            if pd.isna(d):
                continue
            try:
                # Try common formats
                parsed = pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")
                if pd.isna(parsed):
                    parsed = pd.to_datetime(d, format="%m/%d/%Y", errors="coerce")
                if pd.isna(parsed):
                    parsed = pd.to_datetime(d, errors="coerce")
                parsed_dates.append(parsed)
            except:
                continue
        
        # Create dataframe for this pollutant
        pollutant_df = pd.DataFrame({
            "date": date_col[:len(parsed_dates)],
            "time": time_col if isinstance(time_col, str) else time_col[:len(parsed_dates)],
            "value": value_col[:len(parsed_dates)],
        })
        
        # Parse date properly
        pollutant_df["date"] = pd.to_datetime(pollutant_df["date"], errors="coerce")
        
        # Handle time column
        if isinstance(time_col, str):
            pollutant_df["time"] = time_col
        else:
            pollutant_df["time"] = pollutant_df["time"].astype(str).str.replace(".0", "").str.zfill(5)
        
        # Parse value as float, replace missing indicators
        pollutant_df["value"] = pollutant_df["value"].replace(
            ["N/A", "–", "-", "NA", "null", "Missing", ""], np.nan
        )
        pollutant_df["value"] = pd.to_numeric(pollutant_df["value"], errors="coerce")
        
        # Remove rows with missing date or value
        pollutant_df = pollutant_df.dropna(subset=["date", "value"])
        
        # Add metadata
        pollutant_df["pollutant"] = pollutant
        pollutant_df["region"] = region_name
        pollutant_df["filepath_source"] = str(filepath)
        
        all_data.append(pollutant_df)
    
    if not all_data:
        raise ValueError(f"No valid data extracted from {filepath}")
    
    result = pd.concat(all_data, ignore_index=True)
    return result


def load_all_raw_data(data_dir: str = "Pollution Data Raipur") -> pd.DataFrame:
    """
    Load all Excel files from all regions and years.
    
    Args:
        data_dir: Root directory containing region subdirectories
    
    Returns:
        Combined DataFrame with all pollution data
    """
    
    region_mapping = {
        "Bhatagaon DCR": "Bhatagaon",
        "DCR AIIMS": "AIIMS",
        "IGKV DCR": "IGKV",
        "SILTARA DCR": "Siltara"
    }
    
    all_records = []
    data_path = Path(data_dir)
    
    for region_dir in data_path.iterdir():
        if not region_dir.is_dir():
            continue
        
        region_name = region_mapping.get(region_dir.name)
        if not region_name:
            print(f"Warning: Unknown region directory {region_dir.name}")
            continue
        
        # Iterate through years
        for year_dir in region_dir.iterdir():
            if not year_dir.is_dir():
                continue
            
            # Iterate through months
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                
                # Load all Excel files in month directory
                for excel_file in month_dir.glob("*.xlsx"):
                    try:
                        df = load_pollutant_file(str(excel_file), region_name)
                        all_records.append(df)
                        print(f"Loaded: {excel_file.relative_to(data_path)}")
                    except Exception as e:
                        print(f"Error loading {excel_file}: {e}")
    
    if not all_records:
        raise ValueError(f"No data loaded from {data_dir}")
    
    combined = pd.concat(all_records, ignore_index=True)
    
    # Create timestamp from date + time
    combined["timestamp"] = pd.to_datetime(
        combined["date"].astype(str) + " " + combined["time"].astype(str),
        format="%Y-%m-%d %H:%M",
        errors="coerce"
    )
    
    # Remove rows with invalid timestamps
    combined = combined.dropna(subset=["timestamp"])
    
    # Sort by timestamp and region
    combined = combined.sort_values(["region", "pollutant", "timestamp"]).reset_index(drop=True)
    
    return combined
```

---

## Handling Sheet Name Variations

### Problem

Excel files may have inconsistent sheet names across files:

| File 1 | File 2 | File 3 |
|--------|--------|--------|
| "PM2.5" | "PM2_5" | "PM2.5 µg/m³" |
| "PM10" | "PM10 µg/m³" | "PM10" |
| "NO2" | "NO2" | "NO₂" |
| "O3" | "Ozone" | "O3" |

### Solution

Use **fuzzy pattern matching** (implemented above):

```python
# For each sheet name, check if it contains any known pattern
pollutant_patterns = {
    "PM2.5": ["PM2.5", "PM2_5", "pm2.5"],
    "PM10": ["PM10", "pm10"],
    "NO2": ["NO2", "no2"],
    "O3": ["O3", "o3", "Ozone"]
}

for sheet in xls.sheet_names:
    for pollutant, patterns in pollutant_patterns.items():
        if any(pattern in sheet for pattern in patterns):
            # Found the pollutant sheet
            break
```

---

## Data Quality Notes

### Known Issues (See DATA_INVESTIGATION.md for details)

1. **Bhatagaon September 2025 Spike**
   - PM2.5 values >500 µg/m³
   - Determine: Real event vs sensor error
   - Procedure: See DATA_INVESTIGATION.md section 1.2

2. **Missing Data**
   - Canonical pipeline attrition is tracked in Phase 1 outputs (`phase1_investigation_results.json`)
   - Root causes: impossible values, unresolved temporal gaps, and sequence constraints
   - Analysis: Phase 1 data investigation

3. **Region Imbalance**
   - Canonical hourly distribution is near-balanced across all regions (~25% each)
   - Post-sequence imbalance is mild and handled with mild region weights
   - Handled via region weighting (DECISIONS.md section 4.3)

### Validation After Loading

After `load_all_raw_data()` completes:

```python
# Check basic properties
print(f"Records loaded: {len(combined)}")
print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
print(f"Regions: {combined['region'].unique()}")
print(f"Pollutants: {combined['pollutant'].unique()}")
print(f"Missing values per pollutant:")
print(combined.groupby("pollutant")["value"].apply(lambda x: x.isna().sum()))

# Check for duplicates
duplicates = combined.duplicated(
    subset=["region", "pollutant", "timestamp"], keep=False
).sum()
print(f"Duplicate records: {duplicates}")

# Check value ranges
for pollutant in combined["pollutant"].unique():
    subset = combined[combined["pollutant"] == pollutant]["value"]
    print(f"{pollutant}: min={subset.min()}, max={subset.max()}, mean={subset.mean():.1f}")
```

---

## Integration with Phase 1 Data Investigation

After loading raw data:

1. **Save raw data** to `data/raw/pollution_data_raw.csv`:
   ```python
   combined.to_csv("data/raw/pollution_data_raw.csv", index=False)
   ```

2. **Pass to Phase 1 script** (`scripts/data_investigation.py`):
   ```python
   from data_loading import load_all_raw_data
   
   raw_data = load_all_raw_data()
   # Phase 1: Analyze extreme values, data loss, region imbalance
   ```

3. **Outputs documented in DECISIONS.md section 7** after Phase 1 completes

---

## Dependencies

Required packages (in `pyproject.toml`):

```toml
[dependencies]
pandas = ">=2.0.0"
openpyxl = ">=3.10.0"  # For reading Excel files
numpy = ">=1.24.0"
```

Install:
```bash
uv sync
```

---

## Troubleshooting

### Issue: "No pollutant sheets found"

**Cause**: Sheet names don't match expected patterns

**Fix**:
1. Open Excel file manually
2. Check actual sheet names
3. Add new pattern to `pollutant_patterns` dict
4. Report in DECISIONS.md section 4.X for future updates

### Issue: "Date parsing failed"

**Cause**: Date format not in expected formats

**Fix**:
1. Inspect raw dates: `df.iloc[:, 0].unique()[:10]`
2. Add new format to `pd.to_datetime()` attempts
3. Document format in DECISIONS.md section 7

### Issue: "Missing data ratio very high"

**Cause**: Sheet structure different from expected (metadata in first columns instead of data)

**Fix**:
1. Inspect first 10 rows: `df.head(10)`
2. Identify which columns contain actual data
3. Update column extraction logic
4. Document deviation in DECISIONS.md section 4.X

---

## Next Steps

1. **Phase 1**: Run `scripts/data_investigation.py` with loaded raw data
2. **Phase 3**: Use this loader in preprocessing scripts (`preprocess_lstm.py`, `preprocess_xgb.py`)
3. **Reproducibility**: Save raw data snapshot to `data/raw/` with version date

See AGENTS.md section "Phase 1" for data investigation procedures.
