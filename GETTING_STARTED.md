# GETTING_STARTED.md - First 30 Minutes Checklist

## ⚠️ CRITICAL: GIT INSTRUCTIONS
**DO NOT RUN ANY GIT COMMANDS** (git add, git commit, git push, or any git modification commands).
See DECISIONS.md for full git warning. Work directly on files; user handles git operations.

---

## Welcome, Fresh Agent!

This checklist gets you from zero to ready-to-code in ~30 minutes. Follow it step-by-step.

---

## 1. Environment Setup (5 minutes)

### Step 1.1: Verify Python & Dependencies

```bash
# Check Python version (should be 3.11.x)
python --version

# Install uv package manager if missing
pip install uv

# Install all dependencies from pyproject.toml
uv sync

# Verify key libraries installed
python -c "import pandas; import openpyxl; import torch; print('✓ All imports OK')"
```

**Expected output:**
```
Python 3.11.7
✓ All imports OK
```

**Troubleshooting:**
- If `uv sync` fails → Check pyproject.toml exists and dependencies listed
- If imports fail → Run `uv sync` again, then `pip install -e .`

### Step 1.2: Create Directory Structure

```bash
# From project root
mkdir -p data/raw
mkdir -p data/preprocessed_lstm_v1
mkdir -p data/preprocessed_xgb_v1
mkdir -p models
mkdir -p logs
mkdir -p visualizations/phase_1_data_investigation
mkdir -p visualizations/phase_3_preprocessing
mkdir -p visualizations/phase_5_training
mkdir -p visualizations/phase_6_evaluation

# Verify structure
ls -la data/ models/ logs/ visualizations/
```

**Expected output:**
```
data/:
  raw/
  preprocessed_lstm_v1/
  preprocessed_xgb_v1/

models/
logs/
visualizations/:
  phase_1_data_investigation/
  phase_3_preprocessing/
  phase_5_training/
  phase_6_evaluation/
```

---

## 2. Data Verification (5 minutes)

### Step 2.1: Check Raw Data Location

```bash
# Verify pollution data exists
ls -la "Pollution Data Raipur/"

# Expected: 4 subdirectories
# Bhatagaon DCR/
# DCR AIIMS/
# IGKV DCR/
# SILTARA DCR/
```

**Troubleshooting:**
- If directory not found → Check you're in project root (`pwd` should show `ML-Final`)
- If names differ → Search for Excel files: `find . -name "*.xlsx" | head -5`

### Step 2.2: Test Loading One Excel File

```bash
# Run quick test
python << 'EOF'
import pandas as pd
import openpyxl
from pathlib import Path

# Find first Excel file
data_dir = Path("Pollution Data Raipur")
excel_file = list(data_dir.glob("**/*.xlsx"))[0]

print(f"✓ Found: {excel_file}")

# Try loading it
df = pd.read_excel(excel_file)
print(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")
print(f"  First row:\n{df.iloc[0]}")
EOF
```

**Expected output:**
```
✓ Found: Pollution Data Raipur/Bhatagaon DCR/2024/January/data.xlsx
✓ Loaded: 744 rows × 8 columns
  Columns: ['Timestamp', 'PM2.5', 'PM10', 'NO2', 'O3', 'Temperature', 'Humidity', 'Wind_Speed']
  First row: ...
```

**Troubleshooting:**
- If column names differ → See DATA_LOADING.md for expected schema
- If dates wrong format → See DATA_LOADING.md for parsing guidance

---

## 3. Documentation Review (10 minutes)

### Step 3.1: Read Core Docs (In Order)

1. **AGENTS.md** (sections 1-3 only) - 5 minutes
   - Project overview
   - Robustness/reproducibility/transparency principles
   - Documentation structure

2. **DATA_INVESTIGATION.md** (sections 1-3) - 3 minutes
   - Understand what Phase 1 analyzes
   - Don't worry about procedures yet; just know the goals

3. **IMPLEMENTATION_CONTRACT.md** (section "The Contract") - 2 minutes
   - Understand: DECISIONS.md is law
   - Code must match DECISIONS.md or update DECISIONS.md first

### Step 3.2: Understand the Goal

**Phase 1 objectives (3 parallel analyses):**
1. Analyze Bhatagaon Sept 2025 spike (real event vs sensor error?)
2. Analyze data loss root causes in canonical hourly pipeline
3. Quantify post-canonical region imbalance and derive training weights

**Outputs:**
- Plots in `visualizations/phase_1_data_investigation/`
- Updated `DATA_INVESTIGATION.md` with findings
- Updated `DECISIONS.md` section 7 with results

---

## 4. First Script Setup (10 minutes)

### Step 4.1: Create Script Skeleton

Create `scripts/data_investigation.py`:

```python
"""
Phase 1: Data Investigation - Analyze extreme data quality issues

Objectives:
1. Bhatagaon Sept 2025 spike: Real event vs sensor error?
2. Data loss root cause: Where does canonical hourly attrition occur?
3. Region imbalance: Quantify mild post-canonical imbalance and weights

Output:
- Plots: visualizations/phase_1_data_investigation/
- Results: DATA_INVESTIGATION.md (fill sections 1-3)
- Updates: DECISIONS.md section 7
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fixed seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "data_investigation.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("PHASE 1: DATA INVESTIGATION")
logger.info("=" * 80)
logger.info(f"Random seed: {RANDOM_SEED}")
logger.info(f"Started at: {datetime.now().isoformat()}")


def load_raw_data():
    """Load all pollution data from Excel files."""
    logger.info("Loading raw data...")
    # TODO: Implement using DATA_LOADING.md guidance
    pass


def analyze_bhatagaon_spike():
    """Analyze Bhatagaon Sept 2025 spike (real vs sensor error?)."""
    logger.info("Analyzing Bhatagaon September 2025 spike...")
    # TODO: Implement following DATA_INVESTIGATION.md section 1.2
    pass


def analyze_data_loss():
    """Analyze data loss root causes."""
    logger.info("Analyzing data loss root causes...")
    # TODO: Implement following DATA_INVESTIGATION.md section 2.2
    pass


def analyze_region_imbalance():
    """Quantify region imbalance."""
    logger.info("Analyzing region imbalance...")
    # TODO: Implement following DATA_INVESTIGATION.md section 3.2
    pass


def main():
    """Run all Phase 1 analyses."""
    try:
        # Load data once
        data = load_raw_data()
        
        # Run three parallel analyses
        analyze_bhatagaon_spike()
        analyze_data_loss()
        analyze_region_imbalance()
        
        logger.info("Phase 1 complete!")
        print("✓ Phase 1 complete! Check visualizations/phase_1_data_investigation/")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 4.2: Test Script Runs

```bash
# Test that script at least imports correctly
python scripts/data_investigation.py

# Expected output:
# ✓ Phase 1 complete! Check visualizations/phase_1_data_investigation/
# (Or error if load_raw_data not implemented yet - that's OK)
```

---

## 5. Ready to Start Phase 1 (0 minutes remaining)

### Checklist Before Coding

- [ ] Python 3.11.x installed (`python --version`)
- [ ] Dependencies installed (`uv sync` ran successfully)
- [ ] Can import pandas, torch, openpyxl
- [ ] Directory structure created (data/, logs/, visualizations/, etc.)
- [ ] Can load one Excel file successfully
- [ ] Read AGENTS.md sections 1-3
- [ ] Read IMPLEMENTATION_CONTRACT.md "The Contract" section
- [ ] Understand Phase 1 has 3 objectives
- [ ] Script skeleton created in `scripts/data_investigation.py`
- [ ] Script runs (even if functions not implemented)

### Next Steps

1. **Read DATA_LOADING.md** - Learn how to load Excel files
2. **Implement `load_raw_data()` function** - Use DATA_LOADING.md examples
3. **Read DATA_INVESTIGATION.md section 1-3** - Learn what to analyze
4. **Implement analysis functions** - One at a time
5. **Create visualizations** - Following VISUALIZATIONS.md specs
6. **Fill DATA_INVESTIGATION.md** - Document findings
7. **Update DECISIONS.md section 7** - Record results

### Getting Help

If stuck:
1. **Stuck on loading data?** → Read DATA_LOADING.md
2. **Stuck on script structure?** → Read SCRIPT_TEMPLATES.md
3. **Stuck on what Phase 1 means?** → Read DATA_INVESTIGATION.md
4. **Stuck on how to deviate?** → Read IMPLEMENTATION_CONTRACT.md

---

## Common First-Run Issues

### Issue: "ModuleNotFoundError: No module named 'openpyxl'"

**Solution:**
```bash
uv sync
python -m pip install openpyxl pandas torch
```

### Issue: "Excel file not found"

**Solution:**
```bash
# Check working directory
pwd  # Should end in ML-Final

# Find Excel files
find . -name "*.xlsx" -type f | head -5

# Check exact path
ls -la "Pollution Data Raipur/Bhatagaon DCR/" 2024/*/
```

### Issue: "AttributeError: 'NoneType' object has no attribute 'shape'"

**Cause:** `load_raw_data()` function returns None (not implemented yet)

**Solution:** Implement the function using DATA_LOADING.md

### Issue: "uv sync hangs or fails"

**Solution:**
```bash
# Try with verbose output
uv sync --verbose

# Or fall back to pip
pip install -r requirements.txt
```

---

## Verification: You're Ready When...

✓ You can run:
```bash
python scripts/data_investigation.py
```

✓ And see:
```
INFO - Phase 1: DATA INVESTIGATION
INFO - Random seed: 42
INFO - Started at: 2026-04-16T...
... (functions running)
INFO - Phase 1 complete!
✓ Phase 1 complete! Check visualizations/phase_1_data_investigation/
```

✓ And files exist:
```bash
ls -la logs/data_investigation.log
ls -la visualizations/phase_1_data_investigation/
```

**If you see all three ✓, you're ready to start coding!**
