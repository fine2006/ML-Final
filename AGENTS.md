# Air Pollution Forecasting - ML Project

## Project Commands
- Dependency manager: `uv` (defined in pyproject.toml)
- Entry point: `python main.py`
- No test/lint/typecheck commands configured yet

## Data Structure
- Location: `./Pollution Data Raipur/`
- 4 regions: Bhatagaon DCR, DCR AIIMS, IGKV DCR, SILTARA DCR
- Organization: `region/year/month/xlsx files`
- xlsx format: each file contains sheets for every date

## Architecture Notes
- `main.py`: placeholder stub (needs ML model implementation)
- `pyproject.toml`: dependencies array is empty - add required ML libs
- No existing model code - build from scratch per project task

## Project Task
1. Load pollution data (PM2.5/PM10/NO2/O3 target)
2. Preprocess: clean, handle missing values, create lag features, normalize
3. Implement RNN/LSTM model + baseline ML (Random Forest/Linear Regression)
4. Compare RMSE and MAE between models
5. Visualize predictions vs actual

## Setup Required
Before starting, add dependencies to pyproject.toml:
- Data: pandas, numpy, openpyxl
- ML: scikit-learn, torch (or tensorflow)
- Viz: matplotlib