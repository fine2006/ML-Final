# Air Pollution Forecasting Using Recurrent Neural Networks (RNN)

## Project Task
Develop a forecasting model using Recurrent Neural Networks (RNN/LSTM) to predict
a selected air pollutant (e.g., PM2.5, PM10, NO2, O3) using historical pollutant and
meteorological data, and compare its performance with a traditional machine learning
model.

## Major Steps
1. Data preprocessing – clean dataset, handle missing values, and organize time-series
data.
2. Feature engineering – create lag features and normalize input data.
3. Model development – implement a basic RNN or LSTM model for prediction.
4. Baseline comparison – train a simple ML model (e.g., Random Forest or Linear Re-
gression).
5. Model evaluation – compare RNN and ML model performance using RMSE and MAE.
6. Result analysis and visualization – visualize predicted vs actual values and compare
model performance.

## Expected Output
- Trained RNN/LSTM model
- Baseline ML model
- Performance comparison (RNN vs ML)
- Visualization of prediction results

### Notes
- Data is available at ./Pollution Data Raipur/
- Data is split in folders by region, year and month in that order, and each xlsx file has sheet for every date
- uv is used for dependency and project management

## File Tree

```sh
.
├── AGENTS.md
├── main.py
├── Pollution Data Raipur
│   ├── Bhatagaon DCR
│   ├── DCR AIIMS
│   ├── IGKV DCR
│   └── SILTARA DCR
└── pyproject.toml
```
