1. Prepare the data (run once)
# 1️⃣  LSTM‑specific pre‑processing (creates data/preprocessed_lstm_v1/)
python scripts/preprocess_lstm.py
# 2️⃣  XGB‑specific pre‑processing (creates data/preprocessed_xgb_v1/)
python scripts/preprocess_xgb.py
Both scripts read the canonical hourly/quarterly CSV that ensure_canonical_dataset builds, apply the pipeline‑specific missing‑value policy, add the time‑features, split into train/val/test and write a metadata.json. No extra flags are needed for the default split (70 % train / 15 % val / 15 % test).
---
2. Train the four LSTM models (one command per pollutant)
train_lstm.py accepts a single pollutant‑list and a single horizon‑list per invocation, but all pollutants share the same hyper‑parameters.  
Because you need a different configuration for each pollutant, invoke the script four times, overriding the defaults with the values in your dictionaries.
> Useful tip – copy‑paste the commands; they are independent and can be run in parallel (different terminals or background jobs).
2.1 NO₂ (delta‑target)
python scripts/train_lstm.py \
  --pollutants no2 \
  --horizons 168 \
  --seq-len-map 168:168 \
  --hidden-dim 128 \
  --num-layers 2 \
  --num-heads 2 \
  --dropout 0.279 \
  --head-dropout 0.217 \
  --lr 0.00012 \
  --weight-decay 5.67e-05 \
  --target-mode delta_ma3 \
  --delta-baseline-window 3 \
  --min-attn-window 24 \
  --epochs 100 \
  --batch-size 32 \
  --patience 10 \
  --max-grad-norm 1.0 \
  --summary-path models/no2_lstm_h168_summary.json
2.2 O₃ (delta‑target, shorter sequence)
python scripts/train_lstm.py \
  --pollutants o3 \
  --horizons 168 \
  --seq-len-map 168:48 \
  --hidden-dim 128 \
  --num-layers 2 \
  --num-heads 4 \
  --dropout 0.165 \
  --head-dropout 0.291 \
  --lr 0.000095 \
  --weight-decay 2.02e-05 \
  --target-mode delta_ma3 \
  --delta-baseline-window 3 \
  --min-attn-window 24 \
  --epochs 100 \
  --batch-size 32 \
  --patience 10 \
  --max-grad-norm 1.0 \
  --summary-path models/o3_lstm_h168_summary.json
2.3 PM₂.5 (level‑target)
python scripts/train_lstm.py \
  --pollutants pm25 \
  --horizons 168 \
  --seq-len-map 168:168 \
  --hidden-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --dropout 0.40 \
  --head-dropout 0.20 \
  --lr 0.0003267204510288974 \
  --weight-decay 3.520307789017964e-05 \
  --target-mode level \
  --min-attn-window 24 \
  --epochs 100 \
  --batch-size 32 \
  --patience 10 \
  --max-grad-norm 1.0 \
  --summary-path models/pm25_lstm_h168_summary.json
2.4 PM₁₀ (level‑target)
python scripts/train_lstm.py \
  --pollutants pm10 \
  --horizons 168 \
  --seq-len-map 168:168 \
  --hidden-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --dropout 0.35 \
  --head-dropout 0.18 \
  --lr 0.0002667204510288974 \
  --weight-decay 1.520307789017964e-04 \
  --target-mode level \
  --min-attn-window 24 \
  --epochs 100 \
  --batch-size 32 \
  --patience 10 \
  --max-grad-norm 1.0 \
  --summary-path models/pm10_lstm_h168_summary.json
What each command writes
- Checkpoint: models/lstm_quantile_<pollutant>_h168.pt  
- Test‑set predictions: models/lstm_predictions_<pollutant>_h168.npz (contains an array of shape (samples, 4‑quantiles) – you’ll later drop the p99 column).  
- A JSON training summary at the path you supplied (--summary-path).
---
3. Train the XGB baselines (single run)
The XGB script does not expose per‑pollutant hyper‑parameters, and it always trains the four quantiles (p05, p50, p95, p99). That is fine – we’ll simply ignore the p99 column later.
python scripts/train_xgb.py \
  --pollutants pm25,pm10,no2,o3 \
  --horizons 168 \
  --n-estimators 500 \
  --learning-rate 0.1 \
  --max-depth 7 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --device auto
Outcome  
- Model files: models/xgb_quantile_<pollutant>_h168_qXX.json (one for each quantile).  
- Summary JSON: models/xgb_training_summary.json.
---
4. Evaluate – LSTM vs XGB on horizon 168, only p5/p50/p95
The bundled evaluate.py computes all four quantiles, but you can:
1. Run the full evaluator (it will still compute p99, which we’ll simply drop later).  
2. Post‑process the generated QuantilePayload objects or the saved .npz files to keep only the three quantiles you care about.
4.1 Run the evaluator
python scripts/evaluate.py \
  --pollutants pm25,pm10,no2,o3 \
  --horizons 168 \
  --lstm-data-dir data/preprocessed_lstm_v1 \
  --xgb-data-dir data/preprocessed_xgb_v1 \
  --models-dir models \
  --device auto \
  --batch-size 256 \
  --fair-intersection \
  --calibrate-quantiles
What this does
- Loads the LSTM checkpoints (only the h168 ones) and the XGB quantile models.  
- Aligns LSTM and XGB rows on the exact same (region, timestamp) set (--fair-intersection).  
- Fits a quantile‑bias calibration on the validation split for both families (--calibrate-quantiles).  
- Writes:
  - models/evaluation_summary.json (overall metrics, per‑horizon, per‑pollutant, fairness, gates).  
  - models/fair_benchmark_summary.json (compact table ready for a report).  
  - Visualisations under visualizations/phase_6_evaluation/ (calibration plots, fairness bar‑charts, PIT‑KS histograms, operational‑gate tables).
4.2 Strip out the unwanted p99 after evaluation
You have two easy options:
Option A – Use a tiny analysis notebook (recommended)
# analysis.ipynb  (run after the evaluation step)
import numpy as np, json, matplotlib.pyplot as plt
import pandas as pd
# Example for NO2
npz = np.load('models/lstm_predictions_no2_h168.npz')
pred_all = npz['predictions']          # shape (N, 4) → [p05, p50, p95, p99]
targets = npz['targets']
regions = npz['regions']
# Keep only first three quantiles
pred = pred_all[:, :3]                 # (N, 3)
# Compute RMSE / coverage on the three quantiles
p05, p50, p95 = pred.T
rmse = np.sqrt(((targets - p50) ** 2).mean())
coverage = ((targets >= p05) & (targets <= p95)).mean()
print('NO2 h168 – RMSE:', rmse, 'Coverage(p5‑p95):', coverage)
# Repeat for XGB – load the three quantile predictions:
xgb_q05 = np.loadtxt('models/xgb_quantile_no2_h168_q05.csv')
xgb_q50 = np.loadtxt('models/xgb_quantile_no2_h168_q50.csv')
xgb_q95 = np.loadtxt('models/xgb_quantile_no2_h168_q95.csv')
# (or use the .json models to predict on X_test if you prefer)
You can repeat the snippet for each pollutant, collect the numbers in a pandas DataFrame, and plot side‑by‑side bar charts.
Option B – Trim the JSON summary (quick hack)
The evaluation_summary.json contains a field by_horizon with all metrics, including coverage_p05_p95 (which already ignores p99). The only metric that mentions p99 is tail_above_p99. If you only read the keys you need (rmse, mae, r2, crps_approx, coverage_p05_p95, pit_ks_stat, fairness), you can ignore the p99‑related entries.
jq '.lstm.pollutants.no2.h168 | {rmse, mae, r2, crps_approx, coverage_p05_p95, pit_ks_stat, fairness}' models/evaluation_summary.json
The same query works for XGB (.xgb.pollutants.no2.h168).
---
5. Visualise the three‑quantile comparison
The visualizations/phase_6_evaluation/ folder already contains useful plots, but you may want a compact “p5‑p50‑p95” bar chart. Below is a minimal script that reads the three‑quantile predictions and produces a side‑by‑side bar plot for each pollutant.
# plot_comparison.py
import numpy as np, matplotlib.pyplot as plt, json, pandas as pd
def load_lstm(pollutant):
    data = np.load(f'models/lstm_predictions_{pollutant}_h168.npz')
    pred = data['predictions'][:, :3]       # keep p5,p50,p95
    true = data['targets']
    return pred, true
def load_xgb(pollutant):
    # XGB predictions are stored per‑quantile as .json files; we load them via xgboost
    import xgboost as xgb
    X_test = pd.read_csv('data/preprocessed_xgb_v1/X_test.csv')
    feats = [c for c in X_test.columns if c not in {'timestamp','region'}]
    X = X_test[feats].values.astype(np.float32)
    preds = []
    for q in [0.05,0.50,0.95]:
        model = xgb.XGBRegressor()
        model.load_model(f'models/xgb_quantile_{pollutant}_h168_q{int(q*100):02d}.json')
        preds.append(model.predict(X))
    return np.column_stack(preds)          # (N,3)
def rmse(y, p50): return np.sqrt(((y-p50)**2).mean())
def coverage(y, p5, p95): return ((y>=p5) & (y<=p95)).mean()
pollutants = ['pm25','pm10','no2','o3']
rows = []
for pol in pollutants:
    lstm_pred, lstm_true = load_lstm(pol)
    xgb_pred = load_xgb(pol)
    rows.append({
        'pollutant': pol,
        'model': 'LSTM',
        'rmse': rmse(lstm_true, lstm_pred[:,1]),
        'coverage': coverage(lstm_true, lstm_pred[:,0], lstm_pred[:,2])
    })
    rows.append({
        'pollutant': pol,
        'model': 'XGB',
        'rmse': rmse(lstm_true, xgb_pred[:,1]),   # note: use same true values (fair intersection)
        'coverage': coverage(lstm_true, xgb_pred[:,0], xgb_pred[:,2])
    })
df = pd.DataFrame(rows)
# Plot
fig, ax = plt.subplots(1,2, figsize=(12,5), sharey=True)
for i, metric in enumerate(['rmse','coverage']):
    df.pivot(index='pollutant', columns='model', values=metric).plot(kind='bar', ax=ax[i])
    ax[i].set_title(metric.upper())
    ax[i].set_ylabel(metric if i==0 else '')
plt.tight_layout()
plt.savefig('visualizations/phase_6_evaluation/lstm_vs_xgb_p5p50p95.png')
Running python plot_comparison.py will generate a single PNG that directly compares the three‑quantile RMSE and coverage for each pollutant.
---
6. Checklist – what you will have after the run
Artifact	Location	Content
LSTM checkpoints	models/lstm_quantile_<pollutant>_h168.pt	Saved model state, hyper‑params, region mapping, seq_len, target‑mode, etc.
LSTM test predictions	models/lstm_predictions_<pollutant>_h168.npz	predictions (4 quantiles), targets, regions, timestamps.
XGB models (per quantile)	models/xgb_quantile_<pollutant>_h168_qXX.json	XGBoost binary files for p05/p50/p95/p99.
XGB training summary	models/xgb_training_summary.json	Overall RMSE/MAE/R² per pollutant/horizon (includes p99 but you ignore it).
Evaluation summary	models/evaluation_summary.json	All metrics (overall + per‑horizon) for both families, PIT‑KS, fairness, gate flags.
Fair‑intersection table	models/fair_benchmark_summary.json	Compact CSV‑style table ready for a report.
Visualisations	visualizations/phase_6_evaluation/	Calibration bias plots, fairness bar‑charts, PIT‑KS histograms, gate‑summary table, plus the optional lstm_vs_xgb_p5p50p95.png you can generate with the script above.
Optional Python analysis notebook / script	wherever you save it	Loads the .npz files and computes RMSE / coverage for the three quantiles only.
---
7. Summary of the exact command sequence you’ll run (copy‑paste)
# ------------------------------
# 1. Pre‑process both pipelines
# ------------------------------
python scripts/preprocess_lstm.py
python scripts/preprocess_xgb.py
# -------------------------------------------------
# 2. Train the four custom LSTM models (horizon=168)
# -------------------------------------------------
python scripts/train_lstm.py \
  --pollutants no2 \
  --horizons 168 \
  --seq-len-map 168:168 \
  --hidden-dim 128 --num-layers 2 --num-heads 2 \
  --dropout 0.279 --head-dropout 0.217 \
  --lr 0.00012 --weight-decay 5.67e-05 \
  --target-mode delta_ma3 --delta-baseline-window 3 \
  --min-attn-window 24 \
  --summary-path models/no2_lstm_h168_summary.json
python scripts/train_lstm.py \
  --pollutants o3 \
  --horizons 168 \
  --seq-len-map 168:48 \
  --hidden-dim 128 --num-layers 2 --num-heads 4 \
  --dropout 0.165 --head-dropout 0.291 \
  --lr 0.000095 --weight-decay 2.02e-05 \
  --target-mode delta_ma3 --delta-baseline-window 3 \
  --min-attn-window 24 \
  --summary-path models/o3_lstm_h168_summary.json
python scripts/train_lstm.py \
  --pollutants pm25 \
  --horizons 168 \
  --seq-len-map 168:168 \
  --hidden-dim 64 --num-layers 2 --num-heads 4 \
  --dropout 0.40 --head-dropout 0.20 \
  --lr 0.0003267204510288974 --weight-decay 3.520307789017964e-05 \
  --target-mode level \
  --min-attn-window 24 \
  --summary-path models/pm25_lstm_h168_summary.json
python scripts/train_lstm.py \
  --pollutants pm10 \
  --horizons 168 \
  --seq-len-map 168:168 \
  --hidden-dim 64 --num-layers 2 --num-heads 4 \
  --dropout 0.35 --head-dropout 0.18 \
  --lr 0.0002667204510288974 --weight-decay 1.520307789017964e-04 \
  --target-mode level \
  --min-attn-window 24 \
  --summary-path models/pm10_lstm_h168_summary.json
# ------------------------------
# 3. Train XGB baselines (default quantiles)
# ------------------------------
python scripts/train_xgb.py \
  --pollutants pm25,pm10,no2,o3 \
  --horizons 168 \
  --n-estimators 500 \
  --learning-rate 0.1 \
  --max-depth 7 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --device auto
# ------------------------------
# 4. Evaluate (fair intersection, calibration)
# ------------------------------
python scripts/evaluate.py \
  --pollutants pm25,pm10,no2,o3 \
  --horizons 168 \
  --lstm-data-dir data/preprocessed_lstm_v1 \
  --xgb-data-dir data/preprocessed_xgb_v1 \
  --models-dir models \
  --device auto \
  --batch-size 256 \
  --fair-intersection \
  --calibrate-quantiles
# ------------------------------
# 5. (optional) Plot side‑by‑side RMSE & coverage for p5/p50/p95
# ------------------------------
python plot_comparison.py
Run the steps in order. After step 4 you already have the full evaluation JSON and the visualisation folder; step 5 is just a convenience to produce a single bar‑chart that isolates the three quantiles you care about.
---
