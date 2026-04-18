# Notebook Run Script (Aggressive Regularization Sweep)

This runbook is for the new recovery profile:

- Horizons: `h1,h24,h168`
- Sequence lengths: `seq_len=max(24,horizon)` -> `h1=24,h24=24,h168=168`
- LSTM architecture: reduced capacity (`hidden_dim=64`, `num_layers=2`, `num_heads=2`)
- Stronger regularization and earlier stopping (`dropout/head_dropout/weight_decay` + lower patience)
- Evaluation with fair intersection + quantile calibration

All code blocks are notebook-ready with `# %% [code]` headers.

## 1) Setup

```python
# %% [code]
import time
PIPELINE_START_TS = time.time()

!rm -rf ML-Final
!git clone https://github.com/fine2006/ML-Final -b new-approach
!cd ML-Final && uv sync
!cp -r "/kaggle/input/datasets/fine2006/pollution-data-raipur/Pollution Data Raipur" ML-Final/
```

```python
# %% [code]
!cd ML-Final && uv run python -c 'import torch; print(f"torch: {torch.__version__}"); print(f"cuda: {torch.cuda.is_available()}"); print(f"gpu_count: {torch.cuda.device_count()}"); [print(f"gpu[{i}]={torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None'
```

## 2) Phase 1 + preprocessing

```python
# %% [code]
!cd ML-Final && uv run python scripts/data_investigation.py
```

```python
# %% [code]
!export MPLBACKEND=Agg && cd ML-Final && uv run python scripts/preprocess_lstm.py && uv run python scripts/preprocess_xgb.py
```

## 3) Verify XGB models are already present locally

```python
# %% [code]
import pathlib
import re

ROOT = pathlib.Path("ML-Final").resolve()
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

rx = re.compile(r"^xgb_quantile_(pm25|pm10|no2|o3)_h(1|24|168)_q(05|50|95|99)\.json$")

expected = {
    f"xgb_quantile_{p}_h{h}_q{q}.json"
    for p in ["pm25", "pm10", "no2", "o3"]
    for h in [1, 24, 168]
    for q in ["05", "50", "95", "99"]
}

present = {p.name for p in MODELS.glob("xgb_quantile_*.json") if rx.match(p.name)}
missing = sorted(expected - present)

print("xgb model count:", len(present))
if missing:
    print("missing examples:", missing[:10])
    raise RuntimeError(
        f"Missing {len(missing)} expected XGB model files in {MODELS}. "
        "Copy your precomputed XGB artifacts into ML-Final/models before evaluation."
    )
```

## 4) Train LSTM (new aggressive run profile)

This runs all pollutants and all three horizons in one command using:

- `seq_len_map: 1:24,24:24,168:168`
- `hidden_dim=64`, `num_layers=2`, `num_heads=2`
- `dropout=0.35`, `head_dropout=0.30`
- `lr=3e-4`, `weight_decay=5e-4`
- `epochs=40`, `patience=6`
- scheduler: `factor=0.5`, `patience=2`
- gradient clipping: `max_grad_norm=1.0`

```python
# %% [code]
import os
import subprocess
import pathlib

ROOT = pathlib.Path("ML-Final").resolve()
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

subprocess.run(
    [
        "uv", "run", "python", "scripts/train_lstm.py",
        "--pollutants", "pm25,pm10,no2,o3",
        "--horizons", "1,24,168",
        "--seq-len-map", "1:24,24:24,168:168",
        "--device", "cuda",
        "--batch-size", "128",
        "--epochs", "40",
        "--patience", "6",
        "--lr", "3e-4",
        "--hidden-dim", "64",
        "--num-layers", "2",
        "--num-heads", "2",
        "--dropout", "0.35",
        "--head-dropout", "0.30",
        "--weight-decay", "5e-4",
        "--scheduler-factor", "0.5",
        "--scheduler-patience", "2",
        "--max-grad-norm", "1.0",
        "--min-attn-window", "24",
    ],
    cwd=ROOT,
    env=env,
    check=True,
)
```

## 5) Quick sanity check of generated LSTM artifacts

```python
# %% [code]
import json
import pathlib

ROOT = pathlib.Path("ML-Final").resolve()
MODELS = ROOT / "models"

summary_path = MODELS / "lstm_training_summary.json"
if not summary_path.exists():
    raise RuntimeError(f"Missing {summary_path}")

summary = json.loads(summary_path.read_text())
print("summary horizons:", summary.get("horizons"))
print("seq_len_by_horizon:", summary.get("seq_len_by_horizon"))
print("model_hparams:", summary.get("model_hparams"))
print("optimizer_hparams:", summary.get("optimizer_hparams"))

for pollutant in ["pm25", "pm10", "no2", "o3"]:
    for h in [1, 24, 168]:
        ckpt = MODELS / f"lstm_quantile_{pollutant}_h{h}.pt"
        pred = MODELS / f"lstm_predictions_{pollutant}_h{h}.npz"
        if not ckpt.exists() or not pred.exists():
            raise RuntimeError(f"Missing artifacts for {pollutant} h{h}")

print("All expected LSTM checkpoints/predictions are present.")
```

## 6) Evaluate (calibrated + fair intersection)

```python
# %% [code]
import os
import json
import pathlib
import subprocess

ROOT = pathlib.Path("ML-Final").resolve()
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

subprocess.run(
    [
        "uv", "run", "python", "scripts/evaluate.py",
        "--pollutants", "pm25,pm10,no2,o3",
        "--horizons", "1,24,168",
        "--device", "cuda",
        "--fair-intersection",
        "--calibrate-quantiles",
    ],
    cwd=ROOT,
    env=env,
    check=True,
)

summary = json.loads((ROOT / "models" / "evaluation_summary.json").read_text())
fair = json.loads((ROOT / "models" / "fair_benchmark_summary.json").read_text())

print("fair rows:", len(fair.get("rows", [])))
print("calibration enabled:", summary.get("quantile_calibration_enabled"))
print("interval focus:", summary.get("interval_focus"))
print("has split representativeness:", "split_representativeness" in summary)
print("has phase1 context:", "phase1_data_quality_context" in summary)
```

## 7) Print compact LSTM vs XGB table from fair benchmark

```python
# %% [code]
import json
import pathlib

ROOT = pathlib.Path("ML-Final").resolve()
fair_path = ROOT / "models" / "fair_benchmark_summary.json"
fair = json.loads(fair_path.read_text())

rows = fair.get("rows", [])
rows = sorted(rows, key=lambda r: (r["pollutant"], int(r["horizon"])))

print("pollutant,h,n_common,lstm_rmse,xgb_rmse,rmse_impr_pct,lstm_crps,xgb_crps,crps_impr_pct,lstm_cov95,xgb_cov95")
for r in rows:
    print(
        f"{r['pollutant']},{r['horizon']},{r['n_common']},"
        f"{r['lstm_rmse']:.6f},{r['xgb_rmse']:.6f},{r['rmse_improvement_pct']:.2f},"
        f"{r['lstm_crps']:.6f},{r['xgb_crps']:.6f},{r['crps_improvement_pct']:.2f},"
        f"{r['lstm_coverage']:.6f},{r['xgb_coverage']:.6f}"
    )
```

## 8) Predict (single region + selected horizon)

```python
# %% [code]
import pathlib
import subprocess
import time

ROOT = pathlib.Path("ML-Final").resolve()

subprocess.run(
    [
        "uv", "run", "python", "scripts/predict.py",
        "--region", "AIIMS",
        "--horizon", "24",
        "--pollutants", "pm25,pm10,no2,o3",
        "--device", "cuda",
        "--output", "models/prediction_output.json",
    ],
    cwd=ROOT,
    check=True,
)

print("saved:", ROOT / "models" / "prediction_output.json")
print("elapsed hours:", (time.time() - PIPELINE_START_TS) / 3600.0)
```

## 9) Optional second-pass targeted retry (if only a few cells are weak)

Use this only if one or two pollutant/horizon cells remain far behind XGB.

```python
# %% [code]
import pathlib
import subprocess

ROOT = pathlib.Path("ML-Final").resolve()

subprocess.run(
    [
        "uv", "run", "python", "scripts/train_lstm.py",
        "--pollutants", "pm10,no2,o3",
        "--horizons", "168",
        "--seq-len-map", "168:168",
        "--device", "cuda",
        "--batch-size", "128",
        "--epochs", "40",
        "--patience", "6",
        "--lr", "2e-4",
        "--hidden-dim", "64",
        "--num-layers", "2",
        "--num-heads", "2",
        "--dropout", "0.40",
        "--head-dropout", "0.35",
        "--weight-decay", "8e-4",
        "--scheduler-factor", "0.5",
        "--scheduler-patience", "2",
        "--max-grad-norm", "1.0",
        "--min-attn-window", "24",
    ],
    cwd=ROOT,
    check=True,
)
```

Re-run sections 6 and 7 after this optional retry.
