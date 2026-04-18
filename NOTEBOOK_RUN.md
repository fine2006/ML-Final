# Notebook Run Script (Pollutant-Specific h168 Focus)

This runbook focuses on long-horizon (`h168`) model quality with pollutant-specific sweeps.

Core contract:

- Train one model per `(pollutant, horizon)` target.
- Every model still sees all pollutant channels in lookback (`pm25`,`pm10`,`no2`,`o3`) plus weather/time context.
- Start from `h168` only, then move to shorter horizons if needed.

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

## 2.1) Confirm single-latency feature contract in metadata

```python
# %% [code]
import json
import pathlib

ROOT = pathlib.Path("ML-Final").resolve()

lstm_meta = json.loads((ROOT / "data" / "preprocessed_lstm_v1" / "metadata.json").read_text())
xgb_meta = json.loads((ROOT / "data" / "preprocessed_xgb_v1" / "metadata.json").read_text())

print("LSTM feature columns:", lstm_meta["feature_columns"])
assert "temperature" in lstm_meta["feature_columns"]
assert "temperature_lag_1" not in lstm_meta["feature_columns"]
print("LSTM leakage policy:", lstm_meta["leakage_policy"])

print("XGB leakage policy:", xgb_meta["leakage_policy"])
assert xgb_meta["leakage_policy"]["weather_current_values_shifted_by"] == 0
```

## 3) Verify XGB model files exist locally

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
    raise RuntimeError(f"Missing {len(missing)} expected XGB files in {MODELS}")
```

## 4) Utility helpers for experiment runs

```python
# %% [code]
import json
import pathlib
import subprocess

ROOT = pathlib.Path("ML-Final").resolve()
MODELS = ROOT / "models"
EXP = MODELS / "experiments"
EXP.mkdir(parents=True, exist_ok=True)

def run_train(tag, pollutant, seq_len, hidden_dim, num_layers, num_heads, lr, dropout, head_dropout, weight_decay, epochs=80, patience=12, scheduler_patience=4, batch_size=128):
    summary_path = EXP / f"{tag}_summary.json"
    cmd = [
        "uv", "run", "python", "scripts/train_lstm.py",
        "--pollutants", pollutant,
        "--horizons", "168",
        "--seq-len-map", f"168:{seq_len}",
        "--device", "cuda",
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--lr", str(lr),
        "--hidden-dim", str(hidden_dim),
        "--num-layers", str(num_layers),
        "--num-heads", str(num_heads),
        "--dropout", str(dropout),
        "--head-dropout", str(head_dropout),
        "--weight-decay", str(weight_decay),
        "--scheduler-factor", "0.5",
        "--scheduler-patience", str(scheduler_patience),
        "--max-grad-norm", "1.0",
        "--min-attn-window", "24",
        "--summary-path", str(summary_path),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return summary_path

def run_eval(tag):
    cmd = [
        "uv", "run", "python", "scripts/evaluate.py",
        "--pollutants", "pm25,pm10,no2,o3",
        "--horizons", "1,24,168",
        "--device", "cuda",
        "--fair-intersection",
        "--calibrate-quantiles",
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    fair_src = MODELS / "fair_benchmark_summary.json"
    eval_src = MODELS / "evaluation_summary.json"
    fair_dst = EXP / f"{tag}_fair_benchmark_summary.json"
    eval_dst = EXP / f"{tag}_evaluation_summary.json"
    fair_dst.write_text(fair_src.read_text())
    eval_dst.write_text(eval_src.read_text())
    return fair_dst, eval_dst

def print_h168_rows(fair_path):
    fair = json.loads(pathlib.Path(fair_path).read_text())
    rows = [r for r in fair.get("rows", []) if int(r["horizon"]) == 168]
    rows = sorted(rows, key=lambda r: r["pollutant"])
    print("pollutant,lstm_rmse,xgb_rmse,rmse_impr_pct,lstm_crps,xgb_crps,crps_impr_pct,lstm_cov95,xgb_cov95")
    for r in rows:
        print(
            f"{r['pollutant']},{r['lstm_rmse']:.6f},{r['xgb_rmse']:.6f},{r['rmse_improvement_pct']:.2f},"
            f"{r['lstm_crps']:.6f},{r['xgb_crps']:.6f},{r['crps_improvement_pct']:.2f},"
            f"{r['lstm_coverage']:.6f},{r['xgb_coverage']:.6f}"
        )
```

## 5) Baseline eval snapshot before targeted h168 runs

```python
# %% [code]
fair_path, eval_path = run_eval("baseline_before_h168_tuning")
print_h168_rows(fair_path)
print("saved:", fair_path)
```

## 6) One-shot h1+h24 run using best h168 profile

Use the current best long-horizon profile (`64/2/2`, soft regularization) once on
`h1,h24` to refresh short/mid-horizon checkpoints under the new single-latency
feature contract.

```python
# %% [code]
import subprocess
cmd = [
    "uv", "run", "python", "scripts/train_lstm.py",
    "--pollutants", "pm25,pm10,no2,o3",
    "--horizons", "1,24",
    "--seq-len-map", "1:24,24:24",
    "--device", "cuda",
    "--batch-size", "128",
    "--epochs", "80",
    "--patience", "12",
    "--lr", "1.5e-4",
    "--hidden-dim", "64",
    "--num-layers", "2",
    "--num-heads", "2",
    "--dropout", "0.25",
    "--head-dropout", "0.20",
    "--weight-decay", "1e-4",
    "--scheduler-factor", "0.5",
    "--scheduler-patience", "4",
    "--max-grad-norm", "1.0",
    "--min-attn-window", "24",
    "--summary-path", str(EXP / "h1h24_once_from_h168best_summary.json"),
]
print("$", " ".join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)

fair_path, _ = run_eval("h1h24_once_from_h168best")

fair = json.loads(pathlib.Path(fair_path).read_text())
rows = [r for r in fair.get("rows", []) if int(r["horizon"]) in {1, 24}]
rows = sorted(rows, key=lambda r: (int(r["horizon"]), r["pollutant"]))
print("pollutant,h,lstm_rmse,xgb_rmse,rmse_impr_pct,lstm_crps,xgb_crps,crps_impr_pct,lstm_cov95,xgb_cov95")
for r in rows:
    print(
        f"{r['pollutant']},{r['horizon']},{r['lstm_rmse']:.6f},{r['xgb_rmse']:.6f},{r['rmse_improvement_pct']:.2f},"
        f"{r['lstm_crps']:.6f},{r['xgb_crps']:.6f},{r['crps_improvement_pct']:.2f},"
        f"{r['lstm_coverage']:.6f},{r['xgb_coverage']:.6f}"
    )
```

## 7) Pollutant-specific h168 sweeps

Suggested starting profiles (edit as needed):

- `pm25`: softer + longer context
- `pm10`: softer + longer context
- `no2`: slightly larger capacity, moderate context
- `o3`: compact/soft baseline first

```python
# %% [code]
RUNS = [
    # pm25
    {"tag": "h168_pm25_soft336", "pollutant": "pm25", "seq_len": 336, "hidden_dim": 64, "num_layers": 2, "num_heads": 2, "lr": 1.5e-4, "dropout": 0.25, "head_dropout": 0.20, "weight_decay": 1e-4},
    # pm10
    {"tag": "h168_pm10_soft336", "pollutant": "pm10", "seq_len": 336, "hidden_dim": 64, "num_layers": 2, "num_heads": 2, "lr": 1.5e-4, "dropout": 0.25, "head_dropout": 0.20, "weight_decay": 1e-4},
    # no2
    {"tag": "h168_no2_soft336_hd96", "pollutant": "no2", "seq_len": 336, "hidden_dim": 96, "num_layers": 2, "num_heads": 4, "lr": 1.5e-4, "dropout": 0.25, "head_dropout": 0.20, "weight_decay": 1e-4},
    # no2 (regularized large compare)
    {"tag": "h168_no2_reg336_hd96", "pollutant": "no2", "seq_len": 336, "hidden_dim": 96, "num_layers": 2, "num_heads": 4, "lr": 2.0e-4, "dropout": 0.30, "head_dropout": 0.25, "weight_decay": 2e-4, "epochs": 60, "patience": 10, "scheduler_patience": 3},
    # o3
    {"tag": "h168_o3_soft168", "pollutant": "o3", "seq_len": 168, "hidden_dim": 64, "num_layers": 2, "num_heads": 2, "lr": 1.5e-4, "dropout": 0.25, "head_dropout": 0.20, "weight_decay": 1e-4},
    # o3 (larger context)
    {"tag": "h168_o3_soft336_hd96", "pollutant": "o3", "seq_len": 336, "hidden_dim": 96, "num_layers": 2, "num_heads": 4, "lr": 1.5e-4, "dropout": 0.25, "head_dropout": 0.20, "weight_decay": 1e-4},
]

for cfg in RUNS:
    print("\n===", cfg["tag"], "===")
    run_train(**cfg)
    fair_path, _ = run_eval(cfg["tag"])
    print_h168_rows(fair_path)
```

## 8) Compare all h168 sweep outputs saved in `models/experiments`

```python
# %% [code]
import json
import pathlib

ROOT = pathlib.Path("ML-Final").resolve()
EXP = ROOT / "models" / "experiments"

def score_row(r):
    return 0.65 * float(r["rmse_improvement_pct"]) + 0.35 * float(r["crps_improvement_pct"]) - 10.0 * abs(float(r["lstm_coverage"]) - 0.90)

by_pollutant = {p: [] for p in ["pm25", "pm10", "no2", "o3"]}
for fp in sorted(EXP.glob("*_fair_benchmark_summary.json")):
    tag = fp.name.replace("_fair_benchmark_summary.json", "")
    fair = json.loads(fp.read_text())
    for r in fair.get("rows", []):
        if int(r["horizon"]) != 168:
            continue
        p = r["pollutant"]
        if p not in by_pollutant:
            continue
        by_pollutant[p].append((score_row(r), tag, r))

for p in ["pm25", "pm10", "no2", "o3"]:
    print("\n===", p, "h168 ranking ===")
    rows = sorted(by_pollutant[p], key=lambda t: t[0], reverse=True)
    for s, tag, r in rows[:8]:
        print(
            f"{tag}: score={s:+.3f} rmse_impr={float(r['rmse_improvement_pct']):+.2f}% "
            f"crps_impr={float(r['crps_improvement_pct']):+.2f}% cov={float(r['lstm_coverage']):.3f}"
        )
```

## 9) Final h168 report snapshot

```python
# %% [code]
fair_path, eval_path = run_eval("final_h168_snapshot")
print_h168_rows(fair_path)
print("saved:", fair_path)
print("saved:", eval_path)
print("elapsed hours:", (time.time() - PIPELINE_START_TS) / 3600.0)
```
