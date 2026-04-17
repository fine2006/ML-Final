# Notebook Run Script (Kaggle T4x2)

This runbook reflects the current pipeline scope:
- Pollutants: `pm25,pm10,no2,o3`
- Horizons: `h1,h24,h168`
- Evaluation focus: calibrated 95% CI (`p05-p95`), with p99 diagnostics retained

## 1) Setup

```python
# %% [code]
import time
PIPELINE_START_TS = time.time()

! rm -rf ML-Final && git clone https://github.com/fine2006/ML-Final -b new-approach
! cd ML-Final && uv sync
! cp -r "/kaggle/input/datasets/fine2006/pollution-data-raipur/Pollution Data Raipur" ML-Final/
```

```python
# %% [code]
! cd ML-Final && uv run python -c 'import torch; print(f"torch: {torch.__version__}\ncuda: {torch.cuda.is_available()}"); print(f"gpu_count: {torch.cuda.device_count()}"); [print(f"gpu[{i}]={torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None'
```

## 2) Phase 1 + Preprocess

```python
# %% [code]
! cd ML-Final && uv run python scripts/data_investigation.py
```

```python
# %% [code]
! export MPLBACKEND=Agg && cd ML-Final && uv run python scripts/preprocess_lstm.py && uv run python scripts/preprocess_xgb.py
```

## 3) Bring XGB models from local build (skip Kaggle XGB training)

Expected files: `xgb_quantile_{pollutant}_h{1|24|168}_q{05|50|95|99}.json`

```python
# %% [code]
import pathlib, shutil, re

ROOT = pathlib.Path("ML-Final").resolve()
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

rx = re.compile(r"^xgb_quantile_(pm25|pm10|no2|o3)_h(1|24|168)_q(05|50|95|99)\.json$")
src_root = pathlib.Path("/kaggle/input")

candidates = [p for p in src_root.glob("**/xgb_quantile_*.json") if rx.match(p.name)]
if not candidates:
    raise RuntimeError("No xgb_quantile files found under /kaggle/input")

by_name = {}
for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
    by_name.setdefault(p.name, p)

for name, src in by_name.items():
    shutil.copy2(src, MODELS / name)

expected = {
    f"xgb_quantile_{p}_h{h}_q{q}.json"
    for p in ["pm25","pm10","no2","o3"]
    for h in [1,24,168]
    for q in ["05","50","95","99"]
}
present = {p.name for p in MODELS.glob("xgb_quantile_*.json") if rx.match(p.name)}
missing = sorted(expected - present)

print("xgb model count:", len(present))
if missing:
    print("missing examples:", missing[:10])
    raise RuntimeError(f"Missing {len(missing)} expected xgb model files")
```

## 4) LSTM dual-GPU training (horizon-specific)

```python
# %% [code]
import os, time, json, shutil, pathlib, subprocess
from datetime import datetime

ROOT = pathlib.Path("ML-Final").resolve()
MODELS = ROOT / "models"
SUMMARY_SHARED = MODELS / "lstm_training_summary.json"

EXPECTED_POLLUTANTS = {"pm25", "pm10", "no2", "o3"}
GPU_SPLIT = {0: "pm25,pm10", 1: "no2,o3"}

RUNS = [
    {"horizon": 1,   "seq_len": 168, "batch_size": 128, "lr": "1e-4", "epochs": 150, "patience": 24, "min_attn_window": 48},
    {"horizon": 24,  "seq_len": 336, "batch_size": 128, "lr": "8e-5", "epochs": 160, "patience": 22, "min_attn_window": 48},
    {"horizon": 168, "seq_len": 720, "batch_size": 96,  "lr": "6e-5", "epochs": 130, "patience": 18, "min_attn_window": 48},
]

META_KEYS = ["random_seed","device","training_mode","horizons","seq_len_by_horizon","quantiles","min_attn_window"]

def launch(gpu_id: int, pollutants: str, cfg: dict):
    h = cfg["horizon"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "uv","run","python","scripts/train_lstm.py",
        "--pollutants", pollutants,
        "--horizons", str(h),
        "--seq-len-map", f"{h}:{cfg['seq_len']}",
        "--device", "cuda",
        "--batch-size", str(cfg["batch_size"]),
        "--epochs", str(cfg["epochs"]),
        "--patience", str(cfg["patience"]),
        "--lr", cfg["lr"],
        "--min-attn-window", str(cfg["min_attn_window"]),
    ]
    return subprocess.Popen(cmd, cwd=ROOT, env=env)

def try_capture(path: pathlib.Path, captured_meta: dict, captured_pollutants: dict):
    if not path.exists():
        return
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return
    for k in META_KEYS:
        if k in obj:
            captured_meta[k] = obj[k]
    for pol, payload in obj.get("pollutants", {}).items():
        captured_pollutants[pol] = payload

def run_horizon(cfg: dict):
    h = cfg["horizon"]
    if SUMMARY_SHARED.exists():
        SUMMARY_SHARED.unlink()

    captured_meta, captured_pollutants = {}, {}

    p0 = launch(0, GPU_SPLIT[0], cfg)
    time.sleep(1.0)
    p1 = launch(1, GPU_SPLIT[1], cfg)
    procs = {"gpu0": p0, "gpu1": p1}
    snapshotted = set()

    while True:
        try_capture(SUMMARY_SHARED, captured_meta, captured_pollutants)
        for name, p in procs.items():
            if name in snapshotted:
                continue
            rc = p.poll()
            if rc is not None and SUMMARY_SHARED.exists():
                snap = MODELS / f"lstm_training_summary_h{h}_{name}.json"
                shutil.copy2(SUMMARY_SHARED, snap)
                snapshotted.add(name)
                print("captured:", snap.name)
        if all(p.poll() is not None for p in procs.values()):
            break
        time.sleep(2.0)

    rc = {name: p.wait() for name, p in procs.items()}
    print(f"h{h} exit codes:", rc)
    if any(code != 0 for code in rc.values()):
        raise RuntimeError(f"h{h} failed: {rc}")

    for _ in range(3):
        try_capture(SUMMARY_SHARED, captured_meta, captured_pollutants)
        time.sleep(0.5)
    for name in ["gpu0","gpu1"]:
        snap = MODELS / f"lstm_training_summary_h{h}_{name}.json"
        if snap.exists():
            try_capture(snap, captured_meta, captured_pollutants)

    missing = EXPECTED_POLLUTANTS - set(captured_pollutants.keys())
    if missing:
        raise RuntimeError(f"h{h}: missing pollutants in merged summary: {sorted(missing)}")

    merged_h = {
        "created_at": datetime.now().isoformat(),
        "random_seed": captured_meta.get("random_seed", 42),
        "device": captured_meta.get("device", "cuda"),
        "training_mode": captured_meta.get("training_mode", "separated_horizon_models"),
        "horizons": [h],
        "seq_len_by_horizon": {f"h{h}": cfg["seq_len"]},
        "quantiles": captured_meta.get("quantiles", [0.05,0.5,0.95,0.99]),
        "min_attn_window": captured_meta.get("min_attn_window", cfg["min_attn_window"]),
        "pollutants": {k: captured_pollutants[k] for k in sorted(captured_pollutants.keys())},
    }
    out_h = MODELS / f"lstm_training_summary_h{h}.json"
    out_h.write_text(json.dumps(merged_h, indent=2))
    print("wrote", out_h)
    return out_h

h_summaries = [run_horizon(cfg) for cfg in RUNS]

final_pollutants = {}
seq_map = {}
for p in h_summaries:
    obj = json.loads(p.read_text())
    seq_map.update(obj.get("seq_len_by_horizon", {}))
    for pol, payload in obj.get("pollutants", {}).items():
        final_pollutants.setdefault(pol, {})
        for k, v in payload.items():
            if str(k).startswith("h"):
                final_pollutants[pol][k] = v

for pol in EXPECTED_POLLUTANTS:
    for h in [1,24,168]:
        assert f"h{h}" in final_pollutants.get(pol, {}), f"missing {pol} h{h}"

final = {
    "created_at": datetime.now().isoformat(),
    "random_seed": 42,
    "device": "cuda",
    "training_mode": "separated_horizon_models",
    "horizons": [1,24,168],
    "seq_len_by_horizon": seq_map,
    "quantiles": [0.05,0.5,0.95,0.99],
    "min_attn_window": 48,
    "pollutants": {k: final_pollutants[k] for k in sorted(final_pollutants.keys())},
}
SUMMARY_SHARED.write_text(json.dumps(final, indent=2))
print("final summary:", SUMMARY_SHARED)
```

## 5) Evaluate with calibration + representativeness checks

```python
# %% [code]
import os, json, pathlib, subprocess

ROOT = pathlib.Path("ML-Final").resolve()
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

subprocess.run([
    "uv","run","python","scripts/evaluate.py",
    "--pollutants","pm25,pm10,no2,o3",
    "--horizons","1,24,168",
    "--device","cuda",
    "--fair-intersection",
    "--calibrate-quantiles",
], cwd=ROOT, env=env, check=True)

summary = json.loads((ROOT / "models" / "evaluation_summary.json").read_text())
fair = json.loads((ROOT / "models" / "fair_benchmark_summary.json").read_text())

print("fair rows:", len(fair.get("rows", [])))
print("calibration enabled:", summary.get("quantile_calibration_enabled"))
print("interval focus:", summary.get("interval_focus"))
print("has data quality context:", "phase1_data_quality_context" in summary)
```

## 6) Predict (single region, single horizon, all pollutants)

```python
# %% [code]
import pathlib, subprocess

ROOT = pathlib.Path("ML-Final").resolve()
subprocess.run([
    "uv","run","python","scripts/predict.py",
    "--region","AIIMS",
    "--horizon","24",
    "--pollutants","pm25,pm10,no2,o3",
    "--device","cuda",
    "--output","models/prediction_output.json",
], cwd=ROOT, check=True)

print("saved:", ROOT / "models" / "prediction_output.json")
print("elapsed hours:", (time.time() - PIPELINE_START_TS) / 3600.0)
```
