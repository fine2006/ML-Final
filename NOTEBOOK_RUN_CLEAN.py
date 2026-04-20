# %% [code]
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

try:
    import optuna
except Exception:
    optuna = None


PIPELINE_START_TS = time.time()
ROOT = Path("ML-Final")


def run_cmd(cmd, cwd=None, env=None, retries=1, retry_wait=15):
    if isinstance(cmd, list):
        print("$", " ".join(str(x) for x in cmd))
    else:
        print("$", cmd)
    attempt = 1
    while True:
        try:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempt >= retries:
                raise
            wait_s = retry_wait * attempt
            print(
                f"Command failed (exit={exc.returncode}) attempt {attempt}/{retries}; retrying in {wait_s}s"
            )
            time.sleep(wait_s)
            attempt += 1


def patch_requires_python_for_kaggle(pyproject_path: Path) -> None:
    text = pyproject_path.read_text(encoding="utf-8")
    old = 'requires-python = ">=3.13"'
    new = 'requires-python = ">=3.12"'
    if old in text:
        pyproject_path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print("Patched pyproject requires-python to >=3.12 for Kaggle fallback")


def uv_sync_with_fallback(project_root: Path) -> None:
    try:
        run_cmd(["uv", "sync"], cwd=project_root, retries=4, retry_wait=20)
        return
    except subprocess.CalledProcessError:
        print("Primary 'uv sync' failed. Retrying with system Python fallback.")

    patch_requires_python_for_kaggle(project_root / "pyproject.toml")
    env_sync = os.environ.copy()
    env_sync["UV_PYTHON_PREFERENCE"] = "system"
    run_cmd(
        ["uv", "sync", "--python", "3.12"],
        cwd=project_root,
        env=env_sync,
        retries=4,
        retry_wait=20,
    )


# %% [code]
# Clean clone + sync
if ROOT.exists():
    shutil.rmtree(ROOT)

run_cmd(["git", "clone", "https://github.com/fine2006/ML-Final", "-b", "new-approach"])
uv_sync_with_fallback(ROOT)

src_dataset = Path(
    "/kaggle/input/datasets/fine2006/pollution-data-raipur/Pollution Data Raipur"
)
dst_dataset = ROOT / "Pollution Data Raipur"
if dst_dataset.exists():
    shutil.rmtree(dst_dataset)
shutil.copytree(src_dataset, dst_dataset)
print("Copied dataset:", dst_dataset)


# %% [code]
# Quick environment check
run_cmd(
    [
        "uv",
        "run",
        "python",
        "-c",
        (
            "import torch;"
            "print(f'torch: {torch.__version__}');"
            "print(f'cuda: {torch.cuda.is_available()}');"
            "print(f'gpu_count: {torch.cuda.device_count()}');"
            "[print(f'gpu[{i}]={torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] "
            "if torch.cuda.is_available() else None"
        ),
    ],
    cwd=ROOT,
)


# %% [code]
# Phase 1 + preprocessing
run_cmd(["uv", "run", "python", "scripts/data_investigation.py"], cwd=ROOT)

env = os.environ.copy()
env["MPLBACKEND"] = "agg"
run_cmd(["uv", "run", "python", "scripts/preprocess_lstm.py"], cwd=ROOT, env=env)
run_cmd(["uv", "run", "python", "scripts/preprocess_xgb.py"], cwd=ROOT, env=env)


# %% [code]
# Helpers (needed before Phase 2)
repo = ROOT.resolve()
models_dir = repo / "models"
models_dir.mkdir(parents=True, exist_ok=True)
EXP = models_dir / "experiments"
EXP.mkdir(parents=True, exist_ok=True)






# %% [code]
# Phase 2: Optuna hyperparameter tuning for h168 (dual-GPU parallel)
print("\n=== Starting Optuna Hyperparameter Tuning for h168 (dual-GPU) ===")

# GPU 0: pm25
env_gpu0 = os.environ.copy()
env_gpu0["CUDA_VISIBLE_DEVICES"] = "0"

cmd_gpu0 = [
    "uv", "run", "python", "scripts/optuna_tune_h168.py",
    "--pollutants", "pm25",
    "--gpu-id", "0",
    "--output", "optuna_h168_gpu0.json",
    "--n-trials", "25",
]

print("$ CUDA_VISIBLE_DEVICES=0", " ".join(cmd_gpu0))
proc_gpu0 = subprocess.Popen(
    cmd_gpu0,
    cwd=ROOT,
    env=env_gpu0,
)

# GPU 1: pm10
env_gpu1 = os.environ.copy()
env_gpu1["CUDA_VISIBLE_DEVICES"] = "1"

cmd_gpu1 = [
    "uv", "run", "python", "scripts/optuna_tune_h168.py",
    "--pollutants", "pm10",
    "--gpu-id", "1",
    "--output", "optuna_h168_gpu1.json",
    "--n-trials", "25",
]

print("$ CUDA_VISIBLE_DEVICES=1", " ".join(cmd_gpu1))
proc_gpu1 = subprocess.Popen(
    cmd_gpu1,
    cwd=ROOT,
    env=env_gpu1,
)

# Wait for both
print("Waiting for GPU 0 (pm25)...")
ret_gpu0 = proc_gpu0.wait()
print(f"GPU 0 finished with exit code: {ret_gpu0}")

print("Waiting for GPU 1 (pm10)...")
ret_gpu1 = proc_gpu1.wait()
print(f"GPU 1 finished with exit code: {ret_gpu1}")

if ret_gpu0 != 0 or ret_gpu1 != 0:
    raise RuntimeError(f"Optuna failed: GPU0={ret_gpu0}, GPU1={ret_gpu1}")

# Merge results
print("Merging GPU results...")
run_cmd([
    "uv", "run", "python", "scripts/optuna_tune_h168.py",
    "--merge",
    "--files", str(EXP / "optuna_h168_gpu0.json"), str(EXP / "optuna_h168_gpu1.json"),
    "--output", "optuna_h168_best_configs.json",
], cwd=ROOT)


# %% [code]
# Phase 3: Walk-forward cross-validation for h168 (dual-GPU parallel)
# WFCV only for PM25/PM10 (NO2/O3 use pilot configs + holdout)
print("\n=== Starting Walk-Forward Cross-Validation for h168 (PM only) ===")

env_wfcv0 = os.environ.copy()
env_wfcv0["CUDA_VISIBLE_DEVICES"] = "0"

cmd_wfcv0 = [
    "uv", "run", "python", "scripts/walkforward_cv_h168.py",
    "--pollutants", "pm25",
    "--gpu-id", "0",
    "--output", "wfcv_h168_gpu0.json",
]

print("$ CUDA_VISIBLE_DEVICES=0", " ".join(cmd_wfcv0))
proc_wfcv0 = subprocess.Popen(
    cmd_wfcv0,
    cwd=ROOT,
    env=env_wfcv0,
)

env_wfcv1 = os.environ.copy()
env_wfcv1["CUDA_VISIBLE_DEVICES"] = "1"

cmd_wfcv1 = [
    "uv", "run", "python", "scripts/walkforward_cv_h168.py",
    "--pollutants", "pm10",
    "--gpu-id", "1",
    "--output", "wfcv_h168_gpu1.json",
]

print("$ CUDA_VISIBLE_DEVICES=1", " ".join(cmd_wfcv1))
proc_wfcv1 = subprocess.Popen(
    cmd_wfcv1,
    cwd=ROOT,
    env=env_wfcv1,
)

# Wait for both
print("Waiting for WFCV GPU 0 (pm25)...")
ret_wfcv0 = proc_wfcv0.wait()
print(f"WFCV GPU 0 finished with exit code: {ret_wfcv0}")

print("Waiting for WFCV GPU 1 (pm10)...")
ret_wfcv1 = proc_wfcv1.wait()
print(f"WFCV GPU 1 finished with exit code: {ret_wfcv1}")

if ret_wfcv0 != 0 or ret_wfcv1 != 0:
    raise RuntimeError(f"WFCV failed: GPU0={ret_wfcv0}, GPU1={ret_wfcv1}")

# Merge WFCV results
print("Merging WFCV GPU results...")
run_cmd([
    "uv", "run", "python", "scripts/walkforward_cv_h168.py",
    "--merge",
    "--files", str(EXP / "wfcv_h168_gpu0.json"), str(EXP / "wfcv_h168_gpu1.json"),
    "--output", "wfcv_h168_results.json",
], cwd=ROOT)


# %% [code]
# Phase 4: Aggregate h168 results
print("\n=== Starting Results Aggregation for h168 ===")
run_cmd([
    "uv", "run", "python", "scripts/aggregate_results_h168.py"
], cwd=ROOT)


# %% [code]
# Keep XGB frozen by default (use precomputed models)
repo = ROOT.resolve()
models_dir = repo / "models"
models_dir.mkdir(parents=True, exist_ok=True)

TRAIN_XGB = False
if TRAIN_XGB:
    run_cmd(
        [
            "uv",
            "run",
            "python",
            "scripts/train_xgb.py",
            "--pollutants",
            "pm25,pm10,no2,o3",
            "--horizons",
            "1,24,168",
            "--device",
            "auto",
        ],
        cwd=repo,
    )
else:
    print("Skipping XGB retrain (TRAIN_XGB=False); expecting precomputed XGB models in models/.")

rx = re.compile(r"^xgb_quantile_(pm25|pm10|no2|o3)_h(1|24|168)_q(05|50|95|99)\.json$")
expected = {
    f"xgb_quantile_{p}_h{h}_q{q}.json"
    for p in ["pm25", "pm10", "no2", "o3"]
    for h in [1, 24, 168]
    for q in ["05", "50", "95", "99"]
}
present = {p.name for p in models_dir.glob("xgb_quantile_*.json") if rx.match(p.name)}
missing = sorted(expected - present)
print("xgb model count:", len(present))
if missing:
    print("missing examples:", missing[:10])
    raise RuntimeError(f"Missing {len(missing)} expected XGB files in {models_dir}")


# %% [code]
# Helpers
EXP = models_dir / "experiments"
EXP.mkdir(parents=True, exist_ok=True)


def run_eval(tag, horizons="168", pollutants="pm25,pm10,no2,o3"):
    env_eval = os.environ.copy()
    env_eval["MPLBACKEND"] = "agg"
    env_eval["EVAL_EXPERIMENT_TAG"] = str(tag)
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/evaluate.py",
        "--pollutants",
        pollutants,
        "--horizons",
        horizons,
        "--device",
        "auto",
        "--fair-intersection",
        "--calibrate-quantiles",
    ]
    run_cmd(cmd, cwd=repo, env=env_eval)

    fair_src = models_dir / "fair_benchmark_summary.json"
    eval_src = models_dir / "evaluation_summary.json"
    fair_dst = EXP / f"{tag}_fair_benchmark_summary.json"
    eval_dst = EXP / f"{tag}_evaluation_summary.json"
    fair_dst.write_text(fair_src.read_text())
    eval_dst.write_text(eval_src.read_text())
    return fair_dst, eval_dst


print("Helper setup complete.")


# %% [code]
# Final snapshot and runtime
fair_path, eval_path = run_eval("final_h168_operational_snapshot", horizons="168")
print_gate_rows(fair_path)
print("saved:", fair_path)
print("saved:", eval_path)
print("elapsed hours:", (time.time() - PIPELINE_START_TS) / 3600.0)
