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


def build_train_cmd(cfg):
    target_mode = cfg.get("target_mode", "level")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_lstm.py",
        "--pollutants",
        cfg["pollutant"],
        "--horizons",
        "168",
        "--seq-len-map",
        f"168:{cfg['seq_len']}",
        "--target-mode",
        target_mode,
        "--delta-baseline-window",
        str(cfg.get("delta_baseline_window", 3)),
        "--device",
        "cuda",
        "--batch-size",
        str(cfg.get("batch_size", 128)),
        "--epochs",
        str(cfg.get("epochs", 80)),
        "--patience",
        str(cfg.get("patience", 12)),
        "--lr",
        str(cfg["lr"]),
        "--hidden-dim",
        str(cfg["hidden_dim"]),
        "--num-layers",
        str(cfg.get("num_layers", 2)),
        "--num-heads",
        str(cfg.get("num_heads", 2)),
        "--dropout",
        str(cfg["dropout"]),
        "--head-dropout",
        str(cfg.get("head_dropout", 0.2)),
        "--weight-decay",
        str(cfg["weight_decay"]),
        "--scheduler-factor",
        "0.5",
        "--scheduler-patience",
        str(cfg.get("scheduler_patience", 4)),
        "--max-grad-norm",
        "1.0",
        "--min-attn-window",
        "24",
        "--summary-path",
        str(EXP / f"{cfg['tag']}_summary.json"),
    ]
    return cmd


def run_pair(cfg0, cfg1):
    if cfg0["pollutant"] == cfg1["pollutant"]:
        raise ValueError(
            f"Cannot run same pollutant concurrently: {cfg0['pollutant']}"
        )

    jobs = []
    for gpu_id, cfg in ((0, cfg0), (1, cfg1)):
        cmd = build_train_cmd(cfg)
        env_train = os.environ.copy()
        env_train["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = EXP / f"{cfg['tag']}_train_gpu{gpu_id}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        print("$", f"CUDA_VISIBLE_DEVICES={gpu_id}", " ".join(str(x) for x in cmd), f"> {log_path}")
        proc = subprocess.Popen(
            cmd,
            cwd=repo,
            env=env_train,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        jobs.append((cfg, gpu_id, proc, log_handle, log_path))

    failures = []
    for cfg, gpu_id, proc, log_handle, log_path in jobs:
        ret = proc.wait()
        log_handle.close()
        if ret == 0:
            print(f"[{cfg['tag']}] finished on gpu={gpu_id} (log: {log_path})")
        else:
            failures.append((cfg["tag"], gpu_id, ret, str(log_path)))
    if failures:
        raise RuntimeError(f"Dual-GPU training failed: {failures}")


def gate_for_row(row):
    p = row["pollutant"]
    if p in {"pm25", "pm10"}:
        return (
            float(row["lstm_rmse_over_mean"]) < 0.5
            and float(row["lstm_r2"]) > 0.3
        )
    return (
        float(row["lstm_rmse_over_mean"]) < 0.8
        and bool(row.get("beats_persistence", False))
    )


def read_gate_rows(fair_path):
    fair = json.loads(Path(fair_path).read_text())
    op = fair.get("operational_gates", {})
    rows = op.get("rows", [])
    return [r for r in rows if int(r.get("horizon", -1)) == 168]


def print_gate_rows(fair_path):
    rows = sorted(read_gate_rows(fair_path), key=lambda r: r["pollutant"])
    print(
        "pollutant,h,mean,lstm_rmse,xgb_rmse,lstm_r2,xgb_r2,lstm_rmse_over_mean,"
        "xgb_rmse_over_mean,lstm_cov95,lstm_pit_ks,beats_persistence,beats_climatology,gate_pass"
    )
    for r in rows:
        print(
            f"{r['pollutant']},{int(r['horizon'])},{float(r['mean_concentration']):.6f},"
            f"{float(r['lstm_rmse']):.6f},{float(r['xgb_rmse']):.6f},"
            f"{float(r['lstm_r2']):.6f},{float(r['xgb_r2']):.6f},"
            f"{float(r['lstm_rmse_over_mean']):.6f},{float(r['xgb_rmse_over_mean']):.6f},"
            f"{float(r['lstm_coverage']):.6f},{float(r['lstm_pit_ks']):.6f},"
            f"{bool(r.get('beats_persistence', False))},{bool(r.get('beats_climatology', False))},"
            f"{bool(r.get('gate_pass', gate_for_row(r)))}"
        )


def random_space_with_seed(seed=42):
    rng = random.Random(seed)

    def log_uniform(a, b):
        return 10 ** rng.uniform(math.log10(a), math.log10(b))

    runs = []

    # 16 gas rescue
    for i in range(8):
        runs.append(
            {
                "tag": f"pilot_no2_delta_{i+1:02d}",
                "pollutant": "no2",
                "target_mode": "delta_ma3",
                "delta_baseline_window": 3,
                "seq_len": rng.choice([48, 72, 168]),
                "hidden_dim": rng.choice([32, 64, 96, 128]),
                "num_layers": 2,
                "num_heads": rng.choice([2, 4]),
                "lr": log_uniform(5e-5, 5e-4),
                "dropout": round(rng.uniform(0.1, 0.4), 3),
                "head_dropout": round(rng.uniform(0.1, 0.35), 3),
                "weight_decay": log_uniform(1e-5, 1e-2),
                "epochs": 70,
                "patience": 10,
                "scheduler_patience": 3,
            }
        )
    for i in range(8):
        runs.append(
            {
                "tag": f"pilot_o3_delta_{i+1:02d}",
                "pollutant": "o3",
                "target_mode": "delta_ma3",
                "delta_baseline_window": 3,
                "seq_len": rng.choice([48, 72, 168]),
                "hidden_dim": rng.choice([32, 64, 96, 128]),
                "num_layers": 2,
                "num_heads": rng.choice([2, 4]),
                "lr": log_uniform(5e-5, 5e-4),
                "dropout": round(rng.uniform(0.1, 0.4), 3),
                "head_dropout": round(rng.uniform(0.1, 0.35), 3),
                "weight_decay": log_uniform(1e-5, 1e-2),
                "epochs": 70,
                "patience": 10,
                "scheduler_patience": 3,
            }
        )

    # 8 PM refinement
    for i in range(4):
        runs.append(
            {
                "tag": f"pilot_pm25_refine_{i+1:02d}",
                "pollutant": "pm25",
                "target_mode": "level",
                "seq_len": rng.choice([336, 504, 672]),
                "hidden_dim": rng.choice([64, 96, 128, 256]),
                "num_layers": 2,
                "num_heads": rng.choice([2, 4]),
                "lr": log_uniform(5e-5, 5e-4),
                "dropout": round(rng.uniform(0.2, 0.4), 3),
                "head_dropout": round(rng.uniform(0.15, 0.35), 3),
                "weight_decay": log_uniform(1e-5, 2e-3),
                "epochs": 80,
                "patience": 12,
                "scheduler_patience": 4,
            }
        )
    for i in range(4):
        runs.append(
            {
                "tag": f"pilot_pm10_refine_{i+1:02d}",
                "pollutant": "pm10",
                "target_mode": "level",
                "seq_len": rng.choice([336, 504, 672]),
                "hidden_dim": rng.choice([64, 96, 128, 256]),
                "num_layers": 2,
                "num_heads": rng.choice([2, 4]),
                "lr": log_uniform(5e-5, 5e-4),
                "dropout": round(rng.uniform(0.2, 0.4), 3),
                "head_dropout": round(rng.uniform(0.15, 0.35), 3),
                "weight_decay": log_uniform(1e-5, 2e-3),
                "epochs": 80,
                "patience": 12,
                "scheduler_patience": 4,
            }
        )

    return runs


print("Helper setup complete.")


# %% [code]
# Pilot: first 8 runs only, then gate check
ALL_RUNS = random_space_with_seed(seed=42)
PILOT_RUNS = ALL_RUNS[:8]

for i in range(0, len(PILOT_RUNS), 2):
    pair = PILOT_RUNS[i : i + 2]
    if len(pair) == 2:
        print("\n=== dual-gpu pilot batch ===", pair[0]["tag"], "|", pair[1]["tag"], "===")
        run_pair(pair[0], pair[1])
    else:
        cfg = pair[0]
        print("\n=== single-gpu pilot batch ===", cfg["tag"], "===")
        run_pair(cfg, {
            "tag": "_noop_",
            "pollutant": "pm25",
            "target_mode": "level",
            "seq_len": 336,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "lr": 1.5e-4,
            "dropout": 0.25,
            "head_dropout": 0.2,
            "weight_decay": 1e-4,
            "epochs": 1,
            "patience": 1,
            "scheduler_patience": 1,
        })

    for cfg in pair:
        fair_path, _ = run_eval(cfg["tag"], horizons="168", pollutants=cfg["pollutant"])
        print_gate_rows(fair_path)


# %% [code]
# Pilot summary and go/no-go
pilot_tags = {cfg["tag"] for cfg in PILOT_RUNS}
pilot_rows = []
for fp in sorted(EXP.glob("*_fair_benchmark_summary.json")):
    tag = fp.name.replace("_fair_benchmark_summary.json", "")
    if tag not in pilot_tags:
        continue
    fair = json.loads(fp.read_text())
    for r in fair.get("operational_gates", {}).get("rows", []):
        if int(r.get("horizon", -1)) != 168:
            continue
        pilot_rows.append((tag, r))

pass_rows = [x for x in pilot_rows if bool(x[1].get("gate_pass", False))]
gas_rows = [x for x in pilot_rows if x[1]["pollutant"] in {"no2", "o3"}]
gas_positive_signal = [
    x for x in gas_rows
    if bool(x[1].get("beats_persistence", False))
    and float(x[1].get("lstm_coverage", 0.0)) >= 0.80
]

print("pilot_total_rows:", len(pilot_rows))
print("pilot_gate_pass_rows:", len(pass_rows))
print("pilot_gas_positive_signal_rows:", len(gas_positive_signal))

for tag, r in sorted(gas_rows, key=lambda t: (t[1]["pollutant"], t[0])):
    print(
        f"{tag} {r['pollutant']} rmse/mean={float(r['lstm_rmse_over_mean']):.3f} "
        f"cov={float(r['lstm_coverage']):.3f} pit_ks={float(r['lstm_pit_ks']):.3f} "
        f"beat_persist={bool(r.get('beats_persistence', False))}"
    )

AUTO_CONTINUE = len(gas_positive_signal) > 0
print("AUTO_CONTINUE_REMAINING_16:", AUTO_CONTINUE)


# %% [code]
# Remaining 16 runs (only if pilot shows signal)
REMAINING_RUNS = ALL_RUNS[8:]
if AUTO_CONTINUE:
    for i in range(0, len(REMAINING_RUNS), 2):
        pair = REMAINING_RUNS[i : i + 2]
        print("\n=== dual-gpu remaining batch ===", pair[0]["tag"], "|", pair[1]["tag"], "===")
        run_pair(pair[0], pair[1])

        for cfg in pair:
            fair_path, _ = run_eval(cfg["tag"], horizons="168", pollutants=cfg["pollutant"])
            print_gate_rows(fair_path)
else:
    print("Skipping remaining 16 runs due to no positive gas signal in pilot.")


# %% [code]
# Consolidated h168 leaderboard by operational gates
allowed_tags = {cfg["tag"] for cfg in ALL_RUNS}
rows = []
for fp in sorted(EXP.glob("*_fair_benchmark_summary.json")):
    tag = fp.name.replace("_fair_benchmark_summary.json", "")
    if tag not in allowed_tags:
        continue
    fair = json.loads(fp.read_text())
    for r in fair.get("operational_gates", {}).get("rows", []):
        if int(r.get("horizon", -1)) != 168:
            continue
        rows.append((tag, r))

for pollutant in ["pm25", "pm10", "no2", "o3"]:
    subset = [(tag, r) for tag, r in rows if r["pollutant"] == pollutant]
    subset.sort(
        key=lambda t: (
            0 if bool(t[1].get("gate_pass", False)) else 1,
            float(t[1].get("lstm_rmse_over_mean", 999.0)),
            -float(t[1].get("lstm_r2", -999.0)),
            float(abs(float(t[1].get("lstm_coverage", 0.0)) - 0.90)),
            float(t[1].get("lstm_pit_ks", 999.0)),
        )
    )
    print("\n===", pollutant, "operational ranking (h168) ===")
    for tag, r in subset[:12]:
        print(
            f"{tag}: pass={bool(r.get('gate_pass', False))} rmse/mean={float(r['lstm_rmse_over_mean']):.3f} "
            f"r2={float(r['lstm_r2']):+.3f} cov={float(r['lstm_coverage']):.3f} "
            f"pit_ks={float(r['lstm_pit_ks']):.3f} beat_persist={bool(r.get('beats_persistence', False))} "
            f"beat_clim={bool(r.get('beats_climatology', False))}"
        )


# %% [code]
# Final snapshot and runtime
fair_path, eval_path = run_eval("final_h168_operational_snapshot", horizons="168")
print_gate_rows(fair_path)
print("saved:", fair_path)
print("saved:", eval_path)
print("elapsed hours:", (time.time() - PIPELINE_START_TS) / 3600.0)
