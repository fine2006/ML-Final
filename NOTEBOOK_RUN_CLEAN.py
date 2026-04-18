# %% [code]
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

PIPELINE_START_TS = time.time()
ROOT = Path("ML-Final")


def run_cmd(cmd, cwd=None, env=None):
    if isinstance(cmd, list):
        print("$", " ".join(str(x) for x in cmd))
    else:
        print("$", cmd)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


# %% [code]
# Clean clone + sync
if ROOT.exists():
    shutil.rmtree(ROOT)

run_cmd(["git", "clone", "https://github.com/fine2006/ML-Final", "-b", "new-approach"])
run_cmd(["uv", "sync"], cwd=ROOT)

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
# Metadata checks for single-latency contract
repo = ROOT.resolve()
lstm_meta = json.loads((repo / "data" / "preprocessed_lstm_v1" / "metadata.json").read_text())
xgb_meta = json.loads((repo / "data" / "preprocessed_xgb_v1" / "metadata.json").read_text())

print("LSTM feature columns:", lstm_meta["feature_columns"])
print("LSTM leakage policy:", lstm_meta["leakage_policy"])
print("XGB leakage policy:", xgb_meta["leakage_policy"])

assert "temperature" in lstm_meta["feature_columns"]
assert "temperature_lag_1" not in lstm_meta["feature_columns"]
assert xgb_meta["leakage_policy"]["weather_current_values_shifted_by"] == 0


# %% [code]
# Keep XGB frozen by default (use precomputed models)
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


# %% [code]
# Verify expected XGB files
models_dir = repo / "models"
models_dir.mkdir(parents=True, exist_ok=True)

rx = re.compile(r"^xgb_quantile_(pm25|pm10|no2|o3)_h(1|24|168)_q(05|50|95|99)\\.json$")
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


def run_eval(tag, horizons="1,24,168"):
    env_eval = os.environ.copy()
    env_eval["MPLBACKEND"] = "agg"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/evaluate.py",
        "--pollutants",
        "pm25,pm10,no2,o3",
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


def print_rows(fair_path, horizons=None):
    fair = json.loads(Path(fair_path).read_text())
    rows = fair.get("rows", [])
    if horizons is not None:
        hset = {int(h) for h in horizons}
        rows = [r for r in rows if int(r["horizon"]) in hset]
    rows = sorted(rows, key=lambda r: (int(r["horizon"]), r["pollutant"]))

    print(
        "pollutant,h,lstm_rmse,xgb_rmse,rmse_impr_pct,lstm_crps,xgb_crps,"
        "crps_impr_pct,lstm_cov95,xgb_cov95"
    )
    for r in rows:
        print(
            f"{r['pollutant']},{int(r['horizon'])},{float(r['lstm_rmse']):.6f},"
            f"{float(r['xgb_rmse']):.6f},{float(r['rmse_improvement_pct']):+.2f},"
            f"{float(r['lstm_crps']):.6f},{float(r['xgb_crps']):.6f},"
            f"{float(r['crps_improvement_pct']):+.2f},{float(r['lstm_coverage']):.6f},"
            f"{float(r['xgb_coverage']):.6f}"
        )


def assert_lstm_checkpoints(horizons, pollutants=None):
    if pollutants is None:
        pollutants = ["pm25", "pm10", "no2", "o3"]
    missing_ckpts = []
    for p in pollutants:
        for h in horizons:
            ckpt = models_dir / f"lstm_quantile_{p}_h{int(h)}.pt"
            if not ckpt.exists():
                missing_ckpts.append(str(ckpt))
    if missing_ckpts:
        raise RuntimeError(
            "Missing LSTM checkpoints (first 10 shown):\n" + "\n".join(missing_ckpts[:10])
        )
    print("LSTM checkpoints OK for pollutants", pollutants, "horizons", list(horizons))


def train_lock_best_all3():
    # Locked best config discovered previously: h168_05_64_soft_336 profile
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_lstm.py",
        "--pollutants",
        "pm25,pm10,no2,o3",
        "--horizons",
        "1,24,168",
        "--seq-len-map",
        "1:24,24:24,168:336",
        "--device",
        "auto",
        "--batch-size",
        "128",
        "--epochs",
        "80",
        "--patience",
        "12",
        "--lr",
        "1.5e-4",
        "--hidden-dim",
        "64",
        "--num-layers",
        "2",
        "--num-heads",
        "2",
        "--dropout",
        "0.25",
        "--head-dropout",
        "0.20",
        "--weight-decay",
        "1e-4",
        "--scheduler-factor",
        "0.5",
        "--scheduler-patience",
        "4",
        "--max-grad-norm",
        "1.0",
        "--min-attn-window",
        "24",
        "--summary-path",
        str(EXP / "lock_best_all3_summary.json"),
    ]
    run_cmd(cmd, cwd=repo)


def run_train_h168(
    tag,
    pollutant,
    seq_len,
    hidden_dim,
    num_layers,
    num_heads,
    lr,
    dropout,
    head_dropout,
    weight_decay,
    epochs=80,
    patience=12,
    scheduler_patience=4,
    batch_size=128,
):
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_lstm.py",
        "--pollutants",
        pollutant,
        "--horizons",
        "168",
        "--seq-len-map",
        f"168:{seq_len}",
        "--device",
        "auto",
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--lr",
        str(lr),
        "--hidden-dim",
        str(hidden_dim),
        "--num-layers",
        str(num_layers),
        "--num-heads",
        str(num_heads),
        "--dropout",
        str(dropout),
        "--head-dropout",
        str(head_dropout),
        "--weight-decay",
        str(weight_decay),
        "--scheduler-factor",
        "0.5",
        "--scheduler-patience",
        str(scheduler_patience),
        "--max-grad-norm",
        "1.0",
        "--min-attn-window",
        "24",
        "--summary-path",
        str(EXP / f"{tag}_summary.json"),
    ]
    run_cmd(cmd, cwd=repo)


def score_row(r):
    return (
        0.65 * float(r["rmse_improvement_pct"])
        + 0.35 * float(r["crps_improvement_pct"])
        - 10.0 * abs(float(r["lstm_coverage"]) - 0.90)
    )


print("Helper setup complete.")


# %% [code]
# 1) Train LSTM on all three horizons using the previously found best config
train_lock_best_all3()
assert_lstm_checkpoints(horizons=[1, 24, 168])


# %% [code]
# 2) Eval once after lock-train (all three horizons)
fair_path, eval_path = run_eval("baseline_after_lock_all3", horizons="1,24,168")
print_rows(fair_path, horizons=[1, 24, 168])
print("saved:", fair_path)
print("saved:", eval_path)


# %% [code]
# 3) h168-only sweep
RUNS = [
    {
        "tag": "h168_pm25_soft336",
        "pollutant": "pm25",
        "seq_len": 336,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "lr": 1.5e-4,
        "dropout": 0.25,
        "head_dropout": 0.20,
        "weight_decay": 1e-4,
    },
    {
        "tag": "h168_pm10_soft336",
        "pollutant": "pm10",
        "seq_len": 336,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "lr": 1.5e-4,
        "dropout": 0.25,
        "head_dropout": 0.20,
        "weight_decay": 1e-4,
    },
    {
        "tag": "h168_no2_soft336_hd96",
        "pollutant": "no2",
        "seq_len": 336,
        "hidden_dim": 96,
        "num_layers": 2,
        "num_heads": 4,
        "lr": 1.5e-4,
        "dropout": 0.25,
        "head_dropout": 0.20,
        "weight_decay": 1e-4,
    },
    {
        "tag": "h168_no2_reg336_hd96",
        "pollutant": "no2",
        "seq_len": 336,
        "hidden_dim": 96,
        "num_layers": 2,
        "num_heads": 4,
        "lr": 2.0e-4,
        "dropout": 0.30,
        "head_dropout": 0.25,
        "weight_decay": 2e-4,
        "epochs": 60,
        "patience": 10,
        "scheduler_patience": 3,
    },
    {
        "tag": "h168_o3_soft168",
        "pollutant": "o3",
        "seq_len": 168,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "lr": 1.5e-4,
        "dropout": 0.25,
        "head_dropout": 0.20,
        "weight_decay": 1e-4,
    },
    {
        "tag": "h168_o3_soft336_hd96",
        "pollutant": "o3",
        "seq_len": 336,
        "hidden_dim": 96,
        "num_layers": 2,
        "num_heads": 4,
        "lr": 1.5e-4,
        "dropout": 0.25,
        "head_dropout": 0.20,
        "weight_decay": 1e-4,
    },
]

for cfg in RUNS:
    print("\n===", cfg["tag"], "===")
    run_train_h168(**cfg)
    assert_lstm_checkpoints(horizons=[168], pollutants=[cfg["pollutant"]])
    fair_path, _ = run_eval(cfg["tag"], horizons="168")
    print_rows(fair_path, horizons=[168])


# %% [code]
# 4) Leaderboard only on h168
allowed_tags = {"baseline_after_lock_all3"} | {cfg["tag"] for cfg in RUNS}

by_pollutant = {p: [] for p in ["pm25", "pm10", "no2", "o3"]}
global_rows = []

for fp in sorted(EXP.glob("*_fair_benchmark_summary.json")):
    tag = fp.name.replace("_fair_benchmark_summary.json", "")
    if tag not in allowed_tags:
        continue
    fair = json.loads(fp.read_text())
    for r in fair.get("rows", []):
        if int(r["horizon"]) != 168:
            continue
        p = r["pollutant"]
        if p not in by_pollutant:
            continue
        s = score_row(r)
        by_pollutant[p].append((s, tag, r))
        global_rows.append((s, tag, p, r))

for p in ["pm25", "pm10", "no2", "o3"]:
    print("\n===", p, "h168 ranking ===")
    rows = sorted(by_pollutant[p], key=lambda t: t[0], reverse=True)
    for s, tag, r in rows[:10]:
        print(
            f"{tag}: score={s:+.3f} rmse_impr={float(r['rmse_improvement_pct']):+.2f}% "
            f"crps_impr={float(r['crps_improvement_pct']):+.2f}% cov={float(r['lstm_coverage']):.3f}"
        )

print("\n=== global h168 ranking (top 20 rows) ===")
for s, tag, p, r in sorted(global_rows, key=lambda t: t[0], reverse=True)[:20]:
    print(
        f"{tag} [{p}]: score={s:+.3f} rmse_impr={float(r['rmse_improvement_pct']):+.2f}% "
        f"crps_impr={float(r['crps_improvement_pct']):+.2f}% cov={float(r['lstm_coverage']):.3f}"
    )


# %% [code]
# Final h168 snapshot
fair_path, eval_path = run_eval("final_h168_snapshot", horizons="168")
print_rows(fair_path, horizons=[168])
print("saved:", fair_path)
print("saved:", eval_path)
print("elapsed hours:", (time.time() - PIPELINE_START_TS) / 3600.0)
