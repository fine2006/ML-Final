"""
Phase 3-6 orchestrator.

Runs preprocessing and training scripts that follow the canonical
hourly+quarterly merge contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent


def run_cmd(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def uv_run(py_args: list[str]) -> list[str]:
    return ["uv", "run", "python", *py_args]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def print_training_snapshot() -> None:
    lstm_summary = load_json(ROOT / "models" / "lstm_training_summary.json")
    xgb_summary = load_json(ROOT / "models" / "xgb_training_summary.json")

    print("\n" + "=" * 72)
    print("TRAINING SNAPSHOT")
    print("=" * 72)

    if lstm_summary:
        print("\n[LSTM]")
        for pollutant, payload in lstm_summary.get("pollutants", {}).items():
            if isinstance(payload, dict) and "test_metrics" in payload:
                test = payload.get("test_metrics", {})
                print(
                    f"  {pollutant}: "
                    f"RMSE(p50)={test.get('overall_rmse_p50', float('nan')):.4f} "
                    f"CRPS={test.get('overall_crps_approx', float('nan')):.4f}"
                )
                continue

            for key in sorted(payload.keys()):
                if not str(key).startswith("h"):
                    continue
                h_payload = payload.get(key, {})
                if not isinstance(h_payload, dict):
                    continue
                test = h_payload.get("test_metrics", {})
                horizon = h_payload.get("horizon")
                seq_len = h_payload.get("seq_len")
                print(
                    f"  {pollutant} h{horizon} (seq={seq_len}): "
                    f"RMSE(p50)={test.get('overall_rmse_p50', float('nan')):.4f} "
                    f"CRPS={test.get('overall_crps_approx', float('nan')):.4f}"
                )
    else:
        print("\n[LSTM] summary not found")

    if xgb_summary:
        print("\n[XGB]")
        preferred_horizons = xgb_summary.get("horizons", [])
        preferred_h = (
            24
            if 24 in preferred_horizons
            else (preferred_horizons[0] if preferred_horizons else None)
        )
        if preferred_h is None:
            print("  no horizon metrics available")
            return
        for pollutant, payload in xgb_summary.get("pollutants", {}).items():
            metrics = payload.get("metrics", {})
            h_metrics = metrics.get(f"h{preferred_h}", {})
            print(
                f"  {pollutant}: h{preferred_h} RMSE={h_metrics.get('rmse', float('nan')):.4f} "
                f"CRPS={h_metrics.get('crps_approx', float('nan')):.4f}"
            )
    else:
        print("\n[XGB] summary not found")


def print_evaluation_snapshot() -> None:
    eval_summary = load_json(ROOT / "models" / "evaluation_summary.json")
    if not eval_summary:
        print("\n[EVAL] summary not found")
        return

    comparison = eval_summary.get("comparison", [])
    if not comparison:
        print("\n[EVAL] no LSTM/XGB overlap to compare")
        return

    print("\n[EVAL]")
    by_key: dict[tuple[str, int], dict] = {}
    for row in comparison:
        pollutant = str(row.get("pollutant"))
        horizon = int(row.get("horizon"))
        by_key[(pollutant, horizon)] = row

    preferred_horizons = [24, 1, 672]
    shown = 0
    for (pollutant, horizon), row in sorted(by_key.items()):
        if horizon not in preferred_horizons:
            continue
        print(
            "  "
            f"{pollutant} h{horizon}: "
            f"RMSE Δ={row.get('rmse_improvement_pct', float('nan')):.2f}% "
            f"CRPS Δ={row.get('crps_improvement_pct', float('nan')):.2f}%"
        )
        shown += 1
        if shown >= 4:
            break

    if shown == 0:
        first = comparison[0]
        print(
            "  "
            f"{first.get('pollutant')} h{first.get('horizon')}: "
            f"RMSE Δ={first.get('rmse_improvement_pct', float('nan')):.2f}% "
            f"CRPS Δ={first.get('crps_improvement_pct', float('nan')):.2f}%"
        )


def print_prediction_snapshot(prediction_output_path: Path) -> None:
    payload = load_json(prediction_output_path)
    if not payload:
        print("\n[PREDICT] output not found")
        return

    print("\n[PREDICT]")
    print(
        "  "
        f"region={payload.get('region')} "
        f"horizon={payload.get('horizon_hours')}h "
        f"anchor={payload.get('anchor_timestamp')}"
    )
    pollutants = payload.get("pollutants", {})
    for pollutant, pred in pollutants.items():
        q = pred.get("quantiles", {})

        def fmt(value: object) -> str:
            try:
                return f"{float(value):.3f}"
            except (TypeError, ValueError):
                return "nan"

        print(
            "  "
            f"{pollutant}: "
            f"p05={fmt(q.get('p05'))} "
            f"p50={fmt(q.get('p50'))} "
            f"p95={fmt(q.get('p95'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3-6 pipeline runner")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--pollutants",
        type=str,
        default="pm25,pm10,no2,o3",
        help="Comma-separated pollutant list",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,24,672",
        help="Comma-separated horizons in hours",
    )
    parser.add_argument(
        "--lstm-max-train-samples",
        type=int,
        default=None,
        help="Optional cap for LSTM train sequence samples",
    )
    parser.add_argument(
        "--lstm-max-val-samples",
        type=int,
        default=None,
        help="Optional cap for LSTM val sequence samples",
    )
    parser.add_argument(
        "--lstm-max-test-samples",
        type=int,
        default=None,
        help="Optional cap for LSTM test sequence samples",
    )
    parser.add_argument(
        "--lstm-seq-len-map",
        type=str,
        default="",
        help="Optional LSTM horizon:seq_len map, e.g. 1:168,24:336,672:2402",
    )
    parser.add_argument("--fair-bench", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--region", type=str, default="")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--timestamp",
        type=str,
        default="",
        help="Optional inference anchor timestamp for predict mode",
    )
    parser.add_argument(
        "--predict-output",
        type=str,
        default="models/prediction_output.json",
        help="Path to write prediction JSON in predict mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for train/evaluate/predict scripts",
    )
    args = parser.parse_args()

    if args.preprocess_only and args.train_only:
        raise SystemExit("Cannot use both --preprocess-only and --train-only")
    if args.preprocess_only and args.evaluate_only:
        raise SystemExit("Cannot use both --preprocess-only and --evaluate-only")
    if args.train_only and args.evaluate_only:
        raise SystemExit("Cannot use both --train-only and --evaluate-only")
    if args.predict_only and (
        args.preprocess_only or args.train_only or args.evaluate_only
    ):
        raise SystemExit(
            "--predict-only cannot be combined with preprocess/train/evaluate only modes"
        )

    run_preprocess = not args.skip_preprocess
    run_train = not args.skip_train
    run_evaluate = not args.skip_evaluate
    run_predict = False
    if args.preprocess_only:
        run_preprocess = True
        run_train = False
        run_evaluate = False
    if args.train_only:
        run_preprocess = False
        run_train = True
        run_evaluate = False
    if args.evaluate_only:
        run_preprocess = False
        run_train = False
        run_evaluate = True
    if args.predict_only:
        run_preprocess = False
        run_train = False
        run_evaluate = False
        run_predict = True

    if run_predict:
        if not args.region:
            raise SystemExit("--region is required with --predict-only")
        if args.horizon is None:
            raise SystemExit("--horizon is required with --predict-only")

    print("=" * 72)
    print("AIR POLLUTION PIPELINE (PHASE 3-6)")
    print("=" * 72)

    if run_preprocess:
        run_cmd(uv_run(["scripts/preprocess_lstm.py"]))
        run_cmd(uv_run(["scripts/preprocess_xgb.py"]))

    if run_train:
        lstm_cmd = [
            "scripts/train_lstm.py",
            "--pollutants",
            args.pollutants,
            "--horizons",
            args.horizons,
            "--device",
            args.device,
        ]
        if args.lstm_max_train_samples is not None:
            lstm_cmd.extend(["--max-train-samples", str(args.lstm_max_train_samples)])
        if args.lstm_max_val_samples is not None:
            lstm_cmd.extend(["--max-val-samples", str(args.lstm_max_val_samples)])
        if args.lstm_max_test_samples is not None:
            lstm_cmd.extend(["--max-test-samples", str(args.lstm_max_test_samples)])
        if args.lstm_seq_len_map.strip():
            lstm_cmd.extend(["--seq-len-map", args.lstm_seq_len_map])
        run_cmd(uv_run(lstm_cmd))

        run_cmd(
            uv_run(
                [
                    "scripts/train_xgb.py",
                    "--pollutants",
                    args.pollutants,
                    "--horizons",
                    args.horizons,
                    "--device",
                    args.device,
                ]
            )
        )

    if args.fair_bench and run_train and not run_evaluate:
        run_cmd(
            uv_run(
                [
                    "scripts/evaluate.py",
                    "--pollutants",
                    args.pollutants,
                    "--horizons",
                    args.horizons,
                    "--device",
                    args.device,
                    "--fair-intersection",
                ]
            )
        )

    if run_evaluate:
        evaluate_cmd = [
            "scripts/evaluate.py",
            "--pollutants",
            args.pollutants,
            "--horizons",
            args.horizons,
            "--device",
            args.device,
        ]
        if args.fair_bench:
            evaluate_cmd.append("--fair-intersection")
        run_cmd(uv_run(evaluate_cmd))

    if run_predict:
        predict_cmd = [
            "scripts/predict.py",
            "--pollutants",
            args.pollutants,
            "--region",
            args.region,
            "--horizon",
            str(args.horizon),
            "--device",
            args.device,
            "--output",
            args.predict_output,
        ]
        if args.timestamp:
            predict_cmd.extend(["--timestamp", args.timestamp])
        run_cmd(uv_run(predict_cmd))
        print_prediction_snapshot(ROOT / args.predict_output)
        return

    print_training_snapshot()
    if run_evaluate:
        print_evaluation_snapshot()


if __name__ == "__main__":
    main()
