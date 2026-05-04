from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


METRICS = ["mean_return", "std_return", "success_rate", "mean_length"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def make_seed_config(base: dict[str, Any], seed: int, args: argparse.Namespace) -> Path:
    cfg = dict(base)
    cfg["seed"] = seed
    cfg["data"] = dict(base["data"])
    cfg["world_model"] = dict(base["world_model"])
    cfg["eval"] = dict(base["eval"])
    cfg["mpc"] = dict(base["mpc"])

    if args.steps is not None:
        cfg["data"]["num_steps"] = args.steps
    if args.epochs is not None:
        cfg["world_model"]["epochs"] = args.epochs

    cfg["data"]["path"] = f"data/multiseed/transitions_seed{seed}.npz"
    cfg["world_model"]["checkpoint"] = f"runs/multiseed/world_model_seed{seed}.pt"
    cfg["eval"]["output_csv"] = f"runs/multiseed/eval_seed{seed}.csv"
    cfg["eval"]["figure_path"] = f"reports/figures/multiseed_success_seed{seed}.png"

    path = Path(f"configs/generated/multiseed_seed{seed}.yaml")
    save_yaml(cfg, path)
    return path


def read_eval_rows(path: str | Path, seed: int) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["seed"] = str(seed)
    return rows


def write_raw(rows: list[dict[str, str]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "agent", "horizon", *METRICS]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], path: str | Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = row["agent"]
        grouped.setdefault(key, []).append(row)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["agent", "num_seeds"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agent, agent_rows in grouped.items():
            result: dict[str, str | int | float] = {"agent": agent, "num_seeds": len(agent_rows)}
            for metric in METRICS:
                values = [float(row[metric]) for row in agent_rows]
                result[f"{metric}_mean"] = statistics.fmean(values)
                result[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            writer.writerow(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/server.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=None, help="Optional override for data.num_steps.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for world_model.epochs.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base = load_yaml(args.base_config)
    all_rows: list[dict[str, str]] = []

    for seed in args.seeds:
        config_path = make_seed_config(base, seed, args)
        eval_path = Path(f"runs/multiseed/eval_seed{seed}.csv")
        if not (args.skip_existing and eval_path.exists()):
            run([sys.executable, "scripts/run_collect.py", "--config", str(config_path)])
            run([sys.executable, "scripts/run_train_world_model.py", "--config", str(config_path)])
            run([sys.executable, "scripts/run_evaluate.py", "--config", str(config_path)])
        all_rows.extend(read_eval_rows(eval_path, seed))

    write_raw(all_rows, "runs/multiseed_raw.csv")
    summarize(all_rows, "runs/multiseed_summary.csv")
    print("saved runs/multiseed_raw.csv")
    print("saved runs/multiseed_summary.csv")


if __name__ == "__main__":
    main()
