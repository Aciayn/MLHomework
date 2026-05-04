from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def make_smoke_config() -> str:
    with Path("configs/default.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["path"] = "runs/smoke_transitions.npz"
    cfg["data"]["num_steps"] = 512
    cfg["world_model"]["checkpoint"] = "runs/smoke_world_model.pt"
    cfg["world_model"]["epochs"] = 1
    cfg["world_model"]["batch_size"] = 64
    cfg["mpc"]["candidates"] = 64
    cfg["eval"]["horizons"] = [1]
    cfg["eval"]["output_csv"] = "runs/smoke_eval_results.csv"
    cfg["eval"]["figure_path"] = "reports/figures/smoke_success_rate.png"
    cfg["env"]["eval_episodes"] = 3
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(cfg, handle)
    handle.close()
    return handle.name


def main() -> None:
    config = make_smoke_config()
    commands = [
        [sys.executable, "scripts/run_collect.py", "--config", config],
        [sys.executable, "scripts/run_train_world_model.py", "--config", config],
        [
            sys.executable,
            "scripts/plot_training.py",
            "--history",
            "runs/smoke_world_model.history.csv",
            "--output",
            "reports/figures/smoke_world_model_loss.png",
        ],
        [sys.executable, "scripts/run_evaluate.py", "--config", config],
        [
            sys.executable,
            "scripts/render_rollout.py",
            "--config",
            config,
            "--output",
            "reports/figures/smoke_trajectory.png",
        ],
    ]
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
