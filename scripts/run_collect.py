from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world_model_minigrid.data import collect_transitions
from world_model_minigrid.utils import load_config, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    steps = int(args.steps or cfg["data"]["num_steps"])
    stats = collect_transitions(
        env_id=cfg["env"]["id"],
        num_steps=steps,
        output_path=cfg["data"]["path"],
        seed=int(cfg["seed"]),
        policy=cfg["data"]["policy"],
    )
    save_json(stats, Path(cfg["data"]["path"]).with_suffix(".stats.json"))
    print(stats)


if __name__ == "__main__":
    main()
