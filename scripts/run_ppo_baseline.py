from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO

from world_model_minigrid.envs import make_flat_env
from world_model_minigrid.utils import ensure_parent, get_device, load_config, set_seed


FIELDNAMES = ["agent", "horizon", "mean_return", "std_return", "success_rate", "mean_length"]


def evaluate_ppo(model: PPO, env_id: str, episodes: int, seed: int, max_steps: int) -> dict[str, float | str | int]:
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []

    for episode in range(episodes):
        env = make_flat_env(env_id, seed=seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0.0

        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_return += float(reward)
            if terminated or truncated:
                break

        env.close()
        returns.append(episode_return)
        lengths.append(step)
        successes.append(float(episode_return > 0.0))

    return {
        "agent": "PPO",
        "horizon": 0,
        "mean_return": statistics.fmean(returns),
        "std_return": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "success_rate": statistics.fmean(successes),
        "mean_length": statistics.fmean(lengths),
    }


def write_single_result(row: dict[str, float | str | int], path: str | Path) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def append_or_replace_eval(row: dict[str, float | str | int], path: str | Path) -> None:
    output = Path(path)
    rows: list[dict[str, str]] = []
    if output.exists():
        with output.open("r", encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("agent") != row["agent"]]

    normalized = {key: str(row[key]) for key in FIELDNAMES}
    rows.append(normalized)
    ensure_parent(output)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    ppo_cfg = cfg.get("ppo", {})
    device = get_device(cfg.get("device", "auto"))

    train_env = make_flat_env(cfg["env"]["id"], seed=int(cfg["seed"]))
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 256)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        seed=int(cfg["seed"]),
        device=str(device),
        verbose=1,
    )

    total_timesteps = int(args.timesteps or ppo_cfg.get("total_timesteps", 200000))
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    ensure_parent(ppo_cfg.get("model_path", "runs/server_ppo.zip"))
    model.save(ppo_cfg.get("model_path", "runs/server_ppo.zip"))
    train_env.close()

    row = evaluate_ppo(
        model=model,
        env_id=cfg["env"]["id"],
        episodes=int(cfg["env"]["eval_episodes"]),
        seed=int(cfg["seed"]),
        max_steps=int(cfg["env"]["max_steps"]),
    )
    write_single_result(row, ppo_cfg.get("output_csv", "runs/server_ppo_eval.csv"))
    if bool(ppo_cfg.get("append_to_eval_csv", True)):
        append_or_replace_eval(row, cfg["eval"]["output_csv"])

    print({key: (round(value, 6) if isinstance(value, float) and math.isfinite(value) else value) for key, value in row.items()})


if __name__ == "__main__":
    main()
