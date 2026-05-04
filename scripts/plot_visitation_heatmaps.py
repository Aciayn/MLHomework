from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world_model_minigrid.baselines import RandomAgent
from world_model_minigrid.envs import make_env, make_flat_env, preprocess_observation, sample_action_space_size
from world_model_minigrid.mpc import load_mpc_agent
from world_model_minigrid.utils import get_device, load_config


def get_agent_position(env) -> tuple[int, int]:
    pos = env.unwrapped.agent_pos
    return int(pos[0]), int(pos[1])


def find_goal_positions(env) -> list[tuple[int, int]]:
    goals: list[tuple[int, int]] = []
    grid = env.unwrapped.grid
    for x in range(env.unwrapped.width):
        for y in range(env.unwrapped.height):
            cell = grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == "goal":
                goals.append((x, y))
    return goals


def make_agent(agent_name: str, cfg: dict, action_dim: int, device: torch.device, horizon: int):
    if agent_name == "random":
        return RandomAgent(action_dim=action_dim, seed=int(cfg["seed"]))
    if agent_name == "mpc":
        return load_mpc_agent(
            cfg["world_model"]["checkpoint"],
            device=device,
            horizon=horizon,
            candidates=int(cfg["mpc"]["candidates"]),
            gamma=float(cfg["mpc"]["gamma"]),
            seed=int(cfg["seed"]) + horizon,
        )
    if agent_name == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(cfg["ppo"]["model_path"], device=str(device))
    raise ValueError(f"Unknown agent: {agent_name}")


def collect_visitation(agent_name: str, cfg: dict, episodes: int, horizon: int) -> tuple[np.ndarray, dict[str, float], list[tuple[int, int]]]:
    env_factory = make_flat_env if agent_name == "ppo" else make_env
    probe_env = env_factory(cfg["env"]["id"], seed=int(cfg["seed"]))
    width = int(probe_env.unwrapped.width)
    height = int(probe_env.unwrapped.height)
    action_dim = sample_action_space_size(probe_env)
    goal_positions = find_goal_positions(probe_env)
    probe_env.close()

    device = get_device(cfg.get("device", "auto"))
    agent = make_agent(agent_name, cfg, action_dim, torch.device(device), horizon)
    counts = np.zeros((height, width), dtype=np.float64)
    returns: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []

    for episode in range(episodes):
        env = env_factory(cfg["env"]["id"], seed=int(cfg["seed"]) + episode)
        obs, _ = env.reset(seed=int(cfg["seed"]) + episode)
        state = obs if agent_name == "ppo" else preprocess_observation(obs)
        x, y = get_agent_position(env)
        counts[y, x] += 1.0
        episode_return = 0.0

        for step in range(1, int(cfg["env"]["max_steps"]) + 1):
            if agent_name == "ppo":
                action, _ = agent.predict(state, deterministic=True)
                action = int(action)
            else:
                action = agent.act(state)

            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs if agent_name == "ppo" else preprocess_observation(obs)
            x, y = get_agent_position(env)
            counts[y, x] += 1.0
            episode_return += float(reward)
            if terminated or truncated:
                break

        env.close()
        returns.append(episode_return)
        successes.append(float(episode_return > 0.0))
        lengths.append(step)

    metrics = {
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "total_visits": float(counts.sum()),
    }
    return counts, metrics, goal_positions


def save_counts_csv(counts_by_agent: dict[str, np.ndarray], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "x", "y", "count"])
        for agent, counts in counts_by_agent.items():
            height, width = counts.shape
            for y in range(height):
                for x in range(width):
                    writer.writerow([agent, x, y, int(counts[y, x])])


def save_metrics_csv(metrics_by_agent: dict[str, dict[str, float]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["agent", "mean_return", "success_rate", "mean_length", "total_visits"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agent, metrics in metrics_by_agent.items():
            writer.writerow({"agent": agent, **metrics})


def plot_heatmaps(
    counts_by_agent: dict[str, np.ndarray],
    metrics_by_agent: dict[str, dict[str, float]],
    goal_positions: list[tuple[int, int]],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    vmax = max(float(counts.max()) for counts in counts_by_agent.values())
    fig, axes = plt.subplots(1, len(counts_by_agent), figsize=(5.0 * len(counts_by_agent), 4.2), constrained_layout=True)
    if len(counts_by_agent) == 1:
        axes = [axes]

    titles = {
        "random": "Random",
        "mpc": "WM-MPC-H7",
        "ppo": "PPO",
    }
    for ax, (agent, counts) in zip(axes, counts_by_agent.items()):
        image = ax.imshow(counts, cmap="magma", origin="upper", vmin=0.0, vmax=vmax)
        metrics = metrics_by_agent[agent]
        ax.set_title(
            f"{titles.get(agent, agent)}\n"
            f"success={metrics['success_rate']:.2f}, length={metrics['mean_length']:.1f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks(range(counts.shape[1]))
        ax.set_yticks(range(counts.shape[0]))
        ax.grid(color="white", linewidth=0.4, alpha=0.35)
        for goal_x, goal_y in goal_positions:
            ax.scatter(goal_x, goal_y, marker="*", s=180, c="cyan", edgecolors="black", linewidths=0.8)
    fig.colorbar(image, ax=axes, shrink=0.78, label="Visit count")
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"saved heatmap to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--output", default="reports/figures/state_visitation_heatmaps.png")
    parser.add_argument("--counts-csv", default="runs/state_visitation_counts.csv")
    parser.add_argument("--metrics-csv", default="runs/state_visitation_metrics.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    episodes = int(args.episodes or cfg["env"]["eval_episodes"])
    counts_by_agent: dict[str, np.ndarray] = {}
    metrics_by_agent: dict[str, dict[str, float]] = {}
    goal_positions: list[tuple[int, int]] = []

    for agent_name in ["random", "mpc", "ppo"]:
        counts, metrics, goals = collect_visitation(agent_name, cfg, episodes, args.horizon)
        counts_by_agent[agent_name] = counts
        metrics_by_agent[agent_name] = metrics
        if goals:
            goal_positions = goals

    save_counts_csv(counts_by_agent, args.counts_csv)
    save_metrics_csv(metrics_by_agent, args.metrics_csv)
    plot_heatmaps(counts_by_agent, metrics_by_agent, goal_positions, args.output)


if __name__ == "__main__":
    main()
