from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import trange

from world_model_minigrid.baselines import RandomAgent
from world_model_minigrid.envs import make_env, preprocess_observation, sample_action_space_size
from world_model_minigrid.mpc import load_mpc_agent
from world_model_minigrid.utils import ensure_parent, get_device, load_config, set_seed


class Agent(Protocol):
    def act(self, state): ...


def evaluate_agent(env_id: str, agent: Agent, episodes: int, seed: int, max_steps: int) -> dict[str, float]:
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []

    for episode in trange(episodes, desc=f"Evaluating {agent.__class__.__name__}", leave=False):
        env = make_env(env_id, seed=seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        state = preprocess_observation(obs)
        episode_return = 0.0

        for step in range(1, max_steps + 1):
            action = agent.act(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = preprocess_observation(next_obs)
            episode_return += float(reward)
            if terminated or truncated:
                break

        env.close()
        returns.append(episode_return)
        lengths.append(step)
        successes.append(float(episode_return > 0.0))

    return {
        "mean_return": float(pd.Series(returns).mean()),
        "std_return": float(pd.Series(returns).std(ddof=0)),
        "success_rate": float(pd.Series(successes).mean()),
        "mean_length": float(pd.Series(lengths).mean()),
    }


def plot_success_rates(results: pd.DataFrame, output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(results["agent"], results["success_rate"], color="#4c78a8")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Agent")
    ax.set_title("MiniGrid Control Performance")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_from_config(config_path: str = "configs/default.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    env = make_env(cfg["env"]["id"], seed=int(cfg["seed"]))
    action_dim = sample_action_space_size(env)
    env.close()

    episodes = int(cfg["env"]["eval_episodes"])
    max_steps = int(cfg["env"]["max_steps"])
    rows: list[dict[str, float | str | int]] = []

    random_agent = RandomAgent(action_dim=action_dim, seed=int(cfg["seed"]))
    random_metrics = evaluate_agent(cfg["env"]["id"], random_agent, episodes, int(cfg["seed"]), max_steps)
    rows.append({"agent": "Random", "horizon": 0, **random_metrics})

    for horizon in cfg["eval"]["horizons"]:
        mpc_agent = load_mpc_agent(
            cfg["world_model"]["checkpoint"],
            device=torch.device(device),
            horizon=int(horizon),
            candidates=int(cfg["mpc"]["candidates"]),
            gamma=float(cfg["mpc"]["gamma"]),
            seed=int(cfg["seed"]) + int(horizon),
        )
        metrics = evaluate_agent(cfg["env"]["id"], mpc_agent, episodes, int(cfg["seed"]), max_steps)
        rows.append({"agent": f"WM-MPC-H{horizon}", "horizon": int(horizon), **metrics})

    results = pd.DataFrame(rows)
    ensure_parent(cfg["eval"]["output_csv"])
    results.to_csv(cfg["eval"]["output_csv"], index=False)
    plot_success_rates(results, cfg["eval"]["figure_path"])
    print(results)
    print(f"saved evaluation results to {cfg['eval']['output_csv']}")
    print(f"saved figure to {cfg['eval']['figure_path']}")
    return results
