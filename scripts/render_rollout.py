from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world_model_minigrid.baselines import RandomAgent
from world_model_minigrid.envs import make_env, make_flat_env, preprocess_observation, sample_action_space_size
from world_model_minigrid.mpc import load_mpc_agent
from world_model_minigrid.utils import get_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent", choices=["random", "mpc", "ppo"], default="mpc")
    parser.add_argument("--output", default="reports/figures/trajectory.png")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = (
        make_flat_env(cfg["env"]["id"], seed=int(cfg["seed"]), render_mode="rgb_array")
        if args.agent == "ppo"
        else make_env(cfg["env"]["id"], seed=int(cfg["seed"]), render_mode="rgb_array")
    )
    action_dim = sample_action_space_size(env)
    device = get_device(cfg.get("device", "auto"))

    if args.agent == "random":
        agent = RandomAgent(action_dim=action_dim, seed=int(cfg["seed"]))
    elif args.agent == "mpc":
        agent = load_mpc_agent(
            cfg["world_model"]["checkpoint"],
            device=torch.device(device),
            horizon=int(args.horizon or cfg["mpc"]["horizon"]),
            candidates=int(cfg["mpc"]["candidates"]),
            gamma=float(cfg["mpc"]["gamma"]),
            seed=int(cfg["seed"]),
        )
    else:
        from stable_baselines3 import PPO

        agent = PPO.load(cfg["ppo"]["model_path"], device=str(device))

    obs, _ = env.reset(seed=int(cfg["seed"]))
    state = obs if args.agent == "ppo" else preprocess_observation(obs)
    rendered_frames = [env.render()]

    for _ in range(max(1, int(cfg["env"]["max_steps"]))):
        if args.agent == "ppo":
            action, _ = agent.predict(state, deterministic=True)
            action = int(action)
        else:
            action = agent.act(state)
        obs, _, terminated, truncated, _ = env.step(action)
        state = obs if args.agent == "ppo" else preprocess_observation(obs)
        rendered_frames.append(env.render())
        if terminated or truncated or len(rendered_frames) >= args.frames:
            break

    env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cols = len(rendered_frames)
    fig, axes = plt.subplots(1, cols, figsize=(2.4 * cols, 2.4))
    if cols == 1:
        axes = [axes]
    for idx, (ax, frame) in enumerate(zip(axes, rendered_frames)):
        ax.imshow(frame)
        ax.set_title(f"t={idx}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved trajectory visualization to {output}")


if __name__ == "__main__":
    main()
