from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from tqdm import trange

from world_model_minigrid.envs import make_env, preprocess_observation
from world_model_minigrid.utils import ensure_parent


class TransitionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path):
        data = np.load(path)
        self.states = torch.from_numpy(data["states"]).float()
        self.actions = torch.from_numpy(data["actions"]).long()
        self.rewards = torch.from_numpy(data["rewards"]).float().unsqueeze(-1)
        self.next_states = torch.from_numpy(data["next_states"]).float()
        self.dones = torch.from_numpy(data["dones"]).float().unsqueeze(-1)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


def split_dataset(dataset: TransitionDataset, validation_split: float, seed: int):
    val_size = max(1, int(len(dataset) * validation_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def collect_transitions(
    env_id: str,
    num_steps: int,
    output_path: str | Path,
    seed: int = 42,
    policy: str = "random",
) -> dict[str, float]:
    if policy != "random":
        raise ValueError("Only the random data-collection policy is supported by this collector.")

    env = make_env(env_id, seed=seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    state = preprocess_observation(obs)

    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[np.ndarray] = []
    dones: list[float] = []

    episode_returns: list[float] = []
    episode_return = 0.0

    for _ in trange(num_steps, desc="Collecting transitions"):
        action = int(rng.integers(env.action_space.n))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = preprocess_observation(next_obs)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(float(reward))
        next_states.append(next_state)
        dones.append(float(done))

        episode_return += float(reward)
        state = next_state

        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
            state = preprocess_observation(obs)

    env.close()

    ensure_parent(output_path)
    np.savez_compressed(
        output_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
    )

    success_rate = float(np.mean(np.asarray(episode_returns) > 0.0)) if episode_returns else 0.0
    return {
        "num_steps": float(num_steps),
        "episodes": float(len(episode_returns)),
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "success_rate": success_rate,
    }
