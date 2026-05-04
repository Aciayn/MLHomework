from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401 - importing registers MiniGrid environments with Gymnasium.
import numpy as np


IMAGE_VALUE_SCALE = 10.0


@dataclass(frozen=True)
class ObservationSpec:
    image_shape: tuple[int, int, int]
    direction_dim: int = 4

    @property
    def image_size(self) -> int:
        return int(np.prod(self.image_shape))

    @property
    def state_dim(self) -> int:
        return self.image_size + self.direction_dim


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    try:
        env = gym.make(env_id, render_mode=render_mode)
    except gym.error.NameNotFound as exc:
        raise gym.error.NameNotFound(
            f"Environment {env_id!r} is not registered. Make sure `minigrid` is installed "
            "in the active Python environment and that the id includes a version suffix, "
            "for example `MiniGrid-Empty-8x8-v0`."
        ) from exc
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


class FlatMiniGridObservation(gym.ObservationWrapper):
    """Flatten MiniGrid dict observations for standard RL baselines such as PPO."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        image_shape = env.observation_space["image"].shape
        flat_dim = int(np.prod(image_shape)) + 4
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        return preprocess_observation(observation)


def make_flat_env(env_id: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    env = make_env(env_id, seed=seed, render_mode=render_mode)
    return FlatMiniGridObservation(env)


def reset_env(env: gym.Env, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    obs, info = env.reset(seed=seed)
    return preprocess_observation(obs), info


def step_env(env: gym.Env, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    obs, reward, terminated, truncated, info = env.step(action)
    return preprocess_observation(obs), float(reward), bool(terminated), bool(truncated), info


def get_observation_spec(env: gym.Env) -> ObservationSpec:
    obs, _ = env.reset()
    image = np.asarray(obs["image"], dtype=np.float32)
    return ObservationSpec(image_shape=tuple(image.shape))


def preprocess_observation(obs: dict[str, Any]) -> np.ndarray:
    """Convert MiniGrid's dict observation into a compact numeric state vector."""
    image = np.asarray(obs["image"], dtype=np.float32).reshape(-1) / IMAGE_VALUE_SCALE
    direction = int(obs.get("direction", 0))
    direction_one_hot = np.zeros(4, dtype=np.float32)
    direction_one_hot[direction] = 1.0
    return np.concatenate([image, direction_one_hot]).astype(np.float32)


def sample_action_space_size(env: gym.Env) -> int:
    if not hasattr(env.action_space, "n"):
        raise ValueError("This project expects a discrete MiniGrid action space.")
    return int(env.action_space.n)
