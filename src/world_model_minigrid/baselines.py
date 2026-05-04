from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, action_dim: int, seed: int = 42) -> None:
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray) -> int:
        return int(self.rng.integers(self.action_dim))
