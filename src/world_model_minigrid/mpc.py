from __future__ import annotations

import numpy as np
import torch

from world_model_minigrid.models import WorldModel


class MPCAgent:
    def __init__(
        self,
        model: WorldModel,
        action_dim: int,
        device: torch.device,
        horizon: int = 5,
        candidates: int = 512,
        gamma: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.model = model.eval()
        self.action_dim = action_dim
        self.device = device
        self.horizon = horizon
        self.candidates = candidates
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        action_sequences = self.rng.integers(
            low=0,
            high=self.action_dim,
            size=(self.candidates, self.horizon),
            dtype=np.int64,
        )
        states = torch.from_numpy(np.repeat(state[None, :], self.candidates, axis=0)).float().to(self.device)
        returns = torch.zeros(self.candidates, 1, device=self.device)
        discounts = torch.ones(self.candidates, 1, device=self.device)

        for t in range(self.horizon):
            actions = torch.from_numpy(action_sequences[:, t]).long().to(self.device)
            states, rewards, done_probs = self.model.predict_step(states, actions)
            returns += discounts * rewards
            discounts = discounts * self.gamma * (1.0 - done_probs)

        best_idx = int(torch.argmax(returns).item())
        return int(action_sequences[best_idx, 0])


def load_mpc_agent(
    checkpoint_path: str,
    device: torch.device,
    horizon: int,
    candidates: int,
    gamma: float,
    seed: int,
) -> MPCAgent:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WorldModel(
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["config"]["world_model"]["hidden_dim"]),
        num_layers=int(checkpoint["config"]["world_model"]["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return MPCAgent(
        model=model,
        action_dim=int(checkpoint["action_dim"]),
        device=device,
        horizon=horizon,
        candidates=candidates,
        gamma=gamma,
        seed=seed,
    )
