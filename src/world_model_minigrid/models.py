from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)

        layers: list[nn.Module] = [nn.Linear(state_dim + hidden_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        action_features = self.action_embedding(actions)
        features = self.backbone(torch.cat([states, action_features], dim=-1))
        return {
            "next_state": self.next_state_head(features),
            "reward": self.reward_head(features),
            "done_logit": self.done_head(features),
        }

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        reward_loss_weight: float = 2.0,
        done_loss_weight: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = self(states, actions)
        state_loss = F.mse_loss(pred["next_state"], next_states)
        reward_loss = F.mse_loss(pred["reward"], rewards)
        done_loss = F.binary_cross_entropy_with_logits(pred["done_logit"], dones)
        total = state_loss + reward_loss_weight * reward_loss + done_loss_weight * done_loss
        metrics = {
            "loss": float(total.detach().cpu()),
            "state_loss": float(state_loss.detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
            "done_loss": float(done_loss.detach().cpu()),
        }
        return total, metrics

    @torch.no_grad()
    def predict_step(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self(states, actions)
        next_states = pred["next_state"].clamp(0.0, 1.0)
        rewards = pred["reward"]
        dones = torch.sigmoid(pred["done_logit"])
        return next_states, rewards, dones
