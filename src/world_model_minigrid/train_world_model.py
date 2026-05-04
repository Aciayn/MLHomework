from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from world_model_minigrid.data import TransitionDataset, split_dataset
from world_model_minigrid.envs import make_env, sample_action_space_size
from world_model_minigrid.models import WorldModel
from world_model_minigrid.utils import ensure_parent, get_device, load_config, set_seed


def _run_epoch(
    model: WorldModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    reward_loss_weight: float,
    done_loss_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}

    for states, actions, rewards, next_states, dones in tqdm(loader, leave=False):
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        with torch.set_grad_enabled(training):
            loss, metrics = model.loss(
                states,
                actions,
                rewards,
                next_states,
                dones,
                reward_loss_weight=reward_loss_weight,
                done_loss_weight=done_loss_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = states.shape[0]
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        totals["count"] = totals.get("count", 0.0) + batch_size

    count = max(1.0, totals.pop("count", 1.0))
    return {key: value / count for key, value in totals.items()}


def train_from_config(config_path: str = "configs/default.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    dataset = TransitionDataset(cfg["data"]["path"])
    train_set, val_set = split_dataset(dataset, float(cfg["data"]["validation_split"]), int(cfg["seed"]))
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["world_model"]["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["world_model"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    env = make_env(cfg["env"]["id"], seed=int(cfg["seed"]))
    action_dim = sample_action_space_size(env)
    env.close()

    state_dim = int(dataset.states.shape[1])
    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=int(cfg["world_model"]["hidden_dim"]),
        num_layers=int(cfg["world_model"]["num_layers"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["world_model"]["learning_rate"]),
        weight_decay=float(cfg["world_model"]["weight_decay"]),
    )

    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, int(cfg["world_model"]["epochs"]) + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            float(cfg["world_model"]["reward_loss_weight"]),
            float(cfg["world_model"]["done_loss_weight"]),
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            None,
            float(cfg["world_model"]["reward_loss_weight"]),
            float(cfg["world_model"]["done_loss_weight"]),
        )
        row: dict[str, float | int | str] = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

    checkpoint_path = Path(cfg["world_model"]["checkpoint"])
    ensure_parent(checkpoint_path)
    torch.save(
        {
            "model_state": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "config": cfg,
        },
        checkpoint_path,
    )

    history_path = checkpoint_path.with_suffix(".history.csv")
    frame = pd.DataFrame(history)
    frame.to_csv(history_path, index=False)
    print(f"saved checkpoint to {checkpoint_path}")
    print(f"saved training history to {history_path}")
    return frame
