#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/server.yaml}"

echo "[1/3] Rendering Random rollout"
python scripts/render_rollout.py \
  --config "${CONFIG}" \
  --agent random \
  --output reports/figures/server_trajectory_random.png

echo "[2/3] Rendering WM-MPC-H7 rollout"
python scripts/render_rollout.py \
  --config "${CONFIG}" \
  --agent mpc \
  --horizon 7 \
  --output reports/figures/server_trajectory_wm_mpc_h7.png

echo "[3/3] Rendering PPO rollout"
python scripts/render_rollout.py \
  --config "${CONFIG}" \
  --agent ppo \
  --output reports/figures/server_trajectory_ppo.png

echo "Done. Trajectories are saved under reports/figures/."
