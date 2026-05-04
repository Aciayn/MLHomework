#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/server.yaml}"
HISTORY="${2:-runs/server_world_model.history.csv}"

echo "[1/6] Collecting transitions with ${CONFIG}"
python scripts/run_collect.py --config "${CONFIG}"

echo "[2/6] Training world model"
python scripts/run_train_world_model.py --config "${CONFIG}"

echo "[3/6] Plotting training curve"
python scripts/plot_training.py \
  --history "${HISTORY}" \
  --output reports/figures/server_world_model_loss.png

echo "[4/6] Evaluating baselines and MPC"
python scripts/run_evaluate.py --config "${CONFIG}"

echo "[5/6] Rendering MPC rollout"
python scripts/render_rollout.py \
  --config "${CONFIG}" \
  --agent mpc \
  --output reports/figures/server_trajectory.png

echo "[6/6] Generating SVG figures from CSV results"
python scripts/plot_server_results_stdlib.py

echo "Done. Check runs/ and reports/figures/ for outputs."
