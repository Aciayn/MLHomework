#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/server.yaml}"

echo "[1/3] Training and evaluating PPO baseline with ${CONFIG}"
python scripts/run_ppo_baseline.py --config "${CONFIG}"

echo "[2/3] Regenerating SVG figures from CSV results"
python scripts/plot_server_results_stdlib.py

echo "[3/3] PPO result is now appended to runs/server_eval_results.csv"
echo "Please send the updated runs/server_eval_results.csv back if you want the report text rewritten with the PPO result."
