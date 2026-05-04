# Learning to Imagine: MiniGrid World Model Planning

这是我的机器学习课程项目😁：在 MiniGrid 环境中训练轻量级世界模型，并用 shooting-style MPC 在模型内部进行短期想象规划。项目重点展示环境交互数据采集、世界模型学习、规划控制、消融实验。

## 研究问题

给定 MiniGrid 中的转移数据，智能体是否可以学习一个近似环境动力学的模型，并利用该模型预测未来状态和奖励，从而在稀疏奖励任务上获得比随机策略更好的表现？

## 服务器环境

推荐使用 Python 3.10/3.11 与 PyTorch 2.x。强 GPU 服务器可以直接跑多组 seed 和 MPC horizon；小规模 smoke test 也可在 CPU 上完成。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果服务器使用 Conda：

```bash
conda create -n wm-minigrid python=3.10 -y
conda activate wm-minigrid
pip install -r requirements.txt
```

## 运行流程

服务器完整实验推荐使用服务器配置：

```bash
bash scripts/run_server_experiment.sh
```

这会依次完成数据采集、世界模型训练、训练曲线绘制、MPC 评估、轨迹渲染和报告结果表格回填。默认使用 `configs/server.yaml`，输出文件带有 `server_` 前缀，避免覆盖本地小实验结果。

如果已有世界模型实验结果，只想补充无模型强化学习对比，可以单独运行 PPO baseline：

```bash
bash scripts/run_server_extra_ppo.sh
```

该脚本会训练 PPO、评估 PPO，并把 `PPO` 结果追加到 `runs/server_eval_results.csv`。随后会重新生成 `reports/figures/server_success_rate.svg`，用于报告中的对比图。

如果要补充三种策略的轨迹可视化：

```bash
bash scripts/render_policy_comparison.sh
```

这会生成 Random、WM-MPC-H7 和 PPO 三条 rollout 轨迹。

如果要检查结果是否依赖单一随机种子：

```bash
python scripts/run_multiseed_experiment.py --base-config configs/server.yaml --seeds 0 1 2
```

该脚本会对每个 seed 重新采集数据、训练世界模型并评估 Random 与 WM-MPC。若服务器时间有限，可以先用较小规模试跑：

```bash
python scripts/run_multiseed_experiment.py --base-config configs/server.yaml --seeds 0 1 2 --steps 50000 --epochs 30
```

1. 采集 MiniGrid transition 数据：

```bash
python scripts/run_collect.py --config configs/default.yaml
```

2. 训练世界模型：

```bash
python scripts/run_train_world_model.py --config configs/default.yaml
```

3. 画训练曲线：

```bash
python scripts/plot_training.py
```

4. 评估 Random 与 World Model + MPC：

```bash
python scripts/run_evaluate.py --config configs/default.yaml
```

5. 渲染一条 rollout 轨迹：

```bash
python scripts/render_rollout.py --config configs/default.yaml --agent mpc
```

## 输出文件

- `data/transitions_empty8x8.npz`：采集到的转移数据。
- `data/transitions_empty8x8_server.npz`：服务器完整实验采集到的转移数据。
- `runs/world_model.pt`：世界模型 checkpoint。
- `runs/server_world_model.pt`：服务器完整实验 checkpoint。
- `runs/server_ppo.zip`：PPO baseline 模型。
- `runs/world_model.history.csv`：训练与验证损失。
- `runs/eval_results.csv`：各策略成功率、回报、步数。
- `runs/server_ppo_eval.csv`：PPO 单独评估结果。
- `runs/multiseed_raw.csv`：多随机种子完整结果，每行对应一个 seed 和一个 agent。
- `runs/multiseed_summary.csv`：多随机种子均值和标准差汇总。
- `reports/figures/world_model_loss.png`：训练曲线。
- `reports/figures/success_rate.png`：控制表现柱状图。
- `reports/figures/trajectory.png`：策略 rollout 可视化。
- `reports/figures/server_world_model_loss.png`、`reports/figures/server_success_rate.png`、`reports/figures/server_trajectory.png`：服务器实验图表。
- `reports/figures/server_trajectory_random.png`、`reports/figures/server_trajectory_wm_mpc_h7.png`、`reports/figures/server_trajectory_ppo.png`：三种策略轨迹可视化。
- `reports/paper.md`：论文式实验报告。

## 结果整理

实验完成后，可重点检查以下文件：

- `runs/server_eval_results.csv`：主实验结果。
- `runs/multiseed_raw.csv`：多随机种子原始结果。
- `runs/multiseed_summary.csv`：多随机种子汇总结果。
- `reports/figures/server_trajectory_random.png`：Random 轨迹图。
- `reports/figures/server_trajectory_wm_mpc_h7.png`：WM-MPC-H7 轨迹图。
- `reports/figures/server_trajectory_ppo.png`：PPO 轨迹图。

## 最小实验配置

默认使用 `MiniGrid-Empty-8x8-v0`，因为它足够轻量，便于一周内完成完整闭环。若需要扩展，可以改成 DoorKey 或 FourRooms，但这些任务更依赖探索，通常需要更强的数据采集策略或 model-free baseline。
