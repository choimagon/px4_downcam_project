#!/usr/bin/env bash
# Retrain PPO, DDPG and SAC entirely in the MuJoCo moving-QR world.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec /usr/bin/python3 -m landing_rl.train_mujoco \
  --algorithms ppo,ddpg,sac \
  --timesteps "${MUJOCO_TIMESTEPS:-60000}" \
  --eval-episodes "${MUJOCO_EVAL_EPISODES:-20}" \
  --seed 20260830 \
  --models-dir models \
  --artifacts-dir artifacts/rl_training \
  --model-suffix mujoco_moving_qr \
  --metrics-file mujoco_training_metrics.json
