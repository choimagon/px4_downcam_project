#!/usr/bin/env bash
# Train all three landing policies in the official Unitree Go2 MuJoCo scene.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec /usr/bin/python3 -m landing_rl.train_go2_qr \
  --algorithms ppo,ddpg,sac \
  --timesteps "${GO2_TIMESTEPS:-16000}" \
  --eval-episodes "${GO2_EVAL_EPISODES:-20}" \
  --seed 20260830 \
  --models-dir models \
  --artifacts-dir artifacts/rl_training \
  --model-suffix go2_back_qr \
  --metrics-file go2_back_qr_training_metrics.json \
  --locomotion-model "${GO2_LOCOMOTION_MODEL:-models/go2_legged_loco_ppo.zip}"
