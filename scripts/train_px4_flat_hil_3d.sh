#!/usr/bin/env bash
# Train the isolated PX4 HIL landing policies with a 3D velocity action.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TIMESTEPS="${PX4_HIL_3D_TIMESTEPS:-48000}"
EVALUATION_EPISODES="${PX4_HIL_3D_EVAL_EPISODES:-20}"
ARTIFACTS="artifacts/rl_training"
TRAINING_ARTIFACTS="$ARTIFACTS/px4_flat_hil_training"
ONNX_DIR="$ARTIFACTS/px4_flat_hil_onnx"

python3 -m landing_rl.train_go2_qr \
  --algorithms ppo,ddpg,sac \
  --timesteps "$TIMESTEPS" \
  --eval-episodes "$EVALUATION_EPISODES" \
  --seed 20260902 \
  --models-dir models \
  --artifacts-dir "$TRAINING_ARTIFACTS" \
  --model-suffix px4_flat_hil_3d \
  --metrics-file px4_flat_hil_training_metrics.json \
  --locomotion-model models/go2_legged_loco_ppo.zip \
  --full-3d-policy-control

python3 -m landing_rl.export_onnx \
  models/ppo_px4_flat_hil_3d.zip \
  models/ddpg_px4_flat_hil_3d.zip \
  models/sac_px4_flat_hil_3d.zip \
  --output-dir "$ONNX_DIR" \
  --manifest "$ARTIFACTS/px4_flat_hil_onnx_models.json"
