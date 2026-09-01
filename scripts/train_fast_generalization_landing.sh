#!/usr/bin/env bash
# Retrain all policies on fast, short terminal landings and score them only on
# disjoint easy/medium/hard moving-QR trajectories.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec /usr/bin/python3 -m landing_rl.train \
  --algorithms ppo,ddpg,sac \
  --timesteps 60000 \
  --eval-episodes 30 \
  --seed 20260828 \
  --models-dir models \
  --artifacts-dir artifacts/rl_training \
  --model-suffix fast_landing_generalization \
  --metrics-file generalization_training_metrics.json
