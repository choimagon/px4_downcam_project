#!/usr/bin/env bash
# Export all MuJoCo-trained policies to ONNX, then run their ONNX Runtime
# inference across easy/medium/hard moving-QR scenes.  Every MP4 contains the
# synchronized third-person X500 view and the attached down-camera QR view.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="$PROJECT_ROOT/artifacts/rl_training"
MODEL_DIR="$PROJECT_ROOT/models"
cd "$PROJECT_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$ARTIFACTS" "$MODEL_DIR/onnx"

MODELS=()
for algorithm in ppo ddpg sac; do
  model="$MODEL_DIR/${algorithm}_mujoco_moving_qr.zip"
  [[ -f "$model" ]] || { echo "Missing MuJoCo-trained model: $model" >&2; exit 1; }
  MODELS+=("$model")
done

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 -m landing_rl.export_onnx \
  "${MODELS[@]}" \
  --output-dir "$MODEL_DIR/onnx" \
  --manifest "$ARTIFACTS/mujoco_onnx_models.json"
mkdir -p "$ARTIFACTS/onnx"
for onnx_file in "$MODEL_DIR"/onnx/*_mujoco_moving_qr.onnx; do
  cp "$onnx_file" "$ARTIFACTS/onnx/$(basename "$onnx_file")"
done

declare -A SEEDS=( [ppo]=20260911 [ddpg]=20260913 [sac]=20260917 )
declare -A DIFFICULTY_OFFSETS=( [easy]=100 [medium]=0 [hard]=200 )
for algorithm in ppo ddpg sac; do
  for difficulty in easy medium hard; do
    stem="${algorithm}_mujoco_onnx_${difficulty}_follow"
    seed=$(( SEEDS[$algorithm] + DIFFICULTY_OFFSETS[$difficulty] ))
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 -m landing_rl.mujoco_onnx_inference \
      --onnx-model "$MODEL_DIR/onnx/${algorithm}_mujoco_moving_qr.onnx" \
      --difficulty "$difficulty" --seed "$seed" \
      --video-file "$ARTIFACTS/${stem}.mp4" \
      --snapshot-file "$ARTIFACTS/${stem}.png" \
      --log-file "$ARTIFACTS/${stem}.csv" \
      --fps 10
  done
done

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 "$PROJECT_ROOT/scripts/build_mujoco_dashboard.py"
