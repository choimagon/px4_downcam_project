#!/usr/bin/env bash
# Validate, export and transactionally publish nine synchronized Go2 QR MP4s.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="${GO2_ARTIFACTS_DIR:-$PROJECT_ROOT/artifacts/rl_training}"
MODEL_DIR="${GO2_MODELS_DIR:-$PROJECT_ROOT/models}"
LOCOMOTION_MODEL="${GO2_LOCOMOTION_MODEL:-$MODEL_DIR/go2_legged_loco_ppo.zip}"
PYTHON_BIN="${GO2_PYTHON_BIN:-/usr/bin/python3}"
cd "$PROJECT_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
command -v flock >/dev/null || { echo "Missing required command: flock" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "Missing model directory: $MODEL_DIR" >&2; exit 1; }
# train_go2_qr.py holds this same lock for its complete train/evaluate/promote
# transaction, so the accepted hashes cannot change halfway through capture.
exec 9>"$MODEL_DIR/.go2_back_qr_model_set.lock"
flock -x 9

MODELS=()
for algorithm in ppo ddpg sac; do
  model="$MODEL_DIR/${algorithm}_go2_back_qr.zip"
  [[ -f "$model" ]] || { echo "Missing Go2 model: $model" >&2; exit 1; }
  MODELS+=("$model")
done
[[ -f "$LOCOMOTION_MODEL" ]] || { echo "Missing legged-loco-derived Go2 model: $LOCOMOTION_MODEL" >&2; exit 1; }

# Nothing under the currently served flat artifact set is touched until all
# nine inference commands and the staged dashboard have passed validation.
GENERATION_DIR="$(mktemp -d "$ARTIFACTS/.go2_suite.XXXXXX")"
cleanup_generation() {
  case "$GENERATION_DIR" in
    "$ARTIFACTS"/.go2_suite.*) rm -rf -- "$GENERATION_DIR" ;;
    *) echo "Refusing to remove unexpected generation directory: $GENERATION_DIR" >&2 ;;
  esac
}
trap cleanup_generation EXIT
mkdir -p "$GENERATION_DIR/onnx_go2"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/go2_suite_transaction.py" snapshot \
  --project-root "$PROJECT_ROOT" \
  --generation-dir "$GENERATION_DIR" \
  --artifacts-dir "$ARTIFACTS" \
  --models-dir "$MODEL_DIR" \
  --locomotion-model "$LOCOMOTION_MODEL"

MODELS=()
for algorithm in ppo ddpg sac; do
  MODELS+=("$GENERATION_DIR/model_inputs/${algorithm}_go2_back_qr.zip")
done
LOCOMOTION_INPUT="$GENERATION_DIR/model_inputs/go2_legged_loco_ppo.zip"

"$PYTHON_BIN" -m landing_rl.export_onnx \
  "${MODELS[@]}" \
  --output-dir "$GENERATION_DIR/onnx_go2" \
  --manifest "$GENERATION_DIR/go2_back_qr_onnx_models.json"

# Fixed reproducibility seeds. Every run must independently return a stable
# landing; one failed run aborts before publication and preserves the old set.
declare -A VIDEO_SEEDS=(
  [ppo_easy]=20261121 [ppo_medium]=20261021 [ppo_hard]=20261219
  [ddpg_easy]=20261123 [ddpg_medium]=20261023 [ddpg_hard]=20261218
  [sac_easy]=20261127 [sac_medium]=20261027 [sac_hard]=20261203
)
for algorithm in ppo ddpg sac; do
  for difficulty in easy medium hard; do
    stem="${algorithm}_go2_back_qr_onnx_${difficulty}_follow"
    seed="${VIDEO_SEEDS[${algorithm}_${difficulty}]}"
    "$PYTHON_BIN" -m landing_rl.go2_onnx_inference \
      --onnx-model "$GENERATION_DIR/onnx_go2/${algorithm}_go2_back_qr.onnx" \
      --locomotion-model "$LOCOMOTION_INPUT" \
      --difficulty "$difficulty" --seed "$seed" \
      --video-file "$GENERATION_DIR/${stem}.mp4" \
      --snapshot-file "$GENERATION_DIR/${stem}.png" \
      --log-file "$GENERATION_DIR/${stem}.csv" \
      --fps 30
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/go2_suite_transaction.py" record-demo \
      --generation-dir "$GENERATION_DIR" \
      --algorithm "$algorithm" \
      --difficulty "$difficulty" \
      --seed "$seed"
  done
done

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/go2_suite_transaction.py" build-dashboard \
  --project-root "$PROJECT_ROOT" \
  --generation-dir "$GENERATION_DIR" \
  --artifacts-dir "$ARTIFACTS" \
  --models-dir "$MODEL_DIR"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/go2_suite_transaction.py" finalize \
  --generation-dir "$GENERATION_DIR"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/go2_suite_transaction.py" publish \
  --generation-dir "$GENERATION_DIR" \
  --artifacts-dir "$ARTIFACTS" \
  --models-dir "$MODEL_DIR"
