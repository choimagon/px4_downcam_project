#!/usr/bin/env bash
# Run the held-out easy, medium, and hard trajectories for every newly trained
# policy.  Each recording is a live desktop capture: Gazebo third person +
# down camera in the same frame.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="$PROJECT_ROOT/artifacts/rl_training"
mkdir -p "$ARTIFACTS"

DIFFICULTIES=(easy medium hard)
declare -A SPAWN_SEEDS=(
  [ppo:easy]=127 [ppo:medium]=117 [ppo:hard]=137
  [ddpg:easy]=129 [ddpg:medium]=119 [ddpg:hard]=139
  [sac:easy]=123 [sac:medium]=113 [sac:hard]=133
)
declare -A TRAJECTORY_SEEDS=(
  [ppo:easy]=2113 [ppo:medium]=1913 [ppo:hard]=2313
  [ddpg:easy]=2147 [ddpg:medium]=1947 [ddpg:hard]=2347
  [sac:easy]=2177 [sac:medium]=1977 [sac:hard]=2377
)

usage() {
  cat <<'EOF'
Usage: run_generalization_live_suite.sh [--difficulty easy|medium|hard]...

Without --difficulty, records all nine PPO/DDPG/SAC × easy/medium/hard demos.
EOF
}

if [[ $# -gt 0 ]]; then
  DIFFICULTIES=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --difficulty)
        [[ $# -ge 2 ]] || { echo "--difficulty requires easy, medium, or hard." >&2; exit 2; }
        [[ "$2" =~ ^(easy|medium|hard)$ ]] || { echo "Unknown difficulty: $2" >&2; exit 2; }
        DIFFICULTIES+=("$2")
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
fi

STATUS=0

for ALGORITHM in ppo ddpg sac; do
  for DIFFICULTY in "${DIFFICULTIES[@]}"; do
    KEY="${ALGORITHM}:${DIFFICULTY}"
    VIDEO="$ARTIFACTS/${ALGORITHM}_generalization_${DIFFICULTY}_live_dual.mp4"
    CSV="$PROJECT_ROOT/logs/${ALGORITHM}_generalization_${DIFFICULTY}.csv"
    LOG="$PROJECT_ROOT/logs/${ALGORITHM}_generalization_${DIFFICULTY}.log"
    : > "$LOG"
    echo "=== ${ALGORITHM^^}: held-out ${DIFFICULTY} path / live third-person + down-camera capture ===" | tee -a "$LOG"
    if ! timeout --signal=INT --kill-after=15 160 \
      "$PROJECT_ROOT/scripts/run_qr_landing_demo.sh" \
        --model "$PROJECT_ROOT/models/${ALGORITHM}_fast_landing_generalization.zip" \
        --spawn-seed "${SPAWN_SEEDS[$KEY]}" --moving-qr --trajectory-seed "${TRAJECTORY_SEEDS[$KEY]}" \
        --evaluation-difficulty "$DIFFICULTY" \
        --live-dual-video-file "$VIDEO" --log-file "$CSV" \
        --takeoff-altitude 1.4 --climb-rate 0.70 --search-speed 1.50 --max-speed 0.65 --descent-rate 0.13 \
        --alignment-gate 0.13 --descent-hold-gate 0.22 --landing-commit-margin 0.25 --aligned-frames 14 --motion-start-wait-seconds 10 --post-land-record-seconds 4 \
        2>&1 | tee -a "$LOG"; then
      STATUS=1
      echo "${ALGORITHM^^} ${DIFFICULTY} held-out demonstration did not complete; see $LOG" >&2
    fi
  done
done

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 "$PROJECT_ROOT/scripts/build_generalization_dashboard.py"
exit "$STATUS"
