#!/usr/bin/env bash
# Record one reproducible full-monitor third-person landing demo per RL policy.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="$PROJECT_ROOT/artifacts/rl_training"
mkdir -p "$ARTIFACTS"

declare -A SEEDS=( [ppo]=7 [ddpg]=9 [sac]=11 )
STATUS=0
for ALGORITHM in ppo ddpg sac; do
  VIDEO="$ARTIFACTS/${ALGORITHM}_annulus_qr_landing_third_person.mp4"
  CSV="$PROJECT_ROOT/logs/${ALGORITHM}_annulus_qr_landing.csv"
  LOG="$PROJECT_ROOT/logs/${ALGORITHM}_annulus_qr_landing.log"
  : > "$LOG"
  echo "=== ${ALGORITHM^^}: random annulus seed ${SEEDS[$ALGORITHM]} ===" | tee -a "$LOG"
  if ! timeout --signal=INT --kill-after=12 150 \
    "$PROJECT_ROOT/scripts/run_qr_landing_demo.sh" \
      --model "$PROJECT_ROOT/models/${ALGORITHM}_qr_landing.zip" \
      --spawn-seed "${SEEDS[$ALGORITHM]}" \
      --gazebo-video-file "$VIDEO" \
      --log-file "$CSV" \
      --takeoff-altitude 1.4 --search-speed 0.35 --post-land-record-seconds 8 \
      2>&1 | tee -a "$LOG"; then
    STATUS=1
    echo "${ALGORITHM^^} demo did not complete successfully; see $LOG" >&2
  fi
done

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 "$PROJECT_ROOT/scripts/build_annulus_landing_dashboard.py"
exit "$STATUS"
