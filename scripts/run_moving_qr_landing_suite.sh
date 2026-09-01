#!/usr/bin/env bash
# Train-selected policies chase a physically moving QR pad and land on it.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="$PROJECT_ROOT/artifacts/rl_training"
mkdir -p "$ARTIFACTS"

declare -A SEEDS=( [ppo]=17 [ddpg]=19 [sac]=13 )
declare -A TRAJECTORY_SEEDS=( [ppo]=104746 [ddpg]=104748 [sac]=104742 )
STATUS=0
for ALGORITHM in ppo ddpg sac; do
  VIDEO="$ARTIFACTS/${ALGORITHM}_moving_qr_tracking_landing_third_person.mp4"
  CSV="$PROJECT_ROOT/logs/${ALGORITHM}_moving_qr_tracking_landing.csv"
  LOG="$PROJECT_ROOT/logs/${ALGORITHM}_moving_qr_tracking_landing.log"
  : > "$LOG"
  echo "=== ${ALGORITHM^^}: moving QR, annulus seed ${SEEDS[$ALGORITHM]} ===" | tee -a "$LOG"
  if ! timeout --signal=INT --kill-after=15 190 \
    "$PROJECT_ROOT/scripts/run_qr_landing_demo.sh" \
      --model "$PROJECT_ROOT/models/${ALGORITHM}_moving_qr_landing.zip" \
      --spawn-seed "${SEEDS[$ALGORITHM]}" --moving-qr --trajectory-seed "${TRAJECTORY_SEEDS[$ALGORITHM]}" \
      --gazebo-video-file "$VIDEO" --log-file "$CSV" \
      --takeoff-altitude 1.4 --search-speed 0.42 --max-speed 0.42 --descent-rate 0.18 \
      --alignment-gate 0.10 --aligned-frames 12 --motion-start-wait-seconds 42 --post-land-record-seconds 7 \
      2>&1 | tee -a "$LOG"; then
    STATUS=1
    echo "${ALGORITHM^^} moving-target demo did not complete successfully; see $LOG" >&2
  fi
done

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 "$PROJECT_ROOT/scripts/build_moving_qr_landing_dashboard.py"
exit "$STATUS"
