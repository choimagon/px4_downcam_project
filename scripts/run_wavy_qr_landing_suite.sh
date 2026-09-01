#!/usr/bin/env bash
# Train-selected PPO/DDPG/SAC policies make a fast approach to a seeded curved
# QR path, then use a strict vision gate for a deliberately cautious descent.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="$PROJECT_ROOT/artifacts/rl_training"
mkdir -p "$ARTIFACTS"

declare -A SPAWN_SEEDS=( [ppo]=17 [ddpg]=19 [sac]=13 )
declare -A TRAJECTORY_SEEDS=( [ppo]=913 [ddpg]=947 [sac]=977 )
STATUS=0
for ALGORITHM in ppo ddpg sac; do
  VIDEO="$ARTIFACTS/${ALGORITHM}_wavy_qr_tracking_landing_third_person.mp4"
  DRONE_VIDEO="$ARTIFACTS/${ALGORITHM}_wavy_qr_tracking_landing_drone_view.mp4"
  COMBINED_VIDEO="$ARTIFACTS/${ALGORITHM}_wavy_qr_tracking_landing_dual_view.mp4"
  SYNC_FILE="$ARTIFACTS/${ALGORITHM}_wavy_qr_tracking_landing_sync.txt"
  CSV="$PROJECT_ROOT/logs/${ALGORITHM}_wavy_qr_tracking_landing.csv"
  LOG="$PROJECT_ROOT/logs/${ALGORITHM}_wavy_qr_tracking_landing.log"
  : > "$LOG"
  echo "=== ${ALGORITHM^^}: seeded curved / wavy QR, annulus seed ${SPAWN_SEEDS[$ALGORITHM]}, trajectory seed ${TRAJECTORY_SEEDS[$ALGORITHM]} ===" | tee -a "$LOG"
  # Fast 1.50 m/s coarse approach; after visual acquisition, cap the learned
  # centering command at 0.65 m/s for a stable final descent.
  if ! timeout --signal=INT --kill-after=15 150 \
    "$PROJECT_ROOT/scripts/run_qr_landing_demo.sh" \
      --model "$PROJECT_ROOT/models/${ALGORITHM}_wavy_qr_landing.zip" \
      --spawn-seed "${SPAWN_SEEDS[$ALGORITHM]}" --moving-qr --trajectory-seed "${TRAJECTORY_SEEDS[$ALGORITHM]}" \
      --gazebo-video-file "$VIDEO" --drone-video-file "$DRONE_VIDEO" --video-sync-file "$SYNC_FILE" --log-file "$CSV" \
      --takeoff-altitude 1.4 --climb-rate 0.70 --search-speed 1.50 --max-speed 0.65 --descent-rate 0.13 \
      --alignment-gate 0.13 --aligned-frames 14 --motion-start-wait-seconds 10 --post-land-record-seconds 4 \
      2>&1 | tee -a "$LOG"; then
    STATUS=1
    echo "${ALGORITHM^^} wavy-target demo did not complete successfully; see $LOG" >&2
  fi
  if [[ -s "$VIDEO" && -s "$DRONE_VIDEO" ]]; then
    "$PROJECT_ROOT/scripts/compose_dual_view_video.sh" \
      --third-person "$VIDEO" --drone-view "$DRONE_VIDEO" --output "$COMBINED_VIDEO" \
      --sync-file "$SYNC_FILE" || STATUS=1
  else
    STATUS=1
    echo "${ALGORITHM^^} dual-view inputs are missing." >&2
  fi
done

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 "$PROJECT_ROOT/scripts/build_wavy_qr_landing_dashboard.py"
exit "$STATUS"
