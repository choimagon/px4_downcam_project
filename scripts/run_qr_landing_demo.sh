#!/usr/bin/env bash
# End-to-end SITL demo: launch PX4/Gazebo, detect the QR pad, run the trained
# PPO/DDPG-selected policy, and issue guarded PX4 Offboard landing commands.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$PROJECT_ROOT/models/best_qr_landing.zip"
SIM_PID=""
RECORDER_PID=""
GAZEBO_VIDEO_FILE=""
DRONE_VIDEO_FILE=""
LIVE_DUAL_VIDEO_FILE=""
VIDEO_SYNC_FILE=""
INFERENCE_ARGS=()
SPAWN_SEED=""
MOVING_QR=0
MOVING_QR_SPEED=""
MOVING_QR_HEADING_DEG=""
TRAJECTORY_SEED=""
EVALUATION_DIFFICULTY="medium"
SIM_READY_FILE=""

log() { printf '[qr-demo] %s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires a policy ZIP path." >&2; exit 2; }
      MODEL=$2
      shift 2
      ;;
    --gazebo-video-file)
      [[ $# -ge 2 ]] || { echo "--gazebo-video-file requires an MP4 path." >&2; exit 2; }
      GAZEBO_VIDEO_FILE=$2
      shift 2
      ;;
    --drone-video-file)
      [[ $# -ge 2 ]] || { echo "--drone-video-file requires an MP4 path." >&2; exit 2; }
      DRONE_VIDEO_FILE=$2
      shift 2
      ;;
    --live-dual-video-file)
      [[ $# -ge 2 ]] || { echo "--live-dual-video-file requires an MP4 path." >&2; exit 2; }
      LIVE_DUAL_VIDEO_FILE=$2
      shift 2
      ;;
    --video-sync-file)
      [[ $# -ge 2 ]] || { echo "--video-sync-file requires a path." >&2; exit 2; }
      VIDEO_SYNC_FILE=$2
      shift 2
      ;;
    --spawn-seed)
      [[ $# -ge 2 ]] || { echo "--spawn-seed requires an integer." >&2; exit 2; }
      SPAWN_SEED=$2
      shift 2
      ;;
    --moving-qr)
      MOVING_QR=1
      shift
      ;;
    --moving-qr-speed)
      [[ $# -ge 2 ]] || { echo "--moving-qr-speed requires m/s." >&2; exit 2; }
      MOVING_QR_SPEED=$2
      shift 2
      ;;
    --moving-qr-heading-deg)
      [[ $# -ge 2 ]] || { echo "--moving-qr-heading-deg requires degrees." >&2; exit 2; }
      MOVING_QR_HEADING_DEG=$2
      shift 2
      ;;
    --trajectory-seed)
      [[ $# -ge 2 ]] || { echo "--trajectory-seed requires a non-negative integer." >&2; exit 2; }
      TRAJECTORY_SEED=$2
      shift 2
      ;;
    --evaluation-difficulty)
      [[ $# -ge 2 ]] || { echo "--evaluation-difficulty requires easy, medium, or hard." >&2; exit 2; }
      EVALUATION_DIFFICULTY=$2
      shift 2
      ;;
    *)
      INFERENCE_ARGS+=("$1")
      shift
      ;;
  esac
done

cleanup() {
  log 'Cleaning up demo processes.'
  if [[ -n "$RECORDER_PID" ]] && kill -0 "$RECORDER_PID" 2>/dev/null; then
    kill -INT "$RECORDER_PID" 2>/dev/null || true
    wait "$RECORDER_PID" 2>/dev/null || true
  fi
  "$PROJECT_ROOT/scripts/stop_all.sh" --quiet || true
  [[ -z "$SIM_READY_FILE" ]] || rm -f "$SIM_READY_FILE"
}
trap cleanup EXIT INT TERM

[[ -f "$MODEL" ]] || {
  echo "Missing trained policy: $MODEL" >&2
  echo "Run ./scripts/train_qr_landing.sh before starting the end-to-end demo." >&2
  exit 1
}
if [[ -n "$VIDEO_SYNC_FILE" && ( -z "$GAZEBO_VIDEO_FILE" || -z "$DRONE_VIDEO_FILE" ) ]]; then
  echo "--video-sync-file requires both --gazebo-video-file and --drone-video-file." >&2
  exit 2
fi
[[ "$EVALUATION_DIFFICULTY" =~ ^(easy|medium|hard)$ ]] || { echo "--evaluation-difficulty must be easy, medium, or hard." >&2; exit 2; }
if [[ -n "$LIVE_DUAL_VIDEO_FILE" && -n "$GAZEBO_VIDEO_FILE" ]]; then
  echo "Choose either --live-dual-video-file or --gazebo-video-file, not both." >&2
  exit 2
fi

RUN_ALL_ARGS=()
SIM_READY_FILE="$PROJECT_ROOT/logs/px4_mavlink_ready.$$.${RANDOM}"
rm -f "$SIM_READY_FILE"
RUN_ALL_ARGS+=(--px4-ready-file "$SIM_READY_FILE")
[[ -n "$SPAWN_SEED" ]] && RUN_ALL_ARGS+=(--spawn-seed "$SPAWN_SEED")
[[ "$MOVING_QR" -eq 1 ]] && RUN_ALL_ARGS+=(--moving-qr)
[[ -n "$MOVING_QR_SPEED" ]] && RUN_ALL_ARGS+=(--moving-qr-speed "$MOVING_QR_SPEED")
[[ -n "$MOVING_QR_HEADING_DEG" ]] && RUN_ALL_ARGS+=(--moving-qr-heading-deg "$MOVING_QR_HEADING_DEG")
[[ -n "$TRAJECTORY_SEED" ]] && RUN_ALL_ARGS+=(--trajectory-seed "$TRAJECTORY_SEED")
RUN_ALL_ARGS+=(--evaluation-difficulty "$EVALUATION_DIFFICULTY")
[[ -z "$LIVE_DUAL_VIDEO_FILE" ]] && RUN_ALL_ARGS+=(--no-rqt)
"$PROJECT_ROOT/scripts/run_all.sh" "${RUN_ALL_ARGS[@]}" >> "$PROJECT_ROOT/logs/qr_landing_demo_sim.log" 2>&1 &
SIM_PID=$!
log "Started PX4/Gazebo launcher (pid=$SIM_PID)."

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$PROJECT_ROOT/config/cyclonedds.xml"
for _ in $(seq 1 90); do
  [[ "$(ros2 topic type /down_camera/image_raw 2>/dev/null || true)" == "sensor_msgs/msg/Image" ]] && break
  kill -0 "$SIM_PID" 2>/dev/null || {
    echo "PX4/Gazebo launcher exited before the camera became available." >&2
    exit 1
  }
  sleep 1
done
[[ "$(ros2 topic type /down_camera/image_raw 2>/dev/null || true)" == "sensor_msgs/msg/Image" ]] || {
  echo "Timed out waiting for /down_camera/image_raw." >&2
  exit 1
}
log 'Down-camera ROS stream is live.'

# Do not let the presence of a camera frame race PX4 startup.  `run_all.sh`
# creates this unique per-run marker only after its MAVLink endpoint is live.
for _ in $(seq 1 90); do
  [[ -f "$SIM_READY_FILE" ]] && break
  kill -0 "$SIM_PID" 2>/dev/null || {
    echo "PX4/Gazebo launcher exited before the MAVLink endpoint became available." >&2
    exit 1
  }
  sleep 1
done
[[ -f "$SIM_READY_FILE" ]] || {
  echo "Timed out waiting for the PX4 MAVLink endpoint." >&2
  exit 1
}
log 'PX4 MAVLink endpoint is live.'

# rqt_image_view is launched by the simulator as soon as the ROS stream is
# validated, but its X11 window appears a moment later.  Let the compositor
# finish mapping both visible windows before the full-monitor recorder lays
# them out side by side.
sleep 3

RECORDING_FILE="${LIVE_DUAL_VIDEO_FILE:-$GAZEBO_VIDEO_FILE}"
if [[ -n "$RECORDING_FILE" ]]; then
  log "Starting full-monitor recorder: $RECORDING_FILE"
  : > "$PROJECT_ROOT/logs/gazebo_third_person_recording.log"
  RECORDER_ARGS=(--output "$RECORDING_FILE")
  [[ -n "$LIVE_DUAL_VIDEO_FILE" ]] && RECORDER_ARGS+=(--live-dual)
  if [[ -n "$VIDEO_SYNC_FILE" ]]; then
    THIRD_PERSON_START_FILE="${VIDEO_SYNC_FILE}.third_person_started"
    RECORDER_ARGS+=(--start-time-file "$THIRD_PERSON_START_FILE")
  fi
  "$PROJECT_ROOT/scripts/record_gazebo_third_person.sh" "${RECORDER_ARGS[@]}" \
    > "$PROJECT_ROOT/logs/gazebo_third_person_recording.log" 2>&1 &
  RECORDER_PID=$!
  for _ in $(seq 1 35); do
    rg -q '^Recording full monitor ' "$PROJECT_ROOT/logs/gazebo_third_person_recording.log" && break
    kill -0 "$RECORDER_PID" 2>/dev/null || {
      cat "$PROJECT_ROOT/logs/gazebo_third_person_recording.log" >&2
      exit 1
    }
    sleep 1
  done
  rg -q '^Recording full monitor ' "$PROJECT_ROOT/logs/gazebo_third_person_recording.log" || {
    echo "Timed out waiting for Gazebo third-person recording to start." >&2
    exit 1
  }
  log 'Full-monitor recorder is live.'
fi

SPAWN_INFO_FILE="$PROJECT_ROOT/logs/active_annulus_spawn.env"
[[ -f "$SPAWN_INFO_FILE" ]] || { echo "Missing annulus spawn metadata: $SPAWN_INFO_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$SPAWN_INFO_FILE"
TARGET_MOTION_ARGS=(--target-velocity-x 0 --target-velocity-y 0)
if [[ "${MOVING_QR:-0}" -eq 1 ]]; then
  TARGET_MOTION_ARGS=(--target-velocity-x "${TARGET_VX:-0}" --target-velocity-y "${TARGET_VY:-0}")
  [[ -n "$TRAJECTORY_SEED" ]] && TARGET_MOTION_ARGS+=(--trajectory-seed "$TRAJECTORY_SEED")
  TARGET_MOTION_ARGS+=(--trajectory-difficulty "$EVALUATION_DIFFICULTY")
fi
[[ -n "$DRONE_VIDEO_FILE" ]] && INFERENCE_ARGS+=(--video-file "$DRONE_VIDEO_FILE")
if [[ -n "$VIDEO_SYNC_FILE" ]]; then
  DRONE_VIDEO_START_FILE="${VIDEO_SYNC_FILE}.drone_view_started"
  INFERENCE_ARGS+=(--video-start-time-file "$DRONE_VIDEO_START_FILE")
fi
"$PROJECT_ROOT/scripts/run_qr_landing_inference.sh" --model "$MODEL" --enable-actuation \
  --start-x "$SPAWN_X" --start-y "$SPAWN_Y" \
  "${TARGET_MOTION_ARGS[@]}" "${INFERENCE_ARGS[@]}"
log 'Inference process completed.'

if [[ -n "$VIDEO_SYNC_FILE" && -f "${THIRD_PERSON_START_FILE:-}" && -f "${DRONE_VIDEO_START_FILE:-}" ]]; then
  read -r third_person_started < "$THIRD_PERSON_START_FILE"
  read -r drone_view_started < "$DRONE_VIDEO_START_FILE"
  if [[ "$third_person_started" =~ ^[0-9]+([.][0-9]+)?$ && "$drone_view_started" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    mkdir -p "$(dirname "$VIDEO_SYNC_FILE")"
    printf '%s %s\n' "$third_person_started" "$drone_view_started" > "$VIDEO_SYNC_FILE"
  fi
fi
