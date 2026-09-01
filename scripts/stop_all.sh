#!/usr/bin/env bash
# Stop only processes started by this project. A final, project-specific Gazebo
# check handles the server and GUI that PX4 may detach from its parent process.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="$PROJECT_ROOT/PX4-Autopilot"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/run_all.pids"
TREE_PID_FILE="$LOG_DIR/stop_all.tree_pids"
STOP_LOG="$LOG_DIR/stop.log"
quiet=0
[[ "${1:-}" == '--quiet' ]] && quiet=1

log() {
  [[ "$quiet" -eq 1 ]] && return
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$STOP_LOG"
}

stop_group() {
  local label=$1 pid=$2
  [[ "$pid" =~ ^[0-9]+$ ]] || return
  kill -0 "$pid" 2>/dev/null || return
  log "Stopping $label process group ($pid)."
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

# `ros2 run` starts a short-lived Python wrapper in addition to the real
# executable. Stop recorded process trees so that an orphaned parameter_bridge
# or rqt_image_view cannot survive after its wrapper exits.
stop_tree() {
  local label=$1 pid=$2 child
  [[ "$pid" =~ ^[0-9]+$ ]] || return
  printf '%s\n' "$pid" >> "$TREE_PID_FILE"
  while IFS= read -r child; do
    stop_tree "$label child" "$child"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  stop_group "$label" "$pid"
}

force_stop_tree() {
  local label=$1 pid=$2 child
  [[ "$pid" =~ ^[0-9]+$ ]] || return
  while IFS= read -r child; do
    force_stop_tree "$label child" "$child"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  if kill -0 "$pid" 2>/dev/null; then
    log "Force-stopping unresponsive $label process group ($pid)."
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
}

if [[ -f "$PID_FILE" ]]; then
  : > "$TREE_PID_FILE"
  while IFS='=' read -r label pid; do
    stop_tree "$label" "$pid"
  done < "$PID_FILE"

  for _ in 1 2 3 4 5; do
    sleep 1
    remaining=0
    while IFS= read -r pid; do
      kill -0 "$pid" 2>/dev/null && remaining=1
    done < "$TREE_PID_FILE"
    [[ "$remaining" -eq 0 ]] && break
  done

  while IFS= read -r pid; do
    force_stop_tree 'recorded child' "$pid"
  done < "$TREE_PID_FILE"
  rm -f "$PID_FILE"
  rm -f "$TREE_PID_FILE"
fi

# PX4's Gazebo launcher can detach its GUI/server. Match the unique project
# world path rather than using a broad killall command.
while IFS= read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  log "Stopping detached project Gazebo process ($pid)."
  kill -TERM "$pid" 2>/dev/null || true
done < <(pgrep -f "gz sim .*${PX4_DIR}/Tools/simulation/gz/worlds/aruco(_moving_qr)?[.]sdf" 2>/dev/null || true)

# In a failed launch PX4's `make px4_sitl` wrapper and Gazebo server may be
# re-parented to the user systemd session.  They then have no PID-file entry
# for the next run to clean up.  Match both the exact SITL target *and* this
# project's PX4 directory; never touch another project's simulator.
while IFS= read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  [[ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)" == "$PX4_DIR" ]] || continue
  log "Stopping orphaned project PX4 launcher ($pid)."
  kill -TERM "$pid" 2>/dev/null || true
done < <(pgrep -f '^make px4_sitl gz_x500_mono_cam_down$' 2>/dev/null || true)

# Gazebo can ignore TERM while blocked in its GUI/server loop.  Give the
# scoped processes a short graceful window, then force-stop only still-live
# project world servers and exact orphaned PX4 launchers.
sleep 2
while IFS= read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  kill -0 "$pid" 2>/dev/null || continue
  log "Force-stopping detached project Gazebo process ($pid)."
  kill -KILL "$pid" 2>/dev/null || true
done < <(pgrep -f "gz sim .*${PX4_DIR}/Tools/simulation/gz/worlds/aruco(_moving_qr)?[.]sdf" 2>/dev/null || true)
while IFS= read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  [[ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)" == "$PX4_DIR" ]] || continue
  kill -0 "$pid" 2>/dev/null || continue
  log "Force-stopping orphaned project PX4 launcher ($pid)."
  kill -KILL "$pid" 2>/dev/null || true
done < <(pgrep -f '^make px4_sitl gz_x500_mono_cam_down$' 2>/dev/null || true)

exit 0
