#!/usr/bin/env bash
# Capture the full visible monitor while Gazebo shows the third-person world view.
set -euo pipefail

usage() {
  echo "Usage: $0 --output path/to/video.mp4 [--live-dual] [--start-time-file path]" >&2
}

OUTPUT=""
START_TIME_FILE=""
LIVE_DUAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      OUTPUT=$2
      shift 2
      ;;
    --start-time-file)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      START_TIME_FILE=$2
      shift 2
      ;;
    --live-dual)
      LIVE_DUAL=1
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
[[ -n "$OUTPUT" ]] || { usage; exit 2; }

command -v xdotool >/dev/null 2>&1 || { echo "xdotool is required to locate the Gazebo window." >&2; exit 1; }
command -v xdpyinfo >/dev/null 2>&1 || { echo "xdpyinfo is required to determine monitor dimensions." >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required to record the monitor." >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo "DISPLAY is not set; cannot capture the monitor." >&2; exit 1; }

find_gazebo_window() {
  local window_id title window_class
  while IFS= read -r window_id; do
    title=$(xdotool getwindowname "$window_id" 2>/dev/null || true)
    window_class=$(xdotool getwindowclassname "$window_id" 2>/dev/null || true)
    case "$title" in
      "Gazebo Sim"*|"gz sim "*)
        printf '%s\n' "$window_id"
        return 0
        ;;
    esac
    case "$window_class" in
      *gz-gui*|*gazebo*)
        printf '%s\n' "$window_id"
        return 0
        ;;
    esac
  done < <(xdotool search --onlyvisible --name . 2>/dev/null || true)
  return 1
}

find_down_view_window() {
  local window_id title window_class
  while IFS= read -r window_id; do
    title=$(xdotool getwindowname "$window_id" 2>/dev/null || true)
    window_class=$(xdotool getwindowclassname "$window_id" 2>/dev/null || true)
    case "$title" in
      *rqt*|*Rqt*|*Image\ View*) printf '%s\n' "$window_id"; return 0 ;;
    esac
    case "$window_class" in
      *rqt*) printf '%s\n' "$window_id"; return 0 ;;
    esac
  done < <(xdotool search --onlyvisible --name . 2>/dev/null || true)
  return 1
}

WINDOW_ID=""
WINDOW_GEOMETRY=""
for _ in $(seq 1 35); do
  candidate_id=$(find_gazebo_window || true)
  if [[ -n "$candidate_id" ]]; then
    candidate_geometry=$(xdotool getwindowgeometry --shell "$candidate_id" 2>/dev/null || true)
    if [[ "$candidate_geometry" == *$'WIDTH='* && "$candidate_geometry" == *$'HEIGHT='* ]]; then
      WINDOW_ID=$candidate_id
      WINDOW_GEOMETRY=$candidate_geometry
      break
    fi
  fi
  sleep 1
done
[[ -n "$WINDOW_ID" && -n "$WINDOW_GEOMETRY" ]] || { echo "Timed out waiting for a usable visible Gazebo window." >&2; exit 1; }

SCREEN_DIMENSIONS=$(xdpyinfo | awk '/dimensions:/{print $2; exit}')
IFS=x read -r SCREEN_WIDTH SCREEN_HEIGHT <<< "$SCREEN_DIMENSIONS"
[[ "$SCREEN_WIDTH" =~ ^[0-9]+$ && "$SCREEN_HEIGHT" =~ ^[0-9]+$ ]] || { echo "Could not determine monitor dimensions." >&2; exit 1; }
# H.264 / yuv420p requires even dimensions.
SCREEN_WIDTH=$((SCREEN_WIDTH / 2 * 2))
SCREEN_HEIGHT=$((SCREEN_HEIGHT / 2 * 2))

if [[ "$LIVE_DUAL" -eq 1 ]]; then
  DOWN_VIEW_ID=""
  for _ in $(seq 1 35); do
    DOWN_VIEW_ID=$(find_down_view_window || true)
    [[ -n "$DOWN_VIEW_ID" ]] && break
    sleep 1
  done
  [[ -n "$DOWN_VIEW_ID" ]] || { echo "Timed out waiting for rqt_image_view down-camera window." >&2; exit 1; }
  LEFT_WIDTH=$((SCREEN_WIDTH * 2 / 3))
  RIGHT_WIDTH=$((SCREEN_WIDTH - LEFT_WIDTH))
  # This is a real desktop layout, captured once by x11grab. No MP4 streams
  # are stitched together afterward, so third-person and down-camera frames
  # are inherently simultaneous.
  xdotool windowactivate --sync "$WINDOW_ID" 2>/dev/null || true
  xdotool windowsize --sync "$WINDOW_ID" "$LEFT_WIDTH" "$SCREEN_HEIGHT" 2>/dev/null || true
  xdotool windowmove "$WINDOW_ID" 0 0 2>/dev/null || true
  xdotool windowraise "$WINDOW_ID" 2>/dev/null || true
  xdotool windowsize --sync "$DOWN_VIEW_ID" "$RIGHT_WIDTH" "$SCREEN_HEIGHT" 2>/dev/null || true
  xdotool windowmove "$DOWN_VIEW_ID" "$LEFT_WIDTH" 0 2>/dev/null || true
  xdotool windowraise "$DOWN_VIEW_ID" 2>/dev/null || true
  SCENE_X=$((LEFT_WIDTH * 40 / 100))
  SCENE_Y=$((SCREEN_HEIGHT * 58 / 100))
else
  # Single-view compatibility mode for older scripts.
  xdotool windowactivate --sync "$WINDOW_ID" 2>/dev/null || true
  xdotool windowraise "$WINDOW_ID" 2>/dev/null || true
  xdotool key --window "$WINDOW_ID" alt+F10 2>/dev/null || true
  xdotool windowmaximize "$WINDOW_ID" 2>/dev/null || true
  sleep 1
  SCENE_X=$((SCREEN_WIDTH * 32 / 100))
  SCENE_Y=$((SCREEN_HEIGHT * 58 / 100))
fi
xdotool mousemove --sync "$SCENE_X" "$SCENE_Y" 2>/dev/null || true
# Make the aircraft readable in the third-person half of the monitor.  The
# inner 2 m ring still provides spatial context, while the outer ring remains
# partially visible during the approach.
# Gazebo's orbit camera uses wheel-down to move closer to the scene.
xdotool click --repeat 1 --delay 45 40 5 2>/dev/null || true

mkdir -p "$(dirname "$OUTPUT")"
if [[ -n "$START_TIME_FILE" ]]; then
  mkdir -p "$(dirname "$START_TIME_FILE")"
  # Wall time is shared with the inference process and allows the composer to
  # trim the monitor recording's lead-in before stacking both views.
  date +%s.%N > "$START_TIME_FILE"
fi
echo "Recording full monitor ${SCREEN_WIDTH}x${SCREEN_HEIGHT}+0,0 with Gazebo third-person view to $OUTPUT"
exec ffmpeg -y -loglevel warning \
  -f x11grab -draw_mouse 0 -framerate 30 \
  -video_size "${SCREEN_WIDTH}x${SCREEN_HEIGHT}" -i "${DISPLAY}+0,0" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p -movflags +faststart "$OUTPUT"
