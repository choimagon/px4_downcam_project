#!/usr/bin/env bash
# Combine the wide Gazebo third-person capture and the annotated down-camera
# inference capture into one side-by-side H.264 MP4.
set -euo pipefail

THIRD_PERSON=""
DRONE_VIEW=""
OUTPUT=""
SYNC_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --third-person) THIRD_PERSON=${2:?}; shift 2 ;;
    --drone-view) DRONE_VIEW=${2:?}; shift 2 ;;
    --output) OUTPUT=${2:?}; shift 2 ;;
    --sync-file) SYNC_FILE=${2:?}; shift 2 ;;
    *) echo "Usage: $0 --third-person third.mp4 --drone-view drone.mp4 --output combined.mp4 [--sync-file start_times.txt]" >&2; exit 2 ;;
  esac
done

[[ -f "$THIRD_PERSON" && -f "$DRONE_VIEW" && -n "$OUTPUT" ]] || {
  echo "Both input MP4s and --output are required." >&2
  exit 2
}
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required." >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"

THIRD_PERSON_OFFSET=0
if [[ -n "$SYNC_FILE" ]]; then
  [[ -f "$SYNC_FILE" ]] || { echo "Sync metadata is missing: $SYNC_FILE" >&2; exit 2; }
  read -r third_person_started drone_view_started < "$SYNC_FILE"
  [[ "$third_person_started" =~ ^[0-9]+([.][0-9]+)?$ && "$drone_view_started" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "Sync metadata must contain two numeric start times." >&2
    exit 2
  }
  THIRD_PERSON_OFFSET=$(awk -v third="$third_person_started" -v drone="$drone_view_started" 'BEGIN { printf "%.6f", (drone > third ? drone - third : 0) }')
  echo "Synchronizing views: trimming ${THIRD_PERSON_OFFSET}s from the third-person lead-in."
fi

# Preserve both source aspect ratios: the 16:9 third-person view occupies
# 1280×720 within the left pane, while the 4:3 down-camera inference is placed
# in the right pane. Black padding avoids stretching either view.
ffmpeg -y -loglevel warning -ss "$THIRD_PERSON_OFFSET" -i "$THIRD_PERSON" -i "$DRONE_VIEW" \
  -filter_complex "[0:v]setpts=PTS-STARTPTS,fps=30,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:1080:(ow-iw)/2:(oh-ih)/2:black[left];[1:v]setpts=PTS-STARTPTS,fps=30,scale=640:480:force_original_aspect_ratio=decrease,pad=640:1080:(ow-iw)/2:(oh-ih)/2:black[right];[left][right]hstack=inputs=2,fps=30[v]" \
  -map "[v]" -an -shortest -r 30 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart "$OUTPUT"
