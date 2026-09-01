#!/usr/bin/env bash
# Run QR detection and trained-policy inference against an already running run_all.sh session.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$PROJECT_ROOT/config/cyclonedds.xml"
cd "$PROJECT_ROOT"
exec /usr/bin/python3 -m landing_rl.inference "$@"
