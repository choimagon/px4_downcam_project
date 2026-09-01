#!/usr/bin/env bash
# Train PPO and DDPG policies on the same domain-randomized QR landing task.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec /usr/bin/python3 -m landing_rl.train "$@"
