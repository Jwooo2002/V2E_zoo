#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${1:-configs/experiments/smoke_2x4090_real_mamba.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_2x4090.yaml}"

cd "$ROOT"
if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is required for this launch scaffold. Install it separately or run train.py directly." >&2
  exit 127
fi

cmd=(
  accelerate
  launch
  --config_file "$ACCELERATE_CONFIG"
  scripts/run_small_experiment.py
  --experiment "$EXPERIMENT"
)
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
