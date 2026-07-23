#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOTTER="${ROOT}/.venv/bin/hsobs-plot-figure2"
if [[ ! -x "${PLOTTER}" ]]; then
  PLOTTER="hsobs-plot-figure2"
fi

"${PLOTTER}" \
  --malara "${ROOT}/outputs/malara-prod-101.h5" \
  --simulation "${ROOT}/outputs/sim-prod-100.h5" \
  --output "${ROOT}/figs/figure2_timeseries.png" \
  --input-units repo \
  --velocity-frame plasma \
  --start-time 0 \
  --stop-time 300
