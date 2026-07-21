#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOTTER="${ROOT}/.venv/bin/hsobs-plot-figure2"
if [[ ! -x "${PLOTTER}" ]]; then
  PLOTTER="hsobs-plot-figure2"
fi

"${PLOTTER}" \
  --malara "${ROOT}/outputs/synthetic_observations_malara_observatory.h5" \
  --simulation "${ROOT}/outputs/synthetic_observations_simulation_sampled_observatory.h5" \
  --output "${ROOT}/figs/figure2_timeseries.png" \
  --input-units repo \
  --velocity-frame plasma \
  --start-time 0 \
  --stop-time 300
