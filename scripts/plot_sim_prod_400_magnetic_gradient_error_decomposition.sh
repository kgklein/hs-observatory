#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOTTER="${ROOT}/.venv/bin/hsobs-plot-magnetic-gradient"
INPUT="${ROOT}/outputs/sim-prod-400_magnetic_gradients.h5"
OUTPUT="${ROOT}/outputs/sim-prod-400_magnetic_gradient_error_decomposition.png"

if [[ -x "${PLOTTER}" ]]; then
  exec "${PLOTTER}" "${INPUT}" --output "${OUTPUT}" \
    --layout error_decomposition "$@"
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m hs_observatory.cli_magnetic_gradient \
  "${INPUT}" --output "${OUTPUT}" --layout error_decomposition "$@"
