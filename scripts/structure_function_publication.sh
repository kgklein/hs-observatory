#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOTTER="${ROOT}/.venv/bin/hsobs-plot-structure-function"
INPUT="${ROOT}/outputs/malara-prod-001_sf_publication.json"
OUTPUT="${ROOT}/outputs/malara-prod-001_sf_publication.png"

if [[ -x "${PLOTTER}" ]]; then
  "${PLOTTER}" \
    "${INPUT}" \
    --p-value 0.7 \
    --quantity z_plus \
    --output "${OUTPUT}"
else
  PYTHON="${ROOT}/.venv/bin/python"
  if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
  fi
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  "${PYTHON}" -c \
    'import sys; from hs_observatory.analysis.structure_function import plot_structure_function_summary; plot_structure_function_summary(sys.argv[1], p_value=0.7, output_path=sys.argv[2], quantity="z_plus")' \
    "${INPUT}" "${OUTPUT}"
fi
