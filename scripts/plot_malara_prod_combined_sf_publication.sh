#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/analysis/malara-prod-combined_sf-publication.yaml"
PLOTTER="${ROOT}/.venv/bin/hsobs-plot-structure-function-overlay"
PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Examples:
#   scripts/plot_malara_prod_combined_sf_publication.sh
#   scripts/plot_malara_prod_combined_sf_publication.sh \
#       --fit-lambda-range 1000 7000 --min-fit-bins 3
#   scripts/plot_malara_prod_combined_sf_publication.sh \
#       --local-slopes --show-cascade-scales
#   Local diagnostics also support --local-slope-method adjacent and
#   --output-prefix PATH (writes primary/focused PNG/PDF and derived HDF5).
#
# Other useful overrides:
#   --fit-range NAME   use a named fit range saved in the analysis JSON
#   --no-p-model       hide the theoretical p-model curves
#   --output PATH      write the combined PNG to a different path

cd "${ROOT}"
if [[ -x "${PLOTTER}" ]]; then
  "${PLOTTER}" "${CONFIG}" "$@"
else
  "${PYTHON}" -m hs_observatory.cli_structure_function_overlay "${CONFIG}" "$@"
fi
