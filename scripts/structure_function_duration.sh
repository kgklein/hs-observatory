#!/usr/bin/env bash
# Truth-only duration study. No generation or measurement-error MC dispatch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hsobs-duration-mpl}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "${ROOT}/.venv/bin/python" -m hs_observatory.analysis.structure_function_duration "$@"
