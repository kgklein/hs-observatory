#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sf-moment-convergence-mpl}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd "${ROOT}"
exec "${PYTHON}" -m hs_observatory.analysis.structure_function_moment_convergence "$@"
