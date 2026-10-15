#!/usr/bin/env bash
# 500k extension: --budget 500000 --p 0.9 --seed 12345 --resume
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing project Python: ${PYTHON}" >&2
  exit 1
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/malara-direct3d-mpl}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd "${ROOT}"
exec "${PYTHON}" "${ROOT}/scripts/malara_direct3d_convergence.py" run "$@"
