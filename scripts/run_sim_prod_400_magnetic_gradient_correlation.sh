#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/analysis/sim-prod-400_magnetic-gradient-correlation.yaml"
ANALYZER="${ROOT}/.venv/bin/hsobs-analyze"

if [[ -x "${ANALYZER}" ]]; then
  exec "${ANALYZER}" "${CONFIG}"
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m hs_observatory.cli_analyze "${CONFIG}"
