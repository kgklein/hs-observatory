#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT}/analysis/malara-prod-p07_increment-pdf-wasserstein-publication.yaml}"
PLOTTER="${ROOT}/.venv/bin/hsobs-plot-increment-pdf-wasserstein"

if [[ -x "${PLOTTER}" ]]; then
  exec "${PLOTTER}" "${CONFIG}"
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m hs_observatory.cli_increment_pdf_wasserstein "${CONFIG}"
