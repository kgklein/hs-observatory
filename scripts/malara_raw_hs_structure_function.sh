#!/usr/bin/env bash
# Matched raw-q / centered-q / ideal-z+ studies. Never generates observations.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CASES=(500a 502a)
case "${1:-}" in
  500a|502a) CASES=("$1"); shift ;;
  all) shift ;;
  ''|--*) ;;
  *) echo 'Usage: malara_raw_hs_structure_function.sh [all|500a|502a] [options]' >&2; exit 2 ;;
esac
for ARG in "$@"; do
  case "${ARG}" in
    --config|--config=*|--root|--root=*) echo 'Use the Python module for custom configs/roots.' >&2; exit 2 ;;
  esac
done
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hsobs-raw-hs-mpl}"
PYTHON="${HSOBS_PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then PYTHON=python3; fi
for CASE in "${CASES[@]}"; do
  "${PYTHON}" -m hs_observatory.analysis.malara_raw_hs \
    --config "analysis/malara-raw-hs-${CASE}.yaml" "$@"
done
