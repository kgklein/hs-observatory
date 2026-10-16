#!/usr/bin/env bash
# Truth-only pilots. Never invokes hsobs-analyze or a publication plot command.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sf-stage2-mpl}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
if [[ $# -eq 0 ]]; then
  set -- 400 402
fi
for run in "$@"; do
  case "${run}" in
    400|401|402) ;;
    *) echo "Usage: bash scripts/structure_function_stage2.sh [400|401|402 ...]" >&2; exit 2 ;;
  esac
  "${ROOT}/.venv/bin/python" -m hs_observatory.analysis.structure_function_convergence \
    --config "analysis/malara-prod-${run}_sf-stage2-pilot.yaml"
done
