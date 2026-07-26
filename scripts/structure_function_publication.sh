#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOTTER="${ROOT}/.venv/bin/hsobs-plot-structure-function"

usage() {
  echo "Usage: $0 [--pair-counts] {100|101|102} [...]" >&2
  echo "  100: p=0.9 (orange)" >&2
  echo "  101: p=0.7 (green)" >&2
  echo "  102: p=0.5 (blue)" >&2
}

show_pair_counts=false
run_ids=()
for argument in "$@"; do
  case "${argument}" in
    --pair-counts) show_pair_counts=true ;;
    -h|--help) usage; exit 0 ;;
    *) run_ids+=("${argument}") ;;
  esac
done

if [[ ${#run_ids[@]} -eq 0 ]]; then
  usage
  exit 2
fi

plot_run() {
  local run_id="$1"
  local p_value

  case "${run_id}" in
    100) p_value="0.9" ;;
    101) p_value="0.7" ;;
    102) p_value="0.5" ;;
    *)
      echo "Unknown run ID: ${run_id}" >&2
      usage
      return 2
      ;;
  esac

  local input="${ROOT}/outputs/malara-prod-${run_id}_sf_publication.json"
  local output="${ROOT}/outputs/malara-prod-${run_id}_sf_publication.png"

  if [[ ! -f "${input}" ]]; then
    echo "Input not found: ${input}" >&2
    return 1
  fi

  if [[ -x "${PLOTTER}" ]]; then
    local plot_args=(
      "${input}" --p-value "${p_value}" --quantity z_plus --output "${output}"
    )
    if [[ "${show_pair_counts}" == true ]]; then
      plot_args+=(--pair-counts)
    fi
    "${PLOTTER}" "${plot_args[@]}"
  else
    local python="${ROOT}/.venv/bin/python"
    if [[ ! -x "${python}" ]]; then
      python="python3"
    fi
    export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    "${python}" -c \
      'import sys; from hs_observatory.analysis.structure_function import plot_structure_function_summary; plot_structure_function_summary(sys.argv[1], p_value=float(sys.argv[2]), output_path=sys.argv[3], quantity="z_plus", show_pair_counts=sys.argv[4] == "true")' \
      "${input}" "${p_value}" "${output}" "${show_pair_counts}"
  fi
}

for run_id in "${run_ids[@]}"; do
  plot_run "${run_id}"
done
