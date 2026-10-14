#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOTTER="${ROOT}/.venv/bin/hsobs-plot-structure-function"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  echo "Usage: $0 [OPTIONS] {100|101|102|400|401|402} [...]" >&2
  echo "  100, 400: p=0.9 (orange)" >&2
  echo "  101, 401: p=0.7 (green)" >&2
  echo "  102, 402: p=0.5 (blue)" >&2
  echo "Options:" >&2
  echo "  --stage3                      use refined 40X results (defaults to all three)" >&2
  echo "  --recalculate                 opt-in Stage 3 ensemble analysis, then plot" >&2
  echo "  --diagnostics                 Stage 3 fit-window and local-slope figures" >&2
  echo "  --fit-window NAME             Stage 3 primary or sensitivity (default primary)" >&2
  echo "  --pair-counts                 show the baseline-pair count panel" >&2
  echo "  --fit-lambda-range MIN MAX    refit tsi_n over this lambda range [km]" >&2
  echo "  --min-fit-bins N              require at least N bins in the refit" >&2
  echo "  --no-p-model                  hide the p-model reference curve" >&2
  echo "  --no-kurtosis                 plot only the first two summary panels" >&2
  echo "Batch runs automatically use common y-axis limits." >&2
}

show_pair_counts=false
show_p_model=true
show_kurtosis=true
fit_lambda_range=()
min_fit_bins=()
run_ids=()
stage3=false
recalculate=false
diagnostics=false
fit_window=primary
fit_window_requested=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage3) stage3=true ;;
    --recalculate) recalculate=true ;;
    --diagnostics) diagnostics=true ;;
    --fit-window)
      if [[ $# -lt 2 ]]; then usage; exit 2; fi
      fit_window="$2"
      fit_window_requested=true
      shift
      ;;
    --pair-counts) show_pair_counts=true ;;
    --no-p-model) show_p_model=false ;;
    --no-kurtosis) show_kurtosis=false ;;
    --fit-lambda-range)
      if [[ $# -lt 3 ]]; then
        echo "--fit-lambda-range requires MIN and MAX" >&2
        usage
        exit 2
      fi
      fit_lambda_range=("$2" "$3")
      shift 2
      ;;
    --min-fit-bins)
      if [[ $# -lt 2 ]]; then
        echo "--min-fit-bins requires N" >&2
        usage
        exit 2
      fi
      min_fit_bins=("$2")
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *) run_ids+=("$1") ;;
  esac
  shift
done

if [[ "${stage3}" == true ]]; then
  if [[ "${show_pair_counts}" == true || "${show_p_model}" == false ]]; then
    echo "Stage 3 uses two panels with the analytic p-model reference." >&2
    exit 2
  fi
  stage3_args=(--fit-window "${fit_window}")
  if [[ ${#run_ids[@]} -gt 0 ]]; then stage3_args+=(--cases "${run_ids[@]}"); fi
  if [[ "${recalculate}" == true ]]; then stage3_args+=(--recalculate); fi
  if [[ "${diagnostics}" == true ]]; then stage3_args+=(--diagnostics); fi
  if [[ ${#fit_lambda_range[@]} -gt 0 ]]; then
    stage3_args+=(--fit-lambda-range "${fit_lambda_range[@]}")
  fi
  if [[ ${#min_fit_bins[@]} -gt 0 ]]; then
    stage3_args+=(--min-fit-bins "${min_fit_bins[@]}")
  fi
  cd "${ROOT}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sf-stage3-mpl}"
  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  "${PYTHON}" -m hs_observatory.analysis.structure_function_publication "${stage3_args[@]}"
  exit 0
fi
if [[ "${recalculate}" == true || "${diagnostics}" == true || "${fit_window_requested}" == true ]]; then
  echo "--recalculate, --diagnostics and --fit-window require --stage3." >&2
  exit 2
fi

if [[ ${#run_ids[@]} -eq 0 ]]; then
  usage
  exit 2
fi

batch_inputs=()
batch_p_values=()
for run_id in "${run_ids[@]}"; do
  case "${run_id}" in
    100|400) batch_p_values+=("0.9") ;;
    101|401) batch_p_values+=("0.7") ;;
    102|402) batch_p_values+=("0.5") ;;
    *)
      echo "Unknown run ID: ${run_id}" >&2
      usage
      exit 2
      ;;
  esac
  input="${ROOT}/outputs/malara-prod-${run_id}_sf_publication.json"
  if [[ ! -f "${input}" ]]; then
    echo "Input not found: ${input}" >&2
    exit 1
  fi
  batch_inputs+=("${input}")
done

common_structure_ylim=()
common_exponent_ylim=()
common_kurtosis_ylim=()
if [[ ${#run_ids[@]} -gt 1 ]]; then
  limit_arguments=(
    "${show_p_model}" "${show_kurtosis}"
    "${fit_lambda_range[0]:-}" "${fit_lambda_range[1]:-}" "${min_fit_bins[0]:-}"
  )
  for index in "${!run_ids[@]}"; do
    limit_arguments+=("${batch_inputs[index]}" "${batch_p_values[index]}")
  done
  common_limits_output="$(
    "${PYTHON}" -c \
      'import sys; from hs_observatory.analysis.structure_function import structure_function_summary_ylims; args = sys.argv[1:]; show_p_model = args[0] == "true"; show_kurtosis = args[1] == "true"; fit_range = None if args[2] == "" else (float(args[2]), float(args[3])); min_bins = None if args[4] == "" else int(args[4]); pairs = args[5:]; limits = structure_function_summary_ylims(pairs[0::2], p_values=[float(value) for value in pairs[1::2]], fit_lambda_range_km=fit_range, min_fit_bins=min_bins, show_p_model=show_p_model, show_kurtosis=show_kurtosis); print(" ".join(repr(bound) for panel in limits for bound in panel))' \
      "${limit_arguments[@]}"
  )"
  read -r -a common_limits <<< "${common_limits_output}"
  common_structure_ylim=("${common_limits[0]}" "${common_limits[1]}")
  common_exponent_ylim=("${common_limits[2]}" "${common_limits[3]}")
  if [[ "${show_kurtosis}" == true ]]; then
    common_kurtosis_ylim=("${common_limits[4]}" "${common_limits[5]}")
  fi
  echo "Using common y-axis limits for ${#run_ids[@]} runs."
fi

plot_run() {
  local run_id="$1"
  local p_value="$2"

  local input="${ROOT}/outputs/malara-prod-${run_id}_sf_publication.json"
  local output="${ROOT}/outputs/malara-prod-${run_id}_sf_publication.png"

  local plot_args=(
    "${input}" --p-value "${p_value}" --quantity z_plus --output "${output}"
  )
  if [[ "${show_pair_counts}" == true ]]; then
    plot_args+=(--pair-counts)
  fi
  if [[ "${show_p_model}" == false ]]; then
    plot_args+=(--no-p-model)
  fi
  if [[ "${show_kurtosis}" == false ]]; then
    plot_args+=(--no-kurtosis)
  fi
  if [[ ${#fit_lambda_range[@]} -gt 0 ]]; then
    plot_args+=(--fit-lambda-range "${fit_lambda_range[@]}")
  fi
  if [[ ${#min_fit_bins[@]} -gt 0 ]]; then
    plot_args+=(--min-fit-bins "${min_fit_bins[@]}")
  fi
  if [[ ${#common_structure_ylim[@]} -gt 0 ]]; then
    plot_args+=(--structure-ylim "${common_structure_ylim[@]}")
    plot_args+=(--exponent-ylim "${common_exponent_ylim[@]}")
  fi
  if [[ ${#common_kurtosis_ylim[@]} -gt 0 ]]; then
    plot_args+=(--kurtosis-ylim "${common_kurtosis_ylim[@]}")
  fi

  if [[ -x "${PLOTTER}" ]]; then
    "${PLOTTER}" "${plot_args[@]}"
  else
    "${PYTHON}" -c \
      'import sys; from hs_observatory.cli import plot_structure_function_summary_main; raise SystemExit(plot_structure_function_summary_main(sys.argv[1:]))' \
      "${plot_args[@]}"
  fi
}

for index in "${!run_ids[@]}"; do
  plot_run "${run_ids[index]}" "${batch_p_values[index]}"
done
