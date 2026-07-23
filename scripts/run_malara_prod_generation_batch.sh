#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_JOBS=4
JOBS="${HSOBS_JOBS:-${DEFAULT_JOBS}}"
DRY_RUN=false
RESUME=false
RUN_DIR=""

usage() {
  cat <<'EOF'
Usage: scripts/run_malara_prod_generation_batch.sh [options]

Run generation/malara-prod-1*.yaml with bounded parallelism.

Options:
  -j, --jobs N       Run at most N configs concurrently (default: 4)
  -n, --dry-run      List the configs and commands without running them
  -r, --resume DIR   Reuse a prior run directory and skip successful configs
      --run-dir DIR  Store logs/status in DIR (must not already exist)
  -h, --help         Show this help

Each config currently requests four internal Malara workers. The default of four
outer jobs therefore uses up to 16 evaluator workers, plus coordinator processes.
Set HSOBS_JOBS=N to change the default without passing --jobs.
EOF
}

while (($#)); do
  case "$1" in
    -j|--jobs)
      if (($# < 2)); then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      JOBS="$2"
      shift 2
      ;;
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -r|--resume)
      if (($# < 2)); then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      RUN_DIR="$2"
      RESUME=true
      shift 2
      ;;
    --run-dir)
      if (($# < 2)); then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
      fi
      RUN_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Job count must be a positive integer; got: ${JOBS}" >&2
  exit 2
fi

mapfile -d "" CONFIGS < <(
  find "${ROOT}/generation" -maxdepth 1 -type f \
    -name "malara-prod-1*.yaml" -print0 | sort -z
)
if ((${#CONFIGS[@]} == 0)); then
  echo "No configs matched generation/malara-prod-1*.yaml" >&2
  exit 2
fi

if GENERATOR="$(command -v hsobs-generate)"; then
  :
elif [[ -x "${ROOT}/.venv/bin/hsobs-generate" ]]; then
  GENERATOR="${ROOT}/.venv/bin/hsobs-generate"
else
  echo "hsobs-generate was not found on PATH or in ${ROOT}/.venv/bin" >&2
  exit 2
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${ROOT}/logs/malara-prod-generation/$(date +%Y%m%d-%H%M%S)"
elif [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT}/${RUN_DIR}"
fi

if [[ "${DRY_RUN}" == true ]]; then
  printf "Would run %d configs with at most %d concurrent jobs.\n" \
    "${#CONFIGS[@]}" "${JOBS}"
  printf "Each command would be logged under: %s\n" "${RUN_DIR}"
  for config in "${CONFIGS[@]}"; do
    printf '%q %q\n' "${GENERATOR}" "${config}"
  done
  exit 0
fi

if [[ "${RESUME}" == true ]]; then
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Resume directory does not exist: ${RUN_DIR}" >&2
    exit 2
  fi
elif [[ -e "${RUN_DIR}" ]]; then
  echo "Run directory already exists; use --resume to reuse it: ${RUN_DIR}" >&2
  exit 2
else
  mkdir -p "${RUN_DIR}"
fi

SUMMARY="${RUN_DIR}/summary.log"
touch "${SUMMARY}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf "Run directory: %s\n" "${RUN_DIR}" | tee -a "${SUMMARY}"
printf "Processing %d configs with up to %d concurrent jobs.\n" \
  "${#CONFIGS[@]}" "${JOBS}" | tee -a "${SUMMARY}"

printf '%s\0' "${CONFIGS[@]}" |
  xargs -0 -n1 -P "${JOBS}" bash -c '
    root="$1"
    generator="$2"
    run_dir="$3"
    resume="$4"
    config="$5"
    name="${config##*/}"
    name="${name%.yaml}"
    log="${run_dir}/${name}.log"
    status_file="${run_dir}/${name}.status"

    if [[ "${resume}" == true && -f "${status_file}" ]] &&
       [[ "$(<"${status_file}")" == 0 ]]; then
      printf "SKIP %s (already successful)\n" "${config}" |
        tee -a "${run_dir}/summary.log"
      exit 0
    fi

    printf "START %s\n" "${config}" | tee -a "${run_dir}/summary.log"
    cd "${root}" || exit 2
    if "${generator}" "${config}" >"${log}" 2>&1; then
      printf "0\n" >"${status_file}"
      printf "OK    %s\n" "${config}" | tee -a "${run_dir}/summary.log"
    else
      status=$?
      printf "%d\n" "${status}" >"${status_file}"
      printf "FAIL  %s (exit %d; log: %s)\n" "${config}" "${status}" "${log}" |
        tee -a "${run_dir}/summary.log"
      exit "${status}"
    fi
  ' _ "${ROOT}" "${GENERATOR}" "${RUN_DIR}" "${RESUME}"
batch_status=$?

if ((batch_status == 0)); then
  echo "All generation jobs completed successfully." | tee -a "${SUMMARY}"
else
  printf "One or more generation jobs failed (batch exit %d).\n" \
    "${batch_status}" | tee -a "${SUMMARY}"
fi
exit "${batch_status}"
