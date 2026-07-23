#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${1:-6}"
ANALYZER="${ROOT}/.venv/bin/hsobs-analyze"
LOG_DIR="${ROOT}/logs/elsasser-increment-publication"
SUMMARY="${LOG_DIR}/summary.log"

if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [positive-job-count]" >&2
  exit 2
fi
if [[ ! -x "${ANALYZER}" ]]; then
  echo "Analyzer not found or not executable: ${ANALYZER}" >&2
  exit 2
fi

mapfile -d "" CONFIGS < <(
  find "${ROOT}/analysis" -maxdepth 1     -name "malara-prod-1*elsasser-increment-pdf-publication.yaml"     -print0 | sort -z
)
if [[ ${#CONFIGS[@]} -ne 10 ]]; then
  echo "Expected 10 analysis configs, found ${#CONFIGS[@]}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
: >"${SUMMARY}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf "Launching %d analyses with up to %d concurrent jobs\n"   "${#CONFIGS[@]}" "${JOBS}" | tee -a "${SUMMARY}"

printf "%s\0" "${CONFIGS[@]}" |
  xargs -0 -n1 -P "${JOBS}" bash -c '
    root="$1"
    log_dir="$2"
    config="$3"
    name="${config##*/}"
    name="${name%.yaml}"
    log="${log_dir}/${name}.log"
    status_file="${log_dir}/${name}.status"

    if "${root}/.venv/bin/hsobs-analyze" "${config}" >"${log}" 2>&1; then
      printf "0\n" >"${status_file}"
      printf "OK   %s\n" "${config}" | tee -a "${log_dir}/summary.log"
    else
      status=$?
      printf "%d\n" "${status}" >"${status_file}"
      printf "FAIL %s (exit %d)\n" "${config}" "${status}" |
        tee -a "${log_dir}/summary.log"
      exit "${status}"
    fi
  ' _ "${ROOT}" "${LOG_DIR}"
batch_status=$?

if [[ ${batch_status} -eq 0 ]]; then
  echo "All analyses completed successfully." | tee -a "${SUMMARY}"
else
  echo "One or more analyses failed (batch exit ${batch_status})." |
    tee -a "${SUMMARY}"
fi
exit "${batch_status}"
