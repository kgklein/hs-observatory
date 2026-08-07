#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs"
MAX_JOBS=2
CASES=(200 210 220 260 270)
RUN_ID="$(date '+%Y%m%d-%H%M%S')-$$"
RUN_LOG="${LOG_DIR}/grad-run-${RUN_ID}.log"

mkdir -p "${LOG_DIR}"

if [[ -x "${ROOT}/.venv/bin/hsobs-analyze" ]]; then
  ANALYZER="${ROOT}/.venv/bin/hsobs-analyze"
elif command -v hsobs-analyze >/dev/null 2>&1; then
  ANALYZER="$(command -v hsobs-analyze)"
else
  echo "hsobs-analyze was not found in ${ROOT}/.venv/bin or PATH" >&2
  exit 127
fi

announce() {
  local message="$1"
  printf '%s\n' "${message}" | tee -a "${RUN_LOG}"
}

run_case() {
  local case="$1"
  local config="${ROOT}/analysis/sim-prod-${case}_magnetic_gradient.yaml"
  local case_log="${LOG_DIR}/sim-prod-${case}_magnetic-gradient-${RUN_ID}.log"
  local started elapsed status

  started="${SECONDS}"
  announce "[$(date '+%F %T')] START case ${case} (PID ${BASHPID}); log: ${case_log}"

  "${ANALYZER}" "${config}" 2>&1 \
    | sed -u "s/^/[case ${case}] /" \
    | tee -a "${case_log}" "${RUN_LOG}"
  status="${PIPESTATUS[0]}"
  elapsed=$((SECONDS - started))

  if ((status == 0)); then
    announce "[$(date '+%F %T')] DONE  case ${case} (${elapsed}s)"
  else
    announce "[$(date '+%F %T')] FAIL  case ${case} exit=${status} (${elapsed}s)"
  fi
  return "${status}"
}

stop_children() {
  local children=()
  mapfile -t children < <(jobs -pr)
  if ((${#children[@]})); then
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
  announce "[$(date '+%F %T')] INTERRUPTED"
  exit 130
}
trap stop_children INT TERM

announce "[$(date '+%F %T')] Gradient batch starting with ${MAX_JOBS} parallel jobs"
announce "Cases: ${CASES[*]}"
announce "Combined log: ${RUN_LOG}"

failures=0
for case in "${CASES[@]}"; do
  while (($(jobs -pr | wc -l) >= MAX_JOBS)); do
    wait -n || failures=$((failures + 1))
  done
  run_case "${case}" &
done

while (($(jobs -pr | wc -l) > 0)); do
  wait -n || failures=$((failures + 1))
done

if ((failures)); then
  announce "[$(date '+%F %T')] Batch finished with ${failures} failed job(s)"
  exit 1
fi

announce "[$(date '+%F %T')] Batch finished successfully"
