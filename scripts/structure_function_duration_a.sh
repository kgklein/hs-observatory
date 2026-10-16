#!/usr/bin/env bash
# Alternate-angle truth studies; each case has separate configuration/results.
# Usage: bash scripts/structure_function_duration_a.sh [all|500a|502a] [study options]
# Examples: --dry-run; --verify-budget; --plot-only; --durations 300 --seed-count 1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CASES=(500a 502a)
case "${1:-}" in
  500a|502a) CASES=("$1"); shift ;;
  all) shift ;;
  -h|--help)
    echo 'Usage: bash scripts/structure_function_duration_a.sh [all|500a|502a] [study options]'
    echo 'Default: both cases, sequentially, with separate caches, reports and figures.'
    echo 'Study options below are forwarded to each case; --dry-run validates without sampling.'
    exec bash "${ROOT}/scripts/structure_function_duration.sh" --help
    ;;
  ''|--*) ;;
  *) echo 'Expected all, 500a, 502a, or a study option such as --dry-run.' >&2; exit 2 ;;
esac
# Prevent accidental redirection of both studies into the same cache/report.
for ARG in "$@"; do
  case "${ARG}" in
    --config|--config=*|--generation-glob|--generation-glob=*|--output-dir|--output-dir=*)
      echo "${ARG} is managed per case; use structure_function_duration.sh directly for custom paths." >&2
      exit 2
      ;;
  esac
done
for RUN in "${CASES[@]}"; do
  echo "Duration convergence: ${RUN}"
  bash "${ROOT}/scripts/structure_function_duration.sh" \
    --config "analysis/malara-ts-${RUN}_sf-duration.yaml" "$@"
done
