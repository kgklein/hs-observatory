#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOTTER="${ROOT}/.venv/bin/hsobs-plot-increment-pdf-wasserstein"

usage() {
  echo "Usage: $0 [--panels-a-c] [CONFIG]" >&2
  echo "  --panels-a-c  plot only the current panels (a) and (c)" >&2
}

CONFIG="${ROOT}/analysis/malara-prod-p07_increment-pdf-wasserstein-publication.yaml"
plot_args=()
config_supplied=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --panels-a-c) plot_args+=("--panels-a-c") ;;
    -h|--help) usage; exit 0 ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ "${config_supplied}" == true ]]; then
        echo "Only one CONFIG may be supplied" >&2
        usage
        exit 2
      fi
      CONFIG="$1"
      config_supplied=true
      ;;
  esac
  shift
done

if [[ -x "${PLOTTER}" ]]; then
  exec "${PLOTTER}" "${CONFIG}" "${plot_args[@]}"
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m hs_observatory.cli_increment_pdf_wasserstein \
  "${CONFIG}" "${plot_args[@]}"
