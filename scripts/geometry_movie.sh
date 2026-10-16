#!/usr/bin/env bash
# Usage: geometry_movie.sh CDF_PATH START END [options]
# Example: bash scripts/geometry_movie.sh ../HelioSwarm/HS-RT/PhB_SRD8_0x78f \
#   2030-08-25T00:00:00 2030-08-26T00:00:00 --cadence 2 --format mp4
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo 'Usage: geometry_movie.sh CDF_PATH START END [--cadence HOURS] [--kinds all] [--format gif|mp4] [--output-dir DIR] [--fps 5] [--keep-frames] [--dry-run]' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CDF_PATH="$1"
START="$2"
END="$3"
shift 3
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
PYTHON="${HSOBS_PYTHON:-${ROOT}/.venv/bin/python}"
if [[ -z "${HSOBS_PYTHON:-}" && ! -x "${PYTHON}" ]]; then PYTHON=python3; fi
exec "${PYTHON}" -m hs_observatory.geometry_movie geometry-movie \
  --cdf-path "${CDF_PATH}" --start "${START}" --end "${END}" \
  --cadence 1 --output-dir outputs/geometry_movies "$@"
