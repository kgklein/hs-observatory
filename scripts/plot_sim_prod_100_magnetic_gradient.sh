#!/usr/bin/env bash
set -euo pipefail

hsobs-plot-magnetic-gradient \
  outputs/sim-prod-100_magnetic_gradients.h5 \
  --output outputs/sim-prod-100_magnetic_gradients.png
