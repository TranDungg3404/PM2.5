#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m qml_hybrid_vqc20_layers10_readouts.run \
  --data "$ROOT/combined_data.csv" \
  --output-dir "$ROOT/results/qml_hybrid_vqc20_layers10_readouts" \
  --spsa-steps 6 \
  --max-fit-samples 160 \
  "$@"
