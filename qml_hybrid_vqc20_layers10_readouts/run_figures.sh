#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m qml_hybrid_vqc20_layers10_readouts.make_figures \
  --output-dir "$ROOT/results/qml_hybrid_vqc20_layers10_readouts" \
  "$@"
