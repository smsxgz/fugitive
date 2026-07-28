#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPEN_SPIEL_DIR="${ROOT}/third_party/open_spiel"
BUILD_DIR="${OPEN_SPIEL_DIR}/build"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA="$(command -v conda)"
else
  CONDA="${HOME}/anaconda3/bin/conda"
fi

PYTHONPATH="${ROOT}/src:${OPEN_SPIEL_DIR}:${BUILD_DIR}/python" \
  "${CONDA}" run --no-capture-output -n openspiel \
  python -m fugitive.web "$@"
