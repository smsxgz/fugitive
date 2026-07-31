#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPEN_SPIEL_DIR="${ROOT}/third_party/open_spiel"
BUILD_DIR="${OPEN_SPIEL_DIR}/build"
BUILD_JOBS="${BUILD_JOBS:-8}"

export BUILD_TYPE=Testing
export OPEN_SPIEL_BUILD_WITH_ACPC=OFF
export OPEN_SPIEL_BUILD_WITH_GAMUT=OFF
export OPEN_SPIEL_BUILD_WITH_HANABI=OFF
export OPEN_SPIEL_BUILD_WITH_JULIA=OFF
export OPEN_SPIEL_BUILD_WITH_LIBNOP=OFF
export OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF
export OPEN_SPIEL_BUILD_WITH_ORTOOLS=OFF
export OPEN_SPIEL_BUILD_WITH_PYTHON=ON
export OPEN_SPIEL_BUILD_WITH_ROSHAMBO=OFF
export OPEN_SPIEL_BUILD_WITH_XINXIN=OFF
export OPEN_SPIEL_BUILDING_WHEEL=OFF
export OPEN_SPIEL_ENABLE_JAX=OFF
export OPEN_SPIEL_ENABLE_PYTHON_MISC=OFF
export OPEN_SPIEL_ENABLE_PYTORCH=OFF

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA="$(command -v conda)"
else
  CONDA="${HOME}/anaconda3/bin/conda"
fi

ENV_PREFIX="$("${CONDA}" run -n openspiel python -c 'import sys; print(sys.prefix)')"
CXX="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
if [[ ! -x "${CXX}" ]]; then
  CXX="${ENV_PREFIX}/bin/c++"
fi

"${CONDA}" run -n openspiel cmake \
  -S "${OPEN_SPIEL_DIR}/open_spiel" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DPython3_EXECUTABLE="${ENV_PREFIX}/bin/python" \
  -DCMAKE_CXX_COMPILER="${CXX}"

"${CONDA}" run -n openspiel cmake --build "${BUILD_DIR}" \
  --target fugitive_test fugitive_belief_test fugitive_belief_experiment \
  fugitive_baseline_experiment pyspiel \
  --parallel "${BUILD_JOBS}"
"${CONDA}" run -n openspiel ctest --test-dir "${BUILD_DIR}" \
  --output-on-failure --tests-regex '^fugitive(_belief)?_test$'

PYTHONPATH="${ROOT}/src:${OPEN_SPIEL_DIR}:${BUILD_DIR}/python" \
  "${CONDA}" run -n openspiel python -m pytest -q "${ROOT}/tests"
