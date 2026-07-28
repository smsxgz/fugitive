#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/integration/OPEN_SPIEL_VERSION"

OPEN_SPIEL_DIR="${ROOT}/third_party/open_spiel"
GAME_LINK="${OPEN_SPIEL_DIR}/open_spiel/games/fugitive"
PATCH_FILE="${ROOT}/integration/open_spiel-v2.0.1.patch"

find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "${CONDA_EXE}"
  elif command -v conda >/dev/null 2>&1; then
    command -v conda
  elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
    printf '%s\n' "${HOME}/anaconda3/bin/conda"
  else
    echo "conda was not found" >&2
    return 1
  fi
}

clone_commit() {
  local url="$1"
  local commit="$2"
  local destination="$3"
  if [[ -d "${destination}/.git" ]]; then
    local actual_commit
    actual_commit="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${commit}" ]]; then
      echo "Dependency commit mismatch in ${destination}: expected ${commit}, got ${actual_commit}" >&2
      return 1
    fi
    return
  fi
  if [[ -e "${destination}" ]]; then
    echo "${destination} exists and is not a Git checkout" >&2
    return 1
  fi
  if [[ "${#commit}" -eq 40 ]]; then
    mkdir -p "${destination}"
    git -C "${destination}" init --quiet
    git -C "${destination}" remote add origin "${url}"
    git -C "${destination}" fetch --quiet --depth 1 origin "${commit}"
    git -C "${destination}" checkout --quiet --detach FETCH_HEAD
  else
    git clone --quiet --filter=blob:none --no-checkout "${url}" "${destination}"
    git -C "${destination}" checkout --quiet --detach "${commit}"
  fi
}

CONDA="$(find_conda)"
if ! "${CONDA}" run -n openspiel python -c 'import sys; print(sys.prefix)' \
  >/dev/null 2>&1; then
  "${CONDA}" env create --file "${ROOT}/environment.yml"
fi
"${CONDA}" run -n openspiel python -m pip install --quiet \
  --requirement "${ROOT}/requirements.txt"

if [[ ! -d "${OPEN_SPIEL_DIR}/.git" ]]; then
  git clone --quiet --branch "${OPEN_SPIEL_VERSION}" --depth 1 \
    "${OPEN_SPIEL_REPOSITORY}" "${OPEN_SPIEL_DIR}"
fi

actual_commit="$(git -C "${OPEN_SPIEL_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${OPEN_SPIEL_COMMIT}" ]]; then
  echo "OpenSpiel commit mismatch: expected ${OPEN_SPIEL_COMMIT}, got ${actual_commit}" >&2
  exit 1
fi

# Mandatory dependencies from OpenSpiel v2.0.1's install script, resolved to
# exact commits. Optional game dependencies stay disabled.
clone_commit https://github.com/pybind/pybind11.git \
  ed5057ded698e305210269dafa57574ecf964483 \
  "${OPEN_SPIEL_DIR}/pybind11"
clone_commit https://github.com/jblespiau/dds.git \
  091ea94358a4016d4fb6069dea5c452cdc98d0bd \
  "${OPEN_SPIEL_DIR}/open_spiel/games/bridge/double_dummy_solver"
clone_commit https://github.com/abseil/abseil-cpp.git \
  d38452e1ee03523a208362186fd42248ff2609f6 \
  "${OPEN_SPIEL_DIR}/open_spiel/abseil-cpp"
clone_commit https://github.com/nlohmann/json.git \
  9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03 \
  "${OPEN_SPIEL_DIR}/open_spiel/json"
clone_commit https://github.com/pybind/pybind11_json.git \
  d0bf434be9d287d73a963ff28745542daf02c08f \
  "${OPEN_SPIEL_DIR}/open_spiel/pybind11_json"
clone_commit https://github.com/pybind/pybind11_abseil.git \
  73992b54e7d40dfada5e5cd998cad60917f0b3d1 \
  "${OPEN_SPIEL_DIR}/open_spiel/pybind11_abseil"

if [[ -e "${GAME_LINK}" && ! -L "${GAME_LINK}" ]]; then
  echo "${GAME_LINK} exists and is not a symlink" >&2
  exit 1
fi
if [[ ! -L "${GAME_LINK}" ]]; then
  ln -s ../../../../cpp/fugitive "${GAME_LINK}"
elif [[ "$(readlink "${GAME_LINK}")" != "../../../../cpp/fugitive" ]]; then
  echo "${GAME_LINK} points to an unexpected target" >&2
  exit 1
fi

if git -C "${OPEN_SPIEL_DIR}" apply --reverse --check "${PATCH_FILE}" \
  >/dev/null 2>&1; then
  : # Patch is already present.
elif git -C "${OPEN_SPIEL_DIR}" apply --check "${PATCH_FILE}"; then
  git -C "${OPEN_SPIEL_DIR}" apply "${PATCH_FILE}"
else
  echo "OpenSpiel integration patch does not apply cleanly" >&2
  exit 1
fi

echo "OpenSpiel ${OPEN_SPIEL_VERSION} is ready at ${OPEN_SPIEL_DIR}"
