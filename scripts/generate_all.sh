#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "C:/Users/Lenovo/.conda/envs/myEnv/python.exe" ]]; then
    PYTHON_BIN="C:/Users/Lenovo/.conda/envs/myEnv/python.exe"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ -n "${CONFIGS:-}" ]]; then
  # Preserve the existing whitespace-separated multi-config interface.
  # read -d '' consumes the whole list, including newline-separated entries.
  IFS=$' \t\n' read -r -d '' -a config_paths <<< "${CONFIGS}" || true
else
  # The default absolute path is one argument even when the checkout has spaces.
  config_paths=("${PROJECT_ROOT}/configs/small_debug.yaml")
fi
IFS=$' \t\n' read -r -d '' -a seeds <<< "${SEEDS:-42 123}" || true

# Resolve configured relative paths and output roots against this repository,
# independently of the directory from which the batch script was invoked.
cd "${PROJECT_ROOT}"

for config in "${config_paths[@]}"; do
  for seed in "${seeds[@]}"; do
    echo "==> Generating config=${config} seed=${seed}"
    args=("${PROJECT_ROOT}/scripts/generate_static_world.py" "--config" "${config}" "--seed" "${seed}")
    if [[ -n "${OUTPUT_ROOT:-}" ]]; then
      args+=("--output" "${OUTPUT_ROOT}")
    fi
    if [[ "${SKIP_WEATHER_ANIMATION:-${SKIP_WEATHER_GIF:-0}}" == "1" ]]; then
      args+=("--skip-weather-animation")
    fi
    if [[ "${SKIP_STORAGE_ANIMATION:-${SKIP_STORAGE_GIF:-0}}" == "1" ]]; then
      args+=("--skip-storage-animation")
    fi
    args+=("$@")
    "${PYTHON_BIN}" "${args[@]}"
  done
done
