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

CONFIGS="${CONFIGS:-${PROJECT_ROOT}/configs/small_debug.yaml}"
SEEDS="${SEEDS:-42 123}"

for config in ${CONFIGS}; do
  for seed in ${SEEDS}; do
    echo "==> Generating config=${config} seed=${seed}"
    args=("${PROJECT_ROOT}/scripts/generate_static_world.py" "--config" "${config}" "--seed" "${seed}")
    if [[ -n "${OUTPUT_ROOT:-}" ]]; then
      args+=("--output" "${OUTPUT_ROOT}")
    fi
    if [[ "${SKIP_WEATHER_GIF:-0}" == "1" ]]; then
      args+=("--skip-weather-gif")
    fi
    "${PYTHON_BIN}" "${args[@]}"
  done
done
