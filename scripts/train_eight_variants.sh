#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source C:/ProgramData/miniconda3/etc/profile.d/conda.sh
conda activate myEnv

configs=(
  "configs/models/lstm_24to24.yaml"
  "configs/models/lstm_physics_24to24.yaml"
  "configs/models/gru_24to24.yaml"
  "configs/models/gru_physics_24to24.yaml"
  "configs/models/gru_gnn_supervised_24to24.yaml"
  "configs/models/gru_gnn_physics_24to24.yaml"
  "configs/models/attn_gnn_supervised_24to24.yaml"
  "configs/models/attn_gnn_physics_24to24.yaml"
)

names=(
  "LSTM / supervised"
  "LSTM / physics"
  "GRU / supervised"
  "GRU / physics"
  "GRU-GNN / supervised"
  "GRU-GNN / physics"
  "Attention-GNN / supervised"
  "Attention-GNN / physics"
)

for index in "${!configs[@]}"; do
  printf '\n[%d/8] %s\n' "$((index + 1))" "${names[index]}"
  printf 'python scripts/train_physics_gst.py --config %s\n' "${configs[index]}"
  python scripts/train_physics_gst.py --config "${configs[index]}"
done

printf '\nAll eight experiments completed.\n'
