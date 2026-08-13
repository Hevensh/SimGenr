#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source C:/ProgramData/miniconda3/etc/profile.d/conda.sh
conda activate myEnv

python scripts/train_physics_gst.py \
  --config configs/models/attn_gnn_supervised_24to24.yaml

python scripts/train_physics_gst.py \
  --config configs/models/attn_gnn_physics_24to24.yaml
