#!/usr/bin/env bash
set -euo pipefail

# Repo root = .../SCAFL (this script lives in projects/afl_async_base/scripts/)
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
AFL_ROOT="${ROOT}/projects/afl_async_base"
export PYTHONPATH="${AFL_ROOT}:${PYTHONPATH:-}"
cd "${AFL_ROOT}"

CONFIG_DIR="${AFL_ROOT}/configs"

for SEED in 2025 2026 2027; do
  export AFL_SEED="${SEED}"
  for CFG in \
    fedasync_hetero.yaml \
    fedbuff_static_hetero.yaml \
    static_threshold_fedbuff_hetero.yaml \
    scafl_baseline_hetero.yaml
  do
    echo "Running ${CFG} seed=${SEED}"
    AFL_CONFIG="${CONFIG_DIR}/${CFG}" python -m afl.server
  done
done
