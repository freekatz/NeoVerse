#!/usr/bin/env bash
set -euo pipefail

# Online training launcher for large datasets.
#
# This keeps the frozen teacher path online instead of prebuilding all caches.
# It is intended for large-scale runs where full frozen-cache materialization is
# impractical.
#
# Usage examples:
#   bash scripts/train_distill_control_latent_online.sh
#   GPU_LIST=0,1 RUN_NAME=online_v1 bash scripts/train_distill_control_latent_online.sh
#   LAUNCH_MODE=background RUN_NAME=online_v1 bash scripts/train_distill_control_latent_online.sh
#
# Common overrides:
#   MAX_STEPS=50000
#   NUM_WORKERS=4
#   VARIANTS_PER_SCENE=1
#   DATASET_SEED=null          # keep online random sampling
#   FROZEN_CACHE_DIR=/path     # optional opportunistic on-disk cache

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-online}"
export FAST_FROZEN_CACHE="${FAST_FROZEN_CACHE:-0}"
export CONTEXT_SAMPLING_STRATEGY="${CONTEXT_SAMPLING_STRATEGY:-mixed}"
export VARIANTS_PER_SCENE="${VARIANTS_PER_SCENE:-1}"
export PRELOAD_FROZEN_CACHE="${PRELOAD_FROZEN_CACHE:-false}"
export CACHE_FROZEN_OUTPUTS="${CACHE_FROZEN_OUTPUTS:-false}"
export FROZEN_CACHE_WRITE="${FROZEN_CACHE_WRITE:-false}"
export FROZEN_CACHE_READ="${FROZEN_CACHE_READ:-false}"
export DATASET_SEED="${DATASET_SEED:-null}"

exec bash "${SCRIPT_DIR}/train_distill_control_latent.sh" "$@"
