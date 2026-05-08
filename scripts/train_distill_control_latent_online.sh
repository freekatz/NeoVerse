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
#   TEMPORAL_AUGMENTATION=true TEMPORAL_TRAJECTORY_PROFILE=forward_pause

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export RUN_NAME="${RUN_NAME:-online}"
export FAST_FROZEN_CACHE="${FAST_FROZEN_CACHE:-0}"
export CONTEXT_SAMPLING_STRATEGY="${CONTEXT_SAMPLING_STRATEGY:-mixed}"
export VARIANTS_PER_SCENE="${VARIANTS_PER_SCENE:-1}"
export PRELOAD_FROZEN_CACHE="${PRELOAD_FROZEN_CACHE:-false}"
export CACHE_FROZEN_OUTPUTS="${CACHE_FROZEN_OUTPUTS:-false}"
if [[ -n "${FROZEN_CACHE_DIR:-}" || "${USE_FROZEN_CACHE:-0}" == "1" ]]; then
  export FROZEN_CACHE_DIR="${FROZEN_CACHE_DIR:-${CODE_DIR}/outputs/NeoVerseControlLatentDistill/frozen_cache}"
  export TRAIN_FROM_FROZEN_CACHE="${TRAIN_FROM_FROZEN_CACHE:-true}"
  export FROZEN_CACHE_SPLIT="${FROZEN_CACHE_SPLIT:-train}"
  export FROZEN_CACHE_WRITE="${FROZEN_CACHE_WRITE:-false}"
  export FROZEN_CACHE_READ="${FROZEN_CACHE_READ:-true}"
  export FROZEN_CACHE_REQUIRED="${FROZEN_CACHE_REQUIRED:-true}"
else
  export TRAIN_FROM_FROZEN_CACHE="${TRAIN_FROM_FROZEN_CACHE:-false}"
  export FROZEN_CACHE_WRITE="${FROZEN_CACHE_WRITE:-false}"
  export FROZEN_CACHE_READ="${FROZEN_CACHE_READ:-false}"
  export FROZEN_CACHE_REQUIRED="${FROZEN_CACHE_REQUIRED:-false}"
fi
export DATASET_SEED="${DATASET_SEED:-null}"

exec bash "${SCRIPT_DIR}/train_distill_control_latent.sh" "$@"
