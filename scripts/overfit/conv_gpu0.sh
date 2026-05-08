#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_DIR}"

DATA_ROOT="${DATA_ROOT:-data/SpatialVID}"
VIDEO_NAME="${VIDEO_NAME:-c93dd173-51dd-54f9-bead-b835a485db24.mp4}"
VIDEO_ID="${VIDEO_ID:-${VIDEO_NAME%.*}}"
NUM_CONTEXT_VIEWS="${NUM_CONTEXT_VIEWS:-16}"
DATASET_SEED="${DATASET_SEED:-1}"
MAX_STEPS_VALUE="${MAX_STEPS:-20000}"
NUM_EPOCHS_VALUE="${NUM_EPOCHS:-${MAX_STEPS_VALUE}}"

GPU_LIST="${GPU_LIST:-0}" \
RUN_NAME="${RUN_NAME:-overfit_conv_gpu0}" \
DATA_ROOT="${DATA_ROOT}" \
VIDEO_IDS="${VIDEO_ID}" \
DATASET_SEED="${DATASET_SEED}" \
ADAPTER_TYPE="${ADAPTER_TYPE:-conv}" \
FIXED_CLIPS_PER_SCENE="${FIXED_CLIPS_PER_SCENE:-1}" \
TRAJECTORIES_PER_CLIP="${TRAJECTORIES_PER_CLIP:-1}" \
CONTEXT_SAMPLING_STRATEGY="${CONTEXT_SAMPLING_STRATEGY:-uniform_first}" \
TEMPORAL_AUGMENTATION="${TEMPORAL_AUGMENTATION:-false}" \
CACHE_TRAIN_BATCH="${CACHE_TRAIN_BATCH:-true}" \
CACHE_FROZEN_OUTPUTS="${CACHE_FROZEN_OUTPUTS:-true}" \
./cli train online-noaug \
  max_steps="${MAX_STEPS_VALUE}" \
  num_epochs="${NUM_EPOCHS_VALUE}" \
  num_workers="${NUM_WORKERS:-0}" \
  batch_size="${BATCH_SIZE:-1}" \
  print_freq="${PRINT_FREQ:-10}" \
  save_freq="${SAVE_FREQ:-1000}" \
  vis_freq="${VIS_FREQ:-0}" \
  min_num_context_views="${NUM_CONTEXT_VIEWS}" \
  max_num_context_views="${NUM_CONTEXT_VIEWS}" \
  pipeline_kwargs.novel_view_sampling_trans="${NOVEL_VIEW_SAMPLING_TRANS:-[0.0,0.0]}" \
  pipeline_kwargs.culling_prob="${CULLING_PROB:-1.0}" \
  pipeline_kwargs.kernel_size_range="${KERNEL_SIZE_RANGE:-[0,0]}" \
  pipeline_kwargs.color_thresh="${COLOR_THRESH:-[50,50]}" \
  pipeline_kwargs.prompt_drop_prob="${PROMPT_DROP_PROB:-0.0}" \
  pipeline_kwargs.mask_drop_prob="${MASK_DROP_PROB:-0.0}" \
  pipeline_kwargs.condition_drop_prob="${CONDITION_DROP_PROB:-0.0}" \
  "$@"
