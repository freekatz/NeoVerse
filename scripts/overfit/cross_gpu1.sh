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

GPU_LIST="${GPU_LIST:-1}" \
RUN_NAME="${RUN_NAME:-overfit_cross_gpu1}" \
DATA_ROOT="${DATA_ROOT}" \
VIDEO_IDS="${VIDEO_ID}" \
DATASET_SEED="${DATASET_SEED}" \
ADAPTER_TYPE="${ADAPTER_TYPE:-cross_attention_rope}" \
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
  adapter.type=cross_attention_rope \
  adapter.token_dim=null \
  adapter.num_heads="${CROSS_NUM_HEADS:-8}" \
  adapter.num_blocks="${CROSS_NUM_BLOCKS:-4}" \
  adapter.source_pool_hw="${CROSS_SOURCE_POOL_HW:-[16,16]}" \
  adapter.max_source_tokens="${CROSS_MAX_SOURCE_TOKENS:-32768}" \
  adapter.query_chunk_size="${CROSS_QUERY_CHUNK_SIZE:-4096}" \
  adapter.use_rope=true \
  adapter.use_dit_state=false \
  adapter.use_group_embedding="${CROSS_USE_GROUP_EMBEDDING:-true}" \
  adapter.use_local_grid="${CROSS_USE_LOCAL_GRID:-true}" \
  adapter.use_time_film="${CROSS_USE_TIME_FILM:-true}" \
  adapter.time_position_mode="${CROSS_TIME_POSITION_MODE:-reroped}" \
  adapter.rerope_interval="${CROSS_REROPE_INTERVAL:-4.0}" \
  adapter.post_num_res_blocks="${CROSS_POST_NUM_RES_BLOCKS:-2}" \
  "$@"
