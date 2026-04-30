#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

DATA_ROOT="${DATA_ROOT:-data/SpatialVID}"
VIDEO_NAME="${VIDEO_NAME:-c93dd173-51dd-54f9-bead-b835a485db24.mp4}"
VIDEO_ID="${VIDEO_ID:-${VIDEO_NAME%.*}}"
NUM_CONTEXT_VIEWS="${NUM_CONTEXT_VIEWS:-16}"
DATASET_SEED="${DATASET_SEED:-1}"
MAX_STEPS_VALUE="${MAX_STEPS:-20000}"
NUM_EPOCHS_VALUE="${NUM_EPOCHS:-${MAX_STEPS_VALUE}}"

GPU_LIST=0 \
RUN_NAME="${RUN_NAME:-overfit_conv_gpu0}" \
bash scripts/train_distill_control_latent.sh \
  max_steps="${MAX_STEPS_VALUE}" \
  num_epochs="${NUM_EPOCHS_VALUE}" \
  num_workers="${NUM_WORKERS:-0}" \
  batch_size="${BATCH_SIZE:-1}" \
  cache_train_batch="${CACHE_TRAIN_BATCH:-true}" \
  cache_frozen_outputs="${CACHE_FROZEN_OUTPUTS:-true}" \
  print_freq="${PRINT_FREQ:-10}" \
  save_freq="${SAVE_FREQ:-1000}" \
  vis_freq="${VIS_FREQ:-1000}" \
  min_num_context_views="${NUM_CONTEXT_VIEWS}" \
  max_num_context_views="${NUM_CONTEXT_VIEWS}" \
  pipeline_kwargs.novel_view_sampling_trans="${NOVEL_VIEW_SAMPLING_TRANS:-[0.0,0.0]}" \
  pipeline_kwargs.culling_prob="${CULLING_PROB:-1.0}" \
  pipeline_kwargs.kernel_size_range="${KERNEL_SIZE_RANGE:-[0,0]}" \
  pipeline_kwargs.color_thresh="${COLOR_THRESH:-[50,50]}" \
  pipeline_kwargs.prompt_drop_prob="${PROMPT_DROP_PROB:-0.0}" \
  pipeline_kwargs.mask_drop_prob="${MASK_DROP_PROB:-0.0}" \
  pipeline_kwargs.condition_drop_prob="${CONDITION_DROP_PROB:-0.0}" \
  "train_dataset=SpatialVID(split=None, ROOT=\"${DATA_ROOT}\", video_ids=\"${VIDEO_ID}\", use_camera_annotations=\${use_camera_annotations}, continuous_target_frames=\${continuous_target_frames}, force_first_context=\${force_first_context}, timestamp_unit=\"\${timestamp_unit}\", min_interval=1, max_interval=1, height=\${height}, width=\${width}, num_views=\${num_views}, min_num_context_views=${NUM_CONTEXT_VIEWS}, max_num_context_views=${NUM_CONTEXT_VIEWS}, seed=${DATASET_SEED})" \
  "$@"
