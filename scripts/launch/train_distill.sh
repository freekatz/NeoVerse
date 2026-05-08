#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   DRY_RUN=1 bash scripts/launch/train_distill.sh
#   RUN_NAME=neoverse_distill_v1 bash scripts/launch/train_distill.sh
#   GPU_LIST=0,1 RUN_NAME=neoverse_distill_v1 bash scripts/launch/train_distill.sh
#   LAUNCH_MODE=background RUN_NAME=neoverse_distill_v1 bash scripts/launch/train_distill.sh
#   ADAPTER_TYPE=conv RUN_NAME=neoverse_distill_conv bash scripts/launch/train_distill.sh
#   On Volcengine/MLP with MLP_* env vars already set:
#     RUN_NAME=neoverse_distill_v1 bash scripts/launch/train_distill.sh

timestamp_utc() {
  date -u "+%Y%m%d_%H%M%S"
}

log() {
  printf '[train_distill_control][%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

run_cmd() {
  log "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="${MODE:-train}"
LAUNCH_MODE="${LAUNCH_MODE:-foreground}"
DRY_RUN="${DRY_RUN:-0}"

VENV_PATH="${VENV_PATH:-/root/vepfs/envs/neoverse}"
ENV_PYTHON="${ENV_PYTHON:-${VENV_PATH}/bin/python}"
ACCELERATE="${ACCELERATE:-${VENV_PATH}/bin/accelerate}"
CONFIG="${CONFIG:-configs/distill_control_latent.yaml}"

PROJECT_NAME="${PROJECT_NAME:-NeoVerseControlLatentDistill}"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
RUN_TIME="${RUN_TIME:-$(date +%H-%M-%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/${PROJECT_NAME}/${RUN_DATE}/${RUN_TIME}}"
if [[ "${OUTPUT_PATH}" != /* ]]; then
  OUTPUT_PATH="${CODE_DIR}/${OUTPUT_PATH}"
fi
RUN_NAME="${RUN_NAME:-train}"
LOG_DIR="${LOG_DIR:-${OUTPUT_PATH}/logs}"
if [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${CODE_DIR}/${LOG_DIR}"
fi
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run.pid}"
CONFIG_FILE="${CONFIG_FILE:-${LOG_DIR}/run_config.env}"

GPU_LIST="${GPU_LIST:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# Adapter defaults live in configs/distill_control_latent.yaml. Set
# ADAPTER_TYPE=conv or ADAPTER_TYPE=cross_attention_rope to override per run.
ADAPTER_TYPE="${ADAPTER_TYPE:-}"

LAUNCHER="${LAUNCHER:-accelerate}"
USE_DISTRIBUTED="${USE_DISTRIBUTED:-auto}"
DIST_NPROC_PER_NODE="${DIST_NPROC_PER_NODE:-${MLP_WORKER_GPU:-}}"
DIST_NNODES="${DIST_NNODES:-${MLP_WORKER_NUM:-1}}"
DIST_NODE_RANK="${DIST_NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
DIST_MASTER_ADDR="${DIST_MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
DIST_MASTER_PORT="${DIST_MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-}"
ACCELERATE_EXTRA_ARGS="${ACCELERATE_EXTRA_ARGS:-}"

mkdir -p "${LOG_DIR}" "${OUTPUT_PATH}"

case "${MODE}" in
  train) ;;
  *)
    die "Unsupported MODE=${MODE}. Expected: train"
    ;;
esac

if [[ "${LAUNCH_MODE}" == "background" && "${_NEOVERSE_DISTILL_BG:-0}" != "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    die "DRY_RUN=1 cannot be combined with LAUNCH_MODE=background"
  fi
  log "Launching in background"
  log "output_path=${OUTPUT_PATH}"
  log "log_file=${LOG_FILE}"
  nohup env \
    _NEOVERSE_DISTILL_BG=1 \
    LAUNCH_MODE=foreground \
    RUN_NAME="${RUN_NAME}" \
    OUTPUT_PATH="${OUTPUT_PATH}" \
    LOG_DIR="${LOG_DIR}" \
    LOG_FILE="${LOG_FILE}" \
    PID_FILE="${PID_FILE}" \
    CONFIG_FILE="${CONFIG_FILE}" \
    "$0" "$@" >"${LOG_FILE}" 2>&1 &
  child_pid="$!"
  printf '%s\n' "${child_pid}" > "${PID_FILE}"
  log "pid=${child_pid}"
  exit 0
fi

[[ -d "${CODE_DIR}" ]] || die "CODE_DIR does not exist: ${CODE_DIR}"
[[ -d "${VENV_PATH}" ]] || die "VENV_PATH does not exist: ${VENV_PATH}"
[[ -x "${ENV_PYTHON}" ]] || die "ENV_PYTHON is not executable: ${ENV_PYTHON}"
[[ -x "${ACCELERATE}" ]] || die "ACCELERATE is not executable: ${ACCELERATE}"
[[ -f "${CODE_DIR}/${CONFIG}" || -f "${CONFIG}" ]] || die "Missing config: ${CONFIG}"

cd "${CODE_DIR}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED
export TOKENIZERS_PARALLELISM
export PYTORCH_CUDA_ALLOC_CONF
export TORCH_NCCL_ASYNC_ERROR_HANDLING
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

detect_local_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    "${ENV_PYTHON}" - "${CUDA_VISIBLE_DEVICES}" <<'PY'
import sys

items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(len(items))
PY
    return
  fi

  "${ENV_PYTHON}" - <<'PY'
import torch

print(torch.cuda.device_count())
PY
}

if [[ "${USE_DISTRIBUTED}" == "auto" && -z "${DIST_NPROC_PER_NODE}" && "${DIST_NNODES}" == "1" ]]; then
  auto_local_gpu_count="$(detect_local_gpu_count)"
  if [[ "${auto_local_gpu_count}" =~ ^[0-9]+$ && "${auto_local_gpu_count}" -gt 0 ]]; then
    DIST_NPROC_PER_NODE="${auto_local_gpu_count}"
  fi
fi

if [[ -z "${DIST_NPROC_PER_NODE}" ]]; then
  DIST_NPROC_PER_NODE=1
fi

is_distributed=0
if [[ "${USE_DISTRIBUTED}" == "1" ]]; then
  is_distributed=1
elif [[ "${USE_DISTRIBUTED}" == "auto" ]]; then
  if [[ "${DIST_NNODES}" != "1" || "${DIST_NPROC_PER_NODE}" != "1" ]]; then
    is_distributed=1
  fi
fi

if [[ "${DIST_NPROC_PER_NODE}" -le 0 || "${DIST_NNODES}" -le 0 ]]; then
  die "Invalid distributed size: DIST_NPROC_PER_NODE=${DIST_NPROC_PER_NODE}, DIST_NNODES=${DIST_NNODES}"
fi

GLOBAL_NUM_PROCESSES=$((DIST_NPROC_PER_NODE * DIST_NNODES))
MAX_STEPS_VALUE="${MAX_STEPS:-200000}"
NUM_EPOCHS_VALUE="${NUM_EPOCHS:-${MAX_STEPS_VALUE}}"
TRAJECTORIES_PER_CLIP_VALUE="${TRAJECTORIES_PER_CLIP:-${VARIANTS_PER_SCENE:-1}}"

TRAIN_OVERRIDES=(
  "output_path=${OUTPUT_PATH}"
  "num_views=${NUM_VIEWS:-81}"
  "min_num_context_views=${MIN_NUM_CONTEXT_VIEWS:-10}"
  "max_num_context_views=${MAX_NUM_CONTEXT_VIEWS:-20}"
  "continuous_target_frames=${CONTINUOUS_TARGET_FRAMES:-true}"
  "force_first_context=${FORCE_FIRST_CONTEXT:-true}"
  "timestamp_unit=${TIMESTAMP_UNIT:-seconds}"
  "temporal_augmentation=${TEMPORAL_AUGMENTATION:-false}"
  "temporal_trajectory_profile=${TEMPORAL_TRAJECTORY_PROFILE:-forward_pause}"
  "temporal_order=${TEMPORAL_ORDER:-trajectory}"
  "temporal_max_condition_frames=${TEMPORAL_MAX_CONDITION_FRAMES:-8}"
  "context_sampling_strategy=${CONTEXT_SAMPLING_STRATEGY:-mixed}"
  "variants_per_scene=${TRAJECTORIES_PER_CLIP_VALUE}"
  "trajectories_per_clip=${TRAJECTORIES_PER_CLIP_VALUE}"
  "temporal_variant_profile_weights=${TEMPORAL_VARIANT_PROFILE_WEIGHTS:-}"
  "fixed_clips_per_scene=${FIXED_CLIPS_PER_SCENE:-16}"
  "pipeline_kwargs.mask_non_context_targets=${MASK_NON_CONTEXT_TARGETS:-false}"
  "max_steps=${MAX_STEPS_VALUE}"
  "num_epochs=${NUM_EPOCHS_VALUE}"
)

append_override_if_set() {
  local env_name="$1"
  local key="$2"
  local value="${!env_name:-}"
  if [[ -n "${value}" ]]; then
    TRAIN_OVERRIDES+=("${key}=${value}")
  fi
}

append_override_if_set "MODEL_PATH" "model_path"
append_override_if_set "RECONSTRUCTOR_PATH" "reconstructor_path"
append_override_if_set "ENABLE_VRAM_MANAGEMENT" "enable_vram_management"
append_override_if_set "REUSE_RECONSTRUCTOR_TOKENS" "reuse_reconstructor_tokens"
append_override_if_set "SKIP_UNUSED_RECONSTRUCTOR_HEADS" "skip_unused_reconstructor_heads"
append_override_if_set "BATCH_VAE_TEACHER_EMBEDS" "batch_vae_teacher_embeds"
append_override_if_set "REUSE_CONTEXT_FORWARD_TOKENS" "reuse_context_forward_tokens"
append_override_if_set "SEED" "seed"
append_override_if_set "DATASET_SEED" "dataset_seed"
append_override_if_set "DATA_ROOT" "data_root"
append_override_if_set "VIDEO_IDS" "video_ids"
append_override_if_set "VIDEO_PATHS" "video_paths"
append_override_if_set "USE_CAMERA_ANNOTATIONS" "use_camera_annotations"
append_override_if_set "FIXED_CLIPS_PER_SCENE" "fixed_clips_per_scene"
append_override_if_set "CAMERA_CACHE_DIR" "camera_cache_dir"
append_override_if_set "CAMERA_CACHE_REQUIRED" "camera_cache_required"
append_override_if_set "CAMERA_CONDITION_NORMALIZE" "camera_condition_normalization.enabled"
append_override_if_set "CAMERA_CONDITION_NORMALIZE_POSES" "camera_condition_normalization.normalize_poses"
append_override_if_set "CAMERA_CONDITION_NORMALIZE_INTRINSICS" "camera_condition_normalization.normalize_intrinsics"
append_override_if_set "CAMERA_CONDITION_NORMALIZE_PLUCKER" "camera_condition_normalization.normalize_plucker"
append_override_if_set "CAMERA_CONDITION_MIN_TRANSLATION_SCALE" "camera_condition_normalization.min_translation_scale"
append_override_if_set "BATCH_SIZE" "batch_size"
append_override_if_set "NUM_WORKERS" "num_workers"
append_override_if_set "PIN_MEMORY" "pin_memory"
append_override_if_set "PERSISTENT_WORKERS" "persistent_workers"
append_override_if_set "PREFETCH_FACTOR" "prefetch_factor"
append_override_if_set "LEARNING_RATE" "learning_rate"
append_override_if_set "WEIGHT_DECAY" "weight_decay"
append_override_if_set "GRADIENT_ACCUMULATION_STEPS" "gradient_accumulation_steps"
append_override_if_set "CLIP_GRAD" "clip_grad"
append_override_if_set "SAVE_FREQ" "save_freq"
append_override_if_set "PRINT_FREQ" "print_freq"
append_override_if_set "VIS_FREQ" "vis_freq"
append_override_if_set "AUTO_RESUME" "auto_resume"
append_override_if_set "RESUME_FROM" "resume_from"
append_override_if_set "RESUME_OPTIMIZER" "resume_optimizer"
append_override_if_set "SAVE_OPTIMIZER_INTERMEDIATE" "save_optimizer_intermediate"
append_override_if_set "ADAPTER_TYPE" "adapter.type"
append_override_if_set "ADAPTER_OUTPUT_MODE" "adapter.output_mode"
append_override_if_set "ADAPTER_HIDDEN_DIM" "adapter.hidden_dim"
append_override_if_set "ADAPTER_USE_TEXT_CONTEXT" "adapter.use_text_context"
append_override_if_set "ADAPTER_USE_DIT_STATE" "adapter.use_dit_state"
append_override_if_set "CONV_INPUT_CHANNELS" "adapter.input_channels"
append_override_if_set "CONV_NUM_RES_BLOCKS" "adapter.num_res_blocks"
append_override_if_set "CONV_DROPOUT" "adapter.dropout"
append_override_if_set "CONV_CONDITION_TIME_DIM" "adapter.condition_time_dim"
append_override_if_set "CROSS_TOKEN_DIM" "adapter.token_dim"
append_override_if_set "CROSS_NUM_HEADS" "adapter.num_heads"
append_override_if_set "CROSS_NUM_BLOCKS" "adapter.num_blocks"
append_override_if_set "CROSS_SOURCE_POOL_HW" "adapter.source_pool_hw"
append_override_if_set "CROSS_MAX_SOURCE_TOKENS" "adapter.max_source_tokens"
append_override_if_set "CROSS_QUERY_CHUNK_SIZE" "adapter.query_chunk_size"
append_override_if_set "CROSS_USE_ROPE" "adapter.use_rope"
append_override_if_set "CROSS_MAX_TOKEN_GROUPS" "adapter.max_token_groups"
append_override_if_set "CROSS_USE_GROUP_EMBEDDING" "adapter.use_group_embedding"
append_override_if_set "CROSS_USE_LOCAL_GRID" "adapter.use_local_grid"
append_override_if_set "CROSS_USE_TIME_FILM" "adapter.use_time_film"
append_override_if_set "CROSS_TIME_FILM_DIM" "adapter.time_film_dim"
append_override_if_set "CROSS_TIME_POSITION_MODE" "adapter.time_position_mode"
append_override_if_set "CROSS_REROPE_INTERVAL" "adapter.rerope_interval"
append_override_if_set "CROSS_POST_NUM_RES_BLOCKS" "adapter.post_num_res_blocks"
append_override_if_set "CACHE_TRAIN_BATCH" "cache_train_batch"
append_override_if_set "CACHE_FROZEN_OUTPUTS" "cache_frozen_outputs"
append_override_if_set "FROZEN_CACHE_DIR" "frozen_cache_dir"
append_override_if_set "TRAIN_FROM_FROZEN_CACHE" "train_from_frozen_cache"
append_override_if_set "FROZEN_CACHE_PATTERN" "frozen_cache_pattern"
append_override_if_set "FROZEN_CACHE_SPLIT" "frozen_cache_split"
append_override_if_set "FROZEN_CACHE_EVAL_RATIO" "frozen_cache_eval_ratio"
append_override_if_set "FROZEN_CACHE_SPLIT_SEED" "frozen_cache_split_seed"
append_override_if_set "FROZEN_CACHE_SPLIT_MODE" "frozen_cache_split_mode"
append_override_if_set "FROZEN_CACHE_READ" "frozen_cache_read"
append_override_if_set "FROZEN_CACHE_WRITE" "frozen_cache_write"
append_override_if_set "FROZEN_CACHE_REQUIRED" "frozen_cache_required"
append_override_if_set "FROZEN_CACHE_DTYPE" "frozen_cache_dtype"
append_override_if_set "PRELOAD_FROZEN_CACHE" "preload_frozen_cache"
append_override_if_set "PRELOAD_FROZEN_CACHE_DEVICE" "preload_frozen_cache_device"
append_override_if_set "PRELOAD_FROZEN_CACHE_BATCH_SIZE" "preload_frozen_cache_batch_size"
append_override_if_set "PRELOAD_FROZEN_CACHE_LOG_FREQ" "preload_frozen_cache_log_freq"
append_override_if_set "SHUFFLE_PRELOADED_FROZEN_CACHE" "shuffle_preloaded_frozen_cache"
append_override_if_set "EVAL_FREQ" "eval_freq"
append_override_if_set "EVAL_STEPS" "eval_steps"
append_override_if_set "EVAL_BATCH_SIZE" "eval_batch_size"
append_override_if_set "EVAL_NUM_WORKERS" "eval_num_workers"

read -r -a accelerate_extra_args <<< "${ACCELERATE_EXTRA_ARGS}"

cat > "${CONFIG_FILE}" <<EOF
MODE=${MODE}
LAUNCH_MODE=${LAUNCH_MODE}
DRY_RUN=${DRY_RUN}
CODE_DIR=${CODE_DIR}
VENV_PATH=${VENV_PATH}
ENV_PYTHON=${ENV_PYTHON}
ACCELERATE=${ACCELERATE}
CONFIG=${CONFIG}
PROJECT_NAME=${PROJECT_NAME}
RUN_DATE=${RUN_DATE}
RUN_TIME=${RUN_TIME}
RUN_NAME=${RUN_NAME}
OUTPUT_PATH=${OUTPUT_PATH}
FROZEN_CACHE_DIR=${FROZEN_CACHE_DIR:-}
FROZEN_CACHE_REQUIRED=${FROZEN_CACHE_REQUIRED:-}
FROZEN_CACHE_SPLIT=${FROZEN_CACHE_SPLIT:-}
FROZEN_CACHE_EVAL_RATIO=${FROZEN_CACHE_EVAL_RATIO:-}
FROZEN_CACHE_SPLIT_SEED=${FROZEN_CACHE_SPLIT_SEED:-}
LOG_DIR=${LOG_DIR}
LOG_FILE=${LOG_FILE}
PID_FILE=${PID_FILE}
GPU_LIST=${GPU_LIST}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
MIXED_PRECISION=${MIXED_PRECISION}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}
USE_DISTRIBUTED=${USE_DISTRIBUTED}
IS_DISTRIBUTED=${is_distributed}
DIST_NPROC_PER_NODE=${DIST_NPROC_PER_NODE}
DIST_NNODES=${DIST_NNODES}
DIST_NODE_RANK=${DIST_NODE_RANK}
DIST_MASTER_ADDR=${DIST_MASTER_ADDR}
DIST_MASTER_PORT=${DIST_MASTER_PORT}
GLOBAL_NUM_PROCESSES=${GLOBAL_NUM_PROCESSES}
ADAPTER_TYPE=${ADAPTER_TYPE:-}
ADAPTER_HIDDEN_DIM=${ADAPTER_HIDDEN_DIM:-}
CROSS_NUM_HEADS=${CROSS_NUM_HEADS:-}
CROSS_NUM_BLOCKS=${CROSS_NUM_BLOCKS:-}
CROSS_SOURCE_POOL_HW=${CROSS_SOURCE_POOL_HW:-}
CROSS_MAX_SOURCE_TOKENS=${CROSS_MAX_SOURCE_TOKENS:-}
CROSS_QUERY_CHUNK_SIZE=${CROSS_QUERY_CHUNK_SIZE:-}
CROSS_USE_GROUP_EMBEDDING=${CROSS_USE_GROUP_EMBEDDING:-}
CROSS_USE_LOCAL_GRID=${CROSS_USE_LOCAL_GRID:-}
CROSS_USE_TIME_FILM=${CROSS_USE_TIME_FILM:-}
CROSS_TIME_POSITION_MODE=${CROSS_TIME_POSITION_MODE:-}
CROSS_REROPE_INTERVAL=${CROSS_REROPE_INTERVAL:-}
CAMERA_CONDITION_NORMALIZE=${CAMERA_CONDITION_NORMALIZE:-}
CAMERA_CONDITION_NORMALIZE_POSES=${CAMERA_CONDITION_NORMALIZE_POSES:-}
CAMERA_CONDITION_NORMALIZE_INTRINSICS=${CAMERA_CONDITION_NORMALIZE_INTRINSICS:-}
CAMERA_CONDITION_NORMALIZE_PLUCKER=${CAMERA_CONDITION_NORMALIZE_PLUCKER:-}
CAMERA_CONDITION_MIN_TRANSLATION_SCALE=${CAMERA_CONDITION_MIN_TRANSLATION_SCALE:-}
FIXED_CLIPS_PER_SCENE=${FIXED_CLIPS_PER_SCENE:-}
TRAJECTORIES_PER_CLIP=${TRAJECTORIES_PER_CLIP:-}
TRAJECTORIES_PER_CLIP_VALUE=${TRAJECTORIES_PER_CLIP_VALUE}
TEMPORAL_VARIANT_PROFILE_WEIGHTS=${TEMPORAL_VARIANT_PROFILE_WEIGHTS:-}
CAMERA_CACHE_DIR=${CAMERA_CACHE_DIR:-}
CAMERA_CACHE_REQUIRED=${CAMERA_CACHE_REQUIRED:-}
TRAIN_OVERRIDES=${TRAIN_OVERRIDES[*]}
EXTRA_OVERRIDES=$*
EOF

log "mode=${MODE}"
log "run_name=${RUN_NAME}"
log "output_path=${OUTPUT_PATH}"
log "log_file=${LOG_FILE}"
log "config=${CONFIG}"
log "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
log "is_distributed=${is_distributed}"
log "global_num_processes=${GLOBAL_NUM_PROCESSES}"
log "dist_nproc_per_node=${DIST_NPROC_PER_NODE}"
log "dist_nnodes=${DIST_NNODES}"
log "dist_node_rank=${DIST_NODE_RANK}"
log "dist_master_addr=${DIST_MASTER_ADDR}"
log "dist_master_port=${DIST_MASTER_PORT}"
log "default_overrides=${TRAIN_OVERRIDES[*]}"
log "extra_overrides=$*"
log "config_file=${CONFIG_FILE}"

run_cmd bash -n "$0"
run_cmd "${ENV_PYTHON}" -m py_compile \
  tools/train/distill_control_latent.py \
  diffsynth/models/student_adapters.py \
  diffsynth/pipelines/wan_video_neoverse.py \
  diffsynth/models/wan_video_neoverse_controller.py \
  hooks/extract_vggt_tokens.py \
  training/data/base_dataset.py \
  training/data/datasets/spatialvid.py \
  training/data/temporal_trajectory.py \
  tools/cache/build_camera_cache.py \
  tools/cache/build_frozen_cache.py

launch_cmd=(
  "${ACCELERATE}" launch
  --mixed_precision "${MIXED_PRECISION}"
  --num_processes "${GLOBAL_NUM_PROCESSES}"
)

if [[ "${is_distributed}" == "1" ]]; then
  launch_cmd+=(--multi_gpu)
fi

if [[ "${DIST_NNODES}" != "1" ]]; then
  launch_cmd+=(
    --num_machines "${DIST_NNODES}"
    --machine_rank "${DIST_NODE_RANK}"
    --main_process_ip "${DIST_MASTER_ADDR}"
    --main_process_port "${DIST_MASTER_PORT}"
    --same_network
  )
fi

if [[ -n "${NUM_CPU_THREADS_PER_PROCESS}" ]]; then
  launch_cmd+=(--num_cpu_threads_per_process "${NUM_CPU_THREADS_PER_PROCESS}")
fi

if [[ "${#accelerate_extra_args[@]}" -gt 0 ]]; then
  launch_cmd+=("${accelerate_extra_args[@]}")
fi

launch_cmd+=(
  tools/train/distill_control_latent.py "${CONFIG}"
  "${TRAIN_OVERRIDES[@]}"
  "$@"
)

if [[ "${_NEOVERSE_DISTILL_BG:-0}" == "1" ]]; then
  run_cmd "${launch_cmd[@]}"
else
  run_cmd "${launch_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi

log "Finished successfully"
