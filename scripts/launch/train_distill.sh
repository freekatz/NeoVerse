#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   DRY_RUN=1 bash scripts/launch/train_distill.sh
#   RUN_NAME=neoverse_distill_v1 bash scripts/launch/train_distill.sh
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
source "${SCRIPT_DIR}/runtime_env.sh"

MODE="${MODE:-train}"
DRY_RUN="${DRY_RUN:-0}"

CONFIG="${CONFIG:-configs/distill/control_latent.yaml}"

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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

USE_DISTRIBUTED="${USE_DISTRIBUTED:-auto}"
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

cd "${CODE_DIR}"

export PYTHONUNBUFFERED
export TOKENIZERS_PARALLELISM
export PYTORCH_CUDA_ALLOC_CONF
export TORCH_NCCL_ASYNC_ERROR_HANDLING
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
if ! resolve_cuda_visible_devices "${CUDA_VISIBLE_DEVICES:-}" "${MLP_WORKER_GPU:-}" python; then
  die "${ERROR_MESSAGE}"
fi
CUDA_VISIBLE_DEVICES="${RESOLVED_CUDA_VISIBLE_DEVICES}"
LOCAL_VISIBLE_GPU_COUNT="${RESOLVED_LOCAL_VISIBLE_GPU_COUNT}"
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

if ! resolve_distributed_layout \
  "${USE_DISTRIBUTED}" \
  "${DIST_NPROC_PER_NODE:-}" \
  "${DIST_NNODES:-}" \
  "${DIST_NODE_RANK:-}" \
  "${LOCAL_VISIBLE_GPU_COUNT}"; then
  die "${ERROR_MESSAGE}"
fi

read -r -a accelerate_extra_args <<< "${ACCELERATE_EXTRA_ARGS}"

log "mode=${MODE}"
log "run_name=${RUN_NAME}"
log "output_path=${OUTPUT_PATH}"
log "log_file=${LOG_FILE}"
log "config=${CONFIG}"
log "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
log "local_visible_gpu_count=${LOCAL_VISIBLE_GPU_COUNT}"
log "is_distributed=${DIST_IS_DISTRIBUTED}"
log "global_num_processes=${GLOBAL_NUM_PROCESSES}"
log "dist_nproc_per_node=${DIST_NPROC_PER_NODE_VALUE}"
log "dist_nnodes=${DIST_NNODES_VALUE}"
log "dist_node_rank=${DIST_NODE_RANK_VALUE}"
log "dist_master_addr=${DIST_MASTER_ADDR}"
log "dist_master_port=${DIST_MASTER_PORT}"
log "extra_overrides=$*"

run_cmd bash -n "$0"
run_cmd python -m py_compile \
  tools/train/distill_control_latent.py \
  tools/train/launch.py \
  utils/config.py \
  utils/training.py \
  utils/training_module.py \
  utils/swanlab.py \
  models/control_latent/cache.py \
  models/control_latent/camera.py \
  models/control_latent/loss.py \
  models/control_latent/module.py \
  models/control_latent/reconstructor_tokens.py \
  diffsynth/models/student_adapters.py \
  diffsynth/pipelines/wan_video_neoverse.py \
  diffsynth/models/wan_video_neoverse_controller.py \
  utils/datasets/base.py \
  utils/datasets/spatialvid.py \
  utils/datasets/frozen_cache.py \
  models/control_latent/trajectory.py \
  tools/cache/build_camera_cache.py \
  tools/cache/build_frozen_cache.py

launch_cmd=(
  accelerate launch
  --mixed_precision "${MIXED_PRECISION}"
  --num_processes "${GLOBAL_NUM_PROCESSES}"
)

if [[ "${DIST_IS_DISTRIBUTED}" == "1" ]]; then
  launch_cmd+=(--multi_gpu)
fi

if [[ "${DIST_NNODES_VALUE}" != "1" ]]; then
  launch_cmd+=(
    --num_machines "${DIST_NNODES_VALUE}"
    --machine_rank "${DIST_NODE_RANK_VALUE}"
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
  "output_path=${OUTPUT_PATH}"
  "$@"
)

run_cmd "${launch_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"

log "Finished successfully"
