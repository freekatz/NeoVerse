#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU fixed clip + fixed trajectory frozen-cache builder.
#
# Usage examples:
#   bash scripts/launch/build_frozen_cache.sh
#   TRAJECTORIES_PER_CLIP=16 CAMERA_CACHE_DIR=outputs/NeoVerseControlLatentDistill/camera_cache \
#     bash scripts/launch/build_frozen_cache.sh

timestamp_utc() {
  date -u "+%Y%m%d_%H%M%S"
}

log() {
  printf '[build_frozen_cache][%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/runtime_env.sh"

DRY_RUN="${DRY_RUN:-0}"
CONFIG="${CONFIG:-configs/distill/control_latent.yaml}"
PROJECT_NAME="${PROJECT_NAME:-NeoVerseControlLatentDistill}"
RUN_TIME="${RUN_TIME:-$(timestamp_utc)}"
OUTPUT_DIR="${FROZEN_CACHE_DIR:-${OUTPUT_DIR:-${CODE_DIR}/outputs/${PROJECT_NAME}/frozen_cache}}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/outputs/${PROJECT_NAME}/frozen_cache_logs/${RUN_TIME}}"
DEFAULT_CAMERA_CACHE_DIR="${DEFAULT_CAMERA_CACHE_DIR:-${CODE_DIR}/outputs/${PROJECT_NAME}/camera_cache}"
USE_CAMERA_CACHE="${USE_CAMERA_CACHE:-auto}"

run_output_dir="${LOG_DIR}/run_output"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${run_output_dir}"
cd "${CODE_DIR}"

if ! resolve_shard_layout \
  "${CUDA_VISIBLE_DEVICES:-}" \
  "${MLP_WORKER_GPU:-}" \
  "${CACHE_NNODES:-}" \
  "${CACHE_NODE_RANK:-}" \
  "${CACHE_GLOBAL_NUM_SHARDS:-}" \
  "${CACHE_GLOBAL_SHARD_OFFSET:-}" \
  "python"; then
  die "${ERROR_MESSAGE}"
fi

if [[ -z "${CAMERA_CACHE_DIR:-}" ]]; then
  case "${USE_CAMERA_CACHE}" in
    1|true|yes|on)
      CAMERA_CACHE_DIR="${DEFAULT_CAMERA_CACHE_DIR}"
      CAMERA_CACHE_REQUIRED="${CAMERA_CACHE_REQUIRED:-true}"
      ;;
    auto)
      if [[ -d "${DEFAULT_CAMERA_CACHE_DIR}" ]]; then
        CAMERA_CACHE_DIR="${DEFAULT_CAMERA_CACHE_DIR}"
        CAMERA_CACHE_REQUIRED="${CAMERA_CACHE_REQUIRED:-true}"
      fi
      ;;
    0|false|no|off)
      ;;
    *)
      die "Unsupported USE_CAMERA_CACHE=${USE_CAMERA_CACHE}; expected auto/true/false"
      ;;
  esac
fi

common_args=(
  --config "${CONFIG}"
  --output_dir "${OUTPUT_DIR}"
)
if [[ "${OVERWRITE:-0}" == "1" || "${OVERWRITE:-}" == "true" ]]; then
  common_args+=(--overwrite)
fi
if [[ "${CONTINUE_ON_ERROR:-0}" == "1" || "${CONTINUE_ON_ERROR:-}" == "true" ]]; then
  common_args+=(--continue_on_error)
fi

extra_overrides=()
if [[ -n "${CAMERA_CACHE_DIR:-}" ]]; then
  extra_overrides+=("camera_cache_dir=${CAMERA_CACHE_DIR}" "camera_cache_required=${CAMERA_CACHE_REQUIRED:-true}")
fi

log "code_dir=${CODE_DIR}"
log "output_dir=${OUTPUT_DIR}"
log "log_dir=${LOG_DIR}"
log "run_output_dir=${run_output_dir}"
log "gpu_list=${RESOLVED_GPU_LIST}"
log "local_num_shards=${RESOLVED_GPU_COUNT}"
log "cache_nnodes=${SHARD_NNODES}"
log "cache_node_rank=${SHARD_NODE_RANK}"
log "cache_global_num_shards=${SHARD_GLOBAL_NUM_SHARDS}"
log "cache_global_shard_offset=${SHARD_GLOBAL_SHARD_OFFSET}"
log "camera_cache_dir=${CAMERA_CACHE_DIR:-}"
log "camera_cache_required=${CAMERA_CACHE_REQUIRED:-}"
log "common_args=${common_args[*]}"
log "extra_args=$*"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "+ bash -n $0"
  log "+ python -m py_compile tools/cache/build_frozen_cache.py tools/train/distill_control_latent.py utils/config.py models/control_latent/cache.py models/control_latent/camera.py models/control_latent/loss.py models/control_latent/module.py models/control_latent/reconstructor_tokens.py utils/datasets/spatialvid.py utils/datasets/frozen_cache.py models/control_latent/trajectory.py"
else
  bash -n "$0"
  "python" -m py_compile \
    tools/cache/build_frozen_cache.py \
    tools/train/distill_control_latent.py \
    utils/config.py \
    models/control_latent/cache.py \
    models/control_latent/camera.py \
    models/control_latent/loss.py \
    models/control_latent/module.py \
    models/control_latent/reconstructor_tokens.py \
    utils/datasets/spatialvid.py \
    utils/datasets/frozen_cache.py \
    models/control_latent/trajectory.py
fi

pids=()
for shard_index in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[shard_index]}"
  global_shard_index=$((SHARD_GLOBAL_SHARD_OFFSET + shard_index))
  safe_gpu="${gpu//[^A-Za-z0-9_.-]/_}"
  shard_log="${LOG_DIR}/shard_${global_shard_index}_node_${SHARD_NODE_RANK}_gpu_${safe_gpu}.log"
  shard_run_output="${run_output_dir}/shard_${global_shard_index}"
  shard_cmd=(
    "python" tools/cache/build_frozen_cache.py
    "${common_args[@]}"
    --run_output_dir "${shard_run_output}"
    --device cuda:0
    --num_shards "${SHARD_GLOBAL_NUM_SHARDS}"
    --shard_index "${global_shard_index}"
    "${extra_overrides[@]+"${extra_overrides[@]}"}"
    "$@"
  )
  log "+ CUDA_VISIBLE_DEVICES=${gpu} ${shard_cmd[*]} 2>&1 | tee ${shard_log}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="${gpu}" "${shard_cmd[@]}" 2>&1 | tee "${shard_log}"
    ) &
    pids+=("$!")
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN complete"
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  die "One or more frozen cache shards failed. See ${LOG_DIR}"
fi
log "Finished successfully"
