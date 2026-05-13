#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU fixed-clip camera cache builder.
#
# Usage examples:
#   bash scripts/launch/build_camera_cache.sh
#   GPU_LIST=0,1,2,3 bash scripts/launch/build_camera_cache.sh
#   LAUNCH_MODE=background GPU_LIST=0,1 bash scripts/launch/build_camera_cache.sh
#   CAMERA_CACHE_DIR=outputs/NeoVerseControlLatentDistill/camera_cache \
#     FIXED_CLIPS_PER_SCENE=16 bash scripts/launch/build_camera_cache.sh

timestamp_utc() {
  date -u "+%Y%m%d_%H%M%S"
}

log() {
  printf '[build_camera_cache][%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/runtime_env.sh"

LAUNCH_MODE="${LAUNCH_MODE:-foreground}"
DRY_RUN="${DRY_RUN:-0}"
VENV_PATH="${VENV_PATH:-/root/vepfs/envs/neoverse}"
ENV_PYTHON="${ENV_PYTHON:-${PYTHON:-${VENV_PATH}/bin/python}}"
CONFIG="${CONFIG:-configs/distill/control_latent.yaml}"
PROJECT_NAME="${PROJECT_NAME:-NeoVerseControlLatentDistill}"
RUN_TIME="${RUN_TIME:-$(timestamp_utc)}"
OUTPUT_DIR="${CAMERA_CACHE_DIR:-${OUTPUT_DIR:-${CODE_DIR}/outputs/${PROJECT_NAME}/camera_cache}}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/outputs/${PROJECT_NAME}/camera_cache_logs/${RUN_TIME}}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run.pid}"
GPU_LIST="${GPU_LIST:-}"
CACHE_NNODES="${CACHE_NNODES:-${MLP_WORKER_NUM:-1}}"
CACHE_NODE_RANK="${CACHE_NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
CACHE_GLOBAL_NUM_SHARDS="${CACHE_GLOBAL_NUM_SHARDS:-}"
CACHE_GLOBAL_SHARD_OFFSET="${CACHE_GLOBAL_SHARD_OFFSET:-}"

[[ -d "${CODE_DIR}" ]] || die "CODE_DIR does not exist: ${CODE_DIR}"
[[ -x "${ENV_PYTHON}" ]] || die "ENV_PYTHON is not executable: ${ENV_PYTHON}"
[[ -f "${CODE_DIR}/${CONFIG}" || -f "${CONFIG}" ]] || die "Missing config: ${CONFIG}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${CODE_DIR}"

if [[ "${LAUNCH_MODE}" == "background" && "${_NEOVERSE_CAMERA_CACHE_BG:-0}" != "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    die "DRY_RUN=1 cannot be combined with LAUNCH_MODE=background"
  fi
  log "Launching camera cache build in background"
  log "output_dir=${OUTPUT_DIR}"
  log "log_dir=${LOG_DIR}"
  nohup env \
    _NEOVERSE_CAMERA_CACHE_BG=1 \
    LAUNCH_MODE=foreground \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    CAMERA_CACHE_DIR="${OUTPUT_DIR}" \
    LOG_DIR="${LOG_DIR}" \
    PID_FILE="${PID_FILE}" \
    "$0" "$@" >"${LOG_DIR}/launcher.log" 2>&1 &
  child_pid="$!"
  printf '%s\n' "${child_pid}" > "${PID_FILE}"
  log "pid=${child_pid}"
  exit 0
fi

GPU_LIST="$(neoverse_resolve_gpu_list "${GPU_LIST}" "${CUDA_VISIBLE_DEVICES:-}" "${MLP_WORKER_GPU:-}" "${ENV_PYTHON}")"
neoverse_split_csv "${GPU_LIST}"
gpu_ids=("${NEOVERSE_SPLIT_ITEMS[@]}")
if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
  die "GPU_LIST is empty"
fi
neoverse_validate_non_negative_int "CACHE_NNODES" "${CACHE_NNODES}" || die "Invalid CACHE_NNODES=${CACHE_NNODES}"
neoverse_validate_non_negative_int "CACHE_NODE_RANK" "${CACHE_NODE_RANK}" || die "Invalid CACHE_NODE_RANK=${CACHE_NODE_RANK}"
if [[ "${CACHE_NNODES}" -le 0 ]]; then
  die "CACHE_NNODES must be positive, got ${CACHE_NNODES}"
fi
if [[ "${CACHE_NODE_RANK}" -ge "${CACHE_NNODES}" ]]; then
  die "CACHE_NODE_RANK=${CACHE_NODE_RANK} must be smaller than CACHE_NNODES=${CACHE_NNODES}"
fi

local_num_shards="${#gpu_ids[@]}"
if [[ -z "${CACHE_GLOBAL_NUM_SHARDS}" ]]; then
  CACHE_GLOBAL_NUM_SHARDS=$((local_num_shards * CACHE_NNODES))
fi
if [[ -z "${CACHE_GLOBAL_SHARD_OFFSET}" ]]; then
  CACHE_GLOBAL_SHARD_OFFSET=$((CACHE_NODE_RANK * local_num_shards))
fi
neoverse_validate_non_negative_int "CACHE_GLOBAL_NUM_SHARDS" "${CACHE_GLOBAL_NUM_SHARDS}" || die "Invalid CACHE_GLOBAL_NUM_SHARDS=${CACHE_GLOBAL_NUM_SHARDS}"
neoverse_validate_non_negative_int "CACHE_GLOBAL_SHARD_OFFSET" "${CACHE_GLOBAL_SHARD_OFFSET}" || die "Invalid CACHE_GLOBAL_SHARD_OFFSET=${CACHE_GLOBAL_SHARD_OFFSET}"
if [[ "${CACHE_GLOBAL_NUM_SHARDS}" -le 0 ]]; then
  die "CACHE_GLOBAL_NUM_SHARDS must be positive, got ${CACHE_GLOBAL_NUM_SHARDS}"
fi
max_global_shard_index=$((CACHE_GLOBAL_SHARD_OFFSET + local_num_shards - 1))
if [[ "${max_global_shard_index}" -ge "${CACHE_GLOBAL_NUM_SHARDS}" ]]; then
  die "Shard range ${CACHE_GLOBAL_SHARD_OFFSET}-${max_global_shard_index} exceeds CACHE_GLOBAL_NUM_SHARDS=${CACHE_GLOBAL_NUM_SHARDS}"
fi

common_args=(
  --config "${CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --fixed_clips_per_scene "${FIXED_CLIPS_PER_SCENE:-16}"
)

append_arg_if_set() {
  local env_name="$1"
  local option_name="$2"
  local value="${!env_name:-}"
  if [[ -n "${value}" ]]; then
    common_args+=("${option_name}" "${value}")
  fi
}

append_arg_if_set "DATA_ROOT" "--data_root"
append_arg_if_set "RECONSTRUCTOR_PATH" "--reconstructor_path"
append_arg_if_set "TORCH_DTYPE" "--torch_dtype"
append_arg_if_set "LIMIT" "--limit"
append_arg_if_set "VIDEO_IDS" "--video_ids"
append_arg_if_set "VIDEO_PATHS" "--video_paths"
if [[ "${OVERWRITE:-0}" == "1" || "${OVERWRITE:-}" == "true" ]]; then
  common_args+=(--overwrite)
fi

log "code_dir=${CODE_DIR}"
log "output_dir=${OUTPUT_DIR}"
log "log_dir=${LOG_DIR}"
log "gpu_list=${GPU_LIST}"
log "local_num_shards=${local_num_shards}"
log "cache_nnodes=${CACHE_NNODES}"
log "cache_node_rank=${CACHE_NODE_RANK}"
log "cache_global_num_shards=${CACHE_GLOBAL_NUM_SHARDS}"
log "cache_global_shard_offset=${CACHE_GLOBAL_SHARD_OFFSET}"
log "common_args=${common_args[*]}"
log "extra_args=$*"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "+ bash -n $0"
  log "+ ${ENV_PYTHON} -m py_compile tools/cache/build_camera_cache.py training/data/spatialvid.py"
else
  bash -n "$0"
  "${ENV_PYTHON}" -m py_compile tools/cache/build_camera_cache.py training/data/spatialvid.py
fi

pids=()
for shard_index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[shard_index]}"
  global_shard_index=$((CACHE_GLOBAL_SHARD_OFFSET + shard_index))
  safe_gpu="${gpu//[^A-Za-z0-9_.-]/_}"
  shard_log="${LOG_DIR}/shard_${global_shard_index}_node_${CACHE_NODE_RANK}_gpu_${safe_gpu}.log"
  shard_cmd=(
    "${ENV_PYTHON}" tools/cache/build_camera_cache.py
    "${common_args[@]}"
    --device cuda:0
    --num_shards "${CACHE_GLOBAL_NUM_SHARDS}"
    --shard_index "${global_shard_index}"
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
  die "One or more camera cache shards failed. See ${LOG_DIR}"
fi
log "Finished successfully"
