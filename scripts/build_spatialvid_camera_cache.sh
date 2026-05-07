#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU fixed-clip camera cache builder.
#
# Usage examples:
#   bash scripts/build_spatialvid_camera_cache.sh
#   GPU_LIST=0,1,2,3 bash scripts/build_spatialvid_camera_cache.sh
#   LAUNCH_MODE=background GPU_LIST=0,1 bash scripts/build_spatialvid_camera_cache.sh
#   CAMERA_CACHE_DIR=outputs/NeoVerseControlLatentDistill/camera_cache \
#     FIXED_CLIPS_PER_SCENE=16 bash scripts/build_spatialvid_camera_cache.sh

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
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LAUNCH_MODE="${LAUNCH_MODE:-foreground}"
DRY_RUN="${DRY_RUN:-0}"
VENV_PATH="${VENV_PATH:-/root/vepfs/envs/neoverse}"
ENV_PYTHON="${ENV_PYTHON:-${PYTHON:-${VENV_PATH}/bin/python}}"
CONFIG="${CONFIG:-configs/distill_control_latent.yaml}"
PROJECT_NAME="${PROJECT_NAME:-NeoVerseControlLatentDistill}"
RUN_TIME="${RUN_TIME:-$(timestamp_utc)}"
OUTPUT_DIR="${CAMERA_CACHE_DIR:-${OUTPUT_DIR:-${CODE_DIR}/outputs/${PROJECT_NAME}/camera_cache}}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/outputs/${PROJECT_NAME}/camera_cache_logs/${RUN_TIME}}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run.pid}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"

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

IFS=',' read -r -a raw_gpus <<< "${GPU_LIST}"
gpu_ids=()
for gpu in "${raw_gpus[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  if [[ -n "${gpu}" ]]; then
    gpu_ids+=("${gpu}")
  fi
done
if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
  die "GPU_LIST is empty"
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
log "num_shards=${#gpu_ids[@]}"
log "common_args=${common_args[*]}"
log "extra_args=$*"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "+ bash -n $0"
  log "+ ${ENV_PYTHON} -m py_compile experiments/build_spatialvid_camera_cache.py training/data/datasets/spatialvid.py"
else
  bash -n "$0"
  "${ENV_PYTHON}" -m py_compile experiments/build_spatialvid_camera_cache.py training/data/datasets/spatialvid.py
fi

pids=()
for shard_index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[shard_index]}"
  safe_gpu="${gpu//[^A-Za-z0-9_.-]/_}"
  shard_log="${LOG_DIR}/shard_${shard_index}_gpu_${safe_gpu}.log"
  shard_cmd=(
    "${ENV_PYTHON}" experiments/build_spatialvid_camera_cache.py
    "${common_args[@]}"
    --device cuda:0
    --num_shards "${#gpu_ids[@]}"
    --shard_index "${shard_index}"
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
