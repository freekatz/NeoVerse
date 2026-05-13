#!/usr/bin/env bash

ERROR_MESSAGE=""
RESOLVED_GPU_LIST=""
RESOLVED_GPU_COUNT=0
RESOLVED_CUDA_VISIBLE_DEVICES=""
RESOLVED_LOCAL_VISIBLE_GPU_COUNT=0
SHARD_NNODES=1
SHARD_NODE_RANK=0
SHARD_GLOBAL_NUM_SHARDS=1
SHARD_GLOBAL_SHARD_OFFSET=0
DIST_NPROC_PER_NODE_VALUE=1
DIST_NNODES_VALUE=1
DIST_NODE_RANK_VALUE=0
DIST_IS_DISTRIBUTED=0
GLOBAL_NUM_PROCESSES=1
GPU_IDS=()

set_error() {
  ERROR_MESSAGE="$*"
  return 1
}

gpu_list_from_count() {
  local count="${1:-}"
  if [[ ! "${count}" =~ ^[0-9]+$ || "${count}" -le 0 ]]; then
    return 1
  fi

  local gpu_list=""
  local index
  for ((index = 0; index < count; index++)); do
    if [[ -n "${gpu_list}" ]]; then
      gpu_list+=","
    fi
    gpu_list+="${index}"
  done
  printf '%s\n' "${gpu_list}"
}

split_csv() {
  local value="${1:-}"
  local item
  local -a raw_items=()
  SPLIT_ITEMS=()

  IFS=',' read -r -a raw_items <<< "${value}"
  for item in "${raw_items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      SPLIT_ITEMS+=("${item}")
    fi
  done
}

normalize_csv() {
  split_csv "${1:-}"
  local item
  local normalized=""
  for item in "${SPLIT_ITEMS[@]}"; do
    if [[ -n "${normalized}" ]]; then
      normalized+=","
    fi
    normalized+="${item}"
  done
  printf '%s\n' "${normalized}"
}

detect_torch_gpu_count() {
  local python_bin="${1:-}"
  if [[ -z "${python_bin}" ]] || ! command -v "${python_bin}" >/dev/null 2>&1; then
    return 1
  fi

  "${python_bin}" - <<'PY' 2>/dev/null
import torch

print(torch.cuda.device_count())
PY
}

resolve_gpu_list() {
  local explicit_gpu_list="${1:-}"
  local cuda_visible_devices="${2:-}"
  local mlp_worker_gpu="${3:-}"
  local python_bin="${4:-}"
  local detected_count=""

  if [[ -n "${explicit_gpu_list}" ]]; then
    normalize_csv "${explicit_gpu_list}"
    return 0
  fi

  if [[ -n "${cuda_visible_devices}" ]]; then
    normalize_csv "${cuda_visible_devices}"
    return 0
  fi

  if [[ "${mlp_worker_gpu}" =~ ^[0-9]+$ && "${mlp_worker_gpu}" -gt 0 ]]; then
    gpu_list_from_count "${mlp_worker_gpu}"
    return 0
  fi

  detected_count="$(detect_torch_gpu_count "${python_bin}" || true)"
  if [[ "${detected_count}" =~ ^[0-9]+$ && "${detected_count}" -gt 0 ]]; then
    gpu_list_from_count "${detected_count}"
    return 0
  fi

  printf '0\n'
}

validate_non_negative_int() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer, got %s\n' "${name}" "${value}" >&2
    return 1
  fi
}

require_non_negative_int() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    set_error "${name} must be a non-negative integer, got ${value}"
  fi
}

resolve_local_gpus() {
  local cuda_visible_devices="${1:-}"
  local mlp_worker_gpu="${2:-}"
  local python_bin="${3:-python}"

  RESOLVED_GPU_LIST="$(resolve_gpu_list "" "${cuda_visible_devices}" "${mlp_worker_gpu}" "${python_bin}")"
  split_csv "${RESOLVED_GPU_LIST}"
  GPU_IDS=("${SPLIT_ITEMS[@]}")
  RESOLVED_GPU_COUNT="${#GPU_IDS[@]}"
}

resolve_cuda_visible_devices() {
  local cuda_visible_devices="${1:-}"
  local mlp_worker_gpu="${2:-}"
  local python_bin="${3:-python}"

  resolve_local_gpus "${cuda_visible_devices}" "${mlp_worker_gpu}" "${python_bin}" || return 1
  if [[ -n "${cuda_visible_devices}" ]]; then
    RESOLVED_CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"
  else
    RESOLVED_CUDA_VISIBLE_DEVICES="${RESOLVED_GPU_LIST}"
  fi
  RESOLVED_LOCAL_VISIBLE_GPU_COUNT="${RESOLVED_GPU_COUNT}"
}

resolve_shard_layout() {
  local cuda_visible_devices="${1:-}"
  local mlp_worker_gpu="${2:-}"
  local cache_nnodes="${3:-${MLP_WORKER_NUM:-1}}"
  local cache_node_rank="${4:-${MLP_ROLE_INDEX:-0}}"
  local cache_global_num_shards="${5:-}"
  local cache_global_shard_offset="${6:-}"
  local python_bin="${7:-python}"
  local max_global_shard_index=0

  resolve_local_gpus "${cuda_visible_devices}" "${mlp_worker_gpu}" "${python_bin}" || return 1
  require_non_negative_int "CACHE_NNODES" "${cache_nnodes}" || return 1
  require_non_negative_int "CACHE_NODE_RANK" "${cache_node_rank}" || return 1
  if [[ "${cache_nnodes}" -le 0 ]]; then
    set_error "CACHE_NNODES must be positive, got ${cache_nnodes}"
    return 1
  fi
  if [[ "${cache_node_rank}" -ge "${cache_nnodes}" ]]; then
    set_error "CACHE_NODE_RANK=${cache_node_rank} must be smaller than CACHE_NNODES=${cache_nnodes}"
    return 1
  fi

  if [[ -z "${cache_global_num_shards}" ]]; then
    cache_global_num_shards=$((RESOLVED_GPU_COUNT * cache_nnodes))
  fi
  if [[ -z "${cache_global_shard_offset}" ]]; then
    cache_global_shard_offset=$((cache_node_rank * RESOLVED_GPU_COUNT))
  fi

  require_non_negative_int "CACHE_GLOBAL_NUM_SHARDS" "${cache_global_num_shards}" || return 1
  require_non_negative_int "CACHE_GLOBAL_SHARD_OFFSET" "${cache_global_shard_offset}" || return 1
  if [[ "${cache_global_num_shards}" -le 0 ]]; then
    set_error "CACHE_GLOBAL_NUM_SHARDS must be positive, got ${cache_global_num_shards}"
    return 1
  fi

  max_global_shard_index=$((cache_global_shard_offset + RESOLVED_GPU_COUNT - 1))
  if [[ "${max_global_shard_index}" -ge "${cache_global_num_shards}" ]]; then
    set_error "Shard range ${cache_global_shard_offset}-${max_global_shard_index} exceeds CACHE_GLOBAL_NUM_SHARDS=${cache_global_num_shards}"
    return 1
  fi

  SHARD_NNODES="${cache_nnodes}"
  SHARD_NODE_RANK="${cache_node_rank}"
  SHARD_GLOBAL_NUM_SHARDS="${cache_global_num_shards}"
  SHARD_GLOBAL_SHARD_OFFSET="${cache_global_shard_offset}"
}

resolve_distributed_layout() {
  local use_distributed="${1:-auto}"
  local dist_nproc_per_node="${2:-${MLP_WORKER_GPU:-}}"
  local dist_nnodes="${3:-${MLP_WORKER_NUM:-1}}"
  local dist_node_rank="${4:-${MLP_ROLE_INDEX:-0}}"
  local local_visible_gpu_count="${5:-0}"

  case "${use_distributed}" in
    auto|1|0|true|false|yes|no|on|off) ;;
    *)
      set_error "Unsupported USE_DISTRIBUTED=${use_distributed}; expected auto/true/false"
      return 1
      ;;
  esac

  if [[ -z "${dist_nproc_per_node}" ]]; then
    if [[ "${use_distributed}" == "auto" && "${local_visible_gpu_count}" -gt 0 ]]; then
      dist_nproc_per_node="${local_visible_gpu_count}"
    else
      dist_nproc_per_node=1
    fi
  fi

  require_non_negative_int "DIST_NPROC_PER_NODE" "${dist_nproc_per_node}" || return 1
  require_non_negative_int "DIST_NNODES" "${dist_nnodes}" || return 1
  require_non_negative_int "DIST_NODE_RANK" "${dist_node_rank}" || return 1
  if [[ "${dist_nproc_per_node}" -le 0 || "${dist_nnodes}" -le 0 ]]; then
    set_error "Invalid distributed size: DIST_NPROC_PER_NODE=${dist_nproc_per_node}, DIST_NNODES=${dist_nnodes}"
    return 1
  fi
  if [[ "${dist_node_rank}" -ge "${dist_nnodes}" ]]; then
    set_error "DIST_NODE_RANK=${dist_node_rank} must be smaller than DIST_NNODES=${dist_nnodes}"
    return 1
  fi
  if [[ "${local_visible_gpu_count}" -gt 0 && "${dist_nproc_per_node}" -gt "${local_visible_gpu_count}" ]]; then
    set_error "DIST_NPROC_PER_NODE=${dist_nproc_per_node} exceeds visible GPU count ${local_visible_gpu_count} from CUDA_VISIBLE_DEVICES=${RESOLVED_CUDA_VISIBLE_DEVICES}. Set CUDA_VISIBLE_DEVICES or DIST_NPROC_PER_NODE consistently."
    return 1
  fi

  DIST_IS_DISTRIBUTED=0
  case "${use_distributed}" in
    1|true|yes|on)
      DIST_IS_DISTRIBUTED=1
      ;;
    auto)
      if [[ "${dist_nnodes}" != "1" || "${dist_nproc_per_node}" != "1" ]]; then
        DIST_IS_DISTRIBUTED=1
      fi
      ;;
  esac

  DIST_NPROC_PER_NODE_VALUE="${dist_nproc_per_node}"
  DIST_NNODES_VALUE="${dist_nnodes}"
  DIST_NODE_RANK_VALUE="${dist_node_rank}"
  GLOBAL_NUM_PROCESSES=$((dist_nproc_per_node * dist_nnodes))
}
