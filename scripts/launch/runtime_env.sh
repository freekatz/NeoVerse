#!/usr/bin/env bash

neoverse_gpu_list_from_count() {
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

neoverse_split_csv() {
  local value="${1:-}"
  local item
  local -a _neoverse_raw_items=()
  NEOVERSE_SPLIT_ITEMS=()

  IFS=',' read -r -a _neoverse_raw_items <<< "${value}"
  for item in "${_neoverse_raw_items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      NEOVERSE_SPLIT_ITEMS+=("${item}")
    fi
  done
}

neoverse_normalize_csv() {
  neoverse_split_csv "${1:-}"
  local item
  local normalized=""
  for item in "${NEOVERSE_SPLIT_ITEMS[@]}"; do
    if [[ -n "${normalized}" ]]; then
      normalized+=","
    fi
    normalized+="${item}"
  done
  printf '%s\n' "${normalized}"
}

neoverse_detect_torch_gpu_count() {
  local python_bin="${1:-}"
  if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
    return 1
  fi

  "${python_bin}" - <<'PY' 2>/dev/null
import torch

print(torch.cuda.device_count())
PY
}

neoverse_resolve_gpu_list() {
  local explicit_gpu_list="${1:-}"
  local cuda_visible_devices="${2:-}"
  local mlp_worker_gpu="${3:-}"
  local python_bin="${4:-}"
  local detected_count=""

  if [[ -n "${explicit_gpu_list}" ]]; then
    neoverse_normalize_csv "${explicit_gpu_list}"
    return 0
  fi

  if [[ -n "${cuda_visible_devices}" ]]; then
    neoverse_normalize_csv "${cuda_visible_devices}"
    return 0
  fi

  if [[ "${mlp_worker_gpu}" =~ ^[0-9]+$ && "${mlp_worker_gpu}" -gt 0 ]]; then
    neoverse_gpu_list_from_count "${mlp_worker_gpu}"
    return 0
  fi

  detected_count="$(neoverse_detect_torch_gpu_count "${python_bin}" || true)"
  if [[ "${detected_count}" =~ ^[0-9]+$ && "${detected_count}" -gt 0 ]]; then
    neoverse_gpu_list_from_count "${detected_count}"
    return 0
  fi

  printf '0\n'
}

neoverse_validate_non_negative_int() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer, got %s\n' "${name}" "${value}" >&2
    return 1
  fi
}
