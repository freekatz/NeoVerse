#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/launch/compare_reconstruction_context.sh
#   DATASET_INDEX=3 NUM_CONTEXT_VIEWS=20 bash scripts/launch/compare_reconstruction_context.sh
#   USE_CAMERA_ANNOTATIONS=true DATASET_INDEX=0 bash scripts/launch/compare_reconstruction_context.sh

timestamp_utc() {
  date -u "+%Y%m%d_%H%M%S"
}

log() {
  printf '[compare_reconstruction_context][%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_PATH="${VENV_PATH:-/root/vepfs/envs/neoverse}"
ENV_PYTHON="${ENV_PYTHON:-${VENV_PATH}/bin/python}"
CONFIG="${CONFIG:-configs/distill/control_latent.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reconstruction_context_compare/$(timestamp_utc)}"
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${CODE_DIR}/${OUTPUT_DIR}"
fi

DATASET_INDEX="${DATASET_INDEX:-0}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_CONTEXT_VIEWS="${NUM_CONTEXT_VIEWS:-20}"
FPS="${FPS:-15}"
DEVICE="${DEVICE:-cuda}"
ENABLE_VRAM_MANAGEMENT="${ENABLE_VRAM_MANAGEMENT:-0}"
PREFER_GT_TRAJECTORY="${PREFER_GT_TRAJECTORY:-true}"

[[ -d "${CODE_DIR}" ]] || die "CODE_DIR does not exist: ${CODE_DIR}"
[[ -x "${ENV_PYTHON}" ]] || die "ENV_PYTHON is not executable: ${ENV_PYTHON}"
[[ -f "${CODE_DIR}/${CONFIG}" || -f "${CONFIG}" ]] || die "Missing config: ${CONFIG}"

cd "${CODE_DIR}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cmd=(
  "${ENV_PYTHON}" tools/diagnostics/compare_reconstruction_context.py
  --config "${CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_index "${DATASET_INDEX}"
  --num_frames "${NUM_FRAMES}"
  --num_context_views "${NUM_CONTEXT_VIEWS}"
  --fps "${FPS}"
  --device "${DEVICE}"
  --prefer_gt_trajectory "${PREFER_GT_TRAJECTORY}"
)

if [[ -n "${HEIGHT:-}" ]]; then
  cmd+=(--height "${HEIGHT}")
fi
if [[ -n "${WIDTH:-}" ]]; then
  cmd+=(--width "${WIDTH}")
fi
if [[ -n "${SEED:-}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ -n "${USE_CAMERA_ANNOTATIONS:-}" ]]; then
  cmd+=(--use_camera_annotations "${USE_CAMERA_ANNOTATIONS}")
fi
if [[ "${ENABLE_VRAM_MANAGEMENT}" == "1" ]]; then
  cmd+=(--enable_vram_management)
fi

log "output_dir=${OUTPUT_DIR}"
log "dataset_index=${DATASET_INDEX}"
log "num_frames=${NUM_FRAMES}"
log "num_context_views=${NUM_CONTEXT_VIEWS}"
log "+ ${cmd[*]} $*"

"${cmd[@]}" "$@"
