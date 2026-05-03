#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${CODE_DIR}"

python scripts/download/dl3dv_10k/download_dl3dv.py \
  --resolution "${DL3DV_RESOLUTION:-480P}" \
  --file-type "${DL3DV_FILE_TYPE:-images+poses}" \
  --subset "${DL3DV_SUBSET:-1K}" \
  --out-dir "${DL3DV_OUT_DIR:-data/DL3DV-10K}" \
  --artifacts-dir "${DL3DV_ARTIFACTS_DIR:-outputs/download_dl3dv}" \
  --verify-zip \
  "$@"
