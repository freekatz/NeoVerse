#!/usr/bin/env bash
set -euo pipefail

# Prepare SpatialVID-HQ archives for this repo's SpatialVID dataloader.
#
# The downloaded dataset stores group archives as:
#   SOURCE_ROOT/videos/group_0001.tar.gz -> SpatialVID/videos/group_0001/...
#   SOURCE_ROOT/annotations/group_0001.tar.gz -> SpatialVID/annotations/group_0001/...
#
# The local dataloader expects:
#   DEST_ROOT/data/train/SpatialVID_HQ_metadata.csv
#   DEST_ROOT/SpatialVid/HQ/videos/group_0001/...
#   DEST_ROOT/SpatialVid/HQ/annotations/group_0001/...
#
# Usage:
#   SOURCE_ROOT=/root/tos/cmh/datasets/SpatialVID-HQ \
#   DEST_ROOT=/root/vepfs/diffsynth-dev/papers/neoverse/code/data/SpatialVID_full \
#   bash scripts/data/prepare_spatialvid_hq.sh
#
# If one archive is known bad and you want to continue without that group:
#   SKIP_BAD_ARCHIVES=1 bash scripts/data/prepare_spatialvid_hq.sh

timestamp_utc() {
  date -u "+%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[prepare_spatialvid_hq][%s] %s\n' "$(timestamp_utc)" "$*"
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

SOURCE_ROOT="${SOURCE_ROOT:-/root/tos/cmh/datasets/SpatialVID-HQ}"
DEST_ROOT="${DEST_ROOT:-${CODE_DIR}/data/SpatialVID_full}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_DEPTHS="${INCLUDE_DEPTHS:-0}"
SKIP_BAD_ARCHIVES="${SKIP_BAD_ARCHIVES:-0}"
FILTER_METADATA_TO_EXTRACTED="${FILTER_METADATA_TO_EXTRACTED:-${SKIP_BAD_ARCHIVES}}"

[[ -d "${SOURCE_ROOT}" ]] || die "SOURCE_ROOT does not exist: ${SOURCE_ROOT}"
[[ -d "${SOURCE_ROOT}/videos" ]] || die "Missing source videos directory: ${SOURCE_ROOT}/videos"
[[ -d "${SOURCE_ROOT}/annotations" ]] || die "Missing source annotations directory: ${SOURCE_ROOT}/annotations"
[[ -f "${SOURCE_ROOT}/data/train/SpatialVID_HQ_metadata.csv" ]] || die "Missing metadata CSV under ${SOURCE_ROOT}/data/train"

DEST_HQ="${DEST_ROOT}/SpatialVid/HQ"
MARKER_DIR="${DEST_ROOT}/.extract_markers"
TMP_ROOT="${DEST_ROOT}/.extract_tmp"
FAILED_ARCHIVES_FILE="${DEST_ROOT}/failed_archives.txt"

run_cmd mkdir -p "${DEST_ROOT}/data" "${DEST_HQ}" "${MARKER_DIR}" "${TMP_ROOT}"
run_cmd cp -a "${SOURCE_ROOT}/data/." "${DEST_ROOT}/data/"
if [[ "${DRY_RUN}" != "1" ]]; then
  : > "${FAILED_ARCHIVES_FILE}"
fi

extract_kind() {
  local kind="$1"
  local archive
  local group
  local marker
  local temp_dir
  local temp_hq
  local extracted_group
  local final_group
  local archives=()

  shopt -s nullglob
  archives=("${SOURCE_ROOT}/${kind}"/group_*.tar.gz)
  shopt -u nullglob

  [[ "${#archives[@]}" -gt 0 ]] || die "No ${kind} archives found under ${SOURCE_ROOT}/${kind}"

  for archive in "${archives[@]}"; do
    group="$(basename "${archive}" .tar.gz)"
    marker="${MARKER_DIR}/${kind}_${group}.done"
    if [[ -f "${marker}" ]]; then
      log "skip ${kind}/${group}; marker exists"
      continue
    fi
    temp_dir="${TMP_ROOT}/${kind}_${group}.$$"
    temp_hq="${temp_dir}/hq"
    extracted_group="${temp_hq}/${kind}/${group}"
    final_group="${DEST_HQ}/${kind}/${group}"
    log "extract ${kind}/${group}"
    run_cmd rm -rf "${temp_dir}"
    run_cmd mkdir -p "${temp_hq}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      if ! tar -xzf "${archive}" --strip-components=1 -C "${temp_hq}"; then
        rm -rf "${temp_dir}"
        if [[ "${SKIP_BAD_ARCHIVES}" == "1" ]]; then
          log "skip ${kind}/${group}; failed to extract ${archive}"
          rm -rf "${final_group}"
          printf '%s/%s\t%s\n' "${kind}" "${group}" "${archive}" >> "${FAILED_ARCHIVES_FILE}"
          continue
        fi
        die "Failed to extract ${archive}. The archive is probably incomplete or corrupted."
      fi
      [[ -d "${extracted_group}" ]] || die "Archive ${archive} did not contain expected path: ${kind}/${group}"
      mkdir -p "$(dirname "${final_group}")"
      rm -rf "${final_group}"
      mv "${extracted_group}" "${final_group}"
      touch "${marker}"
      rm -rf "${temp_dir}"
    else
      log "+ tar -xzf ${archive} --strip-components=1 -C ${temp_hq}"
      log "+ mv ${extracted_group} ${final_group}"
      log "+ touch ${marker}"
    fi
  done
}

filter_metadata_to_extracted() {
  local metadata_path="${DEST_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
  local backup_path="${DEST_ROOT}/data/train/SpatialVID_HQ_metadata.full.csv"
  [[ -f "${metadata_path}" ]] || die "Missing metadata CSV: ${metadata_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "+ filter metadata to groups with extracted videos and annotations"
    return
  fi
  [[ -f "${backup_path}" ]] || cp -a "${metadata_path}" "${backup_path}"
  python - "${metadata_path}" "${backup_path}" "${MARKER_DIR}" <<'PY'
import os
import re
import sys

import pandas as pd

metadata_path, backup_path, marker_dir = sys.argv[1:4]

def marker_groups(prefix):
    groups = set()
    if not os.path.isdir(marker_dir):
        return groups
    pattern = re.compile(rf"^{re.escape(prefix)}_(group_\d{{4}})\.done$")
    for name in os.listdir(marker_dir):
        match = pattern.match(name)
        if match:
            groups.add(match.group(1))
    return groups

video_groups = marker_groups("videos")
annotation_groups = marker_groups("annotations")
available = video_groups & annotation_groups
if not available:
    raise SystemExit("No groups have both video and annotation extraction markers.")

df = pd.read_csv(backup_path)
group_ids = df["group id"].map(lambda value: f"group_{int(value):04d}")
filtered = df[group_ids.isin(available)].copy()
filtered.to_csv(metadata_path, index=False)
print(
    f"metadata filtered: rows {len(df)} -> {len(filtered)}, "
    f"groups {len(set(group_ids))} -> {len(available)}"
)
PY
}

extract_kind videos
extract_kind annotations

if [[ "${INCLUDE_DEPTHS}" == "1" ]]; then
  [[ -d "${SOURCE_ROOT}/depths" ]] || die "Missing source depths directory: ${SOURCE_ROOT}/depths"
  extract_kind depths
else
  log "skip depths; set INCLUDE_DEPTHS=1 only if a later pipeline needs them"
fi

if [[ "${FILTER_METADATA_TO_EXTRACTED}" == "1" ]]; then
  filter_metadata_to_extracted
fi

log "ready: DATA_ROOT=${DEST_ROOT}"
