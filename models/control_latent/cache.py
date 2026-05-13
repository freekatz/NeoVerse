import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .camera import get_camera_condition_normalization_cfg


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def cache_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.detach().cpu().reshape(-1)[0]
            if value.dtype == torch.bool:
                return bool(value.item())
            if torch.is_floating_point(value):
                return float(value.item())
            return int(value.item())
        return tuple(value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return tuple(value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return cache_scalar(value[0])
    return value


FROZEN_CACHE_SIGNATURE_FIELDS = (
    "video_name",
    "image_name",
    "timestamp",
    "is_target",
    "scene_idx",
    "variant_id",
    "context_strategy",
    "trajectory_index",
    "target_trajectory_index",
    "camera_cache_path",
    "temporal_augmentation",
    "temporal_order",
    "temporal_trajectory_profile",
    "temporal_trajectory_type",
)


def frozen_cache_view_metadata(data):
    views = []
    for view in data:
        views.append({key: cache_scalar(view.get(key)) for key in FROZEN_CACHE_SIGNATURE_FIELDS})
    return views


def frozen_cache_dataset_index(data):
    if not data:
        return None
    idx = data[0].get("idx")
    if isinstance(idx, (list, tuple)) and len(idx) >= 1:
        return cache_scalar(idx[0])
    if isinstance(idx, np.ndarray) and idx.size >= 1:
        return cache_scalar(idx.reshape(-1)[0])
    return cache_scalar(idx)


def frozen_cache_signature(data, cfg):
    views = frozen_cache_view_metadata(data)
    cfg_signature = {
        "model_path": cfg.get("model_path", None),
        "reconstructor_path": cfg.get("reconstructor_path", None),
        "torch_dtype": cfg.get("torch_dtype", None),
        "height": cfg.get("height", None),
        "width": cfg.get("width", None),
        "num_views": cfg.get("num_views", None),
        "use_camera_annotations": cfg.get("use_camera_annotations", None),
        "temporal_augmentation": cfg.get("temporal_augmentation", None),
        "temporal_trajectory_profile": cfg.get("temporal_trajectory_profile", None),
        "temporal_variant_profile_weights": cfg.get("temporal_variant_profile_weights", None),
        "temporal_order": cfg.get("temporal_order", None),
        "temporal_max_condition_frames": cfg.get("temporal_max_condition_frames", None),
        "trajectories_per_clip": cfg.get("trajectories_per_clip", None),
        "camera_condition_normalization": get_camera_condition_normalization_cfg(cfg),
        "pipeline_kwargs": OmegaConf.to_container(cfg.pipeline_kwargs, resolve=True),
        "token": OmegaConf.to_container(cfg.token, resolve=True),
        "vae_tiled": cfg.get("vae_tiled", None),
        "tile_size": list(cfg.get("tile_size", [])),
        "tile_stride": list(cfg.get("tile_stride", [])),
    }
    payload = {"views": views, "config": cfg_signature}
    encoded = json.dumps(payload, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_float_dtype(cfg):
    dtype_name = cfg.get("frozen_cache_dtype", None)
    if dtype_name in {None, "none", "None"}:
        return None
    return getattr(torch, str(dtype_name))


def cache_to_cpu(cache, dtype=None):
    out = {}
    for key, value in cache.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if dtype is not None and torch.is_floating_point(value):
                value = value.to(dtype=dtype)
        out[key] = value
    return out


def cache_to_device(cache, device):
    out = {}
    for key, value in cache.items():
        out[key] = value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def save_frozen_cache(path, cache, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = cache_to_cpu(cache, dtype=cache_float_dtype(cfg))
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    sidecar = {
        key: payload.get(key)
        for key in (
            "cache_format_version",
            "cache_signature",
            "dataset_index",
            "signature_views",
        )
        if key in payload
    }
    if sidecar:
        json_path = str(Path(path).with_suffix(".json"))
        json_tmp_path = f"{json_path}.tmp.{os.getpid()}"
        with open(json_tmp_path, "w") as handle:
            json.dump(sidecar, handle, indent=2, default=json_default)
        os.replace(json_tmp_path, json_path)


def looks_like_frozen_cache(data):
    return isinstance(data, dict) and "teacher" in data and "tokens" in data and "output_grid" in data


def prepare_runtime_cache(cache, device):
    return cache_to_device(cache, device)


def cache_target_is_cpu(device) -> bool:
    return torch.device(device).type == "cpu"


def cache_to_target_device(cache, device, dtype=None):
    if cache_target_is_cpu(device):
        return cache_to_cpu(cache, dtype=dtype)
    return cache_to_device(cache, device)


