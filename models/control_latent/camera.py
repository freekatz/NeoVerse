import numpy as np
import torch
from omegaconf import OmegaConf

from .reconstructor_tokens import compose_views_from_list, pack_worldmirror_token_list


def extract_target_times(source_views, device):
    if isinstance(source_views, list):
        source_views = compose_views_from_list(source_views)
    timestamps = source_views.get("timestamp")
    if timestamps is None or not isinstance(timestamps, torch.Tensor):
        return None
    timestamps = timestamps.to(device=device, dtype=torch.float32)
    target_indices = target_trajectory_indices(source_views, device=device)
    if target_indices is not None and target_indices.shape == timestamps.shape:
        order = continuous_view_order_indices(source_views, device=device)
        if order is not None and order.shape == timestamps.shape:
            timestamps = torch.gather(timestamps, dim=1, index=order)
        return torch.gather(timestamps, dim=1, index=target_indices)
    return torch.sort(timestamps, dim=1).values


def extract_source_times(source_views, device, context_only: bool = True):
    if isinstance(source_views, list):
        source_views = compose_views_from_list(source_views)
    timestamps = source_views.get("timestamp")
    if timestamps is None or not isinstance(timestamps, torch.Tensor):
        return None
    source_views = _ordered_views(source_views, device=device)
    timestamps = source_views["timestamp"].to(device=device, dtype=torch.float32)
    if not context_only or "is_target" not in source_views or not isinstance(source_views["is_target"], torch.Tensor):
        return timestamps
    context_indices = _context_indices_from_ordered(source_views, device=device)
    if context_indices is None:
        return timestamps
    return _gather_view_tensor(timestamps, context_indices)


def _as_tensor_2d(value, device, dtype=None):
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)
    elif isinstance(value, np.ndarray):
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
    elif isinstance(value, (list, tuple)):
        try:
            tensor = torch.as_tensor(value, device=device, dtype=dtype)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _gather_view_tensor(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(tensor.shape[0], -1)
    index = indices.to(device=tensor.device, dtype=torch.long)
    index = index.reshape(index.shape[0], index.shape[1], *([1] * (tensor.ndim - 2)))
    index = index.expand(-1, -1, *tensor.shape[2:])
    return torch.gather(tensor, dim=1, index=index)


def _ordered_views(source_views, device):
    order = continuous_view_order_indices(source_views, device=device)
    if order is None:
        return source_views
    total_views = order.shape[1]
    return {
        key: _gather_view_tensor(value, order)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == total_views
        else value
        for key, value in source_views.items()
    }


def _context_indices_from_ordered(source_views, device):
    if "is_target" not in source_views or not isinstance(source_views["is_target"], torch.Tensor):
        return None
    keep = ~source_views["is_target"].to(device=device).bool()
    if keep.ndim != 2:
        return None
    counts = keep.sum(dim=1)
    if not bool(torch.all(counts == counts[0]).item()):
        raise ValueError("Expected the same number of context views in every batch item.")
    return torch.stack([torch.nonzero(mask, as_tuple=False).flatten() for mask in keep], dim=0).to(device=device)


def continuous_view_order_indices(source_views, device):
    if isinstance(source_views, list):
        source_views = compose_views_from_list(source_views)
    trajectory_index = _as_tensor_2d(source_views.get("trajectory_index"), device=device, dtype=torch.long)
    if trajectory_index is None:
        return None
    return torch.argsort(trajectory_index, dim=1)


def target_trajectory_indices(source_views, device):
    if isinstance(source_views, list):
        source_views = compose_views_from_list(source_views)
    target_index = _as_tensor_2d(source_views.get("target_trajectory_index"), device=device, dtype=torch.long)
    if target_index is None:
        return None
    order = continuous_view_order_indices(source_views, device=device)
    if order is not None and order.shape == target_index.shape:
        target_index = torch.gather(target_index, dim=1, index=order)
    return target_index


def gather_source_cameras_from_target(source_views, target_poses, target_intrs, context_only: bool = True):
    if isinstance(source_views, list):
        source_views = compose_views_from_list(source_views)
    timestamps = source_views.get("timestamp")
    if not isinstance(timestamps, torch.Tensor):
        return None, None

    device = target_poses.device if target_poses is not None else timestamps.device
    source_views = _ordered_views(source_views, device=device)
    pose_key = "camera_poses" if "camera_poses" in source_views else "extrinsics"
    intr_key = "camera_intrs" if "camera_intrs" in source_views else "intrinsics"
    if pose_key in source_views and intr_key in source_views:
        source_poses = source_views[pose_key]
        source_intrs = source_views[intr_key]
        if isinstance(source_poses, torch.Tensor) and isinstance(source_intrs, torch.Tensor):
            source_poses = source_poses.to(device=device)
            source_intrs = source_intrs.to(device=device)
            context_indices = _context_indices_from_ordered(source_views, device=device) if context_only else None
            if context_indices is not None:
                source_poses = _gather_view_tensor(source_poses, context_indices)
                source_intrs = _gather_view_tensor(source_intrs, context_indices)
            return source_poses, source_intrs

    if target_poses is None or target_intrs is None:
        return None, None

    timestamps = source_views["timestamp"].to(device=device, dtype=torch.float32)
    context_indices = _context_indices_from_ordered(source_views, device=device) if context_only else None
    source_times = _gather_view_tensor(timestamps, context_indices) if context_indices is not None else timestamps
    target_times = extract_target_times(source_views, device=device)
    if target_times is None:
        target_times = torch.sort(timestamps, dim=1).values

    gather_indices = []
    for b_idx in range(source_times.shape[0]):
        distance = (target_times[b_idx, None, :] - source_times[b_idx, :, None]).abs()
        gather_indices.append(distance.argmin(dim=-1))
    gather_idx = torch.stack(gather_indices, dim=0).to(device=device)

    pose_idx = gather_idx[:, :, None, None].expand(-1, -1, target_poses.shape[-2], target_poses.shape[-1])
    intr_idx = gather_idx[:, :, None, None].expand(-1, -1, target_intrs.shape[-2], target_intrs.shape[-1])
    source_poses = torch.gather(target_poses, dim=1, index=pose_idx)
    source_intrs = torch.gather(target_intrs, dim=1, index=intr_idx)
    return source_poses, source_intrs


def add_time_metrics(metrics, target_times, source_times, device):
    if target_times is not None:
        target_times_f = target_times.detach().float()
        metrics["time/target_frames"] = torch.tensor(target_times_f.shape[1], device=device, dtype=torch.float32)
        metrics["time/target_start_s"] = target_times_f[:, 0].mean()
        metrics["time/target_end_s"] = target_times_f[:, -1].mean()
        if target_times_f.shape[1] > 1:
            metrics["time/target_dt_s"] = (target_times_f[:, 1:] - target_times_f[:, :-1]).mean()
    if source_times is not None:
        source_times_f = source_times.detach().float()
        metrics["time/source_frames"] = torch.tensor(source_times_f.shape[1], device=device, dtype=torch.float32)
        metrics["time/source_start_s"] = source_times_f[:, 0].mean()
        metrics["time/source_end_s"] = source_times_f[:, -1].mean()
        if target_times is not None:
            metrics["time/source_target_start_delta_s"] = (source_times_f[:, 0] - target_times.detach().float()[:, 0]).abs().mean()


def compact_distill_log_metrics(metrics, prefix, lr=None):
    key_map = {
        "loss": "loss",
        "condition/l1": "l1",
        "condition/cosine": "cosine",
    }
    out = {}
    for source_key, target_key in key_map.items():
        if source_key in metrics:
            out[f"{prefix}/{target_key}"] = metrics[source_key]
    if lr is not None:
        out[f"{prefix}/lr"] = lr
    return out


def detach_optional_tensor(x):
    return x.detach() if isinstance(x, torch.Tensor) else x


def token_pack_from_preprocess(inputs, cfg):
    if not bool(cfg.token.context_only):
        return None
    token_list = inputs.get("reconstructor_context_token_list")
    patch_start_idx = inputs.get("reconstructor_context_patch_start_idx")
    if token_list is None or patch_start_idx is None:
        return None

    expected_layers = tuple(int(layer) for layer in cfg.token.layer_indices)
    actual_layers = tuple(int(layer) for layer in inputs.get("reconstructor_token_layers", ()))
    if actual_layers != expected_layers or len(token_list) != len(expected_layers):
        return None

    source_views = inputs.get("source_views")
    if isinstance(source_views, list):
        source_views = compose_views_from_list(source_views)
    if not isinstance(source_views, dict) or "img" not in source_views:
        return None
    imgs = source_views["img"]
    if not isinstance(imgs, torch.Tensor) or imgs.ndim < 5:
        return None
    if "is_target" in source_views and isinstance(source_views["is_target"], torch.Tensor):
        context_count = int((~source_views["is_target"].bool())[0].sum().item())
        if int(token_list[0].shape[1]) != context_count:
            return None

    return pack_worldmirror_token_list(
        token_list,
        patch_start_idx=int(patch_start_idx),
        height=int(imgs.shape[-2]),
        width=int(imgs.shape[-1]),
        patch_size=int(inputs.get("reconstructor_patch_size", 14)),
        layer_indices=expected_layers,
        include_camera_token=cfg.token.include_camera_token,
    )


def get_camera_condition_normalization_cfg(cfg):
    default = {
        "enabled": False,
        "normalize_poses": True,
        "normalize_intrinsics": True,
        "normalize_plucker": True,
        "min_translation_scale": 1.0,
    }
    value = cfg.get("camera_condition_normalization", None)
    if value is None:
        return default
    if isinstance(value, bool):
        out = dict(default)
        out["enabled"] = value
        return out
    value = OmegaConf.to_container(value, resolve=True) if not isinstance(value, dict) else value
    out = dict(default)
    out.update(value)
    return out


def _homogeneous_poses(poses: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if poses.shape[-2:] == (4, 4):
        return poses, False
    if poses.shape[-2:] != (3, 4):
        raise ValueError(f"Expected camera poses with shape (..., 4, 4) or (..., 3, 4), got {tuple(poses.shape)}")
    row = poses.new_tensor([0, 0, 0, 1]).reshape(*([1] * (poses.ndim - 2)), 1, 4)
    row = row.expand(*poses.shape[:-2], 1, 4)
    return torch.cat([poses, row], dim=-2), True


def _camera_normalization_params(reference_poses: torch.Tensor | None, min_translation_scale: float):
    if reference_poses is None:
        return None, None
    ref_h, _ = _homogeneous_poses(reference_poses.float())
    anchor = ref_h[:, :1]
    anchor_inv = torch.linalg.inv(anchor)
    rel = anchor_inv @ ref_h
    centers = rel[..., :3, 3]
    scale = centers.detach().norm(dim=-1).amax(dim=1, keepdim=True)
    scale = scale.clamp_min(float(min_translation_scale))
    return anchor_inv, scale


def _normalize_poses_to_first(poses: torch.Tensor | None, anchor_inv: torch.Tensor | None, scale: torch.Tensor | None):
    if poses is None or anchor_inv is None or scale is None:
        return poses
    dtype = poses.dtype
    poses_h, squeezed = _homogeneous_poses(poses.float())
    rel = anchor_inv.to(device=poses_h.device, dtype=poses_h.dtype) @ poses_h
    rel = rel.clone()
    rel[..., :3, 3] = rel[..., :3, 3] / scale.to(device=poses_h.device, dtype=poses_h.dtype)[..., None]
    rel = rel.to(dtype=dtype)
    return rel[..., :3, :] if squeezed else rel


def _normalize_intrinsics(intrs: torch.Tensor | None, height: int, width: int):
    if intrs is None:
        return intrs
    out = intrs.clone()
    out[..., 0, :] = out[..., 0, :] / float(width)
    out[..., 1, :] = out[..., 1, :] / float(height)
    return out


def _normalize_plucker_to_first(
    plucker: torch.Tensor | None,
    reference_poses: torch.Tensor | None,
    anchor_inv: torch.Tensor | None,
    scale: torch.Tensor | None,
):
    if plucker is None or reference_poses is None or anchor_inv is None or scale is None:
        return plucker
    if plucker.shape[2] != 6:
        raise ValueError(f"Expected target_plucker with shape B x F x 6 x H x W, got {tuple(plucker.shape)}")
    dtype = plucker.dtype
    ref_h, _ = _homogeneous_poses(reference_poses.float())
    first = ref_h[:, 0].to(device=plucker.device, dtype=torch.float32)
    world_to_first = anchor_inv[:, 0, :3, :3].to(device=plucker.device, dtype=torch.float32)
    first_origin = first[:, :3, 3]
    rays_d = plucker[:, :, :3].float()
    moment = plucker[:, :, 3:].float()
    origin_shift = first_origin[:, None, :, None, None].expand_as(rays_d)
    moment = moment - torch.cross(origin_shift, rays_d, dim=2)
    rays_d = torch.einsum("bij,bfjhw->bfihw", world_to_first, rays_d)
    moment = torch.einsum("bij,bfjhw->bfihw", world_to_first, moment)
    moment = moment / scale.to(device=plucker.device, dtype=torch.float32)[:, :, None, None, None]
    return torch.cat([rays_d, moment], dim=2).to(dtype=dtype)


def normalize_camera_condition_cache(cache: dict, cfg):
    norm_cfg = get_camera_condition_normalization_cfg(cfg)
    if not bool(norm_cfg["enabled"]):
        return cache
    reference_poses = cache.get("target_poses") if cache.get("target_poses") is not None else cache.get("source_poses")
    anchor_inv, scale = _camera_normalization_params(
        reference_poses,
        min_translation_scale=float(norm_cfg["min_translation_scale"]),
    )
    if anchor_inv is None:
        return cache
    cache = dict(cache)
    if bool(norm_cfg["normalize_poses"]):
        cache["target_poses"] = _normalize_poses_to_first(cache.get("target_poses"), anchor_inv, scale)
        cache["source_poses"] = _normalize_poses_to_first(cache.get("source_poses"), anchor_inv, scale)
    if bool(norm_cfg["normalize_intrinsics"]):
        cache["target_intrs"] = _normalize_intrinsics(cache.get("target_intrs"), int(cfg.height), int(cfg.width))
        cache["source_intrs"] = _normalize_intrinsics(cache.get("source_intrs"), int(cfg.height), int(cfg.width))
    if bool(norm_cfg["normalize_plucker"]):
        cache["target_plucker"] = _normalize_plucker_to_first(cache.get("target_plucker"), reference_poses, anchor_inv, scale)
    return cache
