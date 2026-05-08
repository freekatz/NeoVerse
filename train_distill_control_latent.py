import argparse
import datetime
import hashlib
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from diffsynth.models.student_adapters import build_student_adapter
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from hooks.extract_vggt_tokens import compose_views_from_list, extract_vggt_tokens, pack_worldmirror_token_list
from training.data.datasets.spatialvid import SpatialVID


NULL_CONFIG_STRINGS = {"", "none", "None", "null", "Null"}


def is_null_config_value(value):
    return value is None or (isinstance(value, str) and value.strip() in NULL_CONFIG_STRINGS)


def resolved_config_value(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def config_value(cfg, key, default=None):
    return resolved_config_value(cfg.get(key, default))


def none_if_null(value):
    value = resolved_config_value(value)
    return None if is_null_config_value(value) else value


def config_bool(cfg, key, default=False):
    value = config_value(cfg, key, default)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def config_int(cfg, key, default):
    value = config_value(cfg, key, default)
    if is_null_config_value(value):
        value = default
    return int(value)


def config_optional_int(cfg, key, default=None):
    value = config_value(cfg, key, default)
    if is_null_config_value(value):
        return default
    return int(value)


def normalize_filter_values(value):
    value = resolved_config_value(value)
    if is_null_config_value(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if is_null_config_value(text):
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        items = [item.strip().strip("'\"") for item in text.split(",")]
        return [item for item in items if item] or None
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            normalized = normalize_filter_values(item)
            if normalized is None:
                continue
            if isinstance(normalized, list):
                items.extend(normalized)
            else:
                items.append(normalized)
        return items or None
    return [str(value)]


def build_spatialvid_dataset(cfg):
    return SpatialVID(
        split=None,
        ROOT=str(config_value(cfg, "data_root", "data/SpatialVID")),
        video_ids=normalize_filter_values(config_value(cfg, "video_ids", None)),
        video_paths=normalize_filter_values(config_value(cfg, "video_paths", None)),
        use_camera_annotations=config_bool(cfg, "use_camera_annotations", False),
        continuous_target_frames=config_bool(cfg, "continuous_target_frames", True),
        force_first_context=config_bool(cfg, "force_first_context", True),
        timestamp_unit=str(config_value(cfg, "timestamp_unit", "seconds")),
        temporal_augmentation=config_bool(cfg, "temporal_augmentation", False),
        temporal_trajectory_profile=str(config_value(cfg, "temporal_trajectory_profile", "forward_pause")),
        temporal_order=str(config_value(cfg, "temporal_order", "trajectory")),
        temporal_max_condition_frames=config_int(cfg, "temporal_max_condition_frames", 8),
        context_sampling_strategy=str(config_value(cfg, "context_sampling_strategy", "uniform")),
        context_sampling_weights=config_value(cfg, "context_sampling_weights", None),
        variants_per_scene=config_int(cfg, "variants_per_scene", 1),
        trajectories_per_clip=config_optional_int(cfg, "trajectories_per_clip", None),
        temporal_variant_profile_weights=none_if_null(config_value(cfg, "temporal_variant_profile_weights", None)),
        fixed_clips_per_scene=config_int(cfg, "fixed_clips_per_scene", 0),
        camera_cache_dir=none_if_null(config_value(cfg, "camera_cache_dir", None)),
        camera_cache_required=config_bool(cfg, "camera_cache_required", False),
        min_interval=1,
        max_interval=1,
        height=config_int(cfg, "height", 336),
        width=config_int(cfg, "width", 560),
        num_views=config_int(cfg, "num_views", 81),
        min_num_context_views=config_int(cfg, "min_num_context_views", 10),
        max_num_context_views=config_int(cfg, "max_num_context_views", 20),
        seed=config_int(cfg, "dataset_seed", config_int(cfg, "seed", 0)),
    )


def control_key(layer):
    return f"layer_{int(layer)}"


def freeze_module(module: torch.nn.Module):
    module.eval()
    module.requires_grad_(False)


def unwrap_vram_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def build_adapter(pipe: WanVideoNeoVersePipeline, cfg):
    adapter_cfg = OmegaConf.to_container(cfg.adapter, resolve=True)
    adapter_type = adapter_cfg.pop("type")
    use_text_context = bool(adapter_cfg.pop("use_text_context", False))
    control_layers = tuple(pipe.control_branch.control_layers)
    output_dim = int(cfg.get("control_dim", unwrap_vram_module(pipe.control_branch.control_patch_embedding).out_channels))
    common = {
        "output_dim": output_dim,
        "control_layers": control_layers,
        "text_context_dim": output_dim if use_text_context else None,
    }
    if adapter_type == "conv":
        for cross_only_key in (
            "token_dim",
            "num_heads",
            "num_blocks",
            "source_pool_hw",
            "max_source_tokens",
            "query_chunk_size",
            "use_rope",
            "max_token_groups",
            "use_group_embedding",
            "use_local_grid",
            "use_time_film",
            "time_film_dim",
            "time_position_mode",
            "rerope_interval",
            "post_num_res_blocks",
        ):
            adapter_cfg.pop(cross_only_key, None)
        if adapter_cfg.get("input_channels") is None:
            token_dim = int(cfg.token.token_dim)
            token_groups = int(cfg.token.token_groups)
            adapter_cfg["input_channels"] = token_dim * token_groups
        return build_student_adapter(adapter_type, **common, **adapter_cfg)
    if adapter_type in {"cross_attention_rope", "cross_attn_rope"}:
        for conv_only_key in ("input_channels", "num_res_blocks", "dropout", "condition_time_dim"):
            adapter_cfg.pop(conv_only_key, None)
        if adapter_cfg.get("token_dim") is None:
            adapter_cfg["token_dim"] = int(cfg.token.token_dim)
        if adapter_cfg.get("max_token_groups") is None:
            adapter_cfg["max_token_groups"] = max(int(cfg.token.token_groups), 8)
        return build_student_adapter(adapter_type, **common, **adapter_cfg)
    raise ValueError(f"Unsupported adapter type: {adapter_type}")


def prepare_control_state(
    pipe: WanVideoNeoVersePipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
):
    dit = pipe.dit
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep_for_emb = torch.concat(
            [
                torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep_for_emb).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)
    x, grid = dit.patchify(latents, control_camera_latents_input)
    frames, height, width = grid
    freqs = torch.cat(
        [
            dit.freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            dit.freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            dit.freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, 1, -1).to(x.device)
    return x, tuple(int(v) for v in grid), context, t, t_mod, freqs


def conv3d_output_size(length: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def condition_output_grid(control_branch, target_rgb_latents: torch.Tensor) -> tuple[int, int, int]:
    _, _, frames, height, width = target_rgb_latents.shape
    conv = unwrap_vram_module(control_branch.control_patch_embedding)
    kernel = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation
    return (
        conv3d_output_size(frames, kernel[0], stride[0], padding[0], dilation[0]),
        conv3d_output_size(height, kernel[1], stride[1], padding[1], dilation[1]),
        conv3d_output_size(width, kernel[2], stride[2], padding[2], dilation[2]),
    )


def make_teacher_dict(pipe: WanVideoNeoVersePipeline, hints) -> OrderedDict:
    return OrderedDict((control_key(layer), hint.detach()) for layer, hint in zip(pipe.control_branch.control_layers, hints))


def get_layer_weight(weights, layer_key: str) -> float:
    if weights is None:
        return 1.0
    if layer_key in weights:
        return float(weights[layer_key])
    raw = layer_key.replace("layer_", "")
    return float(weights.get(raw, 1.0))


def latent_stats(x: torch.Tensor):
    xf = x.detach().float()
    return {
        "mean": xf.mean(),
        "std": xf.std(unbiased=False),
        "min": xf.amin(),
        "max": xf.amax(),
    }


def compute_distill_loss(student, teacher, cfg):
    if isinstance(student, torch.Tensor):
        student = OrderedDict(condition=student)
    if isinstance(teacher, torch.Tensor):
        teacher = OrderedDict(condition=teacher)
    total = None
    metrics = {}
    for key, teacher_lat in teacher.items():
        student_lat = student[key]
        teacher_lat = teacher_lat.detach()
        diff = student_lat.float() - teacher_lat.float()
        l1 = diff.abs().mean()
        l2 = diff.square().mean()
        cosine = F.cosine_similarity(student_lat.float().flatten(1), teacher_lat.float().flatten(1), dim=1).mean()
        stat = (
            (student_lat.float().mean() - teacher_lat.float().mean()).abs()
            + (student_lat.float().std(unbiased=False) - teacher_lat.float().std(unbiased=False)).abs()
        )
        layer_loss = (
            float(cfg.loss.l1_weight) * l1
            + float(cfg.loss.l2_weight) * l2
            + float(cfg.loss.cosine_weight) * (1 - cosine)
            + float(cfg.loss.stat_weight) * stat
        )
        layer_loss = layer_loss * get_layer_weight(cfg.loss.layer_weights, key)
        total = layer_loss if total is None else total + layer_loss
        metrics[f"{key}/l1"] = l1.detach()
        metrics[f"{key}/l2"] = l2.detach()
        metrics[f"{key}/cosine"] = cosine.detach()
        metrics[f"{key}/stat"] = stat.detach()
        for name, value in latent_stats(teacher_lat).items():
            metrics[f"{key}/teacher_{name}"] = value
        for name, value in latent_stats(student_lat).items():
            metrics[f"{key}/student_{name}"] = value
    metrics["loss"] = total.detach()
    return total, metrics


def normalize_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float()
    x = x - x.amin()
    x = x / x.amax().clamp_min(1e-6)
    return (x.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def save_heatmaps(output_dir: str, step: int, teacher, student, output_grid):
    os.makedirs(output_dir, exist_ok=True)
    key = next(iter(teacher.keys()))
    frames, height, width = output_grid
    teacher_map = teacher[key][0].float().reshape(frames, height, width, -1).mean(dim=-1)
    student_map = student[key][0].float().reshape(frames, height, width, -1).mean(dim=-1)
    error_map = (student_map - teacher_map).abs()
    frame = frames // 2
    for name, tensor in {
        "teacher": teacher_map[frame],
        "student": student_map[frame],
        "abs_error": error_map[frame],
    }.items():
        Image.fromarray(normalize_image(tensor)).save(os.path.join(output_dir, f"step_{step:08d}_{key}_{name}.png"))


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
    timestamps = timestamps.to(device=device, dtype=torch.float32)
    if not context_only or "is_target" not in source_views or not isinstance(source_views["is_target"], torch.Tensor):
        return timestamps
    keep = ~source_views["is_target"].bool()
    if keep.ndim != 2:
        return timestamps
    # SpatialVID and ExampleVideo return context views first, followed by target views.
    count = int(keep[0].sum().item())
    return timestamps[:, :count]


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


target_view_order_indices = continuous_view_order_indices


def _context_count(source_views, fallback_count):
    if "is_target" not in source_views or not isinstance(source_views["is_target"], torch.Tensor):
        return fallback_count
    keep = ~source_views["is_target"].bool()
    if keep.ndim != 2:
        return fallback_count
    return int(keep[0].sum().item())


def gather_source_cameras_from_target(source_views, target_poses, target_intrs, context_only: bool = True):
    if isinstance(source_views, list):
        source_views = compose_views_from_list(source_views)
    timestamps = source_views.get("timestamp")
    if not isinstance(timestamps, torch.Tensor):
        return None, None

    device = target_poses.device if target_poses is not None else timestamps.device
    pose_key = "camera_poses" if "camera_poses" in source_views else "extrinsics"
    intr_key = "camera_intrs" if "camera_intrs" in source_views else "intrinsics"
    if pose_key in source_views and intr_key in source_views:
        source_poses = source_views[pose_key]
        source_intrs = source_views[intr_key]
        if isinstance(source_poses, torch.Tensor) and isinstance(source_intrs, torch.Tensor):
            source_poses = source_poses.to(device=device)
            source_intrs = source_intrs.to(device=device)
            if context_only and "is_target" in source_views and isinstance(source_views["is_target"], torch.Tensor):
                keep = ~source_views["is_target"].bool()
                count = int(keep[0].sum().item()) if keep.ndim == 2 else source_poses.shape[1]
                source_poses = source_poses[:, :count]
                source_intrs = source_intrs[:, :count]
            return source_poses, source_intrs

    if target_poses is None or target_intrs is None:
        return None, None

    timestamps = timestamps.to(device=device, dtype=torch.float32)
    if context_only and "is_target" in source_views and isinstance(source_views["is_target"], torch.Tensor):
        count = _context_count(source_views, timestamps.shape[1])
        source_times = timestamps[:, :count]
    else:
        count = timestamps.shape[1]
        source_times = timestamps

    sorted_times, _ = torch.sort(timestamps, dim=1)
    gather_indices = []
    for b_idx in range(timestamps.shape[0]):
        distance = (sorted_times[b_idx, None, :] - source_times[b_idx, :, None]).abs()
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


def frozen_cache_signature(data, cfg):
    views = []
    for view in data:
        views.append(
            {
                "video_name": cache_scalar(view.get("video_name")),
                "image_name": cache_scalar(view.get("image_name")),
                "timestamp": cache_scalar(view.get("timestamp")),
                "is_target": cache_scalar(view.get("is_target")),
                "scene_idx": cache_scalar(view.get("scene_idx")),
                "variant_id": cache_scalar(view.get("variant_id")),
                "context_strategy": cache_scalar(view.get("context_strategy")),
                "trajectory_index": cache_scalar(view.get("trajectory_index")),
                "target_trajectory_index": cache_scalar(view.get("target_trajectory_index")),
                "camera_cache_path": cache_scalar(view.get("camera_cache_path")),
                "temporal_augmentation": cache_scalar(view.get("temporal_augmentation")),
                "temporal_order": cache_scalar(view.get("temporal_order")),
                "temporal_trajectory_profile": cache_scalar(view.get("temporal_trajectory_profile")),
                "temporal_trajectory_type": cache_scalar(view.get("temporal_trajectory_type")),
            }
        )
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


def frozen_cache_batch_key(cache):
    key = []
    for name in sorted(cache.keys()):
        value = cache[name]
        if isinstance(value, torch.Tensor):
            key.append((name, tuple(value.shape[1:]), str(value.dtype)))
        else:
            key.append((name, json.dumps(value, sort_keys=True, default=json_default)))
    return tuple(key)


def merge_frozen_cache_entries(entries):
    if len(entries) == 1:
        return entries[0]
    merged = {}
    first = entries[0]
    for name, value in first.items():
        if isinstance(value, torch.Tensor):
            merged[name] = torch.cat([entry[name] for entry in entries], dim=0)
        else:
            merged[name] = value
    return merged


def batch_preloaded_frozen_cache(caches, batch_size: int):
    if batch_size <= 1:
        return caches
    buckets = OrderedDict()
    for cache in caches:
        buckets.setdefault(frozen_cache_batch_key(cache), []).append(cache)
    batches = []
    for entries in buckets.values():
        for start in range(0, len(entries), batch_size):
            batches.append(merge_frozen_cache_entries(entries[start : start + batch_size]))
    return batches


def frozen_cache_split_score(path, seed: int) -> float:
    name = Path(path).name
    digest = hashlib.sha256(f"{int(seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def split_frozen_cache_paths(paths, eval_ratio: float, seed: int, mode: str = "hash"):
    mode = str(mode or "hash").strip().lower()
    if mode != "hash":
        raise ValueError(f"Unsupported frozen_cache_split_mode={mode!r}; expected 'hash'.")
    eval_ratio = float(eval_ratio)
    if eval_ratio < 0.0 or eval_ratio >= 1.0:
        raise ValueError(f"frozen_cache_eval_ratio must be in [0, 1), got {eval_ratio}.")
    paths = sorted(str(path) for path in paths)
    if eval_ratio <= 0.0 or len(paths) <= 1:
        return paths, []

    scored = [(frozen_cache_split_score(path, seed), path) for path in paths]
    train_paths = [path for score, path in scored if score >= eval_ratio]
    eval_paths = [path for score, path in scored if score < eval_ratio]
    if len(eval_paths) == 0:
        score, path = min(scored, key=lambda item: item[0])
        eval_paths = [path]
        train_paths = [candidate for _, candidate in scored if candidate != path]
    elif len(train_paths) == 0:
        score, path = max(scored, key=lambda item: item[0])
        train_paths = [path]
        eval_paths = [candidate for _, candidate in scored if candidate != path]
    return sorted(train_paths), sorted(eval_paths)


class FrozenForwardCacheDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir,
        pattern="*.pt",
        split="all",
        eval_ratio=0.05,
        split_seed=0,
        split_mode="hash",
        allow_empty=False,
    ):
        self.cache_dir = str(cache_dir)
        self.pattern = str(pattern or "*.pt")
        self.split = str(split or "all").strip().lower()
        self.eval_ratio = float(eval_ratio)
        self.split_seed = int(split_seed)
        self.split_mode = str(split_mode or "hash")
        root = Path(self.cache_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Frozen cache directory does not exist: {self.cache_dir}")
        all_paths = sorted(str(path) for path in root.glob(self.pattern) if path.is_file())
        if len(all_paths) == 0:
            raise FileNotFoundError(f"No frozen cache files matched {self.pattern!r} in {self.cache_dir}")
        train_paths, eval_paths = split_frozen_cache_paths(
            all_paths,
            eval_ratio=self.eval_ratio,
            seed=self.split_seed,
            mode=self.split_mode,
        )
        if self.split == "all":
            self.paths = all_paths
        elif self.split == "train":
            self.paths = train_paths
        elif self.split == "eval":
            self.paths = eval_paths
        else:
            raise ValueError(f"frozen_cache_split must be one of train/eval/all, got {self.split!r}.")
        self.total_paths = len(all_paths)
        self.split_counts = {"all": len(all_paths), "train": len(train_paths), "eval": len(eval_paths)}
        if len(self.paths) == 0 and not allow_empty:
            raise FileNotFoundError(
                f"No frozen cache files selected for split={self.split!r} from {self.cache_dir} "
                f"(total={len(all_paths)}, eval_ratio={self.eval_ratio})."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[int(idx)]
        cache = torch.load(path, map_location="cpu")
        if not looks_like_frozen_cache(cache):
            raise ValueError(f"Invalid frozen forward cache file: {path}")
        cache = dict(cache)
        cache["cache_path"] = path
        return cache


def collate_frozen_cache_entries(entries):
    return merge_frozen_cache_entries(entries)


def optional_int(value, default=None):
    if value in {None, "", "null", "None"}:
        return default
    return int(value)


def make_dataloader_kwargs(cfg, batch_size, shuffle, num_workers, collate_fn=None):
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(cfg.pin_memory),
    }
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        if cfg.get("prefetch_factor", None) is not None:
            kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return kwargs


def validation_enabled(cfg, train_from_frozen_cache, train_split):
    return (
        bool(train_from_frozen_cache)
        and str(train_split or "train").strip().lower() == "train"
        and int(cfg.get("eval_freq", 0)) > 0
        and int(cfg.get("eval_steps", 0)) > 0
        and float(cfg.get("frozen_cache_eval_ratio", 0.0)) > 0.0
    )


class ControlLatentDistillModule(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        initial_device = "cpu"
        if torch.cuda.is_available():
            initial_device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        pipe = WanVideoNeoVersePipeline.from_pretrained(
            local_model_path=cfg.model_path,
            reconstructor_path=cfg.reconstructor_path,
            pipeline_kwargs=cfg.pipeline_kwargs,
            device=initial_device,
            torch_dtype=getattr(torch, cfg.torch_dtype),
            lora_path=cfg.get("lora_path", None),
            lora_alpha=float(cfg.get("lora_alpha", 1.0)),
            enable_vram_management=bool(cfg.get("enable_vram_management", True)),
            load_dit=bool(cfg.get("load_dit", False)),
            load_text_encoder=bool(cfg.get("load_text_encoder", False)),
            load_vae=bool(cfg.get("load_vae", True)),
        )
        pipe.scheduler.set_timesteps(int(cfg.scheduler_train_steps), training=True)
        pipe.reuse_reconstructor_tokens = bool(cfg.get("reuse_reconstructor_tokens", True))
        pipe.skip_unused_reconstructor_heads = bool(cfg.get("skip_unused_reconstructor_heads", True))
        pipe.batch_vae_teacher_embeds = bool(cfg.get("batch_vae_teacher_embeds", False))
        pipe.reuse_context_forward_tokens = bool(cfg.get("reuse_context_forward_tokens", True))
        freeze_module(pipe)
        object.__setattr__(self, "pipe", pipe)
        self.adapter = build_adapter(pipe, cfg)

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.pipe.device = device
        if dtype is not None:
            self.pipe.torch_dtype = dtype
        self.adapter.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.pipe.eval()
        self.adapter.train(mode)
        return self

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data[0]["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": None,
            "height": data[0]["img"].shape[-2],
            "width": data[0]["img"].shape[-1],
            "num_frames": len(data),
            "source_views": data,
            "control_scale": 1,
            "cfg_scale": 1,
            "tiled": bool(self.cfg.vae_tiled),
            "tile_size": tuple(self.cfg.tile_size),
            "tile_stride": tuple(self.cfg.tile_stride),
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
            "cfg_merge": False,
        }
        # The existing preprocessing unit uses pipe.is_training to decide whether to
        # keep input_video for VAE latents. Temporarily mark one frozen module as
        # training, then restore eval before teacher/student forward.
        old_state = self.pipe.control_branch.training
        self.pipe.control_branch.train(True)
        try:
            with torch.no_grad():
                for unit in self.pipe.units:
                    if unit.__class__.__name__ in {
                        "WanVideoUnit_NoiseInitializer",
                        "WanVideoUnit_InputVideoEmbedder",
                        "WanVideoUnit_PromptEmbedder",
                    }:
                        continue
                    inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                        unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
                    )
        finally:
            self.pipe.control_branch.train(old_state)
            self.pipe.eval()
        return {**inputs_shared, **inputs_posi}

    def build_frozen_forward_cache(self, data):
        inputs = self.forward_preprocess(data)
        with torch.no_grad():
            output_grid = condition_output_grid(self.pipe.control_branch, inputs["target_rgb"])
            sequence_length = int(np.prod(output_grid))
            self.pipe.load_models_to_device(["control_branch"])
            teacher = self.pipe.control_branch.encode_condition(
                inputs["target_rgb"],
                inputs["target_depth"],
                inputs["target_camera_embed"],
                inputs["target_mask"],
                sequence_length=sequence_length,
            )
            token_pack = None
            if bool(self.cfg.get("reuse_reconstructor_tokens", True)):
                token_pack = token_pack_from_preprocess(inputs, self.cfg)
            reused_reconstructor_tokens = token_pack is not None
            if token_pack is None:
                self.pipe.load_models_to_device(["reconstructor"])
                token_pack = extract_vggt_tokens(
                    self.pipe.reconstructor,
                    inputs["source_views"],
                    layer_indices=self.cfg.token.layer_indices,
                    include_camera_token=self.cfg.token.include_camera_token,
                    context_only=bool(self.cfg.token.context_only),
                    ref_view_strategy=str(self.cfg.token.ref_view_strategy),
                    autocast_dtype=getattr(torch, self.cfg.token.autocast_dtype),
                )
            target_times = inputs.get("target_timestamps")
            if isinstance(target_times, torch.Tensor):
                target_times = target_times.to(device=self.pipe.device, dtype=torch.float32)
            else:
                target_times = extract_target_times(inputs["source_views"], device=self.pipe.device)
            source_times = extract_source_times(
                inputs["source_views"],
                device=self.pipe.device,
                context_only=bool(self.cfg.token.context_only),
            )
            source_poses = inputs.get("source_poses")
            source_intrs = inputs.get("source_intrs")
            if not isinstance(source_poses, torch.Tensor) or not isinstance(source_intrs, torch.Tensor):
                source_poses, source_intrs = gather_source_cameras_from_target(
                    inputs["source_views"],
                    inputs.get("target_poses"),
                    inputs.get("target_intrs"),
                    context_only=bool(self.cfg.token.context_only),
                )
            self.pipe.load_models_to_device([])

        target_plucker = inputs.get("target_camera_embed")
        if isinstance(target_plucker, torch.Tensor):
            target_plucker = F.interpolate(
                target_plucker.permute(0, 2, 1, 3, 4),
                size=output_grid,
                mode="trilinear",
            ).permute(0, 2, 1, 3, 4).contiguous()

        cache = {
            "teacher": teacher.detach(),
            "tokens": token_pack["tokens"].detach(),
            "token_dim": int(token_pack["token_dim"]),
            "num_token_groups": int(token_pack["num_token_groups"]),
            "output_grid": output_grid,
            "sequence_length": sequence_length,
            "target_times": detach_optional_tensor(target_times),
            "source_times": detach_optional_tensor(source_times),
            "target_poses": detach_optional_tensor(inputs.get("target_poses")),
            "target_intrs": detach_optional_tensor(inputs.get("target_intrs")),
            "target_plucker": detach_optional_tensor(target_plucker),
            "source_poses": detach_optional_tensor(source_poses),
            "source_intrs": detach_optional_tensor(source_intrs),
            "reused_reconstructor_tokens": bool(reused_reconstructor_tokens),
        }
        return normalize_camera_condition_cache(cache, self.cfg)

    def get_forward_cache(self, data, device=None):
        target_device = self.pipe.device if device is None else device
        frozen_cache_dir = self.cfg.get("frozen_cache_dir", None)
        if frozen_cache_dir:
            cache_path = os.path.join(str(frozen_cache_dir), f"{frozen_cache_signature(data, self.cfg)}.pt")
            if bool(self.cfg.get("frozen_cache_read", True)) and os.path.exists(cache_path):
                return cache_to_target_device(
                    torch.load(cache_path, map_location="cpu"),
                    target_device,
                    dtype=cache_float_dtype(self.cfg) if cache_target_is_cpu(target_device) else None,
                )
            if bool(self.cfg.get("frozen_cache_required", False)):
                raise FileNotFoundError(f"Missing frozen forward cache: {cache_path}")
            cache = self.build_frozen_forward_cache(data)
            if bool(self.cfg.get("frozen_cache_write", False)):
                save_frozen_cache(cache_path, cache, self.cfg)
            return cache_to_target_device(
                cache,
                target_device,
                dtype=cache_float_dtype(self.cfg) if cache_target_is_cpu(target_device) else None,
            )

        if not bool(self.cfg.get("cache_frozen_outputs", False)):
            return cache_to_target_device(
                self.build_frozen_forward_cache(data),
                target_device,
                dtype=cache_float_dtype(self.cfg) if cache_target_is_cpu(target_device) else None,
            )
        if not hasattr(self, "_frozen_forward_cache"):
            self._frozen_forward_cache = self.build_frozen_forward_cache(data)
            cache = self._frozen_forward_cache
            tokens = cache["tokens"]
            teacher = cache["teacher"]
            print(
                "Cached frozen training inputs: "
                f"tokens={tuple(tokens.shape)} teacher={tuple(teacher.shape)} "
                f"output_grid={cache['output_grid']}"
            )
        return cache_to_target_device(
            self._frozen_forward_cache,
            target_device,
            dtype=cache_float_dtype(self.cfg) if cache_target_is_cpu(target_device) else None,
        )

    def forward(self, data):
        cache = prepare_runtime_cache(data, self.pipe.device) if looks_like_frozen_cache(data) else self.get_forward_cache(data)
        student = self.adapter(
            cache["tokens"],
            cache["output_grid"],
            target_times=cache["target_times"],
            target_poses=cache["target_poses"],
            target_intrs=cache["target_intrs"],
            target_plucker=cache["target_plucker"],
            source_times=cache["source_times"],
            source_poses=cache["source_poses"],
            source_intrs=cache["source_intrs"],
        )
        loss, metrics = compute_distill_loss(student, cache["teacher"], self.cfg)
        metrics["token_dim"] = torch.tensor(cache["token_dim"], device=loss.device, dtype=torch.float32)
        metrics["token_groups"] = torch.tensor(cache["num_token_groups"], device=loss.device, dtype=torch.float32)
        metrics["seq_len"] = torch.tensor(cache["sequence_length"], device=loss.device, dtype=torch.float32)
        add_time_metrics(metrics, cache["target_times"], cache["source_times"], loss.device)
        metrics["reused_reconstructor_tokens"] = torch.tensor(
            float(bool(cache.get("reused_reconstructor_tokens", False))),
            device=loss.device,
            dtype=torch.float32,
        )
        self.last_shapes = {
            "teacher": {"condition": tuple(cache["teacher"].shape)},
            "student": {"condition": tuple(student.shape)},
            "tokens": tuple(cache["tokens"].shape),
            "output_grid": cache["output_grid"],
            "target_times": None if cache["target_times"] is None else tuple(cache["target_times"].shape),
            "source_times": None if cache["source_times"] is None else tuple(cache["source_times"].shape),
            "target_time_range_s": None
            if cache["target_times"] is None
            else (
                float(cache["target_times"][0, 0].detach().cpu()),
                float(cache["target_times"][0, -1].detach().cpu()),
            ),
            "source_time_range_s": None
            if cache["source_times"] is None
            else (
                float(cache["source_times"][0, 0].detach().cpu()),
                float(cache["source_times"][0, -1].detach().cpu()),
            ),
            "target_poses": None if cache["target_poses"] is None else tuple(cache["target_poses"].shape),
            "source_poses": None if cache["source_poses"] is None else tuple(cache["source_poses"].shape),
            "reused_reconstructor_tokens": bool(cache.get("reused_reconstructor_tokens", False)),
        }
        self.last_visuals = (
            {"condition": cache["teacher"]},
            {"condition": student.detach()},
            cache["output_grid"],
        )
        return loss, metrics


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(accelerator, model, optimizer, cfg, output_path, step, epoch=None, name=None, include_optimizer=True):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    os.makedirs(output_path, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    ckpt_name = name or "adapter_last.pt"
    payload = {
        "adapter": unwrapped.adapter.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "step": step,
        "epoch": epoch,
        "shapes": getattr(unwrapped, "last_shapes", None),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    ckpt_path = os.path.join(output_path, ckpt_name)
    tmp_path = f"{ckpt_path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)
    OmegaConf.save(cfg, os.path.join(output_path, "config.yaml"))


def resolve_resume_path(cfg):
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        return str(resume_from)
    if bool(cfg.get("auto_resume", False)):
        candidate = os.path.join(str(cfg.output_path), "adapter_last.pt")
        if os.path.exists(candidate):
            return candidate
        if os.path.isdir(str(cfg.output_path)):
            latest_step = -1
            latest_path = None
            for name in os.listdir(str(cfg.output_path)):
                if not (name.startswith("adapter_step_") and name.endswith(".pt")):
                    continue
                try:
                    step = int(name.removeprefix("adapter_step_").removesuffix(".pt"))
                except ValueError:
                    continue
                if step > latest_step:
                    latest_step = step
                    latest_path = os.path.join(str(cfg.output_path), name)
            if latest_path is not None:
                return latest_path
    return None


def load_checkpoint(accelerator, model, optimizer, resume_path, load_optimizer=True):
    checkpoint = torch.load(resume_path, map_location=accelerator.device)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.adapter.load_state_dict(checkpoint["adapter"])
    if load_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = int(checkpoint.get("step", 0))
    if accelerator.is_main_process:
        print(f"Resumed adapter checkpoint from {resume_path} at step={step}")
    return step


def run_validation(accelerator, model, eval_dataloader, cfg):
    max_eval_steps = int(cfg.get("eval_steps", 0))
    if eval_dataloader is None or max_eval_steps <= 0:
        return {}

    model.eval()
    metric_totals = {}
    local_batches = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    try:
        with torch.no_grad():
            for eval_idx, batch in enumerate(eval_dataloader, start=1):
                loss, metrics = model(batch)
                local_batches += 1.0
                metrics = dict(metrics)
                metrics["loss"] = loss.detach()
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        value = value.detach().float().to(device=accelerator.device)
                        metric_totals[key] = metric_totals.get(key, torch.zeros_like(value)) + value
                if eval_idx >= max_eval_steps:
                    break
    finally:
        model.train()

    total_batches = accelerator.gather(local_batches).sum()
    if float(total_batches.detach().cpu()) <= 0:
        return {}
    averaged = {}
    for key, value in metric_totals.items():
        total = accelerator.gather(value).sum()
        averaged[key] = float((total / total_batches).detach().cpu())
    return averaged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. output_path=outputs/run")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    os.makedirs(cfg.output_path, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=6000))],
    )
    seed_everything(int(cfg.seed) + accelerator.process_index)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(cfg.output_path, "config.yaml"))

    train_from_frozen_cache = bool(cfg.get("train_from_frozen_cache", False))
    eval_dataloader = None
    if train_from_frozen_cache:
        if not cfg.get("frozen_cache_dir", None):
            raise ValueError("train_from_frozen_cache=true requires frozen_cache_dir.")
        frozen_cache_split = str(cfg.get("frozen_cache_split", "train") or "train").strip().lower()
        dataset = FrozenForwardCacheDataset(
            cfg.frozen_cache_dir,
            pattern=cfg.get("frozen_cache_pattern", "*.pt"),
            split=frozen_cache_split,
            eval_ratio=float(cfg.get("frozen_cache_eval_ratio", 0.05)),
            split_seed=int(cfg.get("frozen_cache_split_seed", 0)),
            split_mode=cfg.get("frozen_cache_split_mode", "hash"),
        )
        eval_dataset = None
        if validation_enabled(cfg, train_from_frozen_cache, frozen_cache_split):
            eval_dataset = FrozenForwardCacheDataset(
                cfg.frozen_cache_dir,
                pattern=cfg.get("frozen_cache_pattern", "*.pt"),
                split="eval",
                eval_ratio=float(cfg.get("frozen_cache_eval_ratio", 0.05)),
                split_seed=int(cfg.get("frozen_cache_split_seed", 0)),
                split_mode=cfg.get("frozen_cache_split_mode", "hash"),
                allow_empty=True,
            )
            if len(eval_dataset) == 0:
                eval_dataset = None
        if accelerator.is_main_process:
            counts = dataset.split_counts
            print(
                f"Using frozen forward cache dataset: split={frozen_cache_split} selected={len(dataset)} "
                f"total={counts['all']} train={counts['train']} eval={counts['eval']} "
                f"dir={cfg.frozen_cache_dir} pattern={cfg.get('frozen_cache_pattern', '*.pt')!r} "
                f"eval_ratio={float(cfg.get('frozen_cache_eval_ratio', 0.05)):.4g} "
                f"split_seed={int(cfg.get('frozen_cache_split_seed', 0))}"
            )
            if eval_dataset is None:
                print("Frozen cache validation is disabled.")
            else:
                print(
                    f"Using frozen forward eval dataset: selected={len(eval_dataset)} "
                    f"eval_freq={int(cfg.get('eval_freq', 0))} eval_steps={int(cfg.get('eval_steps', 0))}"
                )
    else:
        dataset = build_spatialvid_dataset(cfg)
    collate_fn = collate_frozen_cache_entries if train_from_frozen_cache else None
    dataloader_kwargs = make_dataloader_kwargs(
        cfg,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_fn,
    )
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    if train_from_frozen_cache and eval_dataset is not None:
        eval_batch_size = optional_int(cfg.get("eval_batch_size", None), int(cfg.batch_size))
        eval_num_workers = optional_int(cfg.get("eval_num_workers", None), int(cfg.num_workers))
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            **make_dataloader_kwargs(
                cfg,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=eval_num_workers,
                collate_fn=collate_frozen_cache_entries,
            ),
        )
    model = ControlLatentDistillModule(cfg)
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    if eval_dataloader is None:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    else:
        model, optimizer, dataloader, eval_dataloader = accelerator.prepare(model, optimizer, dataloader, eval_dataloader)
    resume_path = resolve_resume_path(cfg)
    step = 0
    if resume_path is not None:
        step = load_checkpoint(
            accelerator,
            model,
            optimizer,
            resume_path,
            load_optimizer=bool(cfg.get("resume_optimizer", True)),
        )
    if cfg.get("max_steps", None) is not None and step >= int(cfg.max_steps):
        if accelerator.is_main_process:
            print(f"Checkpoint step={step} already reached max_steps={int(cfg.max_steps)}. Exiting.")
        accelerator.end_training()
        return
    writer = SummaryWriter(log_dir=cfg.output_path) if accelerator.is_main_process else None

    cached_train_batch = None
    if bool(cfg.get("cache_train_batch", False)):
        cached_train_batch = next(iter(dataloader))
        if accelerator.is_main_process:
            print("Cached one DataLoader batch for repeated overfit training.")

    preloaded_frozen_batches = None
    if bool(cfg.get("preload_frozen_cache", False)):
        unwrapped = accelerator.unwrap_model(model)
        preloaded_frozen_batches = []
        preload_device = str(cfg.get("preload_frozen_cache_device", "cuda"))
        preload_cache_device = "cpu" if preload_device == "cpu" else accelerator.device
        preload_total = len(dataloader) if hasattr(dataloader, "__len__") else None
        preload_log_freq = int(cfg.get("preload_frozen_cache_log_freq", 20))
        preload_start = time.time()
        if accelerator.is_main_process:
            total_text = "unknown" if preload_total is None else str(preload_total)
            print(f"Preloading frozen cache from DataLoader to {preload_device} ({total_text} batches on this process).")
        for preload_idx, preload_batch in enumerate(dataloader, start=1):
            if looks_like_frozen_cache(preload_batch):
                cache = cache_to_target_device(
                    preload_batch,
                    preload_cache_device,
                    dtype=cache_float_dtype(cfg) if cache_target_is_cpu(preload_cache_device) else None,
                )
            else:
                cache = unwrapped.get_forward_cache(preload_batch, device=preload_cache_device)
            preloaded_frozen_batches.append(cache)
            if accelerator.is_main_process and preload_log_freq > 0 and (
                preload_idx == 1 or preload_idx % preload_log_freq == 0
            ):
                elapsed = time.time() - preload_start
                if preload_total is None:
                    print(f"Preloaded frozen cache batches: {preload_idx} elapsed={elapsed:.1f}s")
                else:
                    print(
                        f"Preloaded frozen cache batches: {preload_idx}/{preload_total} "
                        f"elapsed={elapsed:.1f}s"
                    )
        preloaded_frozen_batches = batch_preloaded_frozen_cache(
            preloaded_frozen_batches,
            int(cfg.get("preload_frozen_cache_batch_size", 1)),
        )
        if accelerator.is_main_process:
            print(
                f"Preloaded {len(preloaded_frozen_batches)} frozen cache batches on this process "
                f"in {time.time() - preload_start:.1f}s."
            )
        if len(preloaded_frozen_batches) == 0:
            raise RuntimeError("preload_frozen_cache=true but no frozen cache entries were loaded.")

    last_log = time.time()
    for epoch in range(int(cfg.num_epochs)):
        if preloaded_frozen_batches is not None:
            if bool(cfg.get("shuffle_preloaded_frozen_cache", True)):
                random.shuffle(preloaded_frozen_batches)
            batch_iter = preloaded_frozen_batches
        elif cached_train_batch is not None:
            batch_iter = (cached_train_batch,)
        else:
            batch_iter = dataloader
        for batch in batch_iter:
            with accelerator.accumulate(model):
                loss, metrics = model(batch)
                accelerator.backward(loss)
                if cfg.clip_grad is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(cfg.clip_grad))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                if step % int(cfg.print_freq) == 0 and accelerator.is_main_process:
                    elapsed = time.time() - last_log
                    last_log = time.time()
                    metrics_float = {k: float(v.detach().float().cpu()) for k, v in metrics.items()}
                    summary = " ".join(
                        f"{k}={v:.4g}" for k, v in metrics_float.items() if k == "loss" or k.endswith("/l1")
                    )
                    print(f"epoch={epoch} step={step} dt={elapsed:.2f}s {summary}")
                    if writer is not None:
                        for key, value in metrics_float.items():
                            writer.add_scalar(key, value, step)
                    shapes = accelerator.unwrap_model(model).last_shapes
                    print(f"token_shape={shapes['tokens']} output_grid={shapes['output_grid']}")
                    first_key = next(iter(shapes["teacher"]))
                    print(f"teacher[{first_key}]={shapes['teacher'][first_key]} student[{first_key}]={shapes['student'][first_key]}")

                if eval_dataloader is not None and int(cfg.get("eval_freq", 0)) > 0 and step % int(cfg.eval_freq) == 0:
                    eval_start = time.time()
                    eval_metrics = run_validation(accelerator, model, eval_dataloader, cfg)
                    if accelerator.is_main_process:
                        if eval_metrics:
                            summary = " ".join(
                                f"eval/{k}={v:.4g}"
                                for k, v in eval_metrics.items()
                                if k == "loss" or k.endswith("/l1")
                            )
                            print(f"eval step={step} dt={time.time() - eval_start:.2f}s {summary}")
                            if writer is not None:
                                for key, value in eval_metrics.items():
                                    writer.add_scalar(f"eval/{key}", value, step)
                        else:
                            print(f"eval step={step} skipped: no eval batches")

                if int(cfg.vis_freq) > 0 and step % int(cfg.vis_freq) == 0 and accelerator.is_main_process:
                    teacher, student, output_grid = accelerator.unwrap_model(model).last_visuals
                    save_heatmaps(os.path.join(cfg.output_path, "visuals"), step, teacher, student, output_grid)

                if int(cfg.save_freq) > 0 and step % int(cfg.save_freq) == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        cfg,
                        cfg.output_path,
                        step,
                        epoch=epoch,
                        name="adapter_last.pt",
                        include_optimizer=bool(cfg.get("save_optimizer_intermediate", False)),
                    )

                if cfg.get("max_steps", None) is not None and step >= int(cfg.max_steps):
                    save_checkpoint(accelerator, model, optimizer, cfg, cfg.output_path, step, epoch=epoch, name="adapter_last.pt")
                    if writer is not None:
                        writer.close()
                    accelerator.end_training()
                    return

    save_checkpoint(accelerator, model, optimizer, cfg, cfg.output_path, step, epoch=int(cfg.num_epochs) - 1, name="adapter_last.pt")
    if writer is not None:
        writer.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
