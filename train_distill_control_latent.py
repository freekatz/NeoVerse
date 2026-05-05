import argparse
import datetime
import hashlib
import json
import os
import random
import time
from collections import OrderedDict

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
from hooks.extract_vggt_tokens import compose_views_from_list, extract_vggt_tokens
from training.data.datasets.spatialvid import SpatialVID


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
        keep = ~source_views["is_target"].bool()
        count = int(keep[0].sum().item()) if keep.ndim == 2 else timestamps.shape[1]
        source_times = timestamps[:, :count]
    else:
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


class ControlLatentDistillModule(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        initial_device = "cpu"
        if bool(cfg.get("enable_vram_management", True)) and torch.cuda.is_available():
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
            target_times = extract_target_times(inputs["source_views"], device=self.pipe.device)
            source_times = extract_source_times(
                inputs["source_views"],
                device=self.pipe.device,
                context_only=bool(self.cfg.token.context_only),
            )
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

        return {
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
        }

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

    dataset = eval(cfg.train_dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    model = ControlLatentDistillModule(cfg)
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
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
            cache = unwrapped.get_forward_cache(preload_batch, device=preload_cache_device)
            if preload_device == "cpu":
                cache = cache_to_cpu(cache, dtype=cache_float_dtype(cfg))
            else:
                cache = cache_to_device(cache, accelerator.device)
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
