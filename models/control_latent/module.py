import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from diffsynth.models.student_adapters import build_student_adapter
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline

from .cache import (
    cache_float_dtype,
    cache_target_is_cpu,
    cache_to_target_device,
    frozen_cache_dataset_index,
    frozen_cache_signature,
    frozen_cache_view_metadata,
    looks_like_frozen_cache,
    prepare_runtime_cache,
    save_frozen_cache,
)
from .camera import (
    add_time_metrics,
    detach_optional_tensor,
    extract_source_times,
    extract_target_times,
    gather_source_cameras_from_target,
    normalize_camera_condition_cache,
    token_pack_from_preprocess,
)
from .loss import compute_distill_loss
from .reconstructor_tokens import extract_vggt_tokens


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
            "cache_format_version": 2,
            "cache_signature": frozen_cache_signature(data, self.cfg),
            "dataset_index": frozen_cache_dataset_index(data),
            "signature_views": frozen_cache_view_metadata(data),
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
