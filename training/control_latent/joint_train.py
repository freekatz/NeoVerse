"""Route A v2 — Queryable World Model training entry.

Inherits the NeoVerse control-injection pathway end-to-end and replaces the
teacher-distillation loss of ``distill.py`` with:
  * Wan v-prediction loss — ``hits`` is fed directly into
    ``NeoVerseControlBranch.hints_from_condition`` (skipping ``encode_condition``);
    the resulting per-layer hints are residually added to Wan DiT exactly as in
    standard NeoVerse inference.
  * Auxiliary L1: ``hits_adapted`` ↔ ``VAE(rebuild rendered low-quality video)``.

Frozen: Wan DiT, control_branch (all parts), VAE, T5 text encoder, reconstructor.
Trainable: QueryModule (CrossAttentionRoPEAdapter + WanAdapter) only.

This is NOT a joint Wan+query optimization. Gradient from the diffusion loss
flows through frozen control_branch and frozen Wan back into the QueryModule.
NeoVerse pretrained weights are reused as-is (same as v0/v1 distill loop).

Maximally reuses ``training.control_latent.distill`` for pipe construction,
data preprocessing, dataset wiring, caching helpers, and accelerator boilerplate.
``main()`` mirrors ``distill.main()`` but swaps the module class. The original
``distill.py`` is left intact as the v0 baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from einops import rearrange
from omegaconf import OmegaConf

from diffsynth.models.student_adapters import build_query_module
from diffsynth.pipelines.wan_video_neoverse import (
    model_fn_wan_video as model_fn_wan_video_neoverse,
)

from training.control_latent import distill as _distill
from training.control_latent.distill import (
    batch_preloaded_frozen_cache,
    collate_frozen_cache_entries,
    compact_distill_log_metrics,
    ControlLatentDistillModule,
    build_spatialvid_dataset,
    cache_float_dtype,
    cache_target_is_cpu,
    cache_to_target_device,
    condition_output_grid,
    detach_optional_tensor,
    extract_source_times,
    extract_target_times,
    freeze_module,
    frozen_cache_dataset_index,
    frozen_cache_signature,
    frozen_cache_view_metadata,
    gather_source_cameras_from_target,
    load_checkpoint,
    looks_like_frozen_cache,
    make_dataloader_kwargs,
    normalize_camera_condition_cache,
    optional_int,
    prepare_runtime_cache,
    resolve_resume_path,
    run_validation,
    save_checkpoint,
    save_frozen_cache,
    seed_everything,
    token_pack_from_preprocess,
    validation_enabled,
)
from training.control_latent.reconstructor_tokens import extract_vggt_tokens
from training.swanlab_utils import init_swanlab_logger

# Module-local version stamp baked into the cache schema. Bump when v2 cache
# layout changes incompatibly. v4 adds text_emb (T5 prompt embeddings).
_V2_CACHE_FORMAT_VERSION = 4
_V2_CACHE_NAMESPACE = "joint_v2"
_V2_CACHE_FILENAME_PREFIX = "v2_"
_DEFAULT_CONTROL_LAYERS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)


# ---------------------------------------------------------------------------
# Cache version guards


def _v2_cache_signature(data, cfg) -> str:
    """Namespace the v0 cache signature so v0/v2 caches never collide."""
    base = frozen_cache_signature(data, cfg)
    payload = f"{_V2_CACHE_NAMESPACE}:v{_V2_CACHE_FORMAT_VERSION}:{base}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def looks_like_v2_cache(data) -> bool:
    """Return True only for v2-v4 caches (NeoVerse-inherit injection schema)."""
    if not isinstance(data, dict):
        return False
    if "teacher" in data:
        return False
    if "tokens" not in data or "output_grid" not in data:
        return False
    if not ("rendered_rgb_latents" in data or "target_rgb_latents" in data):
        return False
    # v4 requires text_emb for the NeoVerse Wan cross-attn KV path.
    version = int(data.get("cache_format_version", 0))
    if version >= 4 and "text_emb" not in data:
        return False
    return True


def _reject_legacy_cache_or_pass(data):
    """Raise on v0 distill cache (contains ``teacher``); otherwise return ``data``."""
    if isinstance(data, dict) and "teacher" in data:
        raise ValueError(
            "Detected a v0 distill cache (key 'teacher' present). joint_train.py "
            "requires a v2 cache (with 'rendered_rgb_latents' / 'target_rgb_latents'). "
            "Use a separate frozen_cache_dir for v2 or delete the v0 caches."
        )
    return data


class V2FrozenForwardCacheDataset(_distill.FrozenForwardCacheDataset):
    """Frozen-cache dataset variant that validates the joint-train v2/v4 schema."""

    def __getitem__(self, idx):
        path = self.paths[int(idx)]
        cache = torch.load(path, map_location="cpu")
        if not looks_like_v2_cache(cache):
            raise ValueError(f"Invalid v2 frozen forward cache file: {path}")
        cache = dict(cache)
        cache["cache_path"] = path
        return cache


# ---------------------------------------------------------------------------
# QueryModule construction


def _resolve_target_grid(value, fallback):
    if value is None:
        return fallback
    grid = tuple(int(x) for x in value)
    if len(grid) != 3:
        raise ValueError(f"target_grid must be a 3-tuple, got {grid}")
    return grid


class _SmokePipe:
    """Tiny pipe stub for smoke-testing QueryModule without loading NeoVerse."""

    def __init__(self, cfg):
        self.device = "cpu"
        self.torch_dtype = getattr(torch, str(cfg.get("torch_dtype", "bfloat16")))
        self.control_branch = SimpleNamespace(
            control_layers=tuple(cfg.get("control_layers", _DEFAULT_CONTROL_LAYERS))
        )

    def eval(self):
        return self

    def load_models_to_device(self, *_args, **_kwargs):
        return None


def build_query_module_from_cfg(pipe, cfg):
    """Build a :class:`QueryModule` (CrossAttentionRoPEAdapter + WanAdapter) from cfg."""
    adapter_cfg = OmegaConf.to_container(cfg.adapter, resolve=True)
    adapter_type = adapter_cfg.pop("type", "cross_attention_rope")
    # Drop Conv-only / unused-by-cross-attn-rope fields (mirrors build_adapter()).
    for key in ("input_channels", "num_res_blocks", "dropout", "condition_time_dim", "use_text_context", "output_mode"):
        adapter_cfg.pop(key, None)
    # hits_dim must match control_branch.dim so hits can be fed directly into
    # NeoVerseControlBranch.hints_from_condition (skipping encode_condition).
    # For Wan2.1-T2V-14B + NeoVerse this is 5120.
    hits_dim = int(cfg.get("hits_dim", cfg.get("control_dim", 5120)))
    adapter_cfg["output_dim"] = hits_dim
    adapter_cfg["control_layers"] = tuple(pipe.control_branch.control_layers)
    if adapter_cfg.get("token_dim") is None:
        adapter_cfg["token_dim"] = int(cfg.token.token_dim)
    max_token_groups = adapter_cfg.get("max_token_groups")
    if max_token_groups is None:
        adapter_cfg["max_token_groups"] = int(cfg.token.token_groups)
    adapter_cfg["output_mode"] = "condition_embedding"

    wan_adapter_cfg = None
    if cfg.get("wan_adapter", None) is not None:
        wac = OmegaConf.to_container(cfg.wan_adapter, resolve=True)
        # base_grid only bounds the learnable embed parameter count; the actual
        # runtime grid is derived from the VAE-encoded rebuild/target video shape
        # and passed to WanAdapter.forward via trilinear interpolation.
        base_grid = _resolve_target_grid(
            wac.get("base_grid", wac.get("target_grid")),  # accept legacy key
            None,
        )
        if base_grid is None:
            raise ValueError("wan_adapter.base_grid must be set (T0, H0, W0)")
        wan_adapter_cfg = {
            "base_grid": base_grid,
            "embed_dim": int(wac.get("embed_dim", 1536)),
            "num_heads": int(wac.get("num_heads", 8)),
            "c_z": int(wac.get("c_z", 16)),
            "mlp_ratio": float(wac.get("mlp_ratio", 4.0)),
        }
    return build_query_module(adapter_type, adapter_cfg, wan_adapter_cfg)


# ---------------------------------------------------------------------------
# Module


class ControlLatentJointModule(torch.nn.Module):
    """Wrap pipe + QueryModule. Pipe is fully frozen; only the QueryModule trains."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.smoke_test = bool(cfg.get("smoke_test", False))
        if self.smoke_test:
            object.__setattr__(self, "_base", None)
            object.__setattr__(self, "pipe", _SmokePipe(cfg))
            self.query_module = build_query_module_from_cfg(self.pipe, cfg)
            return
        # Use the distill module purely to build pipe + reuse its preprocessor.
        # We immediately replace its adapter with our own QueryModule.
        base = ControlLatentDistillModule(cfg)
        del base.adapter  # we don't need the distill adapter
        # Keep the distill wrapper only as a plain helper object for
        # ``forward_preprocess``. If it stays as a registered submodule,
        # ``train()/eval()`` will recurse into ``ControlLatentDistillModule``
        # and hit its deleted ``adapter`` attribute.
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "pipe", base.pipe)
        self.query_module = build_query_module_from_cfg(self.pipe, cfg)
        freeze_module(self.pipe)

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.pipe.device = device
        if dtype is not None:
            self.pipe.torch_dtype = dtype
        self.query_module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.pipe.eval()
        self.query_module.train(mode)
        return self

    # ----- preprocessing & cache -------------------------------------------------

    def forward_preprocess(self, data):
        if self.smoke_test:
            raise RuntimeError("smoke_test mode expects prebuilt v2 cache batches, not raw dataset samples.")
        return self._base.forward_preprocess(data)

    def build_frozen_forward_cache(self, data):
        if self.smoke_test:
            raise RuntimeError("smoke_test mode does not support building frozen caches from raw data.")
        inputs = self.forward_preprocess(data)
        with torch.no_grad():
            output_grid = condition_output_grid(self.pipe.control_branch, inputs["target_rgb"])
            sequence_length = int(np.prod(output_grid))

            # vggt tokens
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

            # cameras / times
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

            # VAE-encode the rebuild rendered low-quality video → hits_adapted supervision target.
            rendered = inputs.get("rebuild_rendered_rgb")
            rendered_rgb_latents = None
            if isinstance(rendered, torch.Tensor):
                self.pipe.load_models_to_device(["vae"])
                rendered_video = rearrange(rendered, "B T H W C -> B T C H W")
                rendered_video = self.pipe.preprocess_video(rendered_video)
                rendered_rgb_latents = self.pipe.vae.encode(
                    rendered_video,
                    device=self.pipe.device,
                    tiled=bool(self.cfg.vae_tiled),
                    tile_size=tuple(self.cfg.tile_size),
                    tile_stride=tuple(self.cfg.tile_stride),
                ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device).detach()

            # The clean target_rgb is already VAE-encoded by WanVideoUnit_4DEmbedder.
            target_rgb_latents = inputs.get("target_rgb")

            # T5 prompt embedding — NeoVerse Wan's native cross-attn KV. Encode
            # the positive prompt once and stash it in the cache; this avoids
            # paying T5 inference cost on every training step.
            text_emb = None
            prompt = inputs.get("prompt") or inputs.get("positive_prompt")
            if prompt is not None and self.pipe.text_encoder is not None:
                self.pipe.load_models_to_device(["text_encoder"])
                text_emb = self.pipe.prompter.encode_prompt(
                    prompt, positive=True, device=self.pipe.device
                ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device).detach()
            self.pipe.load_models_to_device([])

        target_plucker = inputs.get("target_camera_embed")
        if isinstance(target_plucker, torch.Tensor):
            target_plucker = F.interpolate(
                target_plucker.permute(0, 2, 1, 3, 4),
                size=output_grid,
                mode="trilinear",
            ).permute(0, 2, 1, 3, 4).contiguous()

        cache = {
            "cache_format_version": _V2_CACHE_FORMAT_VERSION,
            "cache_namespace": _V2_CACHE_NAMESPACE,
            "cache_signature": _v2_cache_signature(data, self.cfg),
            "dataset_index": frozen_cache_dataset_index(data),
            "signature_views": frozen_cache_view_metadata(data),
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
            "rendered_rgb_latents": detach_optional_tensor(rendered_rgb_latents),
            "target_rgb_latents": detach_optional_tensor(target_rgb_latents),
            "text_emb": detach_optional_tensor(text_emb),
            "reused_reconstructor_tokens": bool(reused_reconstructor_tokens),
        }
        return normalize_camera_condition_cache(cache, self.cfg)

    def get_forward_cache(self, data, device=None):
        if self.smoke_test:
            raise RuntimeError("smoke_test mode expects prebuilt v2 cache batches, not runtime cache generation.")
        target_device = self.pipe.device if device is None else device
        frozen_cache_dir = self.cfg.get("frozen_cache_dir", None)
        if frozen_cache_dir:
            sig = _v2_cache_signature(data, self.cfg)
            cache_path = os.path.join(
                str(frozen_cache_dir), f"{_V2_CACHE_FILENAME_PREFIX}{sig}.pt"
            )
            if bool(self.cfg.get("frozen_cache_read", True)) and os.path.exists(cache_path):
                loaded = torch.load(cache_path, map_location="cpu")
                if not looks_like_v2_cache(loaded):
                    raise ValueError(
                        f"Cache file {cache_path} does not look like a v2 joint_train "
                        "cache (missing 'rendered_rgb_latents'/'target_rgb_latents' or "
                        "contains 'teacher'). Delete the file or use a fresh frozen_cache_dir."
                    )
                return cache_to_target_device(
                    loaded,
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
        return cache_to_target_device(
            self.build_frozen_forward_cache(data),
            target_device,
            dtype=cache_float_dtype(self.cfg) if cache_target_is_cpu(target_device) else None,
        )

    # ----- forward ---------------------------------------------------------------

    def _sample_diffusion_step(self, x0: torch.Tensor):
        """Sample a single diffusion (timestep, noise, noisy_latent, target_velocity)."""
        sched = self.pipe.scheduler
        # Match the existing Wan/NeoVerse training path: sample one scheduler step
        # per optimizer step, then broadcast that timestep across the batch.
        # Keep the index on CPU because scheduler.timesteps is a CPU tensor.
        max_step = int(self.cfg.get("max_timestep_step", len(sched.timesteps)))
        min_step = int(self.cfg.get("min_timestep_step", 0))
        t_idx = torch.randint(low=min_step, high=max_step, size=(1,))
        timestep = sched.timesteps[t_idx].to(device=x0.device, dtype=x0.dtype)
        noise = torch.randn_like(x0)
        noisy_latent = sched.add_noise(x0, noise, timestep=timestep)
        # Wan uses flow-matching v-prediction: target = noise - x0. ``training_target``
        # is honored if the scheduler exposes it.
        if hasattr(sched, "training_target"):
            target_velocity = sched.training_target(x0, noise, timestep=timestep)
        else:
            target_velocity = noise - x0
        return timestep, noisy_latent, target_velocity

    def forward(self, data):
        if looks_like_v2_cache(data):
            cache = prepare_runtime_cache(data, self.pipe.device)
        else:
            data = _reject_legacy_cache_or_pass(data)
            cache = self.get_forward_cache(data)

        if self.smoke_test:
            hits, hits_adapted = self.query_module(
                cache["tokens"],
                cache["output_grid"],
                target_times=cache.get("target_times"),
                target_poses=cache.get("target_poses"),
                target_intrs=cache.get("target_intrs"),
                target_plucker=cache.get("target_plucker"),
                source_times=cache.get("source_times"),
                source_poses=cache.get("source_poses"),
                source_intrs=cache.get("source_intrs"),
                wan_adapter_target_grid=tuple(int(s) for s in cache["rendered_rgb_latents"].shape[-3:]),
            )
            hits_target = cache.get("smoke_hits_target")
            loss_hits = F.mse_loss(hits.float(), hits_target.float()) if hits_target is not None else hits.float().pow(2).mean()
            loss_aux = hits.new_zeros(())
            rendered_target = cache.get("rendered_rgb_latents")
            if hits_adapted is not None and rendered_target is not None:
                loss_aux = F.l1_loss(hits_adapted.float(), rendered_target.float())
            aux_weight = float(self.cfg.get("loss", {}).get("aux_weight", 0.1))
            loss = loss_hits + aux_weight * loss_aux
            metrics = {
                "loss": loss.detach(),
                "loss_diffusion": hits.new_zeros(()),
                "loss_aux": loss_aux.detach(),
                "loss_smoke_hits": loss_hits.detach(),
                "hits_norm": hits.detach().float().norm(dim=-1).mean(),
            }
            if hits_adapted is not None:
                metrics["hits_adapted_norm"] = hits_adapted.detach().float().flatten(1).norm(dim=-1).mean()
            metrics["seq_len"] = torch.tensor(cache["sequence_length"], device=loss.device, dtype=torch.float32)
            self.last_shapes = {
                "hits": tuple(hits.shape),
                "hits_adapted": None if hits_adapted is None else tuple(hits_adapted.shape),
                "tokens": tuple(cache["tokens"].shape),
                "output_grid": cache["output_grid"],
                "mode": "smoke_test",
            }
            return loss, metrics

        # Derive runtime VAE-latent grid from the supervision targets so the
        # WanAdapter's learnable embed gets resampled to the right shape.
        wan_adapter_target_grid = None
        for key in ("rendered_rgb_latents", "target_rgb_latents"):
            tensor = cache.get(key)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 5:
                wan_adapter_target_grid = tuple(int(s) for s in tensor.shape[-3:])
                break

        hits, hits_adapted = self.query_module(
            cache["tokens"],
            cache["output_grid"],
            target_times=cache.get("target_times"),
            target_poses=cache.get("target_poses"),
            target_intrs=cache.get("target_intrs"),
            target_plucker=cache.get("target_plucker"),
            source_times=cache.get("source_times"),
            source_poses=cache.get("source_poses"),
            source_intrs=cache.get("source_intrs"),
            wan_adapter_target_grid=wan_adapter_target_grid,
        )

        # ---- Aux loss: hits_adapted ↔ VAE(rendered rebuild video) ----
        loss_aux = hits.new_zeros(())
        rendered_target = cache.get("rendered_rgb_latents")
        if hits_adapted is not None and rendered_target is not None:
            rendered_target = rendered_target.to(dtype=hits_adapted.dtype)
            if hits_adapted.shape != rendered_target.shape:
                raise ValueError(
                    f"hits_adapted shape {tuple(hits_adapted.shape)} != "
                    f"rendered_rgb_latents shape {tuple(rendered_target.shape)} — "
                    "internal: WanAdapter's runtime target_grid plumbing is broken."
                )
            loss_aux = F.l1_loss(hits_adapted.float(), rendered_target.float())

        # ---- Main loss: Wan diffusion via NeoVerse injection path ----
        # `hits` is fed into control_branch.hints_from_condition (skipping
        # encode_condition); the resulting per-layer hints are residually added
        # to Wan DiT exactly as in NeoVerse v0 inference. `text_emb` occupies
        # Wan's native cross-attn KV slot.
        loss_diff = hits.new_zeros(())
        run_diffusion = bool(self.cfg.get("enable_wan_diffusion_loss", True))
        x0 = cache.get("target_rgb_latents")
        text_emb = cache.get("text_emb")
        if run_diffusion and x0 is not None:
            if text_emb is None:
                raise RuntimeError(
                    "text_emb is required for the NeoVerse-inherit forward path. "
                    "Set load_text_encoder=true in config and rebuild caches "
                    "(v4 format)."
                )
            self.pipe.load_models_to_device(["dit", "control_branch"])
            x0 = x0.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            text_emb = text_emb.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            timestep, noisy_latent, target_velocity = self._sample_diffusion_step(x0)
            pred = model_fn_wan_video_neoverse(
                dit=self.pipe.dit,
                control_branch=self.pipe.control_branch,
                latents=noisy_latent,
                timestep=timestep,
                context=text_emb,                            # T5 emb on Wan's native cross-attn
                raw_control_condition=hits,                  # hits → hints_from_condition
                control_scale=float(self.cfg.get("control_scale", 1.0)),
                use_gradient_checkpointing=bool(
                    self.cfg.get("use_gradient_checkpointing", True)
                ),
            )
            loss_diff = F.mse_loss(pred.float(), target_velocity.float())
            self.pipe.load_models_to_device([])

        diffusion_weight = float(self.cfg.get("loss", {}).get("diffusion_weight", 1.0))
        aux_weight = float(self.cfg.get("loss", {}).get("aux_weight", 0.1))
        loss = diffusion_weight * loss_diff + aux_weight * loss_aux

        metrics = {
            "loss": loss.detach(),
            "loss_diffusion": loss_diff.detach(),
            "loss_aux": loss_aux.detach(),
            "hits_norm": hits.detach().float().norm(dim=-1).mean(),
        }
        if hits_adapted is not None:
            metrics["hits_adapted_norm"] = hits_adapted.detach().float().flatten(1).norm(dim=-1).mean()
        metrics["seq_len"] = torch.tensor(
            cache["sequence_length"], device=loss.device, dtype=torch.float32
        )

        self.last_shapes = {
            "hits": tuple(hits.shape),
            "hits_adapted": None if hits_adapted is None else tuple(hits_adapted.shape),
            "tokens": tuple(cache["tokens"].shape),
            "output_grid": cache["output_grid"],
            "rendered_rgb_latents": None if rendered_target is None else tuple(rendered_target.shape),
            "target_rgb_latents": None if x0 is None else tuple(x0.shape),
        }
        return loss, metrics


def build_smoke_cache(cfg):
    batch_size = int(cfg.get("smoke_batch_size", 1))
    num_views = int(cfg.get("smoke_num_source_views", 3))
    token_groups = int(cfg.token.token_groups)
    token_dim = int(cfg.token.token_dim)
    token_h, token_w = tuple(int(v) for v in cfg.get("smoke_token_hw", (4, 6)))
    output_grid = tuple(int(v) for v in cfg.get("smoke_output_grid", (2, 4, 4)))
    target_grid = tuple(int(v) for v in cfg.get("smoke_wan_target_grid", (2, 4, 4)))
    query_len = int(np.prod(output_grid))
    hits_dim = int(cfg.get("hits_dim", cfg.get("control_dim", 5120)))
    z_channels = int(cfg.get("wan_adapter", {}).get("c_z", 16) if cfg.get("wan_adapter", None) is not None else 16)
    target_frames = int(output_grid[0])

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.float32)

    def eye_batch(frames, size):
        return torch.eye(size, dtype=torch.float32).reshape(1, 1, size, size).expand(batch_size, frames, -1, -1).clone()

    return {
        "cache_format_version": _V2_CACHE_FORMAT_VERSION,
        "cache_namespace": _V2_CACHE_NAMESPACE,
        "cache_signature": "smoke",
        "dataset_index": -1,
        "signature_views": [{"mode": "smoke_test"}],
        "tokens": randn(batch_size, num_views, token_groups, token_h, token_w, token_dim),
        "token_dim": token_dim,
        "num_token_groups": token_groups,
        "output_grid": output_grid,
        "sequence_length": query_len,
        "target_times": torch.linspace(0, 1, target_frames, dtype=torch.float32)[None].expand(batch_size, -1).clone(),
        "source_times": torch.linspace(0, 1, num_views, dtype=torch.float32)[None].expand(batch_size, -1).clone(),
        "target_poses": eye_batch(target_frames, 4),
        "target_intrs": eye_batch(target_frames, 3),
        "target_plucker": randn(batch_size, target_frames, 6, output_grid[1], output_grid[2]),
        "source_poses": eye_batch(num_views, 4),
        "source_intrs": eye_batch(num_views, 3),
        "rendered_rgb_latents": randn(batch_size, z_channels, *target_grid),
        "target_rgb_latents": randn(batch_size, z_channels, *target_grid),
        "text_emb": randn(batch_size, 4, 16),
        "smoke_hits_target": randn(batch_size, query_len, hits_dim),
        "reused_reconstructor_tokens": False,
    }


# ---------------------------------------------------------------------------
# CLI / main


def _build_dataloader(cfg, dataset, shuffle: bool, batch_size: int):
    kwargs = make_dataloader_kwargs(
        cfg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(cfg.num_workers),
        collate_fn=None,
    )
    return torch.utils.data.DataLoader(dataset, **kwargs)


def _frozen_cache_pattern(cfg) -> str:
    pattern = str(cfg.get("frozen_cache_pattern", f"{_V2_CACHE_FILENAME_PREFIX}*.pt") or "").strip()
    return pattern or f"{_V2_CACHE_FILENAME_PREFIX}*.pt"


def _save_checkpoint_v2(accelerator, model, optimizer, cfg, output_path, step, epoch=None, name=None, include_optimizer=True):
    """Like distill.save_checkpoint but saves ``query_module`` state."""
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    os.makedirs(output_path, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    ckpt_name = name or "query_module_last.pt"
    payload = {
        "query_module": unwrapped.query_module.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "step": step,
        "epoch": epoch,
        "shapes": getattr(unwrapped, "last_shapes", None),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    ckpt_path = os.path.join(output_path, ckpt_name)
    torch.save(payload, f"{ckpt_path}.tmp")
    os.replace(f"{ckpt_path}.tmp", ckpt_path)
    OmegaConf.save(cfg, os.path.join(output_path, "config.yaml"))


def _load_checkpoint_v2(accelerator, model, optimizer, resume_path, load_optimizer=True):
    """Like distill.load_checkpoint but restores ``query_module`` state."""
    checkpoint = torch.load(resume_path, map_location=accelerator.device)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.query_module.load_state_dict(checkpoint["query_module"])
    if load_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = int(checkpoint.get("step", 0))
    if accelerator.is_main_process:
        print(f"Resumed query_module checkpoint from {resume_path} at step={step}")
        if load_optimizer and "optimizer" not in checkpoint:
            print("WARNING: checkpoint has no optimizer state; starting fresh.")
    return step


def _resolve_resume_path_v2(cfg):
    """Like distill.resolve_resume_path but looks for query_module_last.pt."""
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        return str(resume_from)
    if bool(cfg.get("auto_resume", False)):
        candidate = os.path.join(str(cfg.output_path), "query_module_last.pt")
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    """v2 training entry — mirrors distill.main() pattern.

    Single-GPU:
        python training/control_latent/joint_train.py \\
            configs/distill/control_latent_v2.yaml

    Multi-GPU (single node, 4 GPUs):
        accelerate launch --num_processes 4 \\
            training/control_latent/joint_train.py \\
            configs/distill/control_latent_v2.yaml

    Multi-node (2 nodes × 4 GPUs via torchrun):
        accelerate launch --multi_gpu --num_machines 2 --num_processes 8 \\
            --machine_rank $RANK --main_process_ip $MASTER_ADDR \\
            training/control_latent/joint_train.py \\
            configs/distill/control_latent_v2.yaml

    Dataset location: set ``data_root`` in the yaml (default: data/SpatialVID).
    Override on CLI: append ``data_root=/path/to/data`` as a positional arg.
    """
    import datetime
    parser = argparse.ArgumentParser()
    # Positional args match distill.py convention: <config> [overrides...]
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist, e.g. data_root=/data output_path=./out")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))],
    )
    seed_everything(int(cfg.get("seed", 0)) + accelerator.process_index)
    os.makedirs(cfg.output_path, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(cfg.output_path, "config.yaml"))

    train_from_frozen_cache = bool(cfg.get("train_from_frozen_cache", False))
    smoke_test = bool(cfg.get("smoke_test", False))
    eval_dataloader = None
    if smoke_test:
        dataset = [build_smoke_cache(cfg)]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda entries: entries[0],
        )
        if accelerator.is_main_process:
            print(
                "Running in smoke_test mode: using a synthetic cache batch and skipping "
                "NeoVerse/Wan pretrained weight loading."
            )
    elif train_from_frozen_cache:
        if not cfg.get("frozen_cache_dir", None):
            raise ValueError("train_from_frozen_cache=true requires frozen_cache_dir.")
        frozen_cache_split = str(cfg.get("frozen_cache_split", "train") or "train").strip().lower()
        cache_pattern = _frozen_cache_pattern(cfg)
        dataset = V2FrozenForwardCacheDataset(
            cfg.frozen_cache_dir,
            pattern=cache_pattern,
            split=frozen_cache_split,
            eval_ratio=float(cfg.get("frozen_cache_eval_ratio", 0.05)),
            split_seed=int(cfg.get("frozen_cache_split_seed", 0)),
            split_mode=cfg.get("frozen_cache_split_mode", "hash"),
        )
        eval_dataset = None
        if validation_enabled(cfg, train_from_frozen_cache, frozen_cache_split):
            eval_dataset = V2FrozenForwardCacheDataset(
                cfg.frozen_cache_dir,
                pattern=cache_pattern,
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
                f"Using v2 frozen cache dataset: split={frozen_cache_split} selected={len(dataset)} "
                f"total={counts['all']} train={counts['train']} eval={counts['eval']} "
                f"dir={cfg.frozen_cache_dir} pattern={cache_pattern!r} "
                f"eval_ratio={float(cfg.get('frozen_cache_eval_ratio', 0.05)):.4g} "
                f"split_seed={int(cfg.get('frozen_cache_split_seed', 0))}"
            )
            if eval_dataset is None:
                print("Frozen cache validation is disabled.")
            else:
                print(
                    f"Using v2 frozen-cache eval dataset: selected={len(eval_dataset)} "
                    f"eval_freq={int(cfg.get('eval_freq', 0))} eval_steps={int(cfg.get('eval_steps', 0))}"
                )
    else:
        dataset = build_spatialvid_dataset(cfg)

    if not smoke_test:
        collate_fn = collate_frozen_cache_entries if train_from_frozen_cache else None
        dataloader = torch.utils.data.DataLoader(
            dataset,
            **make_dataloader_kwargs(
                cfg,
                batch_size=int(cfg.batch_size),
                shuffle=True,
                num_workers=int(cfg.num_workers),
                collate_fn=collate_fn,
            ),
        )

    if train_from_frozen_cache and eval_dataset is not None and not smoke_test:
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

    model = ControlLatentJointModule(cfg)
    trainable_params = [p for p in model.query_module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    if eval_dataloader is None:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    else:
        model, optimizer, dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, dataloader, eval_dataloader
        )

    step = 0
    resume_path = _resolve_resume_path_v2(cfg)
    if resume_path:
        step = _load_checkpoint_v2(
            accelerator, model, optimizer, resume_path,
            load_optimizer=bool(cfg.get("resume_optimizer", True)),
        )
    if cfg.get("max_steps", None) is not None and step >= int(cfg.max_steps):
        if accelerator.is_main_process:
            print(f"Checkpoint step={step} already reached max_steps={int(cfg.max_steps)}. Exiting.")
        accelerator.end_training()
        return

    experiment_logger = (
        init_swanlab_logger(
            cfg,
            default_project="NeoVerseQueryableWorldModel",
            default_experiment_name=os.path.basename(os.path.normpath(str(cfg.output_path))),
            output_path=str(cfg.output_path),
        )
        if accelerator.is_main_process
        else None
    )

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
            print(f"Preloading v2 frozen cache to {preload_device} ({total_text} batches on this process).")
        for preload_idx, preload_batch in enumerate(dataloader, start=1):
            if looks_like_v2_cache(preload_batch):
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
                    print(f"Preloaded v2 frozen cache batches: {preload_idx} elapsed={elapsed:.1f}s")
                else:
                    print(
                        f"Preloaded v2 frozen cache batches: {preload_idx}/{preload_total} "
                        f"elapsed={elapsed:.1f}s"
                    )
        preloaded_frozen_batches = batch_preloaded_frozen_cache(
            preloaded_frozen_batches,
            int(cfg.get("preload_frozen_cache_batch_size", 1)),
        )
        if accelerator.is_main_process:
            print(
                f"Preloaded {len(preloaded_frozen_batches)} v2 frozen-cache batches on this process "
                f"in {time.time() - preload_start:.1f}s."
            )
        if len(preloaded_frozen_batches) == 0:
            raise RuntimeError("preload_frozen_cache=true but no frozen cache entries were loaded.")

    print_freq = int(cfg.get("print_freq", 10))
    save_freq = int(cfg.get("save_freq", 500))
    clip_grad = float(cfg.get("clip_grad", 1.0))
    max_steps = int(cfg.get("max_steps", 20000))
    if smoke_test:
        max_steps = min(max_steps, int(cfg.get("smoke_max_steps", 1)))
        if accelerator.is_main_process:
            print(
                f"Smoke test step budget: max_steps={max_steps} "
                f"(override with smoke_max_steps=... if needed)."
            )
    t0 = time.time()
    last_log = time.time()

    # ``step`` counts OPTIMIZER steps (post-accumulation), not micro-batches.
    epoch = 0
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
                if accelerator.sync_gradients and clip_grad > 0:
                    accelerator.clip_grad_norm_(
                        optimizer.param_groups[0]["params"],
                        clip_grad,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Skip logging/saving on accumulation micro-batches: only after
            # gradients sync (i.e. an actual optimizer step happened).
            if not accelerator.sync_gradients:
                continue

            step += 1

            if accelerator.is_main_process and step % print_freq == 0:
                elapsed = time.time() - last_log
                last_log = time.time()
                lr = optimizer.param_groups[0]["lr"]
                metrics_float = {k: float(v.detach().float().cpu()) for k, v in metrics.items()}
                print(
                    f"[step {step:>6d}] loss={metrics_float['loss']:.4f} "
                    f"diff={metrics_float['loss_diffusion']:.4f} "
                    f"aux={metrics_float['loss_aux']:.4f} "
                    f"lr={lr:.2e} dt={elapsed:.2f}s total={time.time() - t0:.1f}s",
                    flush=True,
                )
                if experiment_logger is not None:
                    experiment_logger.log(compact_distill_log_metrics(metrics_float, "train", lr=lr), step=step)

            if save_freq > 0 and step % save_freq == 0:
                _save_checkpoint_v2(
                    accelerator, model, optimizer, cfg, cfg.output_path, step,
                    epoch=epoch, include_optimizer=bool(cfg.get("resume_optimizer", True)),
                )

            if eval_dataloader is not None and int(cfg.get("eval_freq", 0)) > 0 and step % int(cfg.eval_freq) == 0:
                eval_start = time.time()
                eval_metrics = run_validation(accelerator, model, eval_dataloader, cfg)
                if accelerator.is_main_process:
                    if eval_metrics:
                        summary = " ".join(
                            f"eval/{k}={v:.4g}"
                            for k, v in eval_metrics.items()
                            if k == "loss" or k.startswith("loss_")
                        )
                        print(f"eval step={step} dt={time.time() - eval_start:.2f}s {summary}")
                        if experiment_logger is not None:
                            experiment_logger.log(compact_distill_log_metrics(eval_metrics, "eval"), step=step)
                    else:
                        print(f"eval step={step} produced no metrics.")

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    _save_checkpoint_v2(
        accelerator, model, optimizer, cfg, cfg.output_path, step,
        epoch=epoch, include_optimizer=bool(cfg.get("resume_optimizer", True)),
    )
    if experiment_logger is not None:
        experiment_logger.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
