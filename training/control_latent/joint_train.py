"""Route A v2 — Queryable World Model joint training entry.

Replaces the teacher-distillation loss of ``distill.py`` with:
  * Wan diffusion loss — vanilla Wan inference path (skips ``NeoVerseControlBranch``),
    with ``hits`` filling the T5 context slot. Wan / VAE / reconstructor stay frozen;
    gradient flows back through ``hits`` into the QueryModule.
  * Auxiliary L1: ``hits_adapted`` ↔ ``VAE(rebuild rendered low-quality video)``.

Maximally reuses ``training.control_latent.distill`` for pipe construction,
data preprocessing, dataset wiring, caching, and the optimizer / accelerator
boilerplate. ``main()`` here mirrors ``distill.main()`` but swaps the module
class. The original ``distill.py`` is left intact as the v0 baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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
from diffsynth.pipelines.wan_video import model_fn_wan_video

from training.control_latent import distill as _distill
from training.control_latent.distill import (
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
    prepare_runtime_cache,
    resolve_resume_path,
    save_checkpoint,
    save_frozen_cache,
    seed_everything,
    token_pack_from_preprocess,
)
from training.control_latent.reconstructor_tokens import extract_vggt_tokens


# ---------------------------------------------------------------------------
# QueryModule construction


def _resolve_target_grid(value, fallback):
    if value is None:
        return fallback
    grid = tuple(int(x) for x in value)
    if len(grid) != 3:
        raise ValueError(f"target_grid must be a 3-tuple, got {grid}")
    return grid


def build_query_module_from_cfg(pipe, cfg):
    """Build a :class:`QueryModule` (CrossAttentionRoPEAdapter + WanAdapter) from cfg."""
    adapter_cfg = OmegaConf.to_container(cfg.adapter, resolve=True)
    adapter_type = adapter_cfg.pop("type", "cross_attention_rope")
    # Drop Conv-only / unused-by-cross-attn-rope fields (mirrors build_adapter()).
    for key in ("input_channels", "num_res_blocks", "dropout", "condition_time_dim", "use_text_context", "output_mode"):
        adapter_cfg.pop(key, None)
    # hits_dim defaults to vanilla Wan text_dim (4096) so hits can occupy the T5 slot.
    hits_dim = int(cfg.get("hits_dim", cfg.get("control_dim", 4096)))
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
        wan_adapter_cfg = {
            "target_grid": _resolve_target_grid(wac.get("target_grid"), None),
            "embed_dim": int(wac.get("embed_dim", 1536)),
            "num_heads": int(wac.get("num_heads", 8)),
            "c_z": int(wac.get("c_z", 16)),
            "mlp_ratio": float(wac.get("mlp_ratio", 4.0)),
        }
        if wan_adapter_cfg["target_grid"] is None:
            raise ValueError("wan_adapter.target_grid must be set (T_lat, H_lat, W_lat)")
    return build_query_module(adapter_type, adapter_cfg, wan_adapter_cfg)


# ---------------------------------------------------------------------------
# Module


class ControlLatentJointModule(torch.nn.Module):
    """Wrap pipe + QueryModule. Pipe is fully frozen; only the QueryModule trains."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Use the distill module purely to build pipe + reuse its preprocessor.
        # We immediately replace its adapter with our own QueryModule.
        base = ControlLatentDistillModule(cfg)
        del base.adapter  # we don't need the distill adapter
        self._base = base  # keep for forward_preprocess re-use
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
        return self._base.forward_preprocess(data)

    def build_frozen_forward_cache(self, data):
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
            self.pipe.load_models_to_device([])

        target_plucker = inputs.get("target_camera_embed")
        if isinstance(target_plucker, torch.Tensor):
            target_plucker = F.interpolate(
                target_plucker.permute(0, 2, 1, 3, 4),
                size=output_grid,
                mode="trilinear",
            ).permute(0, 2, 1, 3, 4).contiguous()

        cache = {
            "cache_format_version": 3,  # bump from distill v2
            "cache_signature": frozen_cache_signature(data, self.cfg),
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
            "reused_reconstructor_tokens": bool(reused_reconstructor_tokens),
        }
        return normalize_camera_condition_cache(cache, self.cfg)

    def get_forward_cache(self, data, device=None):
        target_device = self.pipe.device if device is None else device
        frozen_cache_dir = self.cfg.get("frozen_cache_dir", None)
        if frozen_cache_dir:
            cache_path = os.path.join(
                str(frozen_cache_dir), f"{frozen_cache_signature(data, self.cfg)}.pt"
            )
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
        return cache_to_target_device(
            self.build_frozen_forward_cache(data),
            target_device,
            dtype=cache_float_dtype(self.cfg) if cache_target_is_cpu(target_device) else None,
        )

    # ----- forward ---------------------------------------------------------------

    def _sample_diffusion_step(self, x0: torch.Tensor):
        """Sample a single diffusion (timestep, noise, noisy_latent, target_velocity)."""
        sched = self.pipe.scheduler
        bsz = x0.shape[0]
        # Inclusive lower / exclusive upper sample over the scheduler's training timesteps.
        max_step = int(self.cfg.get("max_timestep_step", len(sched.timesteps)))
        min_step = int(self.cfg.get("min_timestep_step", 0))
        t_idx = torch.randint(low=min_step, high=max_step, size=(bsz,), device=x0.device)
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
        cache = (
            prepare_runtime_cache(data, self.pipe.device)
            if looks_like_frozen_cache(data)
            else self.get_forward_cache(data)
        )

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
                    "check wan_adapter.target_grid (T_lat, H/8, W/8)."
                )
            loss_aux = F.l1_loss(hits_adapted.float(), rendered_target.float())

        # ---- Main loss: vanilla Wan diffusion with hits as cross-attn context ----
        loss_diff = hits.new_zeros(())
        run_diffusion = bool(self.cfg.get("enable_wan_diffusion_loss", True))
        x0 = cache.get("target_rgb_latents")
        if run_diffusion and x0 is not None:
            self.pipe.load_models_to_device(["dit"])
            x0 = x0.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            timestep, noisy_latent, target_velocity = self._sample_diffusion_step(x0)
            # vanilla Wan inference (no NeoVerseControlBranch) — hits replaces T5 emb.
            pred = model_fn_wan_video(
                dit=self.pipe.dit,
                x=noisy_latent,
                timestep=timestep,
                context=hits,
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

    dataset = build_spatialvid_dataset(cfg)
    dataloader = _build_dataloader(cfg, dataset, shuffle=True, batch_size=int(cfg.batch_size))

    model = ControlLatentJointModule(cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.query_module.parameters() if p.requires_grad],
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    step = 0
    resume_path = _resolve_resume_path_v2(cfg)
    if resume_path:
        step = _load_checkpoint_v2(
            accelerator, model, optimizer, resume_path,
            load_optimizer=bool(cfg.get("resume_optimizer", True)),
        )

    print_freq = int(cfg.get("print_freq", 10))
    save_freq = int(cfg.get("save_freq", 500))
    clip_grad = float(cfg.get("clip_grad", 1.0))
    max_steps = int(cfg.get("max_steps", 20000))
    t0 = time.time()

    for epoch in range(int(cfg.num_epochs)):
        for batch in dataloader:
            with accelerator.accumulate(model):
                loss, metrics = model(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients and clip_grad > 0:
                    accelerator.clip_grad_norm_(
                        [p for p in model.query_module.parameters() if p.requires_grad],
                        clip_grad,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and step % print_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[step {step:>6d}] loss={metrics['loss'].item():.4f} "
                    f"diff={metrics['loss_diffusion'].item():.4f} "
                    f"aux={metrics['loss_aux'].item():.4f} "
                    f"lr={lr:.2e} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

            if save_freq > 0 and step > 0 and step % save_freq == 0:
                _save_checkpoint_v2(
                    accelerator, model, optimizer, cfg, cfg.output_path, step,
                    epoch=epoch, include_optimizer=bool(cfg.get("resume_optimizer", True)),
                )

            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    _save_checkpoint_v2(
        accelerator, model, optimizer, cfg, cfg.output_path, step,
        epoch=epoch, include_optimizer=bool(cfg.get("resume_optimizer", True)),
    )


if __name__ == "__main__":
    main()
