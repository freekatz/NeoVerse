import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from omegaconf import OmegaConf

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from training.control_latent import distill as train_mod
from training.control_latent.distill import ControlLatentDistillModule


class Timer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.records = defaultdict(float)
        self.counts = defaultdict(int)

    def sync(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def block(self, name):
        timer = self

        class Block:
            def __enter__(self):
                timer.sync()
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                timer.sync()
                timer.records[name] += time.perf_counter() - self.start
                timer.counts[name] += 1

        return Block()

    def wrap_method(self, obj, attr, name):
        if obj is None or not hasattr(obj, attr):
            return
        original = getattr(obj, attr)
        timer = self

        def wrapped(*args, **kwargs):
            block_name = name
            if name == "reconstructor.vgt.forward" and args and isinstance(args[0], torch.Tensor):
                block_name = f"{name}.shape={tuple(args[0].shape)}"
            with timer.block(block_name):
                return original(*args, **kwargs)

        setattr(obj, attr, wrapped)

    def wrap_function(self, module, attr, name):
        original = getattr(module, attr)
        timer = self

        def wrapped(*args, **kwargs):
            with timer.block(name):
                return original(*args, **kwargs)

        setattr(module, attr, wrapped)

    def summary(self):
        return [
            {
                "name": name,
                "seconds": seconds,
                "count": self.counts[name],
                "avg_seconds": seconds / max(self.counts[name], 1),
            }
            for name, seconds in sorted(self.records.items(), key=lambda item: item[1], reverse=True)
        ]


def build_cfg(args):
    cfg = OmegaConf.load(args.config)
    defaults = {
        "output_path": "profile_output/distill_step",
        "num_views": args.num_views,
        "min_num_context_views": args.min_context,
        "max_num_context_views": args.max_context,
        "continuous_target_frames": True,
        "force_first_context": True,
        "timestamp_unit": "seconds",
        "temporal_augmentation": args.temporal_augmentation,
        "temporal_trajectory_profile": args.temporal_profile,
        "temporal_order": "trajectory",
        "temporal_max_condition_frames": 8,
        "context_sampling_strategy": "mixed",
        "variants_per_scene": 1,
        "trajectories_per_clip": None,
        "temporal_variant_profile_weights": "",
        "pipeline_kwargs": {"mask_non_context_targets": False},
        "enable_vram_management": False,
        "reuse_reconstructor_tokens": True,
        "skip_unused_reconstructor_heads": True,
        "reuse_context_forward_tokens": True,
        "dataset_seed": args.dataset_seed,
        "use_camera_annotations": False,
        "camera_condition_normalization": {"enabled": True, "min_translation_scale": 1.0},
        "adapter": {"type": args.adapter_type},
        "batch_vae_teacher_embeds": args.batch_vae_teacher_embeds,
        "num_workers": 0,
        "pin_memory": False,
        "cache_frozen_outputs": False,
        "frozen_cache_dir": None,
        "preload_frozen_cache": False,
    }
    if args.data_root is not None:
        defaults["data_root"] = args.data_root
    if args.extra:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(defaults), OmegaConf.from_dotlist(args.extra))
    else:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(defaults))
    return cfg


def install_wrappers(model, timer):
    pipe = model.pipe
    for unit in pipe.units:
        timer.wrap_method(unit, "process", f"unit.{unit.__class__.__name__}")

    timer.wrap_method(pipe.reconstructor, "forward", "reconstructor.forward")
    if hasattr(pipe.reconstructor, "visual_geometry_transformer"):
        timer.wrap_method(pipe.reconstructor.visual_geometry_transformer, "forward", "reconstructor.vgt.forward")
    if hasattr(pipe.reconstructor, "gs_renderer"):
        timer.wrap_method(pipe.reconstructor.gs_renderer, "render", "reconstructor.gs_renderer.render")
        if hasattr(pipe.reconstructor.gs_renderer, "rasterizer"):
            timer.wrap_method(pipe.reconstructor.gs_renderer.rasterizer, "forward", "target.rasterizer.forward")
    if getattr(pipe, "vae", None) is not None:
        timer.wrap_method(pipe.vae, "encode", "vae.encode")
    timer.wrap_method(pipe.control_branch, "encode_condition", "teacher.encode_condition")
    timer.wrap_method(model.adapter, "forward", "adapter.forward")
    timer.wrap_function(train_mod, "extract_vggt_tokens", "extract_vggt_tokens.context")


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distill/control_latent.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_views", type=int, default=81)
    parser.add_argument("--min_context", type=int, default=10)
    parser.add_argument("--max_context", type=int, default=20)
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--temporal_augmentation", action="store_true", default=True)
    parser.add_argument("--temporal_profile", default="forward_pause")
    parser.add_argument("--adapter_type", default="cross_attention_rope")
    parser.add_argument("--batch_vae_teacher_embeds", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default="profile_output/distill_step_profile.json")
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()

    cfg = build_cfg(args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = eval(cfg.train_dataset, vars(train_mod))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    data_iter = iter(dataloader)

    model = ControlLatentDistillModule(cfg).to(device=device, dtype=getattr(torch, cfg.torch_dtype))
    model.pipe.reconstructor.to(device)
    model.pipe.control_branch.to(device)
    model.pipe.vae.to(device)
    if hasattr(model.pipe.reconstructor, "visual_geometry_transformer"):
        vgt = model.pipe.reconstructor.visual_geometry_transformer
        for name in ("_resnet_mean", "_resnet_std"):
            if hasattr(vgt, name):
                setattr(vgt, name, getattr(vgt, name).to(device))
    model.pipe.load_models_to_device(["reconstructor", "control_branch", "vae"])
    model.train()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))

    timer = Timer(device)
    install_wrappers(model, timer)

    for _ in range(args.warmup):
        batch = to_device(next(data_iter), device)
        loss, _ = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.steps):
        batch = to_device(next(data_iter), device)
        with timer.block("step.total"):
            with timer.block("model.forward"):
                loss, metrics = model(batch)
            with timer.block("backward"):
                loss.backward()
            with timer.block("optimizer.step_zero"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    result = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "loss": float(loss.detach().float().cpu()),
        "metrics": {key: float(value.detach().float().cpu()) for key, value in metrics.items()},
        "records": timer.summary(),
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None
        ),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    for item in result["records"]:
        print(f"{item['name']:45s} {item['seconds']:9.3f}s  n={item['count']}")
    print(f"loss={result['loss']:.6f}")
    if result["peak_cuda_memory_gb"] is not None:
        print(f"peak_cuda_memory_gb={result['peak_cuda_memory_gb']:.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
