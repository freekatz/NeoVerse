import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from training.config_utils import build_spatialvid_dataset
from training.control_latent.cache import frozen_cache_signature
from training.control_latent.module import ControlLatentDistillModule


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_filter_values(value):
    if value in (None, "", "null", "None"):
        return None
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    items = [item.strip().strip("'\"") for item in text.split(",")]
    return [item for item in items if item] or None


def torch_dtype_from_name(name):
    if isinstance(name, torch.dtype):
        return name
    return getattr(torch, str(name))


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def build_cfg(args):
    cfg = OmegaConf.load(args.config)
    fixed_clips_per_scene = (
        args.fixed_clips_per_scene
        if args.fixed_clips_per_scene is not None
        else int(cfg.get("fixed_clips_per_scene", 16) or 16)
    )
    trajectories_per_clip = (
        args.trajectories_per_clip
        if args.trajectories_per_clip is not None
        else int(cfg.get("trajectories_per_clip", cfg.get("variants_per_scene", 8)) or 8)
    )
    dataset_seed = args.dataset_seed
    if dataset_seed is None:
        dataset_seed = cfg.get("dataset_seed", cfg.get("seed", 0))
    if dataset_seed in (None, "null", "None"):
        dataset_seed = 0
    camera_condition_normalize = (
        parse_bool(args.camera_condition_normalize)
        if args.camera_condition_normalize is not None
        else True
    )
    camera_condition_min_translation_scale = (
        float(args.camera_condition_min_translation_scale)
        if args.camera_condition_min_translation_scale is not None
        else 1.0
    )

    overrides = {
        "output_path": args.run_output_dir,
        "data_root": args.data_root or cfg.get("data_root", "data/SpatialVID"),
        "video_ids": normalize_filter_values(args.video_ids if args.video_ids is not None else cfg.get("video_ids", None)),
        "video_paths": normalize_filter_values(args.video_paths if args.video_paths is not None else cfg.get("video_paths", None)),
        "torch_dtype": args.torch_dtype or cfg.get("torch_dtype", "bfloat16"),
        "model_path": args.model_path or cfg.get("model_path", "./models"),
        "reconstructor_path": args.reconstructor_path or cfg.get("reconstructor_path", "./models/NeoVerse/reconstructor.ckpt"),
        "continuous_target_frames": True,
        "temporal_augmentation": True,
        "temporal_order": args.temporal_order or cfg.get("temporal_order", "trajectory"),
        "temporal_trajectory_profile": args.temporal_trajectory_profile or cfg.get("temporal_trajectory_profile", "forward_pause"),
        "temporal_variant_profile_weights": (
            args.temporal_variant_profile_weights
            if args.temporal_variant_profile_weights is not None
            else cfg.get("temporal_variant_profile_weights", "")
        ),
        "context_sampling_strategy": args.context_sampling_strategy or "mixed",
        "variants_per_scene": int(trajectories_per_clip),
        "trajectories_per_clip": int(trajectories_per_clip),
        "fixed_clips_per_scene": int(fixed_clips_per_scene),
        "camera_cache_dir": args.camera_cache_dir if args.camera_cache_dir is not None else cfg.get("camera_cache_dir", ""),
        "camera_cache_required": parse_bool(args.camera_cache_required)
        if args.camera_cache_required is not None
        else parse_bool(cfg.get("camera_cache_required", False)),
        "dataset_seed": int(dataset_seed),
        "camera_condition_normalization": {
            "enabled": bool(camera_condition_normalize),
            "min_translation_scale": float(camera_condition_min_translation_scale),
        },
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "cache_frozen_outputs": False,
        "frozen_cache_dir": args.output_dir,
        "frozen_cache_read": not bool(args.overwrite),
        "frozen_cache_write": True,
        "preload_frozen_cache": False,
        "load_dit": False,
        "load_text_encoder": False,
        "load_vae": True,
    }
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides), OmegaConf.from_dotlist(args.overrides))
    else:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def prepare_model(cfg, device):
    model = ControlLatentDistillModule(cfg).to(device=device, dtype=torch_dtype_from_name(cfg.torch_dtype))
    model.eval()
    model.pipe.device = device
    if getattr(model.pipe, "reconstructor", None) is not None:
        model.pipe.reconstructor.to(device)
        if hasattr(model.pipe.reconstructor, "visual_geometry_transformer"):
            vgt = model.pipe.reconstructor.visual_geometry_transformer
            for name in ("_resnet_mean", "_resnet_std"):
                if hasattr(vgt, name):
                    setattr(vgt, name, getattr(vgt, name).to(device))
    if getattr(model.pipe, "control_branch", None) is not None:
        model.pipe.control_branch.to(device)
    if getattr(model.pipe, "vae", None) is not None:
        model.pipe.vae.to(device)
    model.pipe.load_models_to_device(["reconstructor", "control_branch", "vae"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Build fixed clip/trajectory frozen forward caches for SpatialVID distillation.")
    parser.add_argument("--config", default="configs/distill/control_latent.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default="outputs/NeoVerseControlLatentDistill/frozen_cache")
    parser.add_argument("--run_output_dir", default="outputs/NeoVerseControlLatentDistill/frozen_cache_build")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--reconstructor_path", default=None)
    parser.add_argument("--torch_dtype", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed_clips_per_scene", type=int, default=None)
    parser.add_argument("--trajectories_per_clip", type=int, default=None)
    parser.add_argument("--temporal_trajectory_profile", default=None)
    parser.add_argument("--temporal_variant_profile_weights", default=None)
    parser.add_argument("--temporal_order", default=None)
    parser.add_argument("--context_sampling_strategy", default=None)
    parser.add_argument("--camera_condition_normalize", default=None)
    parser.add_argument("--camera_condition_min_translation_scale", default=None)
    parser.add_argument("--camera_cache_dir", default=None)
    parser.add_argument("--camera_cache_required", default=None)
    parser.add_argument("--dataset_seed", type=int, default=None)
    parser.add_argument("--video_ids", default=None)
    parser.add_argument("--video_paths", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Extra OmegaConf dotlist overrides.")
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards.")

    cfg = build_cfg(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(cfg.output_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_path, f"config_shard{args.shard_index}.yaml"))

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    dataset = build_spatialvid_dataset(cfg)
    model = None

    total = len(dataset) if args.limit is None else min(len(dataset), int(args.limit))
    indices = range(args.shard_index, total, args.num_shards)
    written = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    for idx in tqdm(indices, desc=f"frozen-cache shard {args.shard_index}/{args.num_shards}"):
        batch = default_collate([dataset[idx]])
        signature = frozen_cache_signature(batch, cfg)
        cache_path = os.path.join(str(cfg.frozen_cache_dir), f"{signature}.pt")
        if os.path.exists(cache_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            if model is None:
                model = prepare_model(cfg, device)
            batch = to_device(batch, device)
            with torch.no_grad():
                model.get_forward_cache(batch, device="cpu")
            written += 1
        except Exception as exc:
            failed += 1
            print(f"failed idx={idx}: {exc}", flush=True)
            if not args.continue_on_error:
                raise

    elapsed = time.time() - start_time
    print(f"frozen_cache_dir={cfg.frozen_cache_dir}")
    print(
        f"total={total} shard_index={args.shard_index} num_shards={args.num_shards} "
        f"written={written} skipped={skipped} failed={failed} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
