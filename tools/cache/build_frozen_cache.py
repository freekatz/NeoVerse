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

from models.control_latent.cache import frozen_cache_signature
from models.control_latent.module import ControlLatentDistillModule
from utils.config import build_spatialvid_dataset
from utils.config import resolve_repo_path


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
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.frozen_cache_dir = resolve_repo_path(args.output_dir)
    cfg.output_path = resolve_repo_path(args.run_output_dir)
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.persistent_workers = False
    cfg.preload_frozen_cache = False
    cfg.load_dit = False
    cfg.load_text_encoder = False
    cfg.load_vae = True
    cfg.frozen_cache_write = True
    if args.overwrite:
        cfg.frozen_cache_read = False
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
    parser.add_argument("--output_dir", default="data/frozen_cache")
    parser.add_argument("--run_output_dir", default="outputs/NeoVerseControlLatentDistill/frozen_cache_build")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides (key=value).")
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

    limit = cfg.get("limit", None)
    total = len(dataset) if limit is None else min(len(dataset), int(limit))
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
