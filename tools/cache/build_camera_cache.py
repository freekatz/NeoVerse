import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from diffsynth.models import ModelManager
from training.data.datasets.spatialvid import SpatialVID


def normalize_filter_values(value):
    if value in (None, "", "null", "None"):
        return None
    if isinstance(value, (list, tuple)):
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
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    items = [item.strip().strip("'\"") for item in text.split(",")]
    return [item for item in items if item] or None


def torch_dtype_from_name(name):
    if isinstance(name, torch.dtype):
        return name
    return getattr(torch, str(name))


def build_dataset(cfg, args):
    data_root = args.data_root or cfg.data_root
    fixed_clips_per_scene = (
        args.fixed_clips_per_scene
        if args.fixed_clips_per_scene is not None
        else int(cfg.get("fixed_clips_per_scene", 16) or 16)
    )
    return SpatialVID(
        split=None,
        ROOT=data_root,
        video_ids=normalize_filter_values(args.video_ids if args.video_ids is not None else cfg.get("video_ids", None)),
        video_paths=normalize_filter_values(args.video_paths if args.video_paths is not None else cfg.get("video_paths", None)),
        use_camera_annotations=False,
        continuous_target_frames=True,
        force_first_context=True,
        timestamp_unit=str(cfg.get("timestamp_unit", "seconds")),
        temporal_augmentation=False,
        temporal_trajectory_profile=str(cfg.get("temporal_trajectory_profile", "forward_pause")),
        temporal_order=str(cfg.get("temporal_order", "trajectory")),
        temporal_max_condition_frames=int(cfg.get("temporal_max_condition_frames", 8)),
        context_sampling_strategy="prefix",
        context_sampling_weights=None,
        variants_per_scene=1,
        fixed_clips_per_scene=fixed_clips_per_scene,
        camera_cache_dir=None,
        min_interval=1,
        max_interval=1,
        height=int(cfg.height),
        width=int(cfg.width),
        num_views=int(cfg.num_views),
        min_num_context_views=1,
        max_num_context_views=1,
        seed=int(cfg.get("dataset_seed", cfg.get("seed", 0))),
    )


def load_reconstructor(cfg, args, device):
    dtype = torch_dtype_from_name(args.torch_dtype or cfg.get("torch_dtype", "bfloat16"))
    manager = ModelManager(torch_dtype=dtype, device=device)
    manager.load_model(
        args.reconstructor_path or cfg.reconstructor_path,
        model_names=["reconstructor"],
        device=device,
        torch_dtype=dtype,
    )
    reconstructor = manager.fetch_model("reconstructor")
    if reconstructor is None:
        raise RuntimeError("Failed to load reconstructor.")
    return reconstructor.eval().to(device), dtype


def continuous_views(views):
    return sorted(views, key=lambda view: int(view["trajectory_index"]))


@torch.no_grad()
def reconstruct_cameras(reconstructor, images, dtype):
    device = images.device
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        token_list, _, _, _ = reconstructor.visual_geometry_transformer(images, use_motion=False)
        cam_seq = reconstructor.cam_head(token_list)
        camera_poses, camera_intrs = reconstructor.transform_camera_vector(
            cam_seq[-1],
            images.shape[-2],
            images.shape[-1],
        )
    return camera_poses.float(), camera_intrs.float()


def cache_path(output_dir, views, num_views, height, width):
    first = views[0]
    return SpatialVID.camera_cache_path_for(
        output_dir,
        first["video_name"],
        int(first["clip_start"]),
        num_views,
        height,
        width,
    )


def write_cache(path, views, camera_poses, camera_intrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(
            handle,
            camera_poses=camera_poses.squeeze(0).detach().cpu().numpy().astype(np.float32),
            camera_intrs=camera_intrs.squeeze(0).detach().cpu().numpy().astype(np.float32),
            timestamps=np.asarray([view["timestamp"] for view in views], dtype=np.float32),
            frame_indices=np.asarray([view["frame_index"] for view in views], dtype=np.int64),
            image_names=np.asarray([view["image_name"] for view in views]),
            video_name=str(views[0]["video_name"]),
            clip_start=np.int64(views[0]["clip_start"]),
        )
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Build fixed-clip SpatialVID camera cache with the NeoVerse reconstructor.")
    parser.add_argument("--config", default="configs/distill/control_latent.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default="outputs/NeoVerseControlLatentDistill/camera_cache")
    parser.add_argument("--reconstructor_path", default=None)
    parser.add_argument("--torch_dtype", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed_clips_per_scene", type=int, default=None)
    parser.add_argument("--video_ids", default=None)
    parser.add_argument("--video_paths", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards.")

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    dataset = build_dataset(cfg, args)
    reconstructor, dtype = load_reconstructor(cfg, args, device)

    total = len(dataset) if args.limit is None else min(len(dataset), int(args.limit))
    indices = range(args.shard_index, total, args.num_shards)
    written = 0
    skipped = 0
    seen = 0
    for idx in tqdm(indices, desc=f"camera-cache shard {args.shard_index}/{args.num_shards}"):
        seen += 1
        views = continuous_views(dataset[idx])
        path = cache_path(args.output_dir, views, int(cfg.num_views), int(cfg.height), int(cfg.width))
        if os.path.exists(path) and not args.overwrite:
            skipped += 1
            continue

        images = torch.stack([view["img"] for view in views], dim=0).unsqueeze(0).to(device=device)
        camera_poses, camera_intrs = reconstruct_cameras(reconstructor, images, dtype)
        write_cache(path, views, camera_poses, camera_intrs)
        written += 1

    print(f"camera_cache_dir={args.output_dir}")
    print(f"total={total} shard_index={args.shard_index} num_shards={args.num_shards} seen={seen} written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
