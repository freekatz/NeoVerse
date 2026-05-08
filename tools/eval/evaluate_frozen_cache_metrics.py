import argparse
import csv
import json
import math
import os
import os.path as osp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from scipy import linalg
from tqdm import tqdm

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from diffsynth.data import save_video
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from tools.eval.render_student_comparison import (
    build_render_conditions,
    generate_with_student,
    load_adapter,
    pipeline_condition_kwargs,
    save_eval_comparison_grid,
    save_eval_condition_videos,
    save_eval_gt_comparison_grid,
    save_eval_neoverse_prediction_grid,
    zero_drop_probs,
)
from training.control_latent.distill import (
    FrozenForwardCacheDataset,
    build_spatialvid_dataset,
    frozen_cache_signature,
)
from training.data.datasets.spatialvid import SpatialVID


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path(CODE_DIR) / path


def _is_null(value):
    return value in (None, "", "none", "None", "null", "Null")


def _cache_path_for_dataset(dataset, scene_info, fixed_clip_start):
    if getattr(dataset, "camera_cache_dir", None) is None or fixed_clip_start is None:
        return None
    return dataset.camera_cache_path_for(
        dataset.camera_cache_dir,
        scene_info["id"],
        int(fixed_clip_start),
        dataset.num_views,
        dataset.height,
        dataset.width,
    )


def _spatialvid_signature_views(dataset: SpatialVID, idx: int):
    if dataset.seed is None:
        raise ValueError("Metadata-only cache signature lookup requires a deterministic dataset seed.")

    rng = np.random.default_rng(seed=int(dataset.seed) + int(idx))
    dataset._rng = rng
    num_context_views = rng.integers(dataset.min_num_context_views, dataset.max_num_context_views + 1)
    scene_idx, variant_id, fixed_clip_id, fixed_clip_start = dataset._scene_variant_index(idx)
    scene_info = dataset.scenes.iloc[scene_idx]
    annotation_dir = osp.join(dataset.ROOT, "SpatialVid/HQ", scene_info["annotation path"])
    poses_path = osp.join(annotation_dir, "poses.npy")
    intrinsics_path = osp.join(annotation_dir, "intrinsics.npy")
    has_camera_annotations = (
        bool(dataset.use_camera_annotations)
        and osp.exists(poses_path)
        and osp.exists(intrinsics_path)
    )
    annotation_length = None
    if has_camera_annotations:
        annotation_length = min(len(np.load(poses_path, mmap_mode="r")), len(np.load(intrinsics_path, mmap_mode="r")))

    video_length = int(scene_info["num frames"])
    fps = float(scene_info["fps"])
    reverse = False
    target_trajectory_indices = np.arange(dataset.num_views, dtype=np.int64)
    temporal_trajectory = None
    if dataset.temporal_augmentation:
        target_trajectory_indices, temporal_trajectory = dataset._target_trajectory_indices(
            fps,
            rng,
            num_context_views,
            scene_id=scene_info["id"],
            clip_start=fixed_clip_start,
            variant_id=variant_id,
        )
        clip_length = int(dataset.num_views)
        start = int(fixed_clip_start) if fixed_clip_start is not None else rng.integers(0, max(video_length - clip_length, 0) + 1)
        sample_index = np.arange(start, start + clip_length, dtype=np.int64)
        if has_camera_annotations:
            video_lookup = np.linspace(0, video_length - 1, annotation_length)
            sample_anno_index = np.abs(video_lookup[None, :] - sample_index[:, None]).argmin(axis=1)
        else:
            sample_anno_index = None
    elif dataset.continuous_target_frames:
        clip_length = int(dataset.num_views)
        start = int(fixed_clip_start) if fixed_clip_start is not None else rng.integers(0, max(video_length - clip_length, 0) + 1)
        sample_index = np.arange(start, start + clip_length, dtype=np.int64)
        if has_camera_annotations:
            video_lookup = np.linspace(0, video_length - 1, annotation_length)
            sample_anno_index = np.abs(video_lookup[None, :] - sample_index[:, None]).argmin(axis=1)
        else:
            sample_anno_index = None
    elif has_camera_annotations:
        sample_anno_index, reverse = dataset.sample_from_video(
            annotation_length,
            dataset.num_views,
            dataset.min_interval,
            dataset.max_interval,
            rng,
        )
        video_lookup = np.linspace(0, video_length - 1, annotation_length, dtype=int)
        sample_index = video_lookup[sample_anno_index]
    else:
        sample_index, reverse = dataset.sample_from_video(
            video_length,
            dataset.num_views,
            dataset.min_interval,
            dataset.max_interval,
            rng,
        )
        sample_anno_index = None

    context_local_indices = set(dataset._context_local_indices(num_context_views).tolist())
    context_strategy = getattr(dataset, "_last_context_strategy", dataset.context_sampling_strategy)
    first_sample_index = sample_index[0]
    first_time_index = sample_anno_index[0] if has_camera_annotations else sample_index[0]
    trajectory_type = temporal_trajectory.trajectory_type if temporal_trajectory is not None else "none"
    trajectory_profile = (
        dataset._profile_for_variant(variant_id)
        if dataset.temporal_augmentation
        else dataset.temporal_trajectory_profile
    )
    camera_cache_path = _cache_path_for_dataset(dataset, scene_info, fixed_clip_start)

    context_views = []
    target_views = []
    for view_idx in range(int(dataset.num_views)):
        time_index = sample_anno_index[view_idx] if has_camera_annotations else sample_index[view_idx]
        view = {
            "video_name": scene_info["id"],
            "image_name": f"frame_{sample_index[view_idx]:06d}",
            "timestamp": dataset._timestamp(
                sample_index,
                time_index,
                view_idx,
                first_sample_index,
                first_time_index,
                reverse,
                fps,
            ),
            "scene_idx": int(scene_idx),
            "variant_id": int(variant_id),
            "context_strategy": context_strategy,
            "trajectory_index": int(view_idx),
            "target_trajectory_index": int(target_trajectory_indices[view_idx]),
            "temporal_augmentation": bool(dataset.temporal_augmentation),
            "temporal_order": dataset.temporal_order,
            "temporal_trajectory_profile": trajectory_profile,
            "temporal_trajectory_type": trajectory_type,
        }
        if camera_cache_path is not None:
            view["camera_cache_path"] = camera_cache_path
        if view_idx in context_local_indices:
            view["is_target"] = False
            context_views.append(view)
        else:
            view["is_target"] = True
            target_views.append(view)
    return context_views + target_views


def _signature_views(dataset, idx):
    if isinstance(dataset, SpatialVID):
        return _spatialvid_signature_views(dataset, idx)
    return dataset[int(idx)]


def _select_cache_paths(paths, max_samples=None, sample_seed=0):
    paths = sorted(str(path) for path in paths)
    if max_samples is None or int(max_samples) <= 0 or int(max_samples) >= len(paths):
        return paths
    rng = np.random.default_rng(int(sample_seed))
    indices = rng.choice(len(paths), size=int(max_samples), replace=False)
    return sorted(paths[int(index)] for index in indices.tolist())


def _metadata_sample_from_cache(path):
    sidecar_path = str(Path(path).with_suffix(".json"))
    if not os.path.exists(sidecar_path):
        return None
    try:
        with open(sidecar_path, "r") as handle:
            cache = json.load(handle)
    except Exception:
        return None
    dataset_index = cache.get("dataset_index") if isinstance(cache, dict) else None
    if dataset_index in (None, "", "null", "None"):
        return None
    signature = cache.get("cache_signature") or Path(path).stem
    signature_views = cache.get("signature_views") or []
    first_view = signature_views[0] if signature_views else {}
    num_context_views = None
    if signature_views:
        num_context_views = int(sum(not bool(view.get("is_target")) for view in signature_views))
    return {
        "dataset_index": int(dataset_index),
        "cache_path": str(path),
        "signature": str(signature),
        "video_name": first_view.get("video_name"),
        "first_image_name": first_view.get("image_name"),
        "num_context_views": num_context_views,
        "mapping_source": "cache_metadata",
    }


def _map_cache_paths_to_dataset_indices(dataset, cfg, cache_paths, strict=False):
    metadata_samples = []
    missing_metadata = []
    for path in cache_paths:
        sample = _metadata_sample_from_cache(path)
        if sample is None:
            missing_metadata.append(path)
        else:
            metadata_samples.append(sample)
    if not missing_metadata:
        return metadata_samples, []

    wanted = {Path(path).stem: str(path) for path in cache_paths}
    matches = {}
    for idx in tqdm(range(len(dataset)), desc="map cache signatures"):
        views = _signature_views(dataset, idx)
        signature = frozen_cache_signature(views, cfg)
        if signature in wanted:
            matches[signature] = {
                "dataset_index": int(idx),
                "cache_path": wanted[signature],
                "signature": signature,
                "video_name": views[0].get("video_name"),
                "first_image_name": views[0].get("image_name"),
                "num_context_views": int(sum(not bool(view.get("is_target")) for view in views)),
                "mapping_source": "dataset_signature",
            }
            if len(matches) == len(wanted):
                break
    missing = [path for signature, path in wanted.items() if signature not in matches]
    if missing:
        preview = ", ".join(Path(path).name for path in missing[:8])
        message = (
            f"Could not map {len(missing)} frozen-cache files back to dataset indices. "
            f"First missing files: {preview}. Check that CONFIG matches the cache build config. "
            "Old cache files do not contain dataset_index metadata, so image metrics can only use files "
            "whose signatures match the current dataset/config."
        )
        if strict or not matches:
            raise RuntimeError(message)
        print(f"WARNING: {message}", file=sys.stderr)
    return [matches[Path(path).stem] for path in cache_paths if Path(path).stem in matches], missing


def _to_rgb_tensor(frames):
    if isinstance(frames, torch.Tensor):
        tensor = frames.detach().float().cpu()
        if tensor.ndim == 5 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError(f"Expected video tensor with 4 dims, got shape={tuple(tensor.shape)}")
        if tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        return tensor[:, :3].clamp(0, 1)

    images = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            image = frame.convert("RGB")
        else:
            image = Image.fromarray(np.asarray(frame)).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array).permute(2, 0, 1))
    if not images:
        raise ValueError("Cannot convert an empty video to a tensor.")
    return torch.stack(images, dim=0).clamp(0, 1)


def _resize_like(pred, ref):
    if pred.shape[-2:] == ref.shape[-2:]:
        return pred
    return F.interpolate(pred, size=ref.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)


def _iter_chunks(tensor, batch_size):
    batch_size = max(1, int(batch_size))
    for start in range(0, tensor.shape[0], batch_size):
        yield tensor[start : start + batch_size]


def _mean_psnr(pred, ref, eps=1e-12):
    mse = (pred - ref).pow(2).flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(eps))
    return float(psnr.mean().item())


class InceptionFeatureExtractor:
    def __init__(self, device):
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.device = torch.device(device)
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=weights, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images, batch_size):
        features = []
        for chunk in _iter_chunks(images, batch_size):
            chunk = chunk.to(self.device, dtype=torch.float32)
            chunk = F.interpolate(chunk, size=(299, 299), mode="bilinear", align_corners=False)
            chunk = (chunk - self.mean) / self.std
            feats = self.model(chunk)
            if isinstance(feats, tuple):
                feats = feats[0]
            features.append(feats.detach().float().cpu())
        return torch.cat(features, dim=0).numpy()


def _frechet_distance(real_features, fake_features):
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
    if real.shape[0] < 2 or fake.shape[0] < 2:
        return float("nan")
    mu_real = np.mean(real, axis=0)
    mu_fake = np.mean(fake, axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_fake = np.cov(fake, rowvar=False)
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(cov_real.dot(cov_fake), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((cov_real + offset).dot(cov_fake + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(cov_real) + np.trace(cov_fake) - 2.0 * np.trace(covmean)
    return float(np.real(fid))


class MetricAccumulator:
    def __init__(self, modes, device, batch_size, compute_fid=True, compute_lpips=True, compute_ssim=True):
        self.modes = list(modes)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.paired = {
            mode: {
                "frames": 0,
                "psnr_sum": 0.0,
                "ssim_sum": 0.0,
                "lpips_sum": 0.0,
                "ssim_frames": 0,
                "lpips_frames": 0,
            }
            for mode in self.modes
        }
        self.errors = {}
        self.fid_extractor = None
        self.real_features = []
        self.fake_features = {mode: [] for mode in self.modes}
        if compute_fid:
            try:
                self.fid_extractor = InceptionFeatureExtractor(self.device)
            except Exception as exc:
                self.errors["fid"] = f"{type(exc).__name__}: {exc}"
        self.ssim = None
        if compute_ssim:
            try:
                from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

                self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            except Exception as exc:
                self.errors["ssim"] = f"{type(exc).__name__}: {exc}"
        self.lpips = None
        if compute_lpips:
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

                self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(self.device)
                self.lpips.eval()
            except Exception as exc:
                self.errors["lpips"] = f"{type(exc).__name__}: {exc}"

    def update_real_features(self, ref):
        if self.fid_extractor is None:
            return
        self.real_features.append(self.fid_extractor(ref, self.batch_size))

    @torch.no_grad()
    def update(self, mode, pred, ref):
        pred = _resize_like(pred, ref)
        length = min(pred.shape[0], ref.shape[0])
        pred = pred[:length].float().clamp(0, 1)
        ref = ref[:length].float().clamp(0, 1)
        stats = self.paired[mode]
        stats["frames"] += int(length)
        stats["psnr_sum"] += _mean_psnr(pred, ref) * int(length)
        if self.fid_extractor is not None:
            self.fake_features[mode].append(self.fid_extractor(pred, self.batch_size))
        if self.ssim is not None:
            for pred_chunk, ref_chunk in zip(_iter_chunks(pred, self.batch_size), _iter_chunks(ref, self.batch_size)):
                pred_chunk = pred_chunk.to(self.device)
                ref_chunk = ref_chunk.to(self.device)
                value = self.ssim(pred_chunk, ref_chunk)
                self.ssim.reset()
                stats["ssim_sum"] += float(value.detach().cpu()) * int(pred_chunk.shape[0])
                stats["ssim_frames"] += int(pred_chunk.shape[0])
        if self.lpips is not None:
            for pred_chunk, ref_chunk in zip(_iter_chunks(pred, self.batch_size), _iter_chunks(ref, self.batch_size)):
                pred_chunk = pred_chunk.to(self.device) * 2.0 - 1.0
                ref_chunk = ref_chunk.to(self.device) * 2.0 - 1.0
                value = self.lpips(pred_chunk, ref_chunk)
                self.lpips.reset()
                stats["lpips_sum"] += float(value.detach().cpu()) * int(pred_chunk.shape[0])
                stats["lpips_frames"] += int(pred_chunk.shape[0])

    def sample_metrics(self, pred, ref):
        pred = _resize_like(pred, ref)
        length = min(pred.shape[0], ref.shape[0])
        pred = pred[:length].float().clamp(0, 1)
        ref = ref[:length].float().clamp(0, 1)
        out = {"frames": int(length), "psnr": _mean_psnr(pred, ref)}
        if self.ssim is not None:
            total = 0.0
            count = 0
            for pred_chunk, ref_chunk in zip(_iter_chunks(pred, self.batch_size), _iter_chunks(ref, self.batch_size)):
                value = self.ssim(pred_chunk.to(self.device), ref_chunk.to(self.device))
                self.ssim.reset()
                total += float(value.detach().cpu()) * int(pred_chunk.shape[0])
                count += int(pred_chunk.shape[0])
            out["ssim"] = total / max(count, 1)
        if self.lpips is not None:
            total = 0.0
            count = 0
            for pred_chunk, ref_chunk in zip(_iter_chunks(pred, self.batch_size), _iter_chunks(ref, self.batch_size)):
                value = self.lpips(pred_chunk.to(self.device) * 2.0 - 1.0, ref_chunk.to(self.device) * 2.0 - 1.0)
                self.lpips.reset()
                total += float(value.detach().cpu()) * int(pred_chunk.shape[0])
                count += int(pred_chunk.shape[0])
            out["lpips"] = total / max(count, 1)
        return out

    def compute(self):
        results = {}
        real = np.concatenate(self.real_features, axis=0) if self.real_features else None
        for mode in self.modes:
            stats = self.paired[mode]
            frames = max(int(stats["frames"]), 1)
            result = {
                "frames": int(stats["frames"]),
                "psnr": stats["psnr_sum"] / frames,
            }
            if stats["ssim_frames"] > 0:
                result["ssim"] = stats["ssim_sum"] / stats["ssim_frames"]
            if stats["lpips_frames"] > 0:
                result["lpips"] = stats["lpips_sum"] / stats["lpips_frames"]
            if real is not None and self.fake_features[mode]:
                fake = np.concatenate(self.fake_features[mode], axis=0)
                result["fid"] = _frechet_distance(real, fake)
                result["fid_backend"] = "torchvision_inception_v3_imagenet1k"
            results[mode] = result
        return results


def _resolve_lora(cfg, args):
    official_lora_path = os.path.join(
        str(cfg.model_path),
        "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
    )
    if args.disable_lora:
        return None
    lora_path = args.lora_path or cfg.get("lora_path", None)
    if lora_path is None and os.path.exists(official_lora_path):
        lora_path = official_lora_path
    return lora_path


def _write_rows(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a distill checkpoint on the frozen-cache eval split with image metrics."
    )
    parser.add_argument("config", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--cache_config", default=None)
    parser.add_argument(
        "--cache_config_overrides",
        nargs="*",
        default=None,
        help="OmegaConf dotlist overrides for the cache-build config, e.g. camera_condition_normalization.enabled=false.",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--cache_pattern", default=None)
    parser.add_argument("--split", default="eval", choices=("eval", "train", "all"))
    parser.add_argument("--eval_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--split_mode", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--modes", default="student,teacher")
    parser.add_argument("--metrics", default="fid,lpips,ssim,psnr")
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--save_grids", action="store_true")
    parser.add_argument("--strict_mapping", action="store_true")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--control_scale", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rand_device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile_stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument("--enable_vram_management", action="store_true")
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--dry_run", action="store_true", help="Resolve the eval split and mapping without loading models.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cache_cfg = OmegaConf.load(args.cache_config) if args.cache_config is not None else OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if args.cache_config_overrides:
        cache_cfg = OmegaConf.merge(cache_cfg, OmegaConf.from_dotlist(args.cache_config_overrides))
    args.height = args.height or int(cfg.height)
    args.width = args.width or int(cfg.width)
    args.num_frames = args.num_frames or int(cfg.num_views)

    checkpoint = _resolve_path(args.checkpoint)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(checkpoint.parent / "eval_frozen_cache")
    output_dir = str(_resolve_path(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    cache_dir = args.cache_dir or cache_cfg.get("frozen_cache_dir", None)
    if _is_null(cache_dir):
        raise ValueError("--cache_dir is required when cache_config.frozen_cache_dir is not set.")
    cache_pattern = args.cache_pattern or cache_cfg.get("frozen_cache_pattern", "*.pt")
    eval_ratio = float(args.eval_ratio if args.eval_ratio is not None else cache_cfg.get("frozen_cache_eval_ratio", 0.05))
    split_seed = int(args.split_seed if args.split_seed is not None else cache_cfg.get("frozen_cache_split_seed", 0))
    split_mode = args.split_mode or cache_cfg.get("frozen_cache_split_mode", "hash")

    cache_dataset = FrozenForwardCacheDataset(
        cache_dir,
        pattern=cache_pattern,
        split=args.split,
        eval_ratio=eval_ratio,
        split_seed=split_seed,
        split_mode=split_mode,
    )
    cache_paths = _select_cache_paths(cache_dataset.paths, args.max_samples, args.sample_seed)
    dataset = build_spatialvid_dataset(cache_cfg)
    mapped_samples, unmapped_cache_paths = _map_cache_paths_to_dataset_indices(
        dataset,
        cache_cfg,
        cache_paths,
        strict=args.strict_mapping,
    )
    with open(os.path.join(output_dir, "frozen_cache_eval_samples.json"), "w") as handle:
        json.dump(mapped_samples, handle, indent=2, default=_json_default)
    if unmapped_cache_paths:
        with open(os.path.join(output_dir, "unmapped_cache_files.json"), "w") as handle:
            json.dump([str(path) for path in unmapped_cache_paths], handle, indent=2)

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unsupported = sorted(set(modes) - {"teacher", "student"})
    if unsupported:
        raise ValueError(f"Unsupported mode(s): {', '.join(unsupported)}. Expected teacher,student.")

    metric_names = {item.strip().lower() for item in args.metrics.split(",") if item.strip()}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "cache_dir": str(cache_dir),
                    "cache_config": str(args.cache_config or args.config),
                    "cache_config_overrides": args.cache_config_overrides or [],
                    "split": args.split,
                    "split_counts": cache_dataset.split_counts,
                    "selected_samples": len(mapped_samples),
                    "unmapped_cache_files": len(unmapped_cache_paths),
                    "output_dir": output_dir,
                },
                indent=2,
            )
        )
        return

    lora_path = _resolve_lora(cfg, args)
    use_lora = lora_path is not None
    if args.num_inference_steps is None:
        args.num_inference_steps = 4 if use_lora else 50
    if args.cfg_scale is None:
        args.cfg_scale = 1.0 if use_lora else 5.0
    print(
        f"Resolved generation params: steps={args.num_inference_steps}, "
        f"cfg_scale={args.cfg_scale}, lora_path={lora_path}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=cfg.model_path,
        reconstructor_path=cfg.reconstructor_path,
        pipeline_kwargs=zero_drop_probs(cfg.pipeline_kwargs),
        device=device,
        torch_dtype=getattr(torch, cfg.torch_dtype),
        lora_path=lora_path,
        lora_alpha=float(cfg.get("lora_alpha", 1.0)),
        enable_vram_management=args.enable_vram_management,
    )
    pipe.eval()
    adapter = load_adapter(pipe, cfg, str(checkpoint), pipe.device) if "student" in modes else None
    metrics = MetricAccumulator(
        modes,
        device=pipe.device,
        batch_size=args.metric_batch_size,
        compute_fid="fid" in metric_names,
        compute_lpips="lpips" in metric_names,
        compute_ssim="ssim" in metric_names,
    )

    per_sample_rows = []
    jsonl_path = os.path.join(output_dir, "per_sample_metrics.jsonl")
    open(jsonl_path, "w").close()
    start_time = time.time()
    for sample in tqdm(mapped_samples, desc="eval samples"):
        dataset_index = int(sample["dataset_index"])
        source_views = dataset[dataset_index]
        prompt = source_views[0]["prompt"]
        render_conditions = build_render_conditions(pipe, source_views, args, cfg)
        ref = _to_rgb_tensor(render_conditions["gt_target_rgb"][0])
        metrics.update_real_features(ref)
        eval_frames = {}
        sample_dir = os.path.join(output_dir, f"idx{dataset_index:06d}_{sample['signature'][:10]}")
        if args.save_videos or args.save_grids:
            os.makedirs(sample_dir, exist_ok=True)
            eval_frames.update(save_eval_condition_videos(sample_dir, render_conditions, fps=15))
        row = {
            "dataset_index": dataset_index,
            "cache_path": sample["cache_path"],
            "signature": sample["signature"],
            "video_name": sample.get("video_name"),
            "num_context_views": sample.get("num_context_views"),
        }
        for mode in modes:
            if mode == "teacher":
                video = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    **pipeline_condition_kwargs(render_conditions),
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    sigma_shift=args.sigma_shift,
                    control_scale=args.control_scale,
                    cfg_scale=args.cfg_scale,
                    seed=args.seed,
                    rand_device=args.rand_device,
                    tiled=args.tiled,
                    tile_size=tuple(args.tile_size),
                    tile_stride=tuple(args.tile_stride),
                )
            else:
                video = generate_with_student(pipe, adapter, render_conditions, prompt, cfg, args)
            pred = _to_rgb_tensor(video)
            sample_metric = metrics.sample_metrics(pred, ref)
            metrics.update(mode, pred, ref)
            for name, value in sample_metric.items():
                row[f"{mode}_{name}"] = value
            if args.save_videos or args.save_grids:
                eval_frames[mode] = video
                save_video(video, os.path.join(sample_dir, f"{mode}.mp4"), fps=15)
        if args.save_grids:
            save_eval_comparison_grid(sample_dir, eval_frames, fps=15)
            save_eval_gt_comparison_grid(sample_dir, eval_frames, fps=15)
            save_eval_neoverse_prediction_grid(sample_dir, eval_frames, fps=15)
        per_sample_rows.append(row)
        _write_rows(os.path.join(output_dir, "per_sample_metrics.csv"), per_sample_rows)
        with open(jsonl_path, "a") as handle:
            handle.write(json.dumps(row, default=_json_default) + "\n")

    summary = {
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "cache_config": str(args.cache_config or args.config),
        "cache_config_overrides": args.cache_config_overrides or [],
        "cache_dir": str(cache_dir),
        "cache_pattern": str(cache_pattern),
        "split": args.split,
        "split_counts": cache_dataset.split_counts,
        "selected_samples": len(mapped_samples),
        "unmapped_cache_files": len(unmapped_cache_paths),
        "modes": modes,
        "metrics": metrics.compute(),
        "metric_errors": metrics.errors,
        "output_dir": output_dir,
        "elapsed_seconds": time.time() - start_time,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    print(json.dumps(summary["metrics"], indent=2, default=_json_default))
    if metrics.errors:
        print(f"Metric warnings: {metrics.errors}")


if __name__ == "__main__":
    main()
