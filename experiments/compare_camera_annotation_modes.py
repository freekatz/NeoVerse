import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from diffsynth.data import save_video
from diffsynth.models import ModelManager
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth.utils import ModelConfig
from diffsynth.utils.auxiliary import homo_matrix_inverse
from training.data.datasets.spatialvid import SpatialVID


CAMERA_KEYS = {"camera_poses", "camera_intrs"}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def zero_drop_probs(pipeline_kwargs):
    kwargs = deepcopy(OmegaConf.to_container(pipeline_kwargs, resolve=True))
    kwargs["prompt_drop_prob"] = 0.0
    kwargs["mask_drop_prob"] = 0.0
    kwargs["condition_drop_prob"] = 0.0
    kwargs["culling_prob"] = 1.0
    kwargs["kernel_size_range"] = [0, 0]
    return kwargs


def batch_dataset_views(views, device):
    batched = {}
    for key in views[0].keys():
        values = [view[key] for view in views]
        first = values[0]
        if isinstance(first, torch.Tensor):
            batched[key] = torch.stack(values, dim=0).unsqueeze(0).to(device)
        elif isinstance(first, np.ndarray):
            batched[key] = torch.from_numpy(np.stack(values, axis=0)).unsqueeze(0).to(device)
        elif isinstance(first, bool):
            batched[key] = torch.tensor([values], device=device, dtype=torch.bool)
        elif isinstance(first, (int, np.integer)):
            batched[key] = torch.tensor([values], device=device, dtype=torch.long)
        elif isinstance(first, (float, np.floating)):
            batched[key] = torch.tensor([values], device=device, dtype=torch.float32)
        elif isinstance(first, str):
            batched[key] = [values]
    return batched


def to_device_views(views, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in views.items()}


def drop_camera_annotations(views):
    return {key: value for key, value in views.items() if key not in CAMERA_KEYS}


def slice_context_views(views):
    is_target = views.get("is_target")
    if not isinstance(is_target, torch.Tensor) or is_target.ndim != 2:
        return views, views["img"].shape[1]
    context_num = int((~is_target[0].bool()).sum().item())
    total_views = views["img"].shape[1]
    context_views = {}
    for key, value in views.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == total_views:
            context_views[key] = value[:, :context_num]
        else:
            context_views[key] = value
    context_views["is_target"] = torch.zeros_like(is_target[:, :context_num])
    return context_views, context_num


def sort_target_trajectory(target_poses, target_intrs, target_timestamps, is_target=None):
    sorted_poses = target_poses.clone()
    sorted_intrs = target_intrs.clone()
    sorted_timestamps = target_timestamps.clone()
    sorted_is_target = None if is_target is None else is_target.clone()
    for b_idx in range(target_timestamps.shape[0]):
        order_indices = torch.argsort(target_timestamps[b_idx])
        sorted_timestamps[b_idx] = target_timestamps[b_idx][order_indices]
        sorted_poses[b_idx] = target_poses[b_idx][order_indices]
        sorted_intrs[b_idx] = target_intrs[b_idx][order_indices]
        if sorted_is_target is not None:
            sorted_is_target[b_idx] = sorted_is_target[b_idx][order_indices]
    return sorted_poses, sorted_intrs, sorted_timestamps, sorted_is_target


def tensor_video_to_pil(frames):
    frames = frames.detach().float().cpu().clamp(0, 1)
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim == 4 and frames.shape[1] in {1, 3}:
        frames = frames.permute(0, 2, 3, 1)
    if frames.ndim == 3:
        frames = frames[..., None]
    images = []
    for frame in frames:
        array = (frame * 255).round().to(torch.uint8).numpy()
        if array.shape[-1] == 1:
            array = array[..., 0]
        images.append(Image.fromarray(array))
    return images


def normalize_video(frames, min_value=None, max_value=None):
    frames = frames.detach().float().cpu()
    if min_value is None:
        min_value = float(frames.amin())
    if max_value is None:
        max_value = float(frames.amax())
    return ((frames - min_value) / max(max_value - min_value, 1e-6)).clamp(0, 1)


def label_frame(frame, label):
    image = frame.convert("RGB")
    draw = ImageDraw.Draw(image)
    padding = 5
    try:
        box = draw.textbbox((0, 0), label)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
    except AttributeError:
        text_width, text_height = draw.textsize(label)
    box_width = max(72, text_width + padding * 2)
    box_height = text_height + padding * 2
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0))
    draw.text((padding, padding), label, fill=(255, 255, 255))
    return image


def make_two_col(left_frames, right_frames, left_label, right_label):
    out = []
    for left, right in zip(left_frames, right_frames):
        left = label_frame(left, left_label)
        right = label_frame(right, right_label)
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (0, 0, 0))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        out.append(canvas)
    return out


def make_four_grid(gt_frames, true_frames, false_frames, diff_frames):
    out = []
    length = min(len(gt_frames), len(true_frames), len(false_frames), len(diff_frames))
    for idx in range(length):
        gt = label_frame(gt_frames[idx], "GT sampled frame")
        true = label_frame(true_frames[idx], "USE_CAMERA_ANNOTATIONS=true")
        false = label_frame(false_frames[idx], "USE_CAMERA_ANNOTATIONS=false")
        diff = label_frame(diff_frames[idx], "abs(true - false)")
        width, height = gt.size
        canvas = Image.new("RGB", (width * 2, height * 2), (0, 0, 0))
        canvas.paste(gt, (0, 0))
        canvas.paste(true, (width, 0))
        canvas.paste(false, (0, height))
        canvas.paste(diff, (width, height))
        out.append(canvas)
    return out


def make_diff_video(left, right):
    diff = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    diff_gray = diff.mean(dim=-1, keepdim=True)
    max_value = float(diff_gray.amax())
    return tensor_video_to_pil(normalize_video(diff_gray, min_value=0.0, max_value=max_value))


def video_l1(left, right):
    return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().mean().item())


def video_psnr(left, right):
    mse = float(((left.detach().float().cpu() - right.detach().float().cpu()) ** 2).mean().item())
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def mask_iou(left, right):
    left = left.detach().bool().cpu()
    right = right.detach().bool().cpu()
    inter = (left & right).sum().item()
    union = (left | right).sum().item()
    return float(inter / max(union, 1))


def align_by_nearest_timestamps(reference, candidate, candidate_timestamps, reference_timestamps):
    reference_timestamps = reference_timestamps[0].detach().float().cpu()
    candidate_timestamps = candidate_timestamps[0].detach().float().cpu()
    indices = []
    deltas = []
    for timestamp in reference_timestamps:
        idx = int(torch.argmin((candidate_timestamps - timestamp).abs()).item())
        indices.append(idx)
        deltas.append(float((candidate_timestamps[idx] - timestamp).abs().item()))
    index_tensor = torch.tensor(indices, dtype=torch.long, device=candidate.device)
    return candidate.index_select(1, index_tensor), deltas


def source_timeline_frames(views, target_timestamps):
    source_video = views["img"][0].detach().float().permute(0, 2, 3, 1).cpu().clamp(0, 1)
    source_timestamps = views["timestamp"][0].detach().float().cpu()
    target_timestamps = target_timestamps[0].detach().float().cpu()
    frames = []
    for timestamp in target_timestamps:
        idx = int(torch.argmin((source_timestamps - timestamp).abs()).item())
        frames.append(source_video[idx])
    return tensor_video_to_pil(torch.stack(frames, dim=0))


def context_timeline_frames(views, target_timestamps):
    source_video = views["img"][0].detach().float().permute(0, 2, 3, 1).cpu().clamp(0, 1)
    source_timestamps = views["timestamp"][0].detach().float().cpu()
    target_timestamps = target_timestamps[0].detach().float().cpu()
    context_mask = ~views["is_target"][0].detach().bool().cpu()
    black = torch.zeros_like(source_video[0])
    frames = []
    for timestamp in target_timestamps:
        idx = int(torch.argmin((source_timestamps - timestamp).abs()).item())
        frames.append(source_video[idx] if bool(context_mask[idx]) else black)
    return tensor_video_to_pil(torch.stack(frames, dim=0))


def intrinsics_summary(true_intrs, false_intrs):
    true_intrs = true_intrs[0].detach().float().cpu()
    false_intrs = false_intrs[0].detach().float().cpu()
    fx_rel = ((false_intrs[:, 0, 0] - true_intrs[:, 0, 0]).abs() / true_intrs[:, 0, 0].abs().clamp_min(1e-6))
    fy_rel = ((false_intrs[:, 1, 1] - true_intrs[:, 1, 1]).abs() / true_intrs[:, 1, 1].abs().clamp_min(1e-6))
    cx_abs = (false_intrs[:, 0, 2] - true_intrs[:, 0, 2]).abs()
    cy_abs = (false_intrs[:, 1, 2] - true_intrs[:, 1, 2]).abs()
    return {
        "fx_relative_abs_mean": float(fx_rel.mean().item()),
        "fx_relative_abs_max": float(fx_rel.max().item()),
        "fy_relative_abs_mean": float(fy_rel.mean().item()),
        "fy_relative_abs_max": float(fy_rel.max().item()),
        "cx_abs_px_mean": float(cx_abs.mean().item()),
        "cy_abs_px_mean": float(cy_abs.mean().item()),
    }


def umeyama_align(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    cov = target_centered.T @ source_centered / max(len(source), 1)
    u, singular_values, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    var_source = np.mean(np.sum(source_centered ** 2, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ correction) / max(var_source, 1e-12))
    translation = target_mean - scale * rotation @ source_mean
    aligned = scale * (source @ rotation.T) + translation
    return aligned, scale, rotation, translation


def camera_position_summary(true_poses, false_poses):
    true_poses = true_poses[0].detach().float().cpu().numpy()
    false_poses = false_poses[0].detach().float().cpu().numpy()
    true_xyz = true_poses[:, :3, 3]
    false_xyz = false_poses[:, :3, 3]
    false_aligned, scale, _, _ = umeyama_align(false_xyz, true_xyz)
    errors = np.linalg.norm(false_aligned - true_xyz, axis=1)
    radius = float(np.sqrt(np.mean(np.sum((true_xyz - true_xyz.mean(axis=0)) ** 2, axis=1))))
    return {
        "sim3_aligned_position_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "sim3_aligned_position_mae": float(np.mean(errors)),
        "sim3_aligned_position_rmse_over_gt_radius": float(np.sqrt(np.mean(errors ** 2)) / max(radius, 1e-12)),
        "sim3_scale_false_to_true": float(scale),
        "gt_trajectory_radius": radius,
    }


def save_camera_plot(path, true_poses, false_poses):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    true_xyz = true_poses[0].detach().float().cpu().numpy()[:, :3, 3]
    false_xyz = false_poses[0].detach().float().cpu().numpy()[:, :3, 3]
    false_aligned, _, _, _ = umeyama_align(false_xyz, true_xyz)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    axes[0].plot(true_xyz[:, 0], true_xyz[:, 2], "-o", markersize=2, label="GT annotation")
    axes[0].plot(false_aligned[:, 0], false_aligned[:, 2], "-o", markersize=2, label="WorldMirror aligned")
    axes[0].set_title("top view: x/z")
    axes[0].axis("equal")
    axes[0].legend(fontsize=8)
    axes[1].plot(true_xyz[:, 1], true_xyz[:, 2], "-o", markersize=2, label="GT annotation")
    axes[1].plot(false_aligned[:, 1], false_aligned[:, 2], "-o", markersize=2, label="WorldMirror aligned")
    axes[1].set_title("side view: y/z")
    axes[1].axis("equal")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def load_reconstructor_pipeline(cfg, device, enable_vram_management=False):
    pipe = WanVideoNeoVersePipeline(
        device=device,
        torch_dtype=getattr(torch, cfg.torch_dtype),
        pipeline_kwargs=zero_drop_probs(cfg.pipeline_kwargs),
    )
    model_config = ModelConfig(
        path=cfg.reconstructor_path,
        offload_device="cpu" if enable_vram_management else device,
    )
    model_config.download_if_necessary()
    model_manager = ModelManager()
    model_manager.load_model(
        model_config.path,
        device=model_config.offload_device or device,
        torch_dtype=model_config.offload_dtype or pipe.torch_dtype,
    )
    pipe.reconstructor = model_manager.fetch_model("reconstructor")
    if pipe.reconstructor is None:
        raise RuntimeError(f"Failed to load reconstructor from {cfg.reconstructor_path}")
    pipe.vram_management_enabled = bool(enable_vram_management)
    return pipe


@torch.no_grad()
def build_render_conditions(pipe, views, cfg, height, width, mode_name):
    views = to_device_views(views, pipe.device)
    has_gt_cameras = (
        isinstance(views.get("camera_poses"), torch.Tensor)
        and isinstance(views.get("camera_intrs"), torch.Tensor)
    )
    if has_gt_cameras:
        recon_views, context_num = slice_context_views(views)
    else:
        recon_views = views
        context_num = int((~views["is_target"][0].bool()).sum().item()) if isinstance(views.get("is_target"), torch.Tensor) else views["img"].shape[1]

    pipe.load_models_to_device(["reconstructor"])
    autocast_enabled = str(pipe.device).startswith("cuda")
    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype, enabled=autocast_enabled):
        predictions = pipe.reconstructor(
            recon_views,
            is_inference=False,
            skip_unused_heads=True,
            force_motion_tokens=has_gt_cameras,
        )

    if has_gt_cameras:
        target_poses = views["camera_poses"]
        target_intrs = views["camera_intrs"]
        target_timestamps = views["timestamp"]
        trajectory_reference = "gt_camera_annotations"
        recon_input = "context_only"
    else:
        target_poses = predictions["rendered_extrinsics"]
        target_intrs = predictions["rendered_intrinsics"]
        target_timestamps = predictions["rendered_timestamps"]
        trajectory_reference = "worldmirror_reconstructor_camera"
        recon_input = "all_81_views"

    target_poses, target_intrs, target_timestamps, sorted_is_target = sort_target_trajectory(
        target_poses,
        target_intrs,
        target_timestamps,
        views.get("is_target") if isinstance(views.get("is_target"), torch.Tensor) else None,
    )
    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        predictions["splats"],
        render_viewmats=homo_matrix_inverse(target_poses),
        render_Ks=target_intrs,
        render_timestamps=target_timestamps,
        sh_degree=0,
        width=width,
        height=height,
    )
    target_mask = (target_alpha > float(cfg.pipeline_kwargs.get("alpha_thresh", 0.5))).float()
    if bool(cfg.pipeline_kwargs.get("mask_non_context_targets", False)) and sorted_is_target is not None:
        target_mask = target_mask.masked_fill(sorted_is_target[:, :, None, None, None].bool(), 0)

    return {
        "mode_name": mode_name,
        "source_views": views,
        "context_num": context_num,
        "recon_input": recon_input,
        "trajectory_reference": trajectory_reference,
        "target_rgb": target_rgb.detach().float().cpu(),
        "target_depth": target_depth.detach().float().cpu(),
        "target_mask": target_mask.detach().float().cpu(),
        "target_poses": target_poses.detach().float().cpu(),
        "target_intrs": target_intrs.detach().float().cpu(),
        "target_timestamps": target_timestamps.detach().float().cpu(),
    }


def save_mode_videos(output_dir, mode, fps):
    os.makedirs(output_dir, exist_ok=True)
    save_video(tensor_video_to_pil(mode["target_rgb"][0]), os.path.join(output_dir, "render_rgb.mp4"), fps=fps)
    save_video(
        tensor_video_to_pil(normalize_video(mode["target_depth"][0])),
        os.path.join(output_dir, "render_depth.mp4"),
        fps=fps,
    )
    save_video(tensor_video_to_pil(mode["target_mask"][0]), os.path.join(output_dir, "render_mask.mp4"), fps=fps)


def save_html(output_dir, metadata, sample_dirs):
    videos = [
        ("Summary grid", "summary_grid.mp4"),
        ("True vs false render", "side_by_side_true_vs_false.mp4"),
        ("GT vs true render", "side_by_side_gt_vs_true.mp4"),
        ("GT vs false render", "side_by_side_gt_vs_false.mp4"),
        ("RGB absolute diff", "absdiff_true_vs_false.mp4"),
        ("Context timeline", "context_timeline.mp4"),
    ]
    sections = "\n".join(
        f"""
        <section>
          <h2>{title}</h2>
          <video src="{filename}" controls loop muted playsinline></video>
        </section>
        """
        for title, filename in videos
        if os.path.exists(os.path.join(output_dir, filename))
    )
    sample_links = "\n".join(f'<li><a href="{os.path.relpath(path, output_dir)}/index.html">{name}</a></li>' for name, path in sample_dirs)
    meta = json.dumps(metadata, indent=2, ensure_ascii=False)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>USE_CAMERA_ANNOTATIONS Compare</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; color: #151515; background: #f5f5f5; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; margin-top: 20px; }}
    section {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    h2 {{ margin: 0 0 10px; font-size: 15px; font-weight: 600; }}
    video {{ display: block; width: 100%; background: #000; }}
    pre {{ margin: 16px 0 0; padding: 12px; background: #fff; border: 1px solid #ddd; border-radius: 6px; overflow-x: auto; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>USE_CAMERA_ANNOTATIONS Compare</h1>
  <p>Fixed sampled clip/context. true uses context-only reconstruction + GT annotation camera. false drops camera annotations and uses all 81 views + WorldMirror camera.</p>
  <ul>{sample_links}</ul>
  <div class="grid">{sections}</div>
  <pre>{meta}</pre>
</body>
</html>
"""
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def compare_one_sample(pipe, dataset, index, cfg, output_dir, fps):
    source_views = dataset[int(index)]
    batched_with_gt = batch_dataset_views(source_views, pipe.device)
    if "camera_poses" not in batched_with_gt or "camera_intrs" not in batched_with_gt:
        raise RuntimeError("Dataset sample does not contain camera annotations; cannot compare true mode.")
    batched_without_gt = drop_camera_annotations(batched_with_gt)

    true_mode = build_render_conditions(pipe, batched_with_gt, cfg, int(cfg.height), int(cfg.width), "use_camera_annotations_true")
    false_mode = build_render_conditions(pipe, batched_without_gt, cfg, int(cfg.height), int(cfg.width), "use_camera_annotations_false")

    aligned_false_rgb, false_time_deltas = align_by_nearest_timestamps(
        true_mode["target_rgb"],
        false_mode["target_rgb"],
        false_mode["target_timestamps"],
        true_mode["target_timestamps"],
    )
    aligned_false_mask, _ = align_by_nearest_timestamps(
        true_mode["target_mask"],
        false_mode["target_mask"],
        false_mode["target_timestamps"],
        true_mode["target_timestamps"],
    )
    aligned_false_poses, _ = align_by_nearest_timestamps(
        true_mode["target_poses"],
        false_mode["target_poses"],
        false_mode["target_timestamps"],
        true_mode["target_timestamps"],
    )
    aligned_false_intrs, _ = align_by_nearest_timestamps(
        true_mode["target_intrs"],
        false_mode["target_intrs"],
        false_mode["target_timestamps"],
        true_mode["target_timestamps"],
    )

    true_frames = tensor_video_to_pil(true_mode["target_rgb"][0])
    false_frames = tensor_video_to_pil(aligned_false_rgb[0])
    gt_frames = source_timeline_frames(true_mode["source_views"], true_mode["target_timestamps"])
    context_frames = context_timeline_frames(true_mode["source_views"], true_mode["target_timestamps"])
    diff_frames = make_diff_video(true_mode["target_rgb"], aligned_false_rgb)

    os.makedirs(output_dir, exist_ok=True)
    save_mode_videos(os.path.join(output_dir, "true"), true_mode, fps)
    false_aligned_mode = dict(false_mode)
    false_aligned_mode["target_rgb"] = aligned_false_rgb
    false_aligned_mode["target_mask"] = aligned_false_mask
    false_aligned_mode["target_poses"] = aligned_false_poses
    false_aligned_mode["target_intrs"] = aligned_false_intrs
    save_mode_videos(os.path.join(output_dir, "false_aligned_to_true_timestamps"), false_aligned_mode, fps)
    save_video(gt_frames, os.path.join(output_dir, "gt_81_sorted.mp4"), fps=fps)
    save_video(context_frames, os.path.join(output_dir, "context_timeline.mp4"), fps=fps)
    save_video(true_frames, os.path.join(output_dir, "render_true_gtcam_context.mp4"), fps=fps)
    save_video(false_frames, os.path.join(output_dir, "render_false_worldmirror_all81.mp4"), fps=fps)
    save_video(diff_frames, os.path.join(output_dir, "absdiff_true_vs_false.mp4"), fps=fps)
    save_video(
        make_two_col(true_frames, false_frames, "true: GT camera/context", "false: WorldMirror camera/all81"),
        os.path.join(output_dir, "side_by_side_true_vs_false.mp4"),
        fps=fps,
    )
    save_video(
        make_two_col(gt_frames, true_frames, "GT sampled frame", "true render"),
        os.path.join(output_dir, "side_by_side_gt_vs_true.mp4"),
        fps=fps,
    )
    save_video(
        make_two_col(gt_frames, false_frames, "GT sampled frame", "false render"),
        os.path.join(output_dir, "side_by_side_gt_vs_false.mp4"),
        fps=fps,
    )
    save_video(
        make_four_grid(gt_frames, true_frames, false_frames, diff_frames),
        os.path.join(output_dir, "summary_grid.mp4"),
        fps=fps,
    )
    save_camera_plot(os.path.join(output_dir, "camera_trajectory_aligned.png"), true_mode["target_poses"], aligned_false_poses)

    gt_tensor = torch.stack([
        torch.from_numpy(np.asarray(frame).astype(np.float32) / 255.0)
        for frame in gt_frames
    ], dim=0).unsqueeze(0)

    metrics = {
        "dataset_index": int(index),
        "video_name": source_views[0].get("video_name"),
        "num_frames": int(true_mode["target_rgb"].shape[1]),
        "num_context_views": int(true_mode["context_num"]),
        "true_semantics": {
            "recon_input": true_mode["recon_input"],
            "trajectory_reference": true_mode["trajectory_reference"],
        },
        "false_semantics": {
            "recon_input": false_mode["recon_input"],
            "trajectory_reference": false_mode["trajectory_reference"],
        },
        "timestamp_alignment": {
            "max_abs_delta_sec": float(max(false_time_deltas) if false_time_deltas else 0.0),
            "mean_abs_delta_sec": float(np.mean(false_time_deltas) if false_time_deltas else 0.0),
        },
        "render_true_vs_false": {
            "rgb_l1": video_l1(true_mode["target_rgb"], aligned_false_rgb),
            "rgb_psnr_db": video_psnr(true_mode["target_rgb"], aligned_false_rgb),
            "mask_iou": mask_iou(true_mode["target_mask"], aligned_false_mask),
        },
        "render_vs_gt_sampled_frames": {
            "true_rgb_l1": video_l1(gt_tensor, true_mode["target_rgb"]),
            "true_rgb_psnr_db": video_psnr(gt_tensor, true_mode["target_rgb"]),
            "false_rgb_l1": video_l1(gt_tensor, aligned_false_rgb),
            "false_rgb_psnr_db": video_psnr(gt_tensor, aligned_false_rgb),
        },
        "camera_intrinsics_false_vs_true": intrinsics_summary(true_mode["target_intrs"], aligned_false_intrs),
        "camera_positions_false_vs_true": camera_position_summary(true_mode["target_poses"], aligned_false_poses),
        "paths": {
            "output_dir": os.path.abspath(output_dir),
            "summary_grid": os.path.abspath(os.path.join(output_dir, "summary_grid.mp4")),
            "html": os.path.abspath(os.path.join(output_dir, "index.html")),
        },
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    save_html(output_dir, metrics, [])
    return metrics


def aggregate_metrics(sample_metrics):
    numeric_paths = [
        ("render_true_vs_false", "rgb_l1"),
        ("render_true_vs_false", "rgb_psnr_db"),
        ("render_true_vs_false", "mask_iou"),
        ("render_vs_gt_sampled_frames", "true_rgb_l1"),
        ("render_vs_gt_sampled_frames", "false_rgb_l1"),
        ("camera_intrinsics_false_vs_true", "fx_relative_abs_mean"),
        ("camera_intrinsics_false_vs_true", "fy_relative_abs_mean"),
        ("camera_positions_false_vs_true", "sim3_aligned_position_rmse_over_gt_radius"),
    ]
    aggregate = {"num_samples": len(sample_metrics)}
    for group, key in numeric_paths:
        values = [float(item[group][key]) for item in sample_metrics if np.isfinite(float(item[group][key]))]
        if values:
            aggregate[f"{group}.{key}.mean"] = float(np.mean(values))
            aggregate[f"{group}.{key}.min"] = float(np.min(values))
            aggregate[f"{group}.{key}.max"] = float(np.max(values))
    return aggregate


def parse_indices(text):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Compare USE_CAMERA_ANNOTATIONS=true/false teacher render semantics.")
    parser.add_argument("--config", default="configs/distill_control_latent.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset_indices", default="0")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_context_views", type=int, default=20)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable_vram_management", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.num_views = int(args.num_frames)
    cfg.min_num_context_views = int(args.num_context_views)
    cfg.max_num_context_views = int(args.num_context_views)
    cfg.use_camera_annotations = True
    if args.height is not None:
        cfg.height = int(args.height)
    if args.width is not None:
        cfg.width = int(args.width)
    if args.seed is not None:
        cfg.seed = int(args.seed)
        cfg.dataset_seed = int(args.seed)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("outputs", "camera_annotation_mode_compare", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    pipe = load_reconstructor_pipeline(cfg, device=device, enable_vram_management=args.enable_vram_management)
    pipe.eval()

    dataset = eval(cfg.train_dataset)
    sample_metrics = []
    sample_dirs = []
    for index in parse_indices(args.dataset_indices):
        sample_dir = os.path.join(output_dir, f"sample_{index:04d}")
        print(f"Comparing dataset_index={index}; output_dir={os.path.abspath(sample_dir)}", flush=True)
        metrics = compare_one_sample(pipe, dataset, index, cfg, sample_dir, args.fps)
        sample_metrics.append(metrics)
        sample_dirs.append((f"sample {index}", os.path.abspath(sample_dir)))

    summary = {
        "config": os.path.abspath(args.config),
        "output_dir": os.path.abspath(output_dir),
        "dataset_indices": parse_indices(args.dataset_indices),
        "num_frames": int(cfg.num_views),
        "num_context_views": int(cfg.min_num_context_views),
        "height": int(cfg.height),
        "width": int(cfg.width),
        "aggregate": aggregate_metrics(sample_metrics),
        "samples": sample_metrics,
    }
    with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_html(output_dir, summary, sample_dirs)
    print(f"Saved comparison summary to: {os.path.abspath(output_dir)}")
    print(f"Open: {os.path.abspath(os.path.join(output_dir, 'index.html'))}")


if __name__ == "__main__":
    main()
