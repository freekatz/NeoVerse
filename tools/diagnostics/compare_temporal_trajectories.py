import argparse
import json
import math
import os
import re
import sys
from datetime import datetime

import torch
from omegaconf import OmegaConf

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from diffsynth.data import save_video
from diffsynth.utils.auxiliary import homo_matrix_inverse
from tools.diagnostics.compare_camera_annotation_modes import (
    batch_dataset_views,
    drop_camera_annotations,
    load_reconstructor_pipeline,
    slice_context_views,
    tensor_video_to_pil,
    to_device_views,
)
from training.data.datasets.spatialvid import SpatialVID
from training.data.temporal_trajectory import build_inference_trajectory


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "trajectory"


def parse_trajectory_spec(spec):
    if "=" in spec:
        name, units = spec.split("=", 1)
    else:
        name, units = spec, spec
    backward = False
    if units.endswith("@backward"):
        units = units[: -len("@backward")]
        backward = True
    return slug(name), units, backward


def all_context_views(views):
    out = dict(views)
    is_target = views.get("is_target")
    if isinstance(is_target, torch.Tensor):
        out["is_target"] = torch.zeros_like(is_target, dtype=torch.bool)
    return out


def reorder_views_by_timestamp(views):
    timestamps = views.get("timestamp")
    if not isinstance(timestamps, torch.Tensor) or timestamps.ndim != 2:
        return views
    order = torch.argsort(timestamps, dim=1)
    if order.shape[0] != 1:
        raise ValueError("debug temporal currently expects batch size 1 for timestamp reordering.")
    index = order[0].to(device=timestamps.device)
    num_views = timestamps.shape[1]
    reordered = {}
    for key, value in views.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == num_views:
            reordered[key] = value.index_select(1, index.to(device=value.device))
        elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], list) and len(value[0]) == num_views:
            idx_cpu = index.detach().cpu().tolist()
            reordered[key] = [[value[0][idx] for idx in idx_cpu]]
        else:
            reordered[key] = value
    return reordered


def sorted_input_reference(views):
    images = views["img"].detach().float()
    timestamps = views["timestamp"].detach().float()
    order = torch.argsort(timestamps[0])
    rgb = images[:, order].permute(0, 1, 3, 4, 2).cpu().clamp(0, 1)
    timestamps = timestamps[:, order].cpu()
    return rgb, timestamps


def align_reference_to_timestamps(reference_rgb, reference_timestamps, target_timestamps):
    target_timestamps = target_timestamps.detach().float().cpu()
    aligned = []
    deltas = []
    for b_idx in range(target_timestamps.shape[0]):
        indices = []
        batch_deltas = []
        for timestamp in target_timestamps[b_idx]:
            distance = (reference_timestamps[b_idx] - timestamp).abs()
            index = int(distance.argmin().item())
            indices.append(index)
            batch_deltas.append(float(distance[index].item()))
        index_tensor = torch.tensor(indices, dtype=torch.long)
        aligned.append(reference_rgb[b_idx].index_select(0, index_tensor))
        deltas.append(batch_deltas)
    return torch.stack(aligned, dim=0), deltas


def video_l1(left, right):
    return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().mean().item())


def video_psnr(left, right):
    mse = float(((left.detach().float().cpu() - right.detach().float().cpu()) ** 2).mean().item())
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


@torch.no_grad()
def build_base_render_state(pipe, views, cfg, height, width):
    views = to_device_views(views, pipe.device)
    has_gt_cameras = (
        isinstance(views.get("camera_poses"), torch.Tensor)
        and isinstance(views.get("camera_intrs"), torch.Tensor)
    )
    if has_gt_cameras:
        recon_views, context_num = slice_context_views(views)
    else:
        recon_views = views
        context_num = int((~views["is_target"][0].bool()).sum().item())

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
        camera_source = "gt_camera_annotations"
    else:
        target_poses = predictions["rendered_extrinsics"]
        target_intrs = predictions["rendered_intrinsics"]
        target_timestamps = predictions["rendered_timestamps"]
        camera_source = "worldmirror_reconstructor"

    sorted_poses = target_poses.clone()
    sorted_intrs = target_intrs.clone()
    sorted_timestamps = target_timestamps.clone()
    for b_idx in range(target_timestamps.shape[0]):
        order = torch.argsort(target_timestamps[b_idx])
        sorted_poses[b_idx] = target_poses[b_idx][order]
        sorted_intrs[b_idx] = target_intrs[b_idx][order]
        sorted_timestamps[b_idx] = target_timestamps[b_idx][order]

    return {
        "views": views,
        "splats": predictions["splats"],
        "target_poses": sorted_poses,
        "target_intrs": sorted_intrs,
        "target_timestamps": sorted_timestamps,
        "context_num": context_num,
        "camera_source": camera_source,
        "height": height,
        "width": width,
    }


@torch.no_grad()
def render_with_timestamps(pipe, state, target_timestamps, alpha_thresh):
    target_timestamps = target_timestamps.to(device=pipe.device, dtype=state["target_timestamps"].dtype)
    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        state["splats"],
        render_viewmats=homo_matrix_inverse(state["target_poses"]),
        render_Ks=state["target_intrs"],
        render_timestamps=target_timestamps,
        sh_degree=0,
        width=int(state["width"]),
        height=int(state["height"]),
    )
    return {
        "rgb": target_rgb.detach().float().cpu(),
        "depth": target_depth.detach().float().cpu(),
        "mask": (target_alpha > float(alpha_thresh)).detach().float().cpu(),
        "timestamps": target_timestamps.detach().float().cpu(),
    }


def save_html(output_dir, runs):
    rows = []
    for run in runs:
        rows.append(
            f"<tr><td>{run['name']}</td><td>{run['units']}</td>"
            f"<td>{run['backward']}</td><td>{run['input_psnr_db']:.2f}</td><td>{run['input_l1']:.4f}</td>"
            f"<td><video src=\"{run['rgb_file']}\" controls loop muted></video></td></tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Temporal trajectory comparison</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; }}
    video {{ width: 360px; max-width: 90vw; }}
  </style>
</head>
<body>
  <h1>Temporal trajectory comparison</h1>
  <table>
    <thead><tr><th>Name</th><th>Units</th><th>Backward</th><th>PSNR vs nearest input</th><th>L1 vs nearest input</th><th>RGB render</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(html)


def main():
    parser = argparse.ArgumentParser(description="Render fixed NeoVerse conditions with explicit temporal trajectories.")
    parser.add_argument("--config", default="configs/distill_control_latent.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_views", type=int, default=None)
    parser.add_argument("--min_context", type=int, default=None)
    parser.add_argument("--max_context", type=int, default=None)
    parser.add_argument("--use_camera_annotations", type=parse_bool, default=None)
    parser.add_argument("--reconstruct_from", choices=("all", "context"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--trajectory",
        action="append",
        default=None,
        help="Trajectory spec, e.g. forward=F:81, freeze=Z:81, reverse=R:81, backward=F:81@backward.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    height = int(args.height or cfg.height)
    width = int(args.width or cfg.width)
    num_views = int(args.num_views or cfg.num_views)
    requested_fps = None if args.fps is None else float(args.fps)
    output_dir = args.output_dir or os.path.join(
        "outputs",
        "temporal_trajectory_compare",
        datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    use_camera_annotations = bool(cfg.use_camera_annotations) if args.use_camera_annotations is None else args.use_camera_annotations
    dataset = SpatialVID(
        split=None,
        ROOT=args.data_root or str(cfg.data_root),
        use_camera_annotations=use_camera_annotations,
        continuous_target_frames=True,
        force_first_context=bool(cfg.force_first_context),
        timestamp_unit=str(cfg.timestamp_unit),
        context_sampling_strategy=str(cfg.context_sampling_strategy),
        context_sampling_weights=OmegaConf.to_container(cfg.context_sampling_weights, resolve=True),
        temporal_augmentation=False,
        variants_per_scene=1,
        min_interval=1,
        max_interval=1,
        height=height,
        width=width,
        num_views=num_views,
        min_num_context_views=int(args.min_context or cfg.min_num_context_views),
        max_num_context_views=int(args.max_context or cfg.max_num_context_views),
        seed=int(cfg.dataset_seed) if cfg.get("dataset_seed") is not None and str(cfg.dataset_seed) != "null" else 0,
    )
    views = dataset[int(args.index)]

    pipe = load_reconstructor_pipeline(
        cfg,
        device=torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"),
        enable_vram_management=bool(cfg.get("enable_vram_management", False)),
    )
    batched_views = batch_dataset_views(views, pipe.device)
    if not use_camera_annotations:
        batched_views = drop_camera_annotations(batched_views)
    reference_rgb, reference_timestamps = sorted_input_reference(batched_views)
    recon_views = batched_views if args.reconstruct_from == "context" else all_context_views(reorder_views_by_timestamp(batched_views))

    state = build_base_render_state(pipe, recon_views, cfg, height, width)
    base_timestamps = state["target_timestamps"].detach().float()
    base_time_min = float(base_timestamps.amin().item())
    base_time_max = float(base_timestamps.amax().item())
    base_time_span = max(base_time_max - base_time_min, 1e-8)
    fps = requested_fps if requested_fps is not None else (max(num_views - 1, 1) / base_time_span)
    specs = args.trajectory or [
        "forward=F:81",
        "freeze=Z:81",
        "reverse=R:81",
        "pause=F:30,Z:20,F:31",
        "backward=F:81@backward",
    ]

    runs = []
    save_video(tensor_video_to_pil(reference_rgb[0]), os.path.join(output_dir, "input_sorted.mp4"), fps=15)
    for spec in specs:
        name, units, backward = parse_trajectory_spec(spec)
        trajectory = build_inference_trajectory(
            num_frames=num_views,
            units=units,
            backward=backward,
            fps=fps,
        )
        target_timestamps = torch.tensor(trajectory.temporal_coords, dtype=torch.float32)[None] + base_time_min
        rendered = render_with_timestamps(
            pipe,
            state,
            target_timestamps,
            alpha_thresh=float(cfg.pipeline_kwargs.get("alpha_thresh", 0.5)),
        )
        aligned_input, time_deltas = align_reference_to_timestamps(
            reference_rgb,
            reference_timestamps,
            rendered["timestamps"],
        )
        rgb_file = f"{name}_rgb.mp4"
        depth_file = f"{name}_depth.mp4"
        mask_file = f"{name}_mask.mp4"
        aligned_input_file = f"{name}_nearest_input.mp4"
        save_video(tensor_video_to_pil(rendered["rgb"][0]), os.path.join(output_dir, rgb_file), fps=15)
        save_video(tensor_video_to_pil(rendered["depth"][0]), os.path.join(output_dir, depth_file), fps=15)
        save_video(tensor_video_to_pil(rendered["mask"][0]), os.path.join(output_dir, mask_file), fps=15)
        save_video(tensor_video_to_pil(aligned_input[0]), os.path.join(output_dir, aligned_input_file), fps=15)
        runs.append(
            {
                "name": name,
                "units": units,
                "backward": backward,
                "trajectory_type": trajectory.trajectory_type,
                "rgb_file": rgb_file,
                "depth_file": depth_file,
                "mask_file": mask_file,
                "aligned_input_file": aligned_input_file,
                "input_psnr_db": video_psnr(rendered["rgb"], aligned_input),
                "input_l1": video_l1(rendered["rgb"], aligned_input),
                "max_nearest_input_time_delta": max(max(batch) for batch in time_deltas) if time_deltas else 0.0,
                "timestamp_start": float(target_timestamps[0, 0].item()),
                "timestamp_end": float(target_timestamps[0, -1].item()),
                "timestamp_min": float(target_timestamps.min().item()),
                "timestamp_max": float(target_timestamps.max().item()),
            }
        )

    metadata = {
        "dataset_index": int(args.index),
        "num_views": num_views,
        "fps": fps,
        "fps_source": "argument" if requested_fps is not None else "target_timestamp_range",
        "base_timestamp_min": base_time_min,
        "base_timestamp_max": base_time_max,
        "use_camera_annotations": use_camera_annotations,
        "reconstruct_from": args.reconstruct_from,
        "input_sorted_file": "input_sorted.mp4",
        "camera_source": state["camera_source"],
        "context_num": state["context_num"],
        "runs": runs,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    save_html(output_dir, runs)
    print(f"Wrote temporal trajectory comparison to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
