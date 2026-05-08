import argparse
import json
import os
import sys

import torch
from omegaconf import OmegaConf

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

try:
    from .compare_reconstruction_context import (
        batch_dataset_views,
        load_reconstructor_pipeline,
        render_splats,
        resolve_trajectory,
        run_reconstructor,
        select_exact_timestamp_trajectory,
        subset_views,
        tensor_video_to_pil,
        timestamp_order,
    )
except ImportError:
    from compare_reconstruction_context import (
        batch_dataset_views,
        load_reconstructor_pipeline,
        render_splats,
        resolve_trajectory,
        run_reconstructor,
        select_exact_timestamp_trajectory,
        subset_views,
        tensor_video_to_pil,
        timestamp_order,
    )
from diffsynth.data import save_video
from tools.train.distill_control_latent import build_spatialvid_dataset


def make_local_yaw_turnback(poses, max_degrees=180.0):
    frame_count = poses.shape[1]
    angles = torch.linspace(
        0.0,
        float(max_degrees) * torch.pi / 180.0,
        frame_count,
        device=poses.device,
        dtype=poses.dtype,
    )
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    yaw = torch.zeros((frame_count, 3, 3), device=poses.device, dtype=poses.dtype)
    yaw[:, 0, 0] = cos
    yaw[:, 0, 2] = sin
    yaw[:, 1, 1] = 1.0
    yaw[:, 2, 0] = -sin
    yaw[:, 2, 2] = cos

    out = poses.clone()
    out[:, :, :3, :3] = torch.matmul(poses[:, :, :3, :3], yaw.unsqueeze(0))
    return out


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render sparse-context reconstruction with a camera that gradually turns back."
    )
    parser.add_argument("--config", default="configs/distill_control_latent.yaml")
    parser.add_argument("--output_dir", default="outputs/reconstruction_context_compare/run_dataset0_ctx20_gtcam")
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_context_views", type=int, default=20)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_camera_annotations", type=parse_bool, default=True)
    parser.add_argument("--prefer_gt_trajectory", type=parse_bool, default=True)
    parser.add_argument("--max_degrees", type=float, default=180.0)
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
    if args.height is not None:
        cfg.height = int(args.height)
    if args.width is not None:
        cfg.width = int(args.width)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.use_camera_annotations is not None:
        cfg.use_camera_annotations = bool(args.use_camera_annotations)

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    pipe = load_reconstructor_pipeline(
        cfg,
        device=device,
        enable_vram_management=args.enable_vram_management,
    )
    pipe.eval()

    dataset = build_spatialvid_dataset(cfg)
    source_views = dataset[int(args.dataset_index)]
    batched = batch_dataset_views(source_views, pipe.device)

    order = timestamp_order(batched)
    sorted_all_views = subset_views(batched, order, mark_all_context=True)
    is_context_sorted = batched["is_target"][0].index_select(0, order).logical_not()
    context_indices = torch.nonzero(is_context_sorted, as_tuple=False).flatten()
    sparse_views = subset_views(sorted_all_views, context_indices, mark_all_context=True)

    full_predictions = run_reconstructor(pipe, sorted_all_views)
    sparse_predictions = run_reconstructor(pipe, sparse_views)
    target_poses, target_intrs, target_timestamps, trajectory_reference = resolve_trajectory(
        sorted_all_views,
        full_predictions,
        prefer_gt=bool(args.prefer_gt_trajectory),
    )

    height = int(cfg.height)
    width = int(cfg.width)
    turnback_poses = make_local_yaw_turnback(target_poses, max_degrees=args.max_degrees)
    sparse_rgb, _, _ = render_splats(
        pipe,
        sparse_predictions,
        turnback_poses,
        target_intrs,
        target_timestamps,
        height,
        width,
    )

    exact_poses, exact_intrs, exact_timestamps, exact_target_indices = select_exact_timestamp_trajectory(
        turnback_poses,
        target_intrs,
        target_timestamps,
        sparse_predictions["rendered_timestamps"],
    )
    sparse_exact_rgb, _, _ = render_splats(
        pipe,
        sparse_predictions,
        exact_poses,
        exact_intrs,
        exact_timestamps,
        height,
        width,
    )

    turnback_path = os.path.join(args.output_dir, "render_context_turnback.mp4")
    turnback_nointerp_path = os.path.join(args.output_dir, "render_context_turnback_nointerp.mp4")
    save_video(tensor_video_to_pil(sparse_rgb), turnback_path, fps=args.fps)
    save_video(tensor_video_to_pil(sparse_exact_rgb), turnback_nointerp_path, fps=args.fps)

    metadata_path = os.path.join(args.output_dir, "turnback_metadata.json")
    metadata = {
        "dataset_index": int(args.dataset_index),
        "num_frames": int(args.num_frames),
        "num_context_views": int(args.num_context_views),
        "max_degrees": float(args.max_degrees),
        "trajectory_reference": trajectory_reference,
        "context_sorted_indices": context_indices.detach().cpu().tolist(),
        "turnback_num_frames": int(sparse_rgb.shape[0]),
        "turnback_nointerp_num_frames": int(sparse_exact_rgb.shape[0]),
        "turnback_nointerp_target_indices": exact_target_indices[0],
        "output_dir": os.path.abspath(args.output_dir),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved: {os.path.abspath(turnback_path)}")
    print(f"Saved: {os.path.abspath(turnback_nointerp_path)}")
    print(f"Saved: {os.path.abspath(metadata_path)}")


if __name__ == "__main__":
    main()
