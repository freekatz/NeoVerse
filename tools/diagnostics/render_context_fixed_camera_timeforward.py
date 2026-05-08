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
        normalize_video,
        render_splats,
        resolve_trajectory,
        run_reconstructor,
        subset_views,
        tensor_video_to_pil,
        timestamp_order,
    )
except ImportError:
    from compare_reconstruction_context import (
        batch_dataset_views,
        load_reconstructor_pipeline,
        normalize_video,
        render_splats,
        resolve_trajectory,
        run_reconstructor,
        subset_views,
        tensor_video_to_pil,
        timestamp_order,
    )
from diffsynth.data import save_video
from tools.train.distill_control_latent import build_spatialvid_dataset


def repeat_reference_camera(target_poses, target_intrs, reference_index):
    frame_count = target_poses.shape[1]
    ref_pose = target_poses[:, reference_index : reference_index + 1].expand(-1, frame_count, -1, -1)
    ref_intr = target_intrs[:, reference_index : reference_index + 1].expand(-1, frame_count, -1, -1)
    return ref_pose.contiguous(), ref_intr.contiguous()


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
        description="Render sparse-context splats with a fixed camera and forward-moving timestamps."
    )
    parser.add_argument("--config", default="configs/distill_control_latent.yaml")
    parser.add_argument("--output_dir", default="outputs/reconstruction_context_compare/run_dataset0_ctx20_gtcam")
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_context_views", type=int, default=20)
    parser.add_argument("--reference_index", type=int, default=0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_camera_annotations", type=parse_bool, default=True)
    parser.add_argument("--prefer_gt_trajectory", type=parse_bool, default=True)
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

    if args.reference_index < 0 or args.reference_index >= target_poses.shape[1]:
        raise ValueError(
            f"reference_index must be in [0, {target_poses.shape[1] - 1}], got {args.reference_index}"
        )

    fixed_poses, fixed_intrs = repeat_reference_camera(
        target_poses,
        target_intrs,
        int(args.reference_index),
    )

    height = int(cfg.height)
    width = int(cfg.width)
    rgb, depth, alpha = render_splats(
        pipe,
        sparse_predictions,
        fixed_poses,
        fixed_intrs,
        target_timestamps,
        height,
        width,
    )

    prefix = f"render_context_fixedcam_ref{int(args.reference_index):03d}_timeforward"
    rgb_path = os.path.join(args.output_dir, f"{prefix}.mp4")
    depth_path = os.path.join(args.output_dir, f"depth_context_fixedcam_ref{int(args.reference_index):03d}_timeforward.mp4")
    alpha_path = os.path.join(args.output_dir, f"alpha_context_fixedcam_ref{int(args.reference_index):03d}_timeforward.mp4")
    metadata_path = os.path.join(args.output_dir, f"{prefix}_metadata.json")

    save_video(tensor_video_to_pil(rgb), rgb_path, fps=args.fps)
    save_video(tensor_video_to_pil(normalize_video(depth)), depth_path, fps=args.fps)
    save_video(tensor_video_to_pil(alpha.detach().float().cpu().clamp(0, 1)), alpha_path, fps=args.fps)

    metadata = {
        "dataset_index": int(args.dataset_index),
        "num_frames": int(args.num_frames),
        "num_context_views": int(args.num_context_views),
        "reference_index": int(args.reference_index),
        "reference_timestamp": float(target_timestamps[0, int(args.reference_index)].detach().cpu()),
        "timestamps": target_timestamps[0].detach().cpu().tolist(),
        "trajectory_reference": trajectory_reference,
        "context_sorted_indices": context_indices.detach().cpu().tolist(),
        "output_rgb": os.path.abspath(rgb_path),
        "output_depth": os.path.abspath(depth_path),
        "output_alpha": os.path.abspath(alpha_path),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved: {os.path.abspath(rgb_path)}")
    print(f"Saved: {os.path.abspath(depth_path)}")
    print(f"Saved: {os.path.abspath(alpha_path)}")
    print(f"Saved: {os.path.abspath(metadata_path)}")


if __name__ == "__main__":
    main()
