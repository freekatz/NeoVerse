import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from diffsynth.data import save_video
from diffsynth.models import ModelManager
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth.utils import ModelConfig
from diffsynth.utils.auxiliary import homo_matrix_inverse
from tools.train.distill_control_latent import build_spatialvid_dataset


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


def subset_views(views, indices, mark_all_context=False):
    indices = torch.as_tensor(indices, device=views["img"].device, dtype=torch.long)
    total_views = views["img"].shape[1]
    out = {}
    for key, value in views.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == total_views:
            out[key] = value.index_select(1, indices)
        elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], list) and len(value[0]) == total_views:
            idx_cpu = indices.detach().cpu().tolist()
            out[key] = [[value[0][idx] for idx in idx_cpu]]
        else:
            out[key] = value
    if mark_all_context:
        out["is_target"] = torch.zeros(
            (views["img"].shape[0], indices.numel()),
            device=views["img"].device,
            dtype=torch.bool,
        )
    return out


def timestamp_order(views):
    return torch.argsort(views["timestamp"][0])


def tensor_video_to_pil(frames):
    frames = frames.detach().float().cpu().clamp(0, 1)
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
    box_height = 22
    box_width = max(72, len(label) * 8 + padding * 2)
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0))
    draw.text((padding, 4), label, fill=(255, 255, 255))
    return image


def make_side_by_side(left_frames, right_frames, left_label, right_label):
    out = []
    for left, right in zip(left_frames, right_frames):
        left = label_frame(left, left_label)
        right = label_frame(right, right_label)
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (0, 0, 0))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        out.append(canvas)
    return out


def make_diff_video(left, right):
    diff = (left.detach().float().cpu() - right.detach().float().cpu()).abs().mean(dim=-1, keepdim=True)
    max_value = float(diff.amax())
    return tensor_video_to_pil(normalize_video(diff, min_value=0.0, max_value=max_value))


def save_html(output_dir, files, metadata):
    items = "\n".join(
        f"""
        <section>
          <h2>{title}</h2>
          <video src="{filename}" controls loop muted playsinline></video>
        </section>
        """
        for title, filename in files
    )
    meta = json.dumps(metadata, indent=2, ensure_ascii=False)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>NeoVerse Reconstruction Context Compare</title>
  <style>
    body {{
      margin: 24px;
      font-family: Arial, sans-serif;
      color: #151515;
      background: #f5f5f5;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
      margin-top: 20px;
    }}
    section {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 12px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 15px;
      font-weight: 600;
    }}
    video {{
      display: block;
      width: 100%;
      background: #000;
    }}
    pre {{
      margin: 16px 0 0;
      padding: 12px;
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <h1>Reconstruction Context Compare</h1>
  <div>Same sampled clip and same 81-frame render trajectory.</div>
  <div class="grid">
    {items}
  </div>
  <pre>{meta}</pre>
</body>
</html>
"""
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


@torch.no_grad()
def run_reconstructor(pipe, source_views):
    pipe.load_models_to_device(["reconstructor"])
    autocast_enabled = str(pipe.device).startswith("cuda")
    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype, enabled=autocast_enabled):
        return pipe.reconstructor(source_views, is_inference=False)


@torch.no_grad()
def render_splats(pipe, predictions, target_poses, target_intrs, target_timestamps, height, width):
    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        predictions["splats"],
        render_viewmats=homo_matrix_inverse(target_poses),
        render_Ks=target_intrs,
        render_timestamps=target_timestamps,
        sh_degree=0,
        width=width,
        height=height,
    )
    return target_rgb[0].detach().float().cpu().clamp(0, 1), target_depth[0], target_alpha[0]


def select_exact_timestamp_trajectory(target_poses, target_intrs, target_timestamps, available_timestamps):
    selected_pose_batches = []
    selected_intr_batches = []
    selected_timestamp_batches = []
    selected_indices = []
    for b_idx in range(target_timestamps.shape[0]):
        matches = torch.isclose(
            target_timestamps[b_idx][:, None].float(),
            available_timestamps[b_idx][None, :].float(),
            rtol=0.0,
            atol=1e-6,
        )
        keep = torch.nonzero(matches.any(dim=1), as_tuple=False).flatten()
        if keep.numel() == 0:
            raise RuntimeError(
                f"No target timestamps match reconstructed context timestamps for batch {b_idx}."
            )
        selected_pose_batches.append(target_poses[b_idx].index_select(0, keep))
        selected_intr_batches.append(target_intrs[b_idx].index_select(0, keep))
        selected_timestamp_batches.append(target_timestamps[b_idx].index_select(0, keep))
        selected_indices.append(keep.detach().cpu().tolist())

    selected_counts = {timestamps.shape[0] for timestamps in selected_timestamp_batches}
    if len(selected_counts) != 1:
        raise RuntimeError(
            "Exact timestamp rendering requires the same number of selected frames in each batch."
        )
    return (
        torch.stack(selected_pose_batches, dim=0),
        torch.stack(selected_intr_batches, dim=0),
        torch.stack(selected_timestamp_batches, dim=0),
        selected_indices,
    )


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


def resolve_trajectory(full_views, full_predictions, prefer_gt):
    has_gt_cameras = (
        prefer_gt
        and isinstance(full_views.get("camera_poses"), torch.Tensor)
        and isinstance(full_views.get("camera_intrs"), torch.Tensor)
    )
    if has_gt_cameras:
        poses = full_views["camera_poses"]
        intrs = full_views["camera_intrs"]
        timestamps = full_views["timestamp"]
        reference = "gt_camera_annotations"
    else:
        poses = full_predictions["rendered_extrinsics"]
        intrs = full_predictions["rendered_intrinsics"]
        timestamps = full_predictions["rendered_timestamps"]
        reference = "full_81_reconstructor_camera"

    sorted_poses = poses.clone()
    sorted_intrs = intrs.clone()
    sorted_timestamps = timestamps.clone()
    for b_idx in range(timestamps.shape[0]):
        order = torch.argsort(timestamps[b_idx])
        sorted_poses[b_idx] = poses[b_idx][order]
        sorted_intrs[b_idx] = intrs[b_idx][order]
        sorted_timestamps[b_idx] = timestamps[b_idx][order]
    return sorted_poses, sorted_intrs, sorted_timestamps, reference


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
        description="Compare 81-frame reconstruction rendering with sparse-context reconstruction rendering."
    )
    parser.add_argument("--config", default="configs/distill_control_latent.yaml")
    parser.add_argument("--output_dir", default="outputs/reconstruction_context_compare")
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_context_views", type=int, default=20)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_camera_annotations", type=parse_bool, default=None)
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

    height = int(cfg.height)
    width = int(cfg.width)
    full_rgb, full_depth, full_alpha = render_splats(
        pipe, full_predictions, target_poses, target_intrs, target_timestamps, height, width
    )
    sparse_rgb, sparse_depth, sparse_alpha = render_splats(
        pipe, sparse_predictions, target_poses, target_intrs, target_timestamps, height, width
    )
    exact_poses, exact_intrs, exact_timestamps, exact_target_indices = select_exact_timestamp_trajectory(
        target_poses,
        target_intrs,
        target_timestamps,
        sparse_predictions["rendered_timestamps"],
    )
    sparse_exact_rgb, sparse_exact_depth, sparse_exact_alpha = render_splats(
        pipe, sparse_predictions, exact_poses, exact_intrs, exact_timestamps, height, width
    )

    gt_rgb = sorted_all_views["img"][0].detach().float().cpu().permute(0, 2, 3, 1).clamp(0, 1)
    context_rgb = sparse_views["img"][0].detach().float().cpu().permute(0, 2, 3, 1).clamp(0, 1)

    gt_frames = tensor_video_to_pil(gt_rgb)
    full_frames = tensor_video_to_pil(full_rgb)
    sparse_frames = tensor_video_to_pil(sparse_rgb)
    sparse_exact_frames = tensor_video_to_pil(sparse_exact_rgb)
    context_frames = tensor_video_to_pil(context_rgb)
    side_by_side = make_side_by_side(full_frames, sparse_frames, "all 81 input", f"{len(context_frames)} context input")
    diff_frames = make_diff_video(full_rgb, sparse_rgb)

    save_video(gt_frames, os.path.join(args.output_dir, "gt_81_sorted.mp4"), fps=args.fps)
    save_video(context_frames, os.path.join(args.output_dir, "context_input.mp4"), fps=args.fps)
    save_video(full_frames, os.path.join(args.output_dir, "render_all81.mp4"), fps=args.fps)
    save_video(sparse_frames, os.path.join(args.output_dir, "render_context.mp4"), fps=args.fps)
    save_video(sparse_exact_frames, os.path.join(args.output_dir, "render_context_nointerp.mp4"), fps=args.fps)
    save_video(side_by_side, os.path.join(args.output_dir, "side_by_side_all81_vs_context.mp4"), fps=args.fps)
    save_video(diff_frames, os.path.join(args.output_dir, "absdiff_all81_vs_context.mp4"), fps=args.fps)
    save_video(
        tensor_video_to_pil(normalize_video(full_depth)),
        os.path.join(args.output_dir, "depth_all81.mp4"),
        fps=args.fps,
    )
    save_video(
        tensor_video_to_pil(normalize_video(sparse_depth)),
        os.path.join(args.output_dir, "depth_context.mp4"),
        fps=args.fps,
    )
    save_video(
        tensor_video_to_pil(normalize_video(sparse_exact_depth)),
        os.path.join(args.output_dir, "depth_context_nointerp.mp4"),
        fps=args.fps,
    )
    save_video(
        tensor_video_to_pil(full_alpha.detach().float().cpu().clamp(0, 1)),
        os.path.join(args.output_dir, "alpha_all81.mp4"),
        fps=args.fps,
    )
    save_video(
        tensor_video_to_pil(sparse_alpha.detach().float().cpu().clamp(0, 1)),
        os.path.join(args.output_dir, "alpha_context.mp4"),
        fps=args.fps,
    )
    save_video(
        tensor_video_to_pil(sparse_exact_alpha.detach().float().cpu().clamp(0, 1)),
        os.path.join(args.output_dir, "alpha_context_nointerp.mp4"),
        fps=args.fps,
    )

    context_local_indices = context_indices.detach().cpu().tolist()
    metadata = {
        "dataset_index": int(args.dataset_index),
        "video_name": source_views[0].get("video_name"),
        "num_frames": int(args.num_frames),
        "num_context_views": len(context_local_indices),
        "context_sorted_indices": context_local_indices,
        "height": height,
        "width": width,
        "use_camera_annotations": bool(cfg.use_camera_annotations),
        "trajectory_reference": trajectory_reference,
        "output_dir": os.path.abspath(args.output_dir),
        "render_context_nointerp_num_frames": len(sparse_exact_frames),
        "render_context_nointerp_target_indices": exact_target_indices[0],
        "render_context_nointerp_timestamps": exact_timestamps[0].detach().cpu().tolist(),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    save_html(
        args.output_dir,
        [
            ("Side by side: all 81 vs context", "side_by_side_all81_vs_context.mp4"),
            ("All 81-frame input reconstruction render", "render_all81.mp4"),
            ("Sparse context reconstruction render", "render_context.mp4"),
            ("Sparse context exact-timestamp render", "render_context_nointerp.mp4"),
            ("Absolute RGB difference", "absdiff_all81_vs_context.mp4"),
            ("Ground-truth sampled 81 frames", "gt_81_sorted.mp4"),
            ("Context input frames", "context_input.mp4"),
            ("All 81 alpha", "alpha_all81.mp4"),
            ("Context alpha", "alpha_context.mp4"),
            ("Context exact-timestamp alpha", "alpha_context_nointerp.mp4"),
        ],
        metadata,
    )
    print(f"Saved reconstruction comparison to: {os.path.abspath(args.output_dir)}")
    print(f"Open: {os.path.abspath(os.path.join(args.output_dir, 'index.html'))}")


if __name__ == "__main__":
    main()
