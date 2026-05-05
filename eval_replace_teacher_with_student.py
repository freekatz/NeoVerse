import argparse
import os
import numpy as np
from copy import deepcopy

import imageio
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffsynth.data import save_video
from diffsynth.utils.auxiliary import homo_matrix_inverse
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline, model_fn_wan_video
from hooks.extract_vggt_tokens import extract_vggt_tokens
from train_distill_control_latent import (
    build_adapter,
    extract_source_times,
    extract_target_times,
    gather_source_cameras_from_target,
    prepare_control_state,
)
from training.data.datasets.spatialvid import SpatialVID


def zero_drop_probs(pipeline_kwargs):
    kwargs = deepcopy(OmegaConf.to_container(pipeline_kwargs, resolve=True))
    kwargs["prompt_drop_prob"] = 0.0
    kwargs["mask_drop_prob"] = 0.0
    kwargs["condition_drop_prob"] = 0.0
    return kwargs


def pipeline_condition_kwargs(render_conditions):
    allowed = {
        "source_views",
        "target_rgb",
        "target_depth",
        "target_mask",
        "target_poses",
        "target_intrs",
    }
    return {key: value for key, value in render_conditions.items() if key in allowed}


def load_adapter(pipe, cfg, checkpoint_path, device):
    adapter = build_adapter(pipe, cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["adapter"] if isinstance(checkpoint, dict) and "adapter" in checkpoint else checkpoint
    adapter.load_state_dict(state_dict, strict=True)
    adapter.to(device=device, dtype=pipe.torch_dtype)
    adapter.eval()
    return adapter


def batch_dataset_views(views, device):
    batched = {}
    keys = views[0].keys()
    for key in keys:
        values = [view[key] for view in views]
        first = values[0]
        if isinstance(first, torch.Tensor):
            if first.ndim == 3:
                batched[key] = torch.stack(values, dim=0).unsqueeze(0).to(device)
            else:
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


def _normalize_video_tensor(frames, min_value=None, max_value=None):
    frames = frames.detach().float().cpu()
    if min_value is None:
        min_value = float(frames.amin())
    if max_value is None:
        max_value = float(frames.amax())
    frames = (frames - min_value) / max(max_value - min_value, 1e-6)
    return frames.clamp(0, 1)


def _tensor_to_pil_video(frames):
    frames = frames.detach().float().cpu().clamp(0, 1)
    if frames.ndim == 3:
        frames = frames[..., None]
    images = []
    for frame in frames:
        array = (frame * 255).round().to(torch.uint8).numpy()
        if array.shape[-1] == 1:
            array = array[..., 0]
        images.append(Image.fromarray(array))
    return images


def _read_video(path):
    reader = imageio.get_reader(path)
    try:
        return [Image.fromarray(frame).convert("RGB") for frame in reader]
    finally:
        reader.close()


def _frame_size(frames_by_name):
    for frames in frames_by_name.values():
        if frames:
            return _to_pil_rgb(frames[0]).size
    raise ValueError("No frames available for comparison grid.")


def _to_pil_rgb(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    return Image.fromarray(np.array(frame)).convert("RGB")


def _resize_filter():
    return getattr(Image, "Resampling", Image).LANCZOS


def _pad_video(frames, length, size):
    frames = [_to_pil_rgb(frame).resize(size, _resize_filter()) for frame in frames[:length]]
    frames.extend(Image.new("RGB", size, (0, 0, 0)) for _ in range(length - len(frames)))
    return frames


def _draw_boxed_label(image, label, corner="top_left"):
    draw = ImageDraw.Draw(image)
    padding = 7
    try:
        text_box = draw.textbbox((0, 0), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
    except AttributeError:
        text_width, text_height = draw.textsize(label)
    box_width = text_width + padding * 2
    box_height = text_height + padding * 2
    if corner == "top_right":
        x0 = max(image.width - box_width, 0)
    else:
        x0 = 0
    y0 = 0
    draw.rectangle((x0, y0, x0 + box_width, y0 + box_height), fill=(0, 0, 0))
    draw.text((x0 + padding, y0 + padding), label, fill=(255, 255, 255))


def _label_frame(frame, label, top_right_label=None):
    image = _to_pil_rgb(frame).copy()
    _draw_boxed_label(image, label, corner="top_left")
    if top_right_label:
        _draw_boxed_label(image, top_right_label, corner="top_right")
    return image


def _make_grid_video(frames_by_name, labels, length=None, top_right_labels=None):
    size = _frame_size(frames_by_name)
    if length is None:
        length = max(len(frames) for frames in frames_by_name.values())
    videos = {name: _pad_video(frames_by_name.get(name, []), length, size) for name in labels}
    top_right_labels = top_right_labels or {}
    grid_frames = []
    for frame_idx in range(length):
        tiles = [
            _label_frame(videos[name][frame_idx], label, top_right_labels.get(name))
            for name, label in labels.items()
        ]
        canvas = Image.new("RGB", (size[0] * 2, size[1] * 2), (0, 0, 0))
        canvas.paste(tiles[0], (0, 0))
        canvas.paste(tiles[1], (size[0], 0))
        canvas.paste(tiles[2], (0, size[1]))
        canvas.paste(tiles[3], (size[0], size[1]))
        grid_frames.append(canvas)
    return grid_frames


def _context_timeline_video(views, source_video, target_timestamps):
    is_target = views.get("is_target")
    if not isinstance(is_target, torch.Tensor) or is_target.ndim != 2:
        return []

    context_mask = ~is_target[0].detach().cpu().bool()
    target_timestamps = target_timestamps[0].detach().float().cpu()
    black = torch.zeros_like(source_video[0])
    source_timestamps = views.get("timestamp")
    if isinstance(source_timestamps, torch.Tensor) and source_timestamps.ndim == 2:
        source_timestamps = source_timestamps[0].detach().float().cpu()
        aligned_frames = []
        for target_ts in target_timestamps:
            src_idx = int(torch.argmin((source_timestamps - target_ts).abs()).item())
            if bool(context_mask[src_idx]):
                aligned_frames.append(source_video[src_idx])
            else:
                aligned_frames.append(black)
        return _tensor_to_pil_video(torch.stack(aligned_frames, dim=0))

    total_frames = min(source_video.shape[0], target_timestamps.shape[0], context_mask.shape[0])
    aligned = source_video[:total_frames].clone()
    aligned[~context_mask[:total_frames]] = 0
    if total_frames < target_timestamps.shape[0]:
        aligned = torch.cat(
            [aligned, black[None].repeat(target_timestamps.shape[0] - total_frames, 1, 1, 1)],
            dim=0,
        )
    return _tensor_to_pil_video(aligned)


def _source_timeline_video(views, source_video, target_timestamps=None):
    source_timestamps = views.get("timestamp")
    if isinstance(source_timestamps, torch.Tensor) and source_timestamps.ndim == 2:
        source_timestamps = source_timestamps[0].detach().float().cpu()
        if isinstance(target_timestamps, torch.Tensor) and target_timestamps.ndim == 2:
            target_timestamps = target_timestamps[0].detach().float().cpu()
            aligned = []
            for target_ts in target_timestamps:
                src_idx = int(torch.argmin((source_timestamps - target_ts).abs()).item())
                aligned.append(source_video[src_idx])
            return _tensor_to_pil_video(torch.stack(aligned, dim=0))
        order = torch.argsort(source_timestamps)
        return _tensor_to_pil_video(source_video[order])
    return _tensor_to_pil_video(source_video)


def _video_psnr(reference_frames, predicted_frames):
    length = min(len(reference_frames), len(predicted_frames))
    if length <= 0:
        return None
    mse_values = []
    for frame_idx in range(length):
        reference = _to_pil_rgb(reference_frames[frame_idx])
        predicted = _to_pil_rgb(predicted_frames[frame_idx])
        if predicted.size != reference.size:
            predicted = predicted.resize(reference.size, _resize_filter())
        reference = np.asarray(reference, dtype=np.float32) / 255.0
        predicted = np.asarray(predicted, dtype=np.float32) / 255.0
        mse_values.append(np.mean((reference - predicted) ** 2))
    mse = float(np.mean(mse_values))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _format_psnr(value):
    if value is None:
        return None
    if np.isinf(value):
        return "PSNR inf"
    return f"PSNR {value:.2f} dB"


def save_eval_comparison_grid(output_dir, frames_by_name=None, fps=15, filename="comparison_grid.mp4"):
    labels = {
        "input_context_views": "Input context views",
        "rendered_degraded_rgb": "Rendered degraded RGB",
        "student": "Student",
        "teacher": "Teacher",
    }
    frames_by_name = dict(frames_by_name or {})
    disk_names = {
        "rendered_degraded_rgb": ["rendered_degraded_rgb.mp4"],
        "input_context_views": ["input_context_views_timeline.mp4"],
        "student": ["student.mp4"],
        "teacher": ["teacher.mp4"],
    }
    missing = {}
    for name, video_names in disk_names.items():
        if name in frames_by_name and frames_by_name[name]:
            continue
        for video_name in video_names:
            path = os.path.join(output_dir, video_name)
            if os.path.exists(path):
                frames_by_name[name] = _read_video(path)
                break
        if name not in frames_by_name or not frames_by_name[name]:
            missing[name] = video_names
    if missing:
        missing_text = ", ".join(f"{name} ({'/'.join(paths)})" for name, paths in missing.items())
        print(f"Skip comparison grid; missing videos: {missing_text}")
        return None

    length = max(len(frames_by_name["rendered_degraded_rgb"]), len(frames_by_name["student"]), len(frames_by_name["teacher"]))
    grid_frames = _make_grid_video(frames_by_name, labels, length=length)
    output_path = os.path.join(output_dir, filename)
    save_video(grid_frames, output_path, fps=fps)
    return output_path


def save_eval_gt_comparison_grid(output_dir, frames_by_name=None, fps=15, filename="comparison_grid_gt.mp4"):
    labels = {
        "gt_81_sorted": "GT 81 frames",
        "rendered_degraded_rgb": "Rendered degraded RGB",
        "student": "Student",
        "teacher": "Teacher",
    }
    frames_by_name = dict(frames_by_name or {})
    disk_names = {
        "gt_81_sorted": ["gt_81_sorted.mp4"],
        "rendered_degraded_rgb": ["rendered_degraded_rgb.mp4"],
        "student": ["student.mp4"],
        "teacher": ["teacher.mp4"],
    }
    missing = {}
    for name, video_names in disk_names.items():
        if name in frames_by_name and frames_by_name[name]:
            continue
        for video_name in video_names:
            path = os.path.join(output_dir, video_name)
            if os.path.exists(path):
                frames_by_name[name] = _read_video(path)
                break
        if name not in frames_by_name or not frames_by_name[name]:
            missing[name] = video_names
    if missing:
        missing_text = ", ".join(f"{name} ({'/'.join(paths)})" for name, paths in missing.items())
        print(f"Skip GT comparison grid; missing videos: {missing_text}")
        return None

    length = max(len(frames_by_name["gt_81_sorted"]), len(frames_by_name["student"]), len(frames_by_name["teacher"]))
    top_right_labels = {
        "student": _format_psnr(_video_psnr(frames_by_name["gt_81_sorted"], frames_by_name["student"])),
        "teacher": _format_psnr(_video_psnr(frames_by_name["gt_81_sorted"], frames_by_name["teacher"])),
    }
    grid_frames = _make_grid_video(frames_by_name, labels, length=length, top_right_labels=top_right_labels)
    output_path = os.path.join(output_dir, filename)
    save_video(grid_frames, output_path, fps=fps)
    return output_path


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


def save_eval_condition_videos(output_dir, render_conditions, camera_embed=None, fps=15):
    os.makedirs(output_dir, exist_ok=True)
    saved_frames = {}
    views = render_conditions["source_views"]
    source_video = views["img"][0].detach().float().permute(0, 2, 3, 1).cpu().clamp(0, 1)
    source_frames = _tensor_to_pil_video(source_video)
    saved_frames["input_source_views"] = source_frames
    save_video(source_frames, os.path.join(output_dir, "input_source_views.mp4"), fps=fps)
    gt_81_frames = _source_timeline_video(views, source_video, render_conditions["target_timestamps"])
    saved_frames["gt_81_sorted"] = gt_81_frames
    save_video(gt_81_frames, os.path.join(output_dir, "gt_81_sorted.mp4"), fps=fps)

    is_target = views.get("is_target")
    if isinstance(is_target, torch.Tensor) and is_target.ndim == 2:
        context_mask = ~is_target[0].detach().cpu().bool()
        target_mask = is_target[0].detach().cpu().bool()
        context_video = source_video[context_mask]
        if len(context_video) > 0:
            context_frames = _tensor_to_pil_video(context_video)
            saved_frames["input_context_views_raw"] = context_frames
            save_video(context_frames, os.path.join(output_dir, "input_context_views.mp4"), fps=fps)
            context_timeline_frames = _context_timeline_video(
                views,
                source_video,
                render_conditions["target_timestamps"],
            )
            saved_frames["input_context_views"] = context_timeline_frames
            save_video(context_timeline_frames, os.path.join(output_dir, "input_context_views_timeline.mp4"), fps=fps)
        target_input_video = source_video[target_mask]
        if len(target_input_video) > 0:
            target_input_frames = _tensor_to_pil_video(target_input_video)
            saved_frames["input_target_views_gt"] = target_input_frames
            save_video(target_input_frames, os.path.join(output_dir, "input_target_views_gt.mp4"), fps=fps)

    target_rgb = render_conditions["target_rgb"][0].detach().float().cpu().clamp(0, 1)
    target_rgb_frames = _tensor_to_pil_video(target_rgb)
    saved_frames["rendered_degraded_rgb"] = target_rgb_frames
    save_video(target_rgb_frames, os.path.join(output_dir, "rendered_degraded_rgb.mp4"), fps=fps)

    target_depth = render_conditions["target_depth"][0].detach().float().cpu()
    save_video(
        _tensor_to_pil_video(_normalize_video_tensor(target_depth)),
        os.path.join(output_dir, "rendered_degraded_depth.mp4"),
        fps=fps,
    )

    target_mask_video = render_conditions["target_mask"][0].detach().float().cpu().clamp(0, 1)
    save_video(_tensor_to_pil_video(target_mask_video), os.path.join(output_dir, "rendered_degraded_mask.mp4"), fps=fps)

    if camera_embed is not None:
        camera_embed = camera_embed[0].detach().float().cpu()
        camera_dir = (camera_embed[:, :3].permute(0, 2, 3, 1).clamp(-1, 1) + 1) * 0.5
        save_video(_tensor_to_pil_video(camera_dir), os.path.join(output_dir, "target_plucker_dir.mp4"), fps=fps)
        camera_moment = camera_embed[:, 3:].permute(0, 2, 3, 1)
        scale = camera_moment.abs().amax().clamp_min(1e-6)
        camera_moment = (camera_moment / scale).clamp(-1, 1) * 0.5 + 0.5
        save_video(_tensor_to_pil_video(camera_moment), os.path.join(output_dir, "target_plucker_moment.mp4"), fps=fps)
    return saved_frames


@torch.no_grad()
def build_render_conditions(pipe, source_views, args, cfg):
    views = batch_dataset_views(source_views, pipe.device) if isinstance(source_views, list) else source_views
    views = {
        key: value.to(pipe.device) if isinstance(value, torch.Tensor) else value
        for key, value in views.items()
    }
    pipe.load_models_to_device(["reconstructor"])
    has_gt_cameras = (
        isinstance(views.get("camera_poses"), torch.Tensor)
        and isinstance(views.get("camera_intrs"), torch.Tensor)
    )
    if has_gt_cameras:
        recon_views, context_num = slice_context_views(views)
    else:
        recon_views = views
        context_num = int((~views["is_target"][0].bool()).sum().item()) if isinstance(views.get("is_target"), torch.Tensor) else views["img"].shape[1]
    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype, enabled=str(pipe.device).startswith("cuda")):
        predictions = pipe.reconstructor(recon_views, is_inference=False)
    if has_gt_cameras:
        target_poses = views["camera_poses"]
        target_intrs = views["camera_intrs"]
        target_timestamps = views["timestamp"]
    else:
        target_poses = predictions["rendered_extrinsics"]
        target_intrs = predictions["rendered_intrinsics"]
        target_timestamps = predictions["rendered_timestamps"]
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
        width=args.width,
        height=args.height,
    )
    alpha_thresh = float(cfg.pipeline_kwargs.get("alpha_thresh", 0.5))
    target_mask = (target_alpha > alpha_thresh).float()
    if bool(cfg.pipeline_kwargs.get("mask_non_context_targets", False)) and sorted_is_target is not None:
        target_mask = target_mask.masked_fill(sorted_is_target[:, :, None, None, None].bool(), 0)
    return {
        "source_views": views,
        "target_rgb": target_rgb,
        "target_depth": target_depth,
        "target_mask": target_mask,
        "target_poses": target_poses,
        "target_intrs": target_intrs,
        "target_timestamps": target_timestamps,
    }


@torch.no_grad()
def prepare_inputs(pipe, prompt, render_conditions, args):
    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=args.sigma_shift)
    inputs_posi = {"prompt": prompt if isinstance(prompt, list) else [prompt]}
    inputs_nega = {"negative_prompt": args.negative_prompt if isinstance(args.negative_prompt, list) else [args.negative_prompt]}
    inputs_shared = {
        "input_video": None,
        "denoising_strength": 1.0,
        "control_scale": args.control_scale,
        "source_views": render_conditions["source_views"],
        "target_rgb": render_conditions["target_rgb"],
        "target_depth": render_conditions["target_depth"],
        "target_mask": render_conditions["target_mask"],
        "target_poses": render_conditions["target_poses"],
        "target_intrs": render_conditions["target_intrs"],
        "seed": args.seed,
        "rand_device": args.rand_device,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "cfg_scale": args.cfg_scale,
        "sigma_shift": args.sigma_shift,
        "tiled": args.tiled,
        "tile_size": tuple(args.tile_size),
        "tile_stride": tuple(args.tile_stride),
        "sliding_window_size": None,
        "sliding_window_stride": None,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    return inputs_shared, inputs_posi, inputs_nega


@torch.no_grad()
def compute_student_condition(pipe, adapter, token_pack, inputs_shared, output_grid, cfg):
    target_times = extract_target_times(inputs_shared["source_views"], device=pipe.device)
    source_times = extract_source_times(
        inputs_shared["source_views"],
        device=pipe.device,
        context_only=bool(cfg.token.context_only),
    )
    source_poses, source_intrs = gather_source_cameras_from_target(
        inputs_shared["source_views"],
        inputs_shared.get("target_poses"),
        inputs_shared.get("target_intrs"),
        context_only=bool(cfg.token.context_only),
    )
    return adapter(
        token_pack["tokens"],
        output_grid,
        target_times=target_times,
        target_poses=inputs_shared.get("target_poses"),
        target_intrs=inputs_shared.get("target_intrs"),
        target_plucker=inputs_shared.get("target_camera_embed"),
        source_times=source_times,
        source_poses=source_poses,
        source_intrs=source_intrs,
    ).to(dtype=pipe.torch_dtype, device=pipe.device)


@torch.no_grad()
def compute_student_hints(pipe, student_condition, context, latents, timestep, cfg):
    x, output_grid, context_emb, _, t_mod, freqs = prepare_control_state(
        pipe,
        latents,
        timestep,
        context,
        fuse_vae_embedding_in_latents=bool(cfg.get("fuse_vae_embedding_in_latents", False)),
    )
    hints = pipe.control_branch.hints_from_condition(
        student_condition,
        x,
        context_emb,
        t_mod,
        freqs,
        False,
        False,
    )
    return {f"layer_{layer}": hint for layer, hint in zip(pipe.control_branch.control_layers, hints)}


@torch.no_grad()
def generate_with_student(pipe, adapter, render_conditions, prompt, cfg, args):
    inputs_shared, inputs_posi, inputs_nega = prepare_inputs(pipe, prompt, render_conditions, args)
    pipe.load_models_to_device(["reconstructor"])
    token_pack = extract_vggt_tokens(
        pipe.reconstructor,
        inputs_shared["source_views"],
        layer_indices=cfg.token.layer_indices,
        include_camera_token=cfg.token.include_camera_token,
        context_only=bool(cfg.token.context_only),
        ref_view_strategy=str(cfg.token.ref_view_strategy),
        autocast_dtype=getattr(torch, cfg.token.autocast_dtype),
    )
    token_pack["tokens"] = token_pack["tokens"].to(device=pipe.device)
    _, output_grid, _, _, _, _ = prepare_control_state(
        pipe,
        inputs_shared["latents"],
        pipe.scheduler.timesteps[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device),
        inputs_posi["context"],
        fuse_vae_embedding_in_latents=bool(cfg.get("fuse_vae_embedding_in_latents", False)),
    )
    student_condition = compute_student_condition(pipe, adapter, token_pack, inputs_shared, output_grid, cfg)

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep_value in enumerate(tqdm(pipe.scheduler.timesteps, desc="student")):
        timestep = timestep_value.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        hints_posi = compute_student_hints(
            pipe, student_condition, inputs_posi["context"], inputs_shared["latents"], timestep, cfg
        )
        model_inputs_posi = {
            **inputs_shared,
            **inputs_posi,
            "target_rgb": None,
            "precomputed_control_hints": hints_posi,
        }
        noise_pred_posi = model_fn_wan_video(**models, **model_inputs_posi, timestep=timestep)
        if args.cfg_scale != 1.0:
            hints_nega = compute_student_hints(
                pipe, student_condition, inputs_nega["context"], inputs_shared["latents"], timestep, cfg
            )
            model_inputs_nega = {
                **inputs_shared,
                **inputs_nega,
                "target_rgb": None,
                "precomputed_control_hints": hints_nega,
            }
            noise_pred_nega = model_fn_wan_video(**models, **model_inputs_nega, timestep=timestep)
            noise_pred = noise_pred_nega + args.cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        inputs_shared["latents"] = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"])

    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        inputs_shared["latents"],
        device=pipe.device,
        tiled=args.tiled,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
    )
    video = pipe.vae_output_to_video(video)
    pipe.load_models_to_device([])
    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--output_dir", default="./outputs/distill_eval")
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--modes", default="teacher,student")
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
    parser.add_argument("--no_save_conditions", action="store_true")
    parser.add_argument("--no_save_comparison_grid", action="store_true")
    args = parser.parse_args()
    requested_modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if not requested_modes:
        raise ValueError("--modes must contain at least one mode")
    supported_modes = {"teacher", "student"}
    unsupported_modes = sorted(set(requested_modes) - supported_modes)
    if unsupported_modes:
        raise ValueError(f"Unsupported mode(s): {', '.join(unsupported_modes)}. Supported modes: teacher, student")

    cfg = OmegaConf.load(args.config)
    args.height = args.height or int(cfg.height)
    args.width = args.width or int(cfg.width)
    args.num_frames = args.num_frames or int(cfg.num_views)
    official_lora_path = os.path.join(
        str(cfg.model_path),
        "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
    )
    lora_path = None
    if not args.disable_lora:
        lora_path = args.lora_path or cfg.get("lora_path", None)
        if lora_path is None and os.path.exists(official_lora_path):
            lora_path = official_lora_path
    use_lora = lora_path is not None
    if args.num_inference_steps is None:
        args.num_inference_steps = 4 if use_lora else 50
    if args.cfg_scale is None:
        args.cfg_scale = 1.0 if use_lora else 5.0
    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"Resolved generation params: steps={args.num_inference_steps}, "
        f"cfg_scale={args.cfg_scale}, lora_path={lora_path}"
    )

    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=cfg.model_path,
        reconstructor_path=cfg.reconstructor_path,
        pipeline_kwargs=zero_drop_probs(cfg.pipeline_kwargs),
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=getattr(torch, cfg.torch_dtype),
        lora_path=lora_path,
        lora_alpha=float(cfg.get("lora_alpha", 1.0)),
        enable_vram_management=args.enable_vram_management,
    )
    pipe.eval()
    adapter = load_adapter(pipe, cfg, args.checkpoint, pipe.device) if "student" in requested_modes else None

    dataset = eval(cfg.train_dataset)
    source_views = dataset[int(args.dataset_index)]
    prompt = source_views[0]["prompt"]
    render_conditions = build_render_conditions(pipe, source_views, args, cfg)
    eval_frames = {}
    if not args.no_save_conditions:
        camera_embed_for_vis = None
        camera_unit = next((unit for unit in pipe.units if unit.__class__.__name__ == "WanVideoUnit_CameraProcesser"), None)
        if camera_unit is not None:
            camera_embed_for_vis = camera_unit.convert_plucker_map(
                render_conditions["target_poses"],
                render_conditions["target_intrs"],
                args.height,
                args.width,
            )
        eval_frames.update(save_eval_condition_videos(args.output_dir, render_conditions, camera_embed=camera_embed_for_vis))
    for mode in requested_modes:
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
        elif mode == "student":
            video = generate_with_student(pipe, adapter, render_conditions, prompt, cfg, args)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        eval_frames[mode] = video
        save_video(video, os.path.join(args.output_dir, f"{mode}.mp4"), fps=15)
    if not args.no_save_comparison_grid:
        grid_path = save_eval_comparison_grid(args.output_dir, eval_frames, fps=15)
        if grid_path is not None:
            print(f"Saved comparison grid: {grid_path}")
        gt_grid_path = save_eval_gt_comparison_grid(args.output_dir, eval_frames, fps=15)
        if gt_grid_path is not None:
            print(f"Saved GT comparison grid: {gt_grid_path}")


if __name__ == "__main__":
    main()
