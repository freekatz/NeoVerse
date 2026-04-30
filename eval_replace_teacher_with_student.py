import argparse
import os
import numpy as np
from copy import deepcopy

import torch
from omegaconf import OmegaConf
from PIL import Image
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


def save_eval_condition_videos(output_dir, render_conditions, camera_embed=None, fps=15):
    os.makedirs(output_dir, exist_ok=True)
    views = render_conditions["source_views"]
    source_video = views["img"][0].detach().float().permute(0, 2, 3, 1).cpu().clamp(0, 1)
    save_video(_tensor_to_pil_video(source_video), os.path.join(output_dir, "input_source_views.mp4"), fps=fps)

    is_target = views.get("is_target")
    if isinstance(is_target, torch.Tensor) and is_target.ndim == 2:
        context_mask = ~is_target[0].detach().cpu().bool()
        target_mask = is_target[0].detach().cpu().bool()
        context_video = source_video[context_mask]
        if len(context_video) > 0:
            save_video(_tensor_to_pil_video(context_video), os.path.join(output_dir, "input_context_views.mp4"), fps=fps)
        target_input_video = source_video[target_mask]
        if len(target_input_video) > 0:
            save_video(_tensor_to_pil_video(target_input_video), os.path.join(output_dir, "input_target_views_gt.mp4"), fps=fps)

    target_rgb = render_conditions["target_rgb"][0].detach().float().cpu().clamp(0, 1)
    save_video(_tensor_to_pil_video(target_rgb), os.path.join(output_dir, "rendered_degraded_rgb.mp4"), fps=fps)

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
    for b_idx in range(len(target_rgb)):
        if isinstance(views.get("is_target"), torch.Tensor):
            target_mask[b_idx][context_num:] = 0
        order_indices = torch.argsort(target_timestamps[b_idx])
        target_rgb[b_idx] = target_rgb[b_idx][order_indices]
        target_depth[b_idx] = target_depth[b_idx][order_indices]
        target_mask[b_idx] = target_mask[b_idx][order_indices]
        target_poses[b_idx] = target_poses[b_idx][order_indices]
        target_intrs[b_idx] = target_intrs[b_idx][order_indices]
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
def compute_student_hints(pipe, student_condition, inputs_shared, context, latents, timestep, mode, cfg):
    x, output_grid, context_emb, _, t_mod, freqs = prepare_control_state(
        pipe,
        latents,
        timestep,
        context,
        fuse_vae_embedding_in_latents=bool(cfg.get("fuse_vae_embedding_in_latents", False)),
    )
    condition = student_condition
    if mode == "combined":
        teacher_condition = pipe.control_branch.encode_condition(
            inputs_shared["target_rgb"],
            inputs_shared["target_depth"],
            inputs_shared["target_camera_embed"],
            inputs_shared["target_mask"],
            sequence_length=x.shape[1],
        )
        condition = 0.5 * (teacher_condition + student_condition)

    hints = pipe.control_branch.hints_from_condition(
        condition,
        x,
        context_emb,
        t_mod,
        freqs,
        False,
        False,
    )
    return {f"layer_{layer}": hint for layer, hint in zip(pipe.control_branch.control_layers, hints)}


@torch.no_grad()
def generate_with_student(pipe, adapter, render_conditions, prompt, cfg, args, mode="student"):
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
    for progress_id, timestep_value in enumerate(tqdm(pipe.scheduler.timesteps, desc=f"{mode}")):
        timestep = timestep_value.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        hints_posi = compute_student_hints(
            pipe, student_condition, inputs_shared, inputs_posi["context"], inputs_shared["latents"], timestep, mode, cfg
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
                pipe, student_condition, inputs_shared, inputs_nega["context"], inputs_shared["latents"], timestep, mode, cfg
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
    parser.add_argument("--modes", default="teacher,student,combined")
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
    args = parser.parse_args()

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
    adapter = load_adapter(pipe, cfg, args.checkpoint, pipe.device)

    dataset = eval(cfg.train_dataset)
    source_views = dataset[int(args.dataset_index)]
    prompt = source_views[0]["prompt"]
    render_conditions = build_render_conditions(pipe, source_views, args, cfg)
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
        save_eval_condition_videos(args.output_dir, render_conditions, camera_embed=camera_embed_for_vis)
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        if mode == "teacher":
            video = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                **render_conditions,
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
        elif mode in {"student", "combined"}:
            video = generate_with_student(pipe, adapter, render_conditions, prompt, cfg, args, mode=mode)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        save_video(video, os.path.join(args.output_dir, f"{mode}.mp4"), fps=15)


if __name__ == "__main__":
    main()
