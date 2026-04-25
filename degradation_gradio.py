import argparse
import copy
import os
import threading
import time
from contextlib import nullcontext

for proxy_env in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(proxy_env, None)

import gradio as gr
import imageio
import numpy as np
import torch
from decord import VideoReader
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF


OUTPUT_ROOT = "outputs/degradation_gradio"
SAMPLE_VIDEO = "examples/videos/tree_and_building.mp4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

_MODEL_LOCK = threading.Lock()
_RECONSTRUCTOR = None
_RECONSTRUCTOR_PATH = None


def center_crop(image, resolution):
    width, height = image.size
    target_width, target_height = resolution
    scale = max(target_width / width, target_height / height)
    resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_width) // 2
    top = (resized.height - target_height) // 2
    return resized.crop((left, top, left + target_width, top + target_height))


def process_image(image, resolution, resize_mode):
    image = image.convert("RGB")
    if resize_mode == "resize":
        return image.resize(resolution, Image.Resampling.LANCZOS)
    return center_crop(image, resolution)


def extract_video_path(video_value):
    if video_value is None:
        return None
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        return video_value.get("video") or video_value.get("path") or video_value.get("name")
    if isinstance(video_value, (list, tuple)) and video_value:
        return video_value[0]
    return None


def load_frames(video_value, image_value, num_frames, resolution, resize_mode):
    video_path = extract_video_path(video_value)
    if video_path is None and image_value is None:
        video_path = SAMPLE_VIDEO

    if video_path is not None:
        reader = VideoReader(video_path)
        indices = np.linspace(0, len(reader) - 1, int(num_frames), dtype=int)
        raw_frames = reader.get_batch(indices).asnumpy()
        return [process_image(Image.fromarray(frame), resolution, resize_mode) for frame in raw_frames], video_path

    image = image_value if isinstance(image_value, Image.Image) else Image.fromarray(np.asarray(image_value))
    frame = process_image(image, resolution, resize_mode)
    return [frame.copy() for _ in range(int(num_frames))], "uploaded image"


def save_video(frames, path, fps=12):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with imageio.get_writer(path, fps=fps, quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    return path


def tensor_rgb_to_frames(video_tensor):
    video_tensor = video_tensor.detach().float().cpu().clamp(0, 1)
    frames = []
    for frame in video_tensor:
        image = (frame * 255.0).round().to(torch.uint8).numpy()
        frames.append(Image.fromarray(image))
    return frames


def tensor_mask_to_frames(mask_tensor):
    mask_tensor = mask_tensor.detach().float().cpu().clamp(0, 1)
    frames = []
    for frame in mask_tensor:
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        image = (frame * 255.0).round().to(torch.uint8).numpy()
        frames.append(Image.fromarray(np.stack([image] * 3, axis=-1)))
    return frames


def add_label(image, label):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    pad = 8
    text_box = draw.textbbox((0, 0), label)
    box_w = text_box[2] - text_box[0] + pad * 2
    box_h = text_box[3] - text_box[1] + pad * 2
    draw.rectangle((0, 0, box_w, box_h), fill=(12, 14, 18))
    draw.text((pad, pad), label, fill=(245, 245, 245))
    return image


def build_comparison_frames(source_frames, baseline_frames, degraded_frames, mask_frames):
    output = []
    labels = ("GT", "Reconstructed", "Degraded", "Mask")
    for frames in zip(source_frames, baseline_frames, degraded_frames, mask_frames):
        labeled = [add_label(frame.copy(), label) for frame, label in zip(frames, labels)]
        w, h = labeled[0].size
        gap = 8
        canvas = Image.new("RGB", (w * 2 + gap, h * 2 + gap), (8, 10, 14))
        canvas.paste(labeled[0], (0, 0))
        canvas.paste(labeled[1], (w + gap, 0))
        canvas.paste(labeled[2], (0, h + gap))
        canvas.paste(labeled[3], (w + gap, h + gap))
        output.append(canvas)
    return output


def homo_matrix_inverse(homo_matrix):
    assert homo_matrix.shape[-2:] in ((4, 4), (3, 4))
    R = homo_matrix[..., :3, :3].reshape(-1, 3, 3)
    T = homo_matrix[..., :3, 3:4].reshape(-1, 3, 1)
    R_inv = R.transpose(-1, -2)
    T_inv = -torch.bmm(R_inv, T)
    homo_inv = torch.eye(4, device=homo_matrix.device, dtype=homo_matrix.dtype)[None].repeat(R_inv.shape[0], 1, 1)
    homo_inv[:, :3, :3] = R_inv
    homo_inv[:, :3, 3:4] = T_inv
    return homo_inv.reshape(*homo_matrix.shape[:-2], 4, 4)


def get_reconstructor(reconstructor_path):
    global _RECONSTRUCTOR, _RECONSTRUCTOR_PATH
    with _MODEL_LOCK:
        if _RECONSTRUCTOR is not None and _RECONSTRUCTOR_PATH == reconstructor_path:
            return _RECONSTRUCTOR

        from diffsynth.models import ModelManager

        manager = ModelManager(device=str(DEVICE), torch_dtype=DTYPE)
        manager.load_model(reconstructor_path, device=str(DEVICE), torch_dtype=DTYPE)
        reconstructor = manager.fetch_model("reconstructor")
        if reconstructor is None:
            raise RuntimeError(f"Could not load a NeoVerse reconstructor from {reconstructor_path}")
        reconstructor.eval()
        _RECONSTRUCTOR = reconstructor
        _RECONSTRUCTOR_PATH = reconstructor_path
        return _RECONSTRUCTOR


def build_training_views(frames, num_context_views):
    num_frames = len(frames)
    context_indices = np.unique(np.linspace(0, num_frames - 1, int(num_context_views), dtype=int))
    context_indices = context_indices[:num_frames]
    target_indices = np.array([idx for idx in range(num_frames) if idx not in set(context_indices)], dtype=int)
    ordered_indices = np.concatenate([context_indices, target_indices], axis=0)

    images = torch.stack([TF.to_tensor(frames[int(idx)]) for idx in ordered_indices], dim=0).unsqueeze(0).to(DEVICE)
    is_target = torch.tensor(
        [[False] * len(context_indices) + [True] * len(target_indices)],
        dtype=torch.bool,
        device=DEVICE,
    )
    timestamps = torch.tensor([ordered_indices.tolist()], dtype=torch.int64, device=DEVICE)
    views = {
        "img": images,
        "is_target": is_target,
        "is_static": torch.zeros_like(is_target),
        "timestamp": timestamps,
    }
    return views, len(context_indices), ordered_indices


def count_gaussians(splats):
    total = 0
    per_frame = []
    for batch in splats:
        batch_counts = []
        for gaussian in batch:
            count = int(gaussian.means.shape[0])
            total += count
            batch_counts.append(count)
        per_frame.append(batch_counts)
    return total, per_frame


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.amp.autocast("cuda", dtype=DTYPE)
    return nullcontext()


def run_demo(
    video_input,
    image_input,
    reconstructor_path,
    num_frames,
    num_context_views,
    width,
    height,
    resize_mode,
    degradation_mode,
    trans_min,
    trans_max,
    culling_prob,
    kernel_size,
    occlusion_thresh,
    alpha_thresh,
):
    if DEVICE.type != "cuda":
        raise gr.Error("这个真实退化 demo 需要 CUDA/GPU，因为 reconstructor 和 gsplat rasterizer 依赖 GPU。")

    start = time.time()
    num_frames = int(num_frames)
    num_context_views = max(1, min(int(num_context_views), num_frames))
    width = max(14, int(round(float(width) / 14.0)) * 14)
    height = max(14, int(round(float(height) / 14.0)) * 14)
    trans_min, trans_max = sorted((float(trans_min), float(trans_max)))
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    frames, input_name = load_frames(video_input, image_input, num_frames, (width, height), resize_mode)
    source_views, context_num, ordered_indices = build_training_views(frames, num_context_views)
    reconstructor = get_reconstructor(reconstructor_path)

    from diffsynth.pipelines.wan_video_neoverse import WanVideoUnit_4DPreprocesser

    degrader = WanVideoUnit_4DPreprocesser(
        novel_view_sampling_trans=[trans_min, trans_max],
        novel_view_sampling_max_rot=0.0,
        culling_prob=float(culling_prob),
        kernel_size_range=[kernel_size, kernel_size],
        occlusion_thresh=float(occlusion_thresh),
        alpha_thresh=float(alpha_thresh),
    )

    with torch.inference_mode(), autocast_context():
        recon_output = reconstructor(source_views, is_inference=False)

    before_total, before_per_frame = count_gaussians(recon_output["splats"])
    render_viewmats = homo_matrix_inverse(recon_output["rendered_extrinsics"])
    render_Ks = recon_output["rendered_intrinsics"]
    render_timestamps = recon_output["rendered_timestamps"]

    with torch.inference_mode(), autocast_context():
        baseline_rgb, _, baseline_alpha = reconstructor.gs_renderer.rasterizer.forward(
            copy.deepcopy(recon_output["splats"]),
            render_viewmats=render_viewmats,
            render_Ks=render_Ks,
            render_timestamps=render_timestamps,
            sh_degree=0,
            width=width,
            height=height,
        )

    novel_context_poses = degrader.novel_view_sampling(
        recon_output["rendered_extrinsics"][:, :context_num],
        recon_output["gs_depth"].squeeze(-1),
    )

    if degradation_mode == "Gaussian culling":
        active_kernel = 0
    elif degradation_mode == "Average geometry filter":
        active_kernel = kernel_size
    else:
        active_kernel = 0 if np.random.rand() < float(culling_prob) else kernel_size

    with torch.inference_mode():
        degraded_splats = degrader.degradation_simulation(
            recon_output["splats"],
            novel_context_poses,
            recon_output["rendered_intrinsics"][:, :context_num],
            (height, width),
            kernel_size=active_kernel,
            occlusion_thresh=float(occlusion_thresh),
        )
    with torch.inference_mode(), autocast_context():
        degraded_rgb, _, degraded_alpha = reconstructor.gs_renderer.rasterizer.forward(
            degraded_splats,
            render_viewmats=render_viewmats,
            render_Ks=render_Ks,
            render_timestamps=render_timestamps,
            sh_degree=0,
            width=width,
            height=height,
        )

    after_total, after_per_frame = count_gaussians(degraded_splats)
    order = torch.argsort(render_timestamps[0])
    source_video = source_views["img"][0, order].permute(0, 2, 3, 1)
    baseline_rgb = baseline_rgb[0, order]
    degraded_rgb = degraded_rgb[0, order]
    alpha_mask = (degraded_alpha > float(alpha_thresh)).float()
    train_mask = alpha_mask.clone()
    train_mask[:, context_num:] = 0.0
    train_mask = train_mask[0, order]

    source_frames = tensor_rgb_to_frames(source_video)
    baseline_frames = tensor_rgb_to_frames(baseline_rgb)
    degraded_frames = tensor_rgb_to_frames(degraded_rgb)
    mask_frames = tensor_mask_to_frames(train_mask)
    comparison_frames = build_comparison_frames(source_frames, baseline_frames, degraded_frames, mask_frames)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, stamp)
    comparison_path = save_video(comparison_frames, os.path.join(run_dir, "comparison.mp4"), fps=12)
    baseline_path = save_video(baseline_frames, os.path.join(run_dir, "baseline_render.mp4"), fps=12)
    degraded_path = save_video(degraded_frames, os.path.join(run_dir, "degraded_render.mp4"), fps=12)
    mask_path = save_video(mask_frames, os.path.join(run_dir, "training_mask.mp4"), fps=12)

    elapsed = time.time() - start
    mode_text = "visibility-based Gaussian culling" if active_kernel == 0 else f"average geometry filter, kernel={active_kernel}"
    removed = before_total - after_total
    status = (
        f"Input: {input_name}\n"
        f"Frames: {num_frames}, context views: {context_num}, ordered indices: {ordered_indices.tolist()}\n"
        f"Degradation: {mode_text}\n"
        f"Gaussians: {before_total:,} -> {after_total:,} ({removed:,} removed or repositioned/cull-kept)\n"
        f"Per-context before: {before_per_frame[0][:context_num]}\n"
        f"Per-context after: {after_per_frame[0][:context_num]}\n"
        f"Output dir: {run_dir}\n"
        f"Elapsed: {elapsed:.1f}s"
    )

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return comparison_path, degraded_path, baseline_path, mask_path, status


def build_ui(default_reconstructor):
    with gr.Blocks(title="NeoVerse Degradation Simulation Demo") as demo:
        gr.Markdown(
            "# NeoVerse Degradation Simulation\n"
            "可视化 paper 里的 training-time degradation：随机 novel trajectory、Gaussian culling、average geometry filter，以及训练用 mask。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="输入视频", value=SAMPLE_VIDEO)
                image_input = gr.Image(label="或输入单张图片", type="pil")
                reconstructor_path = gr.Textbox(label="Reconstructor checkpoint", value=default_reconstructor)
                with gr.Row():
                    num_frames = gr.Slider(5, 41, value=21, step=2, label="帧数")
                    num_context_views = gr.Slider(2, 21, value=7, step=1, label="Context views")
                with gr.Row():
                    width = gr.Slider(224, 672, value=560, step=14, label="宽")
                    height = gr.Slider(168, 448, value=336, step=14, label="高")
                resize_mode = gr.Radio(["center_crop", "resize"], value="center_crop", label="Resize mode")
                degradation_mode = gr.Radio(
                    ["Gaussian culling", "Average geometry filter", "Random training policy"],
                    value="Average geometry filter",
                    label="退化模式",
                )
                with gr.Row():
                    trans_min = gr.Slider(0.0, 0.5, value=0.01, step=0.01, label="Novel shift min")
                    trans_max = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Novel shift max")
                with gr.Row():
                    culling_prob = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Random culling prob")
                    kernel_size = gr.Slider(3, 101, value=31, step=2, label="Average filter kernel")
                with gr.Row():
                    occlusion_thresh = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Occlusion threshold")
                    alpha_thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Alpha threshold")
                gr.Examples(
                    examples=[
                        [
                            "examples/videos/tree_and_building.mp4",
                            None,
                            default_reconstructor,
                            21,
                            7,
                            560,
                            336,
                            "center_crop",
                            "Average geometry filter",
                            0.01,
                            0.10,
                            0.30,
                            31,
                            0.10,
                            0.50,
                        ],
                        [
                            "examples/videos/driving.mp4",
                            None,
                            default_reconstructor,
                            21,
                            7,
                            560,
                            336,
                            "center_crop",
                            "Gaussian culling",
                            0.01,
                            0.12,
                            0.30,
                            31,
                            0.10,
                            0.50,
                        ],
                        [
                            None,
                            "examples/videos/room.png",
                            default_reconstructor,
                            13,
                            5,
                            560,
                            336,
                            "center_crop",
                            "Average geometry filter",
                            0.01,
                            0.10,
                            0.30,
                            41,
                            0.10,
                            0.50,
                        ],
                    ],
                    inputs=[
                        video_input,
                        image_input,
                        reconstructor_path,
                        num_frames,
                        num_context_views,
                        width,
                        height,
                        resize_mode,
                        degradation_mode,
                        trans_min,
                        trans_max,
                        culling_prob,
                        kernel_size,
                        occlusion_thresh,
                        alpha_thresh,
                    ],
                    label="内置示例",
                )
                run_btn = gr.Button("运行退化模拟", variant="primary")
            with gr.Column(scale=1):
                comparison = gr.Video(label="四宫格对比")
                degraded = gr.Video(label="退化渲染")
                baseline = gr.Video(label="未退化重建渲染")
                mask = gr.Video(label="训练 mask")
                status = gr.Textbox(label="运行日志", lines=10)

        run_btn.click(
            fn=run_demo,
            inputs=[
                video_input,
                image_input,
                reconstructor_path,
                num_frames,
                num_context_views,
                width,
                height,
                resize_mode,
                degradation_mode,
                trans_min,
                trans_max,
                culling_prob,
                kernel_size,
                occlusion_thresh,
                alpha_thresh,
            ],
            outputs=[comparison, degraded, baseline, mask, status],
        )
    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt")
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    demo = build_ui(args.reconstructor_path)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
