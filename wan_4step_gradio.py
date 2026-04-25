import argparse
import gc
import glob
import os
import threading
import time
from pathlib import Path

for proxy_env in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(proxy_env, None)

import gradio as gr
import torch

from diffsynth import save_video
from diffsynth.lora import LightX2VLoRALoader
from diffsynth.models import ModelManager, load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for 4-step distilled Wan text-to-video.")
    parser.add_argument("--model_path", type=str, default="models", help="Model root directory.")
    parser.add_argument("--model_id", type=str, default="NeoVerse", help="Model folder name under model_path.")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="models/NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        help="Path to LightX2V 4-step distilled LoRA.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--low_vram", action="store_true", help="Load models on CPU and enable offload wrappers.")
    parser.add_argument("--output_dir", type=str, default="outputs/wan_4step_gradio")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
OUTPUT_DIR = Path(ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PIPE = None
LOAD_LOCK = threading.Lock()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _torch_dtype():
    return DTYPE_MAP[ARGS.dtype]


def _model_dir():
    return Path(ARGS.model_path) / ARGS.model_id


def _check_required_files():
    model_dir = _model_dir()
    diffusion_files = sorted(
        str(path)
        for path in model_dir.glob("diffusion_pytorch_model-*.safetensors")
        if not path.name.endswith(".index.json")
    )
    required = [
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        model_dir / "Wan2.1_VAE.pth",
        model_dir / "google" / "umt5-xxl",
        Path(ARGS.lora_path),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if not diffusion_files:
        missing.append(str(model_dir / "diffusion_pytorch_model-*.safetensors"))
    if missing:
        raise FileNotFoundError("Missing required model files:\n" + "\n".join(missing))
    return diffusion_files, str(required[0]), str(required[1]), str(required[3])


def _load_pipe():
    global PIPE
    if PIPE is not None:
        return PIPE, "模型已加载。"

    with LOAD_LOCK:
        if PIPE is not None:
            return PIPE, "模型已加载。"

        diffusion_files, text_encoder_path, vae_path, lora_path = _check_required_files()
        torch_dtype = _torch_dtype()
        load_device = "cpu" if ARGS.low_vram else ARGS.device

        started_at = time.perf_counter()
        manager = ModelManager(torch_dtype=torch_dtype, device=load_device)
        manager.load_model(diffusion_files, device=load_device, torch_dtype=torch_dtype)
        manager.load_model(text_encoder_path, device=load_device, torch_dtype=torch_dtype)
        manager.load_model(vae_path, device=load_device, torch_dtype=torch_dtype)

        pipe = WanVideoPipeline.from_model_manager(manager, torch_dtype=torch_dtype, device=ARGS.device)

        patch_device = "cpu" if ARGS.low_vram else ARGS.device
        lora_state = load_state_dict(lora_path, torch_dtype=torch_dtype, device=patch_device)
        LightX2VLoRALoader(device=patch_device, torch_dtype=torch_dtype).load(pipe.dit, lora_state, alpha=1.0)
        del lora_state

        if ARGS.low_vram and ARGS.device.startswith("cuda"):
            pipe.enable_vram_management()

        PIPE = pipe
        elapsed = time.perf_counter() - started_at
        return PIPE, f"模型加载完成，用时 {elapsed:.1f}s。"


def _unload_pipe():
    global PIPE
    with LOAD_LOCK:
        PIPE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    return "模型已卸载，显存缓存已清理。"


def _progress_wrapper(progress):
    def wrap(iterable):
        total = len(iterable)
        for index, item in enumerate(iterable):
            progress((index, total), desc=f"Denoising {index + 1}/{total}")
            yield item
        progress((total, total), desc="Decoding video")

    return wrap


def load_model_ui():
    _, status = _load_pipe()
    return status


def unload_model_ui():
    return _unload_pipe()


def generate(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_inference_steps,
    fps,
    tiled,
    progress=gr.Progress(track_tqdm=False),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("请输入 prompt。")

    if int(num_frames) % 4 != 1:
        num_frames = (int(num_frames) + 2) // 4 * 4 + 1
    width = int(width)
    height = int(height)
    seed = int(seed)
    fps = int(fps)

    pipe, load_status = _load_pipe()
    progress(0, desc="Generating")

    started_at = time.perf_counter()
    frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        seed=seed,
        rand_device=ARGS.device,
        height=height,
        width=width,
        num_frames=int(num_frames),
        cfg_scale=float(cfg_scale),
        num_inference_steps=int(num_inference_steps),
        tiled=bool(tiled),
        progress_bar_cmd=_progress_wrapper(progress),
    )

    output_path = OUTPUT_DIR / f"wan4step_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4"
    save_video(frames, str(output_path), fps=fps)
    elapsed = time.perf_counter() - started_at

    status = (
        f"{load_status}\n"
        f"生成完成：{output_path}\n"
        f"参数：{width}x{height}, frames={num_frames}, steps={num_inference_steps}, "
        f"cfg={cfg_scale}, seed={seed}, fps={fps}\n"
        f"耗时：{elapsed:.1f}s"
    )
    return str(output_path), status


def build_demo():
    with gr.Blocks(title="Wan 4-step Distilled T2V") as demo:
        gr.Markdown(
            "# Wan 4-step Distilled 文生视频测试\n"
            "加载本地 `NeoVerse` 里的 Wan 2.1 T2V 权重，并合并 LightX2V 4-step distilled LoRA。"
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    value="A cinematic shot of a small robot walking through a futuristic laboratory, smooth camera motion, high detail.",
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    lines=2,
                    value="low quality, blurry, distorted, watermark, text",
                )

                with gr.Row():
                    width = gr.Dropdown(
                        label="宽度",
                        choices=[416, 512, 560, 640, 704, 832],
                        value=560,
                    )
                    height = gr.Dropdown(
                        label="高度",
                        choices=[240, 288, 320, 336, 384, 480],
                        value=336,
                    )
                    num_frames = gr.Slider(label="帧数", minimum=17, maximum=81, value=81, step=4)

                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    cfg_scale = gr.Slider(label="CFG scale", minimum=1.0, maximum=8.0, value=1.0, step=0.1)
                    num_inference_steps = gr.Slider(label="推理步数", minimum=4, maximum=50, value=4, step=1)
                    fps = gr.Slider(label="FPS", minimum=8, maximum=24, value=16, step=1)

                tiled = gr.Checkbox(label="VAE tiled decode/encode", value=True)

                with gr.Row():
                    load_btn = gr.Button("加载模型")
                    unload_btn = gr.Button("卸载模型")
                    generate_btn = gr.Button("生成视频", variant="primary")

                status = gr.Textbox(label="状态", lines=7)

            with gr.Column(scale=1):
                output_video = gr.Video(label="输出视频")
                gr.Markdown(
                    "建议先用默认 `560x336 / 81 frames / 4 steps / cfg=1.0` 做 smoke test。\n\n"
                    "当前仓库默认用 `steps=4`、`cfg=1.0` 跑这个 distilled LoRA；调大步数可以跑，但就不是复现仓库默认设置。"
                )

        load_btn.click(load_model_ui, outputs=status)
        unload_btn.click(unload_model_ui, outputs=status)
        generate_btn.click(
            generate,
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                num_frames,
                seed,
                cfg_scale,
                num_inference_steps,
                fps,
                tiled,
            ],
            outputs=[output_video, status],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=2)
    demo.launch(server_name=ARGS.server_name, server_port=ARGS.server_port, share=ARGS.share)
