import warnings
from collections import OrderedDict
from contextlib import nullcontext
import torch, warnings, os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import repeat, rearrange
from typing import Optional, Union
from tqdm import tqdm
try:
    from gsplat.rendering import rasterization
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,
        _quat_scale_to_covar_preci,
    )
except ModuleNotFoundError:
    rasterization = None
    _fully_fused_projection = None
    _quat_scale_to_covar_preci = None

from .base import BasePipeline, PipelineUnit, PipelineUnitRunner
from .schedulers import FlowMatchScheduler
from .loaders import ModelConfig, ModelPool, load_state_dict
from .vram import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
    WanAutoCastLayerNorm,
)
from .lora import GeneralLoRALoader
from .data import save_video
from .utils.auxiliary import (
    homo_matrix_inverse,
    average_filter,
    fast_perceptual_color_distance,
    pixel_to_world_coords,
)
from .models.wan_video_text_encoder import (
    WanTextEncoder,
    T5RelativeEmbedding,
    T5LayerNorm,
    HuggingfaceTokenizer,
)
from wan.modules.vae_neoverse import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from .control_branch import NeoVerseControlBranch
from .adapters.wan import model_fn_official_wan_neoverse, sinusoidal_embedding_1d
try:
    from wan.modules.model import WanRMSNorm
except ImportError:
    WanRMSNorm = torch.nn.Module
RMSNorm = WanRMSNorm


def _autocast_context(device, dtype):
    device = torch.device(device)
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.amp.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def build_vis_output_path(save_root, source_views, filename):
    output_path = save_root + "/"
    if source_views is not None and "dataset" in source_views:
        output_path += source_views["dataset"][0][0] + "/"
    if source_views is not None and "video_name" in source_views:
        output_path += source_views["video_name"][0][0] + "/"
    output_path += filename
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


def save_tensor_video_for_vis(save_root, source_views, filename, frames, pattern="B T C H W", min_value=0.0, max_value=1.0):
    if save_root is None:
        return
    if pattern == "B T C H W":
        frames = rearrange(frames, "B T C H W -> (B T) H W C")
    elif pattern == "B T H W C":
        frames = rearrange(frames, "B T H W C -> (B T) H W C")
    elif pattern == "B T H W":
        frames = rearrange(frames, "B T H W -> (B T) H W")
    else:
        raise ValueError(f"Unsupported video visualization pattern: {pattern}")

    frames = frames.detach().to(device="cpu", dtype=torch.float32)
    scale = max(max_value - min_value, 1e-6)
    frames = ((frames - min_value) * (255.0 / scale)).clip(0, 255).to(dtype=torch.uint8)

    video = []
    for frame in frames:
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame.squeeze(-1)
        video.append(Image.fromarray(frame.numpy()))

    save_video(video, build_vis_output_path(save_root, source_views, filename), fps=15)


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        return {"context": self.encode_prompt(pipe, prompt)}


class WanVideoNeoVersePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, pipeline_kwargs={}):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.control_branch: NeoVerseControlBranch = None
        self.reconstructor = None
        self.in_iteration_models = ("dit", "control_branch")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_4DPreprocesser(**pipeline_kwargs),
            WanVideoUnit_CameraProcesser(),
            WanVideoUnit_RandomDrop(**pipeline_kwargs),
            WanVideoUnit_4DEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_wan_video
        self.save_root = None
        self.is_training = False
        self.trainable_models = []

    def load_lora(self, module, path=None, state_dict=None, alpha=1, lora_type="neoverse"):
        if lora_type != "neoverse":
            raise ValueError(f"Unsupported lora_type {lora_type}.")
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        if path is not None:
            lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        loader.load(module, lora, alpha=alpha)

    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        with _autocast_context(self.device, self.torch_dtype):
            noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    WanRMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
                max_num_param=num_persistent_param_in_dit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.control_branch is not None:
            dtype = next(iter(self.control_branch.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.control_branch,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.GroupNorm: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                " for details.")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None and name in self.trainable_models:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    @staticmethod
    def from_pretrained(
        local_model_path: str = "models",
        reconstructor_path: str = "models/NeoVerse/reconstructor.ckpt",
        pipeline_kwargs: dict = {},
        lora_path: Optional[str] = None,
        lora_alpha: float = 1.0,
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_vram_management: bool = False,
        load_dit: bool = True,
        load_text_encoder: bool = True,
        load_vae: bool = True,
    ):
        """Load a NeoVerse pipeline.

        Loads the base WAN dit / text_encoder / vae / tokenizer via WanVideoPipeline.from_pretrained,
        then attaches the reconstructor (via ModelPool) and the NeoVerseControlBranch
        (extracted from the same diffusion checkpoint by filtering 'control*' keys).
        """
        offload_device = "cpu" if enable_vram_management else device

        # 1) Build NeoVerse pipeline shell.
        pipe = WanVideoNeoVersePipeline(device=device, torch_dtype=torch_dtype, pipeline_kwargs=pipeline_kwargs)

        # 2) Locate the combined diffusion checkpoint (contains WanModel + NeoVerseControlBranch weights).
        diffusion_cfg = ModelConfig(
            local_model_path=local_model_path,
            model_id="NeoVerse",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device=offload_device,
        )
        diffusion_cfg.download_if_necessary()

        # 3) Load the WAN side (dit / text_encoder / vae / tokenizer) without the DiffSynth pipeline wrapper.
        base_model_configs = []
        if load_dit:
            base_model_configs.append(ModelConfig(path=diffusion_cfg.path, offload_device=offload_device))
        if load_text_encoder:
            text_cfg = ModelConfig(
                local_model_path=local_model_path,
                model_id="NeoVerse",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device=offload_device,
            )
            text_cfg.download_if_necessary()
            base_model_configs.append(text_cfg)
        if load_vae:
            vae_cfg = ModelConfig(
                local_model_path=local_model_path,
                model_id="NeoVerse",
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device=offload_device,
            )
            vae_cfg.download_if_necessary()
            base_model_configs.append(vae_cfg)

        tokenizer_config = ModelConfig(
            local_model_path=local_model_path, model_id="NeoVerse", origin_file_pattern="google/*"
        )
        tokenizer_config.download_if_necessary()

        model_pool = ModelPool()
        for model_config in base_model_configs:
            model_config.download_if_necessary()
            vram_config = model_config.vram_config()
            default_vram_config = model_pool.default_vram_config()
            for key, value in default_vram_config.items():
                if vram_config.get(key) is None:
                    vram_config[key] = value
            vram_config["computation_dtype"] = torch_dtype
            vram_config["computation_device"] = offload_device
            vram_config["onload_dtype"] = torch_dtype
            vram_config["onload_device"] = offload_device
            model_pool.auto_load_model(
                model_config.path,
                vram_config=vram_config,
                clear_parameters=model_config.clear_parameters,
                state_dict=model_config.state_dict,
            )

        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder") if load_text_encoder else None
        pipe.dit = model_pool.fetch_model("wan_video_dit") if load_dit else None
        if pipe.dit is not None and pipe.dit.__class__.__module__.startswith("wan."):
            pipe.model_fn = model_fn_official_wan_neoverse
        pipe.vae = model_pool.fetch_model("wan_video_vae") if load_vae else None
        pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean="whitespace")

        # 4) Load NeoVerseControlBranch from the same combined diffusion checkpoint.
        ctrl_state_dict = load_state_dict(diffusion_cfg.path, torch_dtype=torch_dtype, device=offload_device)
        converter = NeoVerseControlBranch.state_dict_converter()
        ctrl_state, ctrl_config = converter.from_civitai(ctrl_state_dict)
        pipe.control_branch = NeoVerseControlBranch(**ctrl_config)
        incompatible = pipe.control_branch.load_state_dict(ctrl_state, assign=True, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "Failed to load NeoVerse control branch cleanly: "
                f"missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}."
            )
        pipe.control_branch.to(dtype=torch_dtype, device=offload_device).eval()

        # 5) Load reconstructor via ModelPool (registered in MODEL_CONFIGS by hash).
        recon_pool = ModelPool()
        recon_vram_config = recon_pool.default_vram_config()
        recon_vram_config["computation_dtype"] = torch_dtype
        recon_vram_config["computation_device"] = offload_device
        recon_vram_config["onload_dtype"] = torch_dtype
        recon_vram_config["onload_device"] = offload_device
        recon_pool.auto_load_model(reconstructor_path, vram_config=recon_vram_config)
        pipe.reconstructor = recon_pool.fetch_model("reconstructor")

        # 6) Size division factor.
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # 7) Distilled LoRA for faster inference.
        if lora_path is not None:
            if pipe.dit is None:
                raise ValueError("LoRA loading requires load_dit=True.")
            assert os.path.exists(lora_path), f"LoRA path {lora_path} does not exist."
            pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha, lora_type="neoverse")
            print(f"Loaded LoRA from {lora_path}")

        if enable_vram_management:
            pipe.enable_vram_management()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_video: Optional[list] = None,
        denoising_strength: Optional[float] = 1.0,
        control_scale: Optional[float] = 1.0,
        source_views: Optional[dict] = None,
        target_rgb: Optional[torch.Tensor] = None,
        target_depth: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        target_intrs: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        cfg_scale: Optional[float] = 5.0,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple] = (30, 52),
        tile_stride: Optional[tuple] = (15, 26),
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        progress_bar_cmd=tqdm,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        inputs_posi = {
            "prompt": prompt if isinstance(prompt, list) else [prompt],
            "positive": True,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt if isinstance(negative_prompt, list) else [negative_prompt],
            "positive": False,
        }
        inputs_shared = {
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_scale": control_scale, "source_views": source_views,
            "target_rgb": target_rgb, "target_depth": target_depth, "target_mask": target_mask,
            "target_poses": target_poses, "target_intrs": target_intrs,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        self.load_models_to_device(["vae"])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


class WanVideoUnit_4DPreprocesser(PipelineUnit):
    def __init__(
        self,
        novel_view_sampling_trans=[0.1, 0.5],
        novel_view_sampling_max_rot=0.0,
        culling_prob=0.3,
        kernel_size_range=[11, 101],
        occlusion_thresh=0.1,
        alpha_thresh=0.5,
        color_thresh=[50, 100],
        mask_non_context_targets=True,
        **kwargs,
    ):
        super().__init__(
            input_params=("source_views", "target_rgb", "target_depth", "target_mask", "target_poses", "target_intrs"),
            onload_model_names=("reconstructor",)
        )
        self.novel_view_sampling_trans = novel_view_sampling_trans
        self.novel_view_sampling_max_rot = novel_view_sampling_max_rot
        self.culling_prob = culling_prob
        self.kernel_size_range = kernel_size_range
        self.occlusion_thresh = occlusion_thresh
        self.alpha_thresh = alpha_thresh
        self.color_thresh = color_thresh
        self.mask_non_context_targets = bool(mask_non_context_targets)

    def _order_indices(self, source_views, b_idx, fallback_timestamps):
        trajectory_index = source_views.get("trajectory_index")
        if not isinstance(trajectory_index, torch.Tensor) and isinstance(trajectory_index, (np.ndarray, list, tuple)):
            try:
                trajectory_index = torch.as_tensor(trajectory_index, device=fallback_timestamps.device, dtype=torch.long)
            except (TypeError, ValueError):
                trajectory_index = None
        if isinstance(trajectory_index, torch.Tensor):
            if trajectory_index.ndim >= 2:
                trajectory_index = trajectory_index[b_idx]
            return torch.argsort(trajectory_index.to(device=fallback_timestamps.device))
        return torch.argsort(fallback_timestamps[b_idx])

    @staticmethod
    def _gather_view_tensor(tensor, indices):
        if indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(tensor.shape[0], -1)
        index = indices.to(device=tensor.device, dtype=torch.long)
        index = index.reshape(index.shape[0], index.shape[1], *([1] * (tensor.ndim - 2)))
        index = index.expand(-1, -1, *tensor.shape[2:])
        return torch.gather(tensor, dim=1, index=index)

    def _gather_views(self, views, indices):
        total_views = views["img"].shape[1]
        gathered = {}
        for key, value in views.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == total_views:
                gathered[key] = self._gather_view_tensor(value, indices)
            else:
                gathered[key] = value
        return gathered

    def _continuous_order(self, source_views):
        timestamps = source_views["timestamp"]
        orders = []
        for b_idx in range(timestamps.shape[0]):
            orders.append(self._order_indices(source_views, b_idx, timestamps))
        return torch.stack(orders, dim=0).to(device=timestamps.device, dtype=torch.long)

    def _target_trajectory_indices(self, source_views, continuous_order):
        timestamps = source_views["timestamp"]
        target_map = source_views.get("target_trajectory_index")
        if not isinstance(target_map, torch.Tensor) and isinstance(target_map, (np.ndarray, list, tuple)):
            try:
                target_map = torch.as_tensor(target_map, device=timestamps.device, dtype=torch.long)
            except (TypeError, ValueError):
                target_map = None
        if isinstance(target_map, torch.Tensor):
            target_map = target_map.to(device=timestamps.device, dtype=torch.long)
            if target_map.ndim == 1:
                target_map = target_map.unsqueeze(0)
            return self._gather_view_tensor(target_map.unsqueeze(-1), continuous_order).squeeze(-1)
        batch_size, num_views = timestamps.shape
        return torch.arange(num_views, device=timestamps.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    def process(self, pipe: WanVideoNeoVersePipeline, source_views, target_rgb, target_depth, target_mask, target_poses, target_intrs):
        if source_views is None:
            return {}

        if isinstance(source_views, list):
            source_views = self.compose_batches_from_list(source_views)
        continuous_order = self._continuous_order(source_views)
        continuous_views = self._gather_views(source_views, continuous_order)
        target_indices = self._target_trajectory_indices(source_views, continuous_order)
        target_input_video = self._gather_view_tensor(continuous_views["img"], target_indices)
        continuous_is_context = ~continuous_views["is_target"].bool()
        context_counts = continuous_is_context.sum(dim=1)
        if not bool(torch.all(context_counts == context_counts[0]).item()):
            raise ValueError("NeoVerse preprocessing expects the same number of context views in every batch item.")
        context_indices = torch.stack(
            [torch.nonzero(mask, as_tuple=False).flatten() for mask in continuous_is_context],
            dim=0,
        ).to(device=continuous_views["img"].device, dtype=torch.long)
        continuous_context_views = self._gather_views(continuous_views, context_indices)
        if pipe.is_training:
            input_video = target_input_video.clone()
            assert len(input_video) == 1, "During training, only batch size 1 is supported."
        else:
            input_video = None

        if target_rgb is not None and target_depth is not None and target_mask is not None and target_poses is not None and target_intrs is not None:
            return {
                "input_video": input_video,
                "target_rgb": target_rgb,
                "target_depth": target_depth,
                "target_mask": target_mask,
                "target_poses": target_poses,
                "target_intrs": target_intrs,
            }

        has_gt_cameras = (
            isinstance(source_views.get("camera_poses"), torch.Tensor)
            and isinstance(source_views.get("camera_intrs"), torch.Tensor)
        )
        has_camera_cache = has_gt_cameras and "camera_cache_path" in source_views
        has_annotation_cameras = has_gt_cameras and not has_camera_cache
        if has_annotation_cameras:
            recon_source_views = dict(continuous_context_views)
            recon_source_views["is_target"] = torch.zeros_like(continuous_context_views["is_target"])
        elif has_camera_cache:
            recon_source_views = dict(continuous_context_views)
            recon_source_views.pop("camera_poses", None)
            recon_source_views.pop("camera_intrs", None)
            recon_source_views["is_target"] = torch.zeros_like(recon_source_views["is_target"])
        else:
            recon_source_views = dict(continuous_views)

        pipe.load_models_to_device(self.onload_model_names)
        with _autocast_context(pipe.device, pipe.torch_dtype):
            skip_unused_heads = bool(getattr(pipe, "skip_unused_reconstructor_heads", False))
            force_motion_tokens = (has_annotation_cameras or has_camera_cache) and bool(getattr(pipe, "reuse_context_forward_tokens", True))
            if bool(getattr(pipe, "reuse_reconstructor_tokens", False)):
                try:
                    recon_output = pipe.reconstructor(
                        recon_source_views,
                        is_inference=False,
                        return_token_list=True,
                        skip_unused_heads=skip_unused_heads,
                        force_motion_tokens=force_motion_tokens,
                    )
                except TypeError:
                    recon_output = pipe.reconstructor(recon_source_views, is_inference=False)
            else:
                try:
                    recon_output = pipe.reconstructor(
                        recon_source_views,
                        is_inference=False,
                        skip_unused_heads=skip_unused_heads,
                        force_motion_tokens=force_motion_tokens,
                    )
                except TypeError:
                    recon_output = pipe.reconstructor(recon_source_views, is_inference=False)
        if has_annotation_cameras:
            render_extrinsics = continuous_views["camera_poses"]
            render_intrinsics = continuous_views["camera_intrs"]
            render_timestamps = continuous_views["timestamp"]
            context_extrinsics = recon_source_views["camera_poses"]
            context_intrinsics = recon_source_views["camera_intrs"]
        elif has_camera_cache:
            render_extrinsics = continuous_views["camera_poses"]
            render_intrinsics = continuous_views["camera_intrs"]
            render_timestamps = continuous_views["timestamp"]
            context_extrinsics = recon_output["rendered_extrinsics"]
            context_intrinsics = recon_output["rendered_intrinsics"]
            cached_context_extrinsics = self._gather_view_tensor(render_extrinsics, context_indices)
            cached_norm = cached_context_extrinsics[:, :, :3, 3].norm(dim=-1).mean(dim=1, keepdim=True)
            pred_norm = context_extrinsics[:, :, :3, 3].norm(dim=-1).mean(dim=1, keepdim=True)
            scale_factor = (pred_norm / (cached_norm + 1e-6)).to(dtype=render_extrinsics.dtype)
            render_extrinsics = render_extrinsics.clone()
            render_extrinsics[:, :, :3, 3] = render_extrinsics[:, :, :3, 3] * scale_factor.unsqueeze(-1)
        else:
            render_extrinsics = recon_output["rendered_extrinsics"]
            render_intrinsics = recon_output["rendered_intrinsics"]
            render_timestamps = recon_output["rendered_timestamps"]
            context_extrinsics = self._gather_view_tensor(render_extrinsics, context_indices)
            context_intrinsics = self._gather_view_tensor(render_intrinsics, context_indices)
        novel_context_poses = self.novel_view_sampling(
            context_extrinsics,
            recon_output["gs_depth"].squeeze(-1),
        )
        if self.culling_prob >= 1.0:
            kernel_size = 0
        elif self.culling_prob <= 0.0 and self.kernel_size_range[0] == self.kernel_size_range[1]:
            kernel_size = int(self.kernel_size_range[0])
        elif np.random.rand() < self.culling_prob:
            kernel_size = 0
        else:
            kernel_size = np.random.randint(self.kernel_size_range[0], self.kernel_size_range[1]+1)
        H, W = source_views["img"].shape[-2:]
        splats = self.degradation_simulation(
            recon_output["splats"],
            novel_context_poses,
            context_intrinsics,
            (H, W),
            kernel_size=kernel_size,
            occlusion_thresh=self.occlusion_thresh,
        )

        target_poses = self._gather_view_tensor(render_extrinsics, target_indices)
        target_intrs = self._gather_view_tensor(render_intrinsics, target_indices)
        target_timestamps = self._gather_view_tensor(render_timestamps.unsqueeze(-1), target_indices).squeeze(-1)
        source_poses = self._gather_view_tensor(render_extrinsics, context_indices)
        source_intrs = self._gather_view_tensor(render_intrinsics, context_indices)
        source_timestamps = self._gather_view_tensor(render_timestamps.unsqueeze(-1), context_indices).squeeze(-1)
        target_is_target = None
        if (
            isinstance(source_views.get("is_target"), torch.Tensor)
            and source_views["is_target"].shape[:2] == render_timestamps.shape[:2]
        ):
            continuous_is_target = self._gather_view_tensor(source_views["is_target"], continuous_order)
            target_is_target = self._gather_view_tensor(continuous_is_target, target_indices)

        rendered_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
            splats,
            render_viewmats=homo_matrix_inverse(target_poses),   # c2w -> w2c
            render_Ks=target_intrs,
            render_timestamps=target_timestamps,
            sh_degree=0, width=W, height=H,
        )
        target_rgb = rearrange(target_input_video, "B T C H W -> B T H W C")
        target_mask = target_alpha > self.alpha_thresh
        if self.mask_non_context_targets and target_is_target is not None:
            target_mask = target_mask.masked_fill(target_is_target[:, :, None, None, None].bool(), False)

        if input_video is not None:
            if self.color_thresh[0] == self.color_thresh[1]:
                color_thresh = float(self.color_thresh[0])
            else:
                color_thresh = np.random.uniform(self.color_thresh[0], self.color_thresh[1])
            color_mask = fast_perceptual_color_distance(
                rearrange(input_video, "B T C H W -> B T H W C"),
                rendered_rgb,
            ) < color_thresh
            target_mask = target_mask & color_mask.unsqueeze(-1)
        target_mask = target_mask.float()

        if pipe.save_root is not None:
            for_save = rearrange(input_video, "B T C H W -> (B T) H W C")
            for_save = (for_save * 255).clip(0, 255)
            video = [Image.fromarray(image.to(device="cpu", dtype=torch.uint8).numpy()) for image in for_save]
            output_path = pipe.save_root + "/"
            if "dataset" in source_views:
                output_path += source_views["dataset"][0][0] + "/"
            if "video_name" in source_views:
                output_path += source_views["video_name"][0][0] + "/"
            output_path += "gt.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_video(video, output_path, fps=15)
        output = {
            "source_views": source_views,
            "input_video": input_video,
            "target_rgb": target_rgb,
            "target_depth": target_depth,
            "target_mask": target_mask,
            "target_poses": target_poses,
            "target_intrs": target_intrs,
            "target_timestamps": target_timestamps,
            "source_timestamps": source_timestamps,
            "source_poses": source_poses,
            "source_intrs": source_intrs,
        }
        if "token_list" in recon_output:
            output["reconstructor_token_list"] = recon_output["token_list"]
            output["reconstructor_patch_start_idx"] = recon_output.get("patch_start_idx")
        if "context_token_list" in recon_output:
            output["reconstructor_context_token_list"] = recon_output["context_token_list"]
            output["reconstructor_context_patch_start_idx"] = recon_output.get("context_patch_start_idx")
        if hasattr(pipe.reconstructor, "visual_geometry_transformer"):
            output["reconstructor_token_layers"] = tuple(pipe.reconstructor.visual_geometry_transformer.intermediate_idxs)
            output["reconstructor_patch_size"] = int(getattr(pipe.reconstructor.visual_geometry_transformer, "patch_size", 14))
        return output

    def compose_batches_from_list(self, batch):
        batched_inputs = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                batched_inputs[key] = torch.stack([b[key] for b in batch], dim=1)
            elif isinstance(batch[0][key], np.ndarray):
                batched_inputs[key] = np.stack([b[key] for b in batch], axis=1)
            elif isinstance(batch[0][key], (int, float, str, bool, list)):
                batched_inputs[key] = [b[key] for b in batch]
            else:
                continue
        return batched_inputs

    def novel_view_sampling(self, poses, depths):
        batch_size = len(poses)
        novel_context_poses_list = []
        for batch_idx in range(batch_size):
            novel_context_poses = self._generate_novel_pose(
                poses[batch_idx], depths[batch_idx]
            )
            novel_context_poses_list.append(novel_context_poses)
        novel_context_poses = torch.stack(novel_context_poses_list, dim=0)
        return novel_context_poses

    def _generate_novel_pose(self, original_poses, depth_maps):
        trans_min, trans_max = float(self.novel_view_sampling_trans[0]), float(self.novel_view_sampling_trans[1])
        if trans_min == 0.0 and trans_max == 0.0 and float(self.novel_view_sampling_max_rot) == 0.0:
            return original_poses.clone()

        N_context = original_poses.shape[0]
        original_pos = original_poses[:, :3, 3]
        original_rotation = original_poses[:, :3, :3]
        original_forward = original_rotation[..., 2]

        valid_depth = depth_maps > 0
        scene_depth = (depth_maps * valid_depth).sum(dim=(1, 2)) / valid_depth.sum(dim=(1, 2)).clamp_min(1e-6)
        view_center = original_pos + scene_depth[:, None] * original_forward

        if N_context > 1:
            global_movement_dir = original_pos[-1] - original_pos[0]
            global_movement_dir = F.normalize(global_movement_dir.unsqueeze(0), dim=-1)
        else:
            global_movement_dir = original_forward.unsqueeze(0)

        up = torch.tensor([0.0, -1.0, 0.0], device=original_poses.device).unsqueeze(0)

        use_left = torch.rand(1, device=original_poses.device).item() > 0.5
        if use_left:
            perpendicular_dir = torch.cross(up, global_movement_dir, dim=-1)
        else:
            perpendicular_dir = torch.cross(global_movement_dir, up, dim=-1)

        perpendicular_dir = F.normalize(perpendicular_dir, dim=-1)
        random_direction = perpendicular_dir.expand(N_context, -1)
        translation_distance = torch.rand((1, 1), device=original_poses.device) * (trans_max - trans_min) + trans_min
        new_pos = original_pos + random_direction * translation_distance

        new_forward = F.normalize(view_center - new_pos, dim=-1)

        angle_rad = torch.deg2rad(torch.tensor(self.novel_view_sampling_max_rot, device=original_poses.device))
        random_rotation_angle = (torch.rand((N_context, 1), device=original_poses.device) - 0.5) * 2 * angle_rad

        random_axis = torch.randn((N_context, 3), device=original_poses.device)
        random_axis = random_axis - torch.sum(random_axis * new_forward, dim=-1, keepdim=True) * new_forward
        random_axis = F.normalize(random_axis, dim=-1)

        cos_angle = torch.cos(random_rotation_angle)
        sin_angle = torch.sin(random_rotation_angle)
        new_forward = (cos_angle * new_forward +
                      sin_angle * torch.cross(random_axis, new_forward, dim=-1))
        new_forward = F.normalize(new_forward, dim=-1)

        up = torch.tensor([0.0, 1.0, 0.0], device=original_poses.device)[None].repeat(N_context, 1)
        right = torch.cross(up, new_forward, dim=-1)
        right = F.normalize(right, dim=-1)
        new_up = torch.cross(new_forward, right, dim=-1)

        new_rotation = torch.stack([right, new_up, new_forward], dim=-1)

        new_pose = torch.zeros_like(original_poses)
        new_pose[:, :3, :3] = new_rotation
        new_pose[:, :3, 3] = new_pos
        new_pose[:, 3, 3] = 1.0
        return new_pose

    def degradation_simulation(self, gaussians, novel_context_poses, context_intrinsics, image_size_hw, kernel_size, occlusion_thresh=0.1):
        if rasterization is None or _fully_fused_projection is None or _quat_scale_to_covar_preci is None:
            raise ModuleNotFoundError("NeoVerse degradation simulation requires gsplat.")
        batch_size = len(gaussians)
        novel_context_world2cam = homo_matrix_inverse(novel_context_poses)
        h, w = image_size_hw
        for b_idx in range(batch_size):
            for s_idx in range(len(novel_context_poses[b_idx])):
                cur_gaussian = gaussians[b_idx][s_idx]
                cur_extrinsic = novel_context_world2cam[b_idx][s_idx]
                cur_intrinsic = context_intrinsics[b_idx][s_idx]
                if cur_gaussian.means.shape[0] == 0:
                    continue

                covars, _ = _quat_scale_to_covar_preci(
                    cur_gaussian.rotations,
                    cur_gaussian.scales,
                    True,
                    False,
                    triu=False
                )
                radii, means2d, depths, conics, compensations = _fully_fused_projection(
                    cur_gaussian.means,
                    covars,
                    cur_extrinsic[None],
                    cur_intrinsic[None],
                    w, h,
                )
                valid_gs_indices = torch.where((radii[0, :, 0] > 0) & (radii[0, :, 1] > 0))[0]
                if len(valid_gs_indices) == 0:
                    continue
                gs_x = means2d[0, valid_gs_indices, 0].round().long().clamp(0, w - 1)
                gs_y = means2d[0, valid_gs_indices, 1].round().long().clamp(0, h - 1)
                gs_depths = depths[0, valid_gs_indices]

                rendered_depths, rendered_alphas, _ = rasterization(
                    means=cur_gaussian.means,
                    quats=cur_gaussian.rotations,
                    scales=cur_gaussian.scales,
                    opacities=cur_gaussian.opacities,
                    colors=cur_gaussian.harmonics,
                    viewmats=cur_extrinsic[None],
                    Ks=cur_intrinsic[None],
                    width=w,
                    height=h,
                    sh_degree=0,
                    packed=False,
                    render_mode="ED",
                )
                rendered_depths = rendered_depths[0, ..., 0]
                visible_mask = (gs_depths < (rendered_depths[gs_y, gs_x] + occlusion_thresh))
                visible_indices = valid_gs_indices[visible_mask]
                if kernel_size == 0:
                    cur_gaussian.keep_indices(visible_indices)
                else:
                    smoothed_depths = average_filter(rendered_depths, kernel_size=kernel_size)
                    smoothed_gs_depths = smoothed_depths[gs_y[visible_mask], gs_x[visible_mask]]
                    visible_gs_x = gs_x[visible_mask]
                    visible_gs_y = gs_y[visible_mask]

                    world_coords = pixel_to_world_coords(
                        visible_gs_x, visible_gs_y, smoothed_gs_depths,
                        cur_intrinsic, cur_extrinsic
                    )
                    world_coords = world_coords.to(dtype=cur_gaussian.means.dtype)
                    cur_gaussian.means[visible_indices] = world_coords
                    cur_gaussian.keep_indices(visible_indices)
        return gaussians


class WanVideoUnit_CameraProcesser(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("target_poses", "target_intrs", "height", "width"),
        )

    def process(self, pipe: WanVideoNeoVersePipeline, target_poses, target_intrs, height, width):
        if target_poses is None:
            return {}
        camera_maps = self.convert_plucker_map(target_poses, target_intrs, height, width)
        return {
            "target_camera_embed": camera_maps,
        }

    def convert_plucker_map(self, poses, intrinsics, height, width):
        batch_size, num_frames = poses.shape[:2]
        device = poses.device
        dtype = poses.dtype

        poses = poses.reshape(batch_size * num_frames, 4, 4)
        intrinsics = intrinsics.reshape(batch_size * num_frames, 3, 3)

        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij"
        )
        pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)
        pixel_coords = pixel_coords[None].expand(batch_size * num_frames, -1, -1)

        rays_d_cam = intrinsics.inverse() @ pixel_coords

        rotation_matrices = poses[:, :3, :3]
        rays_d_world = rotation_matrices @ rays_d_cam
        rays_d_world = F.normalize(rays_d_world, dim=1)
        rays_d_world = rays_d_world.reshape(batch_size * num_frames, 3, height, width)

        rays_o_world = poses[:, :3, 3]
        rays_o_world = rays_o_world[..., None, None].expand_as(rays_d_world)

        moment = torch.cross(rays_o_world, rays_d_world, dim=1)
        plucker_embedding = torch.cat([rays_d_world, moment], dim=1)
        plucker_embedding = plucker_embedding.reshape(batch_size, num_frames, 6, height, width)
        return plucker_embedding


class WanVideoUnit_RandomDrop(PipelineUnit):
    def __init__(self, prompt_drop_prob=0.0, mask_drop_prob=0.0, condition_drop_prob=0.0, **kwargs):
        super().__init__(take_over=True)
        self.prompt_drop_prob = prompt_drop_prob
        self.mask_drop_prob = mask_drop_prob
        self.condition_drop_prob = condition_drop_prob

    def process(self, pipe: WanVideoNeoVersePipeline, inputs_shared, inputs_posi, inputs_nega):
        prompt = inputs_posi["prompt"]
        target_rgb = inputs_shared["target_rgb"]
        target_depth = inputs_shared["target_depth"]
        target_camera_embed = inputs_shared["target_camera_embed"]
        target_mask = inputs_shared["target_mask"]
        assert len(prompt) == target_rgb.shape[0] == target_depth.shape[0] == target_camera_embed.shape[0] == target_mask.shape[0], "Batch size must be the same for prompt and target conditions"

        batch_size = target_rgb.shape[0]
        for b_idx in range(batch_size):
            if np.random.rand() < self.prompt_drop_prob:
                prompt[b_idx] = ""
            if np.random.rand() < self.mask_drop_prob:
                target_mask[b_idx] *= 0
            if np.random.rand() < self.condition_drop_prob:
                target_rgb[b_idx] *= 0
                target_depth[b_idx] *= 0
                target_camera_embed[b_idx] *= 0
                target_mask[b_idx] *= 0
        inputs_posi["prompt"] = prompt
        inputs_shared["target_rgb"] = target_rgb
        inputs_shared["target_depth"] = target_depth
        inputs_shared["target_camera_embed"] = target_camera_embed
        inputs_shared["target_mask"] = target_mask
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_4DEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("source_views", "target_rgb", "target_depth", "target_camera_embed", "target_mask",
                          "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoNeoVersePipeline,
        source_views, target_rgb, target_depth, target_camera_embed, target_mask,
        tiled, tile_size, tile_stride
    ):
        if target_rgb is not None:
            batch_size = len(target_rgb)
            try:
                target_d_max = torch.quantile(target_depth.reshape(batch_size, -1), 0.98, dim=-1).clamp_min(1e-6)
            except:
                target_d_max = target_depth.reshape(batch_size, -1).max(dim=-1).values.clamp_min(1e-6)
            d_max = target_d_max[:, None, None, None, None]

            pipe.load_models_to_device(self.onload_model_names)
            target_rgb = rearrange(target_rgb, "B T H W C -> B T C H W")

            if pipe.save_root is not None:
                save_tensor_video_for_vis(pipe.save_root, source_views, "target_rgb.mp4", target_rgb)

            target_depth = repeat(target_depth, "B T H W 1 -> B T C H W", C=3)

            if pipe.save_root is not None:
                target_depth_vis = (target_depth / d_max).clamp(0, 1)
                save_tensor_video_for_vis(pipe.save_root, source_views, "target_depth.mp4", target_depth_vis)

            target_rgb = pipe.preprocess_video(target_rgb)
            target_depth = pipe.preprocess_video(target_depth, normalize=d_max).clamp(min=-1, max=1)
            if bool(getattr(pipe, "batch_vae_teacher_embeds", False)):
                target_batch = torch.cat([target_rgb, target_depth], dim=0)
                target_latents = pipe.vae.encode(target_batch, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                target_rgb_latents, target_depth_latents = target_latents[:batch_size], target_latents[batch_size:]
            else:
                target_rgb_latents = pipe.vae.encode(target_rgb, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                target_depth_latents = pipe.vae.encode(target_depth, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            if pipe.save_root is not None:
                save_tensor_video_for_vis(pipe.save_root, source_views, "target_mask.mp4", target_mask, pattern="B T H W C")

                target_camera_dir = target_camera_embed[:, :, :3].clamp(-1, 1)
                target_camera_moment = target_camera_embed[:, :, 3:]
                target_camera_moment_scale = target_camera_moment.float().abs().reshape(batch_size, -1).amax(dim=-1).clamp_min(1e-6)[:, None, None, None, None]
                target_camera_moment = (target_camera_moment / target_camera_moment_scale).clamp(-1, 1)
                save_tensor_video_for_vis(
                    pipe.save_root,
                    source_views,
                    "target_camera_dir.mp4",
                    target_camera_dir,
                    min_value=-1.0,
                    max_value=1.0,
                )
                save_tensor_video_for_vis(
                    pipe.save_root,
                    source_views,
                    "target_camera_moment.mp4",
                    target_camera_moment,
                    min_value=-1.0,
                    max_value=1.0,
                )
            return {
                "target_rgb": target_rgb_latents,
                "target_depth": target_depth_latents,
                "target_camera_embed": target_camera_embed.to(dtype=pipe.torch_dtype, device=pipe.device),
                "target_mask": rearrange(target_mask, "B T H W C -> B T C H W").to(dtype=pipe.torch_dtype, device=pipe.device),
            }
        else:
            return {}


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoNeoVersePipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device"))

    def process(self, pipe: WanVideoNeoVersePipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoNeoVersePipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


def model_fn_wan_video(
    dit: WanModel,
    control_branch: NeoVerseControlBranch = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    control_scale=1.0,
    target_rgb=None,
    target_depth=None,
    target_camera_embed=None,
    target_mask=None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    precomputed_control_hints=None,
    **kwargs,
):
    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    control_hints = None
    if precomputed_control_hints is not None:
        if isinstance(precomputed_control_hints, dict):
            control_hints = []
            for layer in control_branch.control_layers:
                layer_key = f"layer_{layer}"
                if layer_key in precomputed_control_hints:
                    control_hints.append(precomputed_control_hints[layer_key])
                elif layer in precomputed_control_hints:
                    control_hints.append(precomputed_control_hints[layer])
                else:
                    control_hints.append(precomputed_control_hints[str(layer)])
            control_hints = tuple(control_hints)
        else:
            control_hints = tuple(precomputed_control_hints)
    elif target_rgb is not None:
        control_hints = control_branch(
            x, target_rgb, target_depth, target_camera_embed, target_mask,
            context, t_mod, freqs, use_gradient_checkpointing, use_gradient_checkpointing_offload,
        )

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

        if control_hints is not None and block_id in control_branch.control_layers_mapping:
            current_control_hint = control_hints[control_branch.control_layers_mapping[block_id]]
            current_control_hint = current_control_hint.to(device=x.device, dtype=x.dtype)
            x = x + current_control_hint * control_scale

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


NeoVersePipeline = WanVideoNeoVersePipeline
