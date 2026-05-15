from PIL import Image
import numpy as np
import torch
from einops import reduce, repeat

from .utils.device import get_device_type, parse_device_type


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        output_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None,
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.output_params = output_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def fetch_input_params(self):
        params = []
        if self.input_params is not None:
            params.extend(self.input_params)
        if self.input_params_posi is not None:
            params.extend(self.input_params_posi.values())
        if self.input_params_nega is not None:
            params.extend(self.input_params_nega.values())
        return sorted(set(params))

    def fetch_output_params(self):
        return [] if self.output_params is None else list(self.output_params)

    def process(self, pipe, **kwargs) -> dict:
        return {}


class PipelineUnitRunner:
    def __call__(self, unit: PipelineUnit, pipe, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict):
        if unit.take_over:
            return unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)
        if unit.seperate_cfg:
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)

            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_nega.update(processor_outputs)
            return inputs_shared, inputs_posi, inputs_nega

        processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
        inputs_shared.update(unit.process(pipe, **processor_inputs))
        return inputs_shared, inputs_posi, inputs_nega


class BasePipeline(torch.nn.Module):
    def __init__(
        self,
        device=get_device_type(),
        torch_dtype=torch.float16,
        height_division_factor=64,
        width_division_factor=64,
        time_division_factor=None,
        time_division_remainder=None,
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.device_type = parse_device_type(device)
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False
        self.unit_runner = PipelineUnitRunner()

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
            self.device_type = parse_device_type(device)
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None, verbose=1):
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            if verbose > 0:
                print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            if verbose > 0:
                print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        if num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
            if verbose > 0:
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
        return height, width, num_frames

    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        image = torch.tensor(np.array(image, dtype=np.float32), dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        return repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))

    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1, normalize=None):
        if isinstance(video, torch.Tensor):
            out = video.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
            if normalize is not None:
                return out / normalize
            return out * ((max_value - min_value) / 1.0) + min_value if out.min() >= 0 and out.max() <= 1 else out
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        return torch.stack(video, dim=pattern.index("T") // 2)

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        return Image.fromarray(image.to(device="cpu", dtype=torch.uint8).numpy())

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        return [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]

    def load_models_to_device(self, model_names):
        if not self.vram_management_enabled:
            return
        for name, model in self.named_children():
            if name not in model_names and hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                for module in model.modules():
                    if hasattr(module, "offload"):
                        module.offload()
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()
        for name, model in self.named_children():
            if name in model_names and hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                for module in model.modules():
                    if hasattr(module, "onload"):
                        module.onload()

    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        return noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)

    def get_vram(self):
        if torch.device(self.device).type != "cuda":
            return 0
        return torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)

    def get_module(self, model, name):
        if "." not in name:
            return getattr(model, name)
        head, tail = name.split(".", 1)
        return self.get_module(model[int(head)] if head.isdigit() else getattr(model, head), tail)

    def freeze_except(self, model_names, lora_base_model=None):
        self.eval()
        self.requires_grad_(False)
        for name in model_names:
            module = self.get_module(self, name)
            if module is None:
                print(f"No {name} models in the pipeline.")
                continue
            module.train()
            module.requires_grad_(True)
