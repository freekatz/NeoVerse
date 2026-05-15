import json
import os

import torch

from neoverse.control_branch import NeoVerseControlBranch
from neoverse.loaders.file import load_state_dict
from neoverse.loaders.converters.wan_video_dit import WanVideoDiTStateDictConverter


def split_legacy_neoverse_state_dict(state_dict):
    """Split a legacy NeoVerse diffusion checkpoint into new package parts.

    Legacy NeoVerse diffusion checkpoints store the Wan DiT weights and the
    NeoVerse control branch in one state dict. The new package keeps these
    concepts separate, so this converter is intentionally a migration tool,
    not a runtime dependency.
    """
    base_state = WanVideoDiTStateDictConverter(state_dict)
    control_converter = NeoVerseControlBranch.state_dict_converter()
    control_state, control_config = control_converter.from_civitai(state_dict)
    return {
        "wan_dit": base_state,
        "control_branch": control_state,
        "control_config": control_config,
    }


def load_and_split_legacy_neoverse(path, torch_dtype=None, device="cpu"):
    return split_legacy_neoverse_state_dict(load_state_dict(path, torch_dtype=torch_dtype, device=device))


def save_state_dict(path, state_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(state_dict, path)
    else:
        torch.save(state_dict, path)


def save_split_legacy_neoverse(input_path, output_dir, torch_dtype=None, device="cpu"):
    parts = load_and_split_legacy_neoverse(input_path, torch_dtype=torch_dtype, device=device)
    os.makedirs(output_dir, exist_ok=True)
    save_state_dict(os.path.join(output_dir, "wan_dit.safetensors"), parts["wan_dit"])
    save_state_dict(os.path.join(output_dir, "control_branch.safetensors"), parts["control_branch"])
    with open(os.path.join(output_dir, "control_config.json"), "w") as handle:
        json.dump(parts["control_config"], handle, indent=2)
    return parts
