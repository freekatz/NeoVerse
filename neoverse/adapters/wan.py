import torch

from wan.modules.model import sinusoidal_embedding_1d


def patchify_tensor(dit, x: torch.Tensor):
    x = dit.patch_embedding(x)
    grid = tuple(int(v) for v in x.shape[2:])
    x = x.flatten(2).transpose(1, 2)
    return x, grid


def unpatchify_tensor(dit, x: torch.Tensor, grid):
    f, h, w = grid
    pt, ph, pw = dit.patch_size
    c = dit.out_dim
    return torch.einsum(
        "bfhwtpqc->bctfhpwq",
        x.reshape(x.shape[0], f, h, w, pt, ph, pw, c),
    ).reshape(x.shape[0], c, f * pt, h * ph, w * pw)


def official_rope_freqs(dit, f: int, h: int, w: int, device):
    return dit.freqs.to(device)


def expanded_rope_freqs_for_legacy_control(dit, f: int, h: int, w: int, device):
    head_dim = dit.dim // dit.num_heads
    c = head_dim // 2
    freqs = dit.freqs.to(device).split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    return torch.cat(
        [
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)


def model_fn_official_wan_neoverse(
    dit,
    control_branch=None,
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
    precomputed_control_hints=None,
    **kwargs,
):
    """NeoVerse denoise loop for official `wan.modules.model.WanModel`.

    This is the migration bridge. It mirrors the old NeoVerse manual block loop
    but calls official Wan blocks with their native `(e, seq_lens, grid_sizes,
    freqs, context, context_lens)` signature.
    """
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x, grid = patchify_tensor(dit, latents)
    f, h, w = grid
    seq_len = f * h * w
    seq_lens = torch.tensor([seq_len] * x.shape[0], dtype=torch.long, device=x.device)
    grid_sizes = torch.tensor([grid] * x.shape[0], dtype=torch.long, device=x.device)
    freqs = official_rope_freqs(dit, f, h, w, x.device)

    control_hints = None
    if precomputed_control_hints is not None:
        control_hints = tuple(precomputed_control_hints.values()) if isinstance(precomputed_control_hints, dict) else tuple(precomputed_control_hints)
    elif target_rgb is not None and control_branch is not None:
        legacy_freqs = expanded_rope_freqs_for_legacy_control(dit, f, h, w, x.device)
        control_hints = control_branch(
            x,
            target_rgb,
            target_depth,
            target_camera_embed,
            target_mask,
            context,
            t_mod,
            legacy_freqs,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
        )

    for block_id, block in enumerate(dit.blocks):
        x = block(
            x,
            e=t_mod,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=None,
        )
        if control_hints is not None and block_id in control_branch.control_layers_mapping:
            hint = control_hints[control_branch.control_layers_mapping[block_id]].to(device=x.device, dtype=x.dtype)
            x = x + hint * control_scale

    x = dit.head(x, t)
    return unpatchify_tensor(dit, x, grid)
