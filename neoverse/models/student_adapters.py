from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


def _num_groups(channels: int, requested: int = 32) -> int:
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    out_dtype = values.dtype
    values = values.float()
    half = dim // 2
    if half == 0:
        return values[..., None]
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = values[..., None] * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb.to(dtype=out_dtype)


def _resize_5d(x: torch.Tensor, size: tuple[int, int, int], mode: str = "nearest-exact") -> torch.Tensor:
    try:
        return F.interpolate(x, size=size, mode=mode)
    except ValueError:
        fallback = "nearest" if mode == "nearest-exact" else mode
        return F.interpolate(x, size=size, mode=fallback)


def _interp_sequence(values: torch.Tensor, frames: int) -> torch.Tensor:
    if values.shape[1] == frames:
        return values
    values = values.transpose(1, 2)
    values = F.interpolate(values.float(), size=frames, mode="linear", align_corners=False)
    return values.transpose(1, 2)


def _shared_time_scale(
    batch_size: int,
    device,
    dtype,
    *time_values: torch.Tensor | None,
) -> torch.Tensor:
    scales = []
    for values in time_values:
        if values is None:
            continue
        values = values.to(device=device, dtype=dtype)
        if values.ndim == 1:
            values = values[None].expand(batch_size, -1)
        scales.append(values.reshape(batch_size, -1).detach().abs().amax(dim=1, keepdim=True))
    if not scales:
        return torch.ones(batch_size, 1, device=device, dtype=dtype)
    return torch.stack(scales, dim=0).amax(dim=0).clamp_min(1.0)


def _control_key(layer: int | str) -> str:
    if isinstance(layer, str):
        return layer if layer.startswith("layer_") else f"layer_{layer}"
    return f"layer_{int(layer)}"


class CausalConv3d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, dilation=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kt, kh, kw = self.kernel_size
        dt, dh, dw = self.dilation
        pad_t = (kt - 1) * dt
        pad_h = ((kh - 1) * dh) // 2
        pad_w = ((kw - 1) * dw) // 2
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_t, 0))
        return self.conv(x)


class CausalResBlock(torch.nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        groups = _num_groups(channels)
        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(groups, channels),
            torch.nn.SiLU(),
            CausalConv3d(channels, channels, kernel_size=3),
            torch.nn.GroupNorm(groups, channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            CausalConv3d(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConditionGridEncoder(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        time_dim: int = 128,
        pose_dim: int = 128,
        ray_dim: int = 64,
        pos_dim: int = 32,
        diffusion_time_dim: int = 128,
        text_context_dim: int | None = None,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.pose_dim = pose_dim
        self.ray_dim = ray_dim
        self.pos_dim = pos_dim
        self.diffusion_time_dim = diffusion_time_dim
        self.pose_mlp = torch.nn.Sequential(
            torch.nn.Linear(25, pose_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(pose_dim, pose_dim),
        )
        self.ray_proj = torch.nn.Sequential(
            torch.nn.Conv3d(6, ray_dim, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(ray_dim, ray_dim, kernel_size=1),
        )
        self.pos_proj = torch.nn.Sequential(
            torch.nn.Conv3d(6, pos_dim, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(pos_dim, pos_dim, kernel_size=1),
        )
        self.text_context_dim = text_context_dim
        self.text_proj = (
            torch.nn.Sequential(
                torch.nn.Linear(text_context_dim, diffusion_time_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(diffusion_time_dim, diffusion_time_dim),
            )
            if text_context_dim is not None
            else None
        )
        total = time_dim + pose_dim + ray_dim + pos_dim + diffusion_time_dim
        if self.text_proj is not None:
            total += diffusion_time_dim
        self.out = torch.nn.Sequential(
            torch.nn.Conv3d(total, out_channels, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=1),
        )

    def _target_time(
        self,
        target_times: torch.Tensor | None,
        bsz: int,
        frames: int,
        device,
        dtype,
        time_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_times is None:
            values = torch.linspace(0, 1, frames, device=device, dtype=dtype)[None].expand(bsz, -1)
        else:
            values = target_times.to(device=device, dtype=dtype)
            if values.ndim == 1:
                values = values[None].expand(bsz, -1)
            values = _interp_sequence(values[..., None], frames).squeeze(-1)
            denom = time_scale.to(device=device, dtype=dtype) if time_scale is not None else values.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
            values = values / denom
        return values

    def _pose_embedding(
        self,
        target_poses: torch.Tensor | None,
        target_intrs: torch.Tensor | None,
        bsz: int,
        frames: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if target_poses is None:
            pose = torch.eye(4, device=device, dtype=dtype).reshape(1, 1, 16).expand(bsz, frames, -1)
        else:
            pose = target_poses.to(device=device, dtype=dtype).reshape(target_poses.shape[0], target_poses.shape[1], 16)
            pose = _interp_sequence(pose, frames)
        if target_intrs is None:
            intr = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 9).expand(bsz, frames, -1)
        else:
            intr = target_intrs.to(device=device, dtype=dtype).reshape(target_intrs.shape[0], target_intrs.shape[1], 9)
            intr = _interp_sequence(intr, frames)
        pose_intr = torch.cat([pose, intr], dim=-1).to(dtype=self.pose_mlp[0].weight.dtype)
        return self.pose_mlp(pose_intr).to(dtype=dtype)

    def _position_embedding(self, bsz: int, frames: int, height: int, width: int, device, dtype) -> torch.Tensor:
        t = torch.linspace(-1, 1, frames, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
        coords = torch.stack(
            [
                tt,
                yy,
                xx,
                torch.sin(math.pi * tt),
                torch.sin(math.pi * yy),
                torch.sin(math.pi * xx),
            ],
            dim=0,
        )
        coords = coords[None].expand(bsz, -1, -1, -1, -1)
        return self.pos_proj(coords)

    def forward(
        self,
        output_grid: tuple[int, int, int],
        *,
        batch_size: int,
        device,
        dtype,
        target_times: torch.Tensor | None = None,
        target_poses: torch.Tensor | None = None,
        target_intrs: torch.Tensor | None = None,
        target_plucker: torch.Tensor | None = None,
        diffusion_timestep: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
        time_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frames, height, width = output_grid
        time_values = self._target_time(target_times, batch_size, frames, device, dtype, time_scale=time_scale)
        time_emb = sinusoidal_embedding(time_values, self.time_dim)
        time_emb = time_emb.permute(0, 2, 1)[:, :, :, None, None].expand(-1, -1, -1, height, width)

        pose_emb = self._pose_embedding(target_poses, target_intrs, batch_size, frames, device, dtype)
        pose_emb = pose_emb.permute(0, 2, 1)[:, :, :, None, None].expand(-1, -1, -1, height, width)

        if target_plucker is None:
            ray_emb = torch.zeros(batch_size, self.ray_dim, frames, height, width, device=device, dtype=dtype)
        else:
            rays = target_plucker.to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)
            rays = _resize_5d(rays, (frames, height, width), mode="trilinear")
            ray_emb = self.ray_proj(rays)

        pos_emb = self._position_embedding(batch_size, frames, height, width, device, dtype)

        if diffusion_timestep is None:
            diffusion_timestep = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            diffusion_timestep = diffusion_timestep.to(device=device, dtype=dtype).flatten()
            if diffusion_timestep.numel() == 1:
                diffusion_timestep = diffusion_timestep.expand(batch_size)
        diffusion_emb = sinusoidal_embedding(diffusion_timestep, self.diffusion_time_dim)
        diffusion_emb = diffusion_emb[:, :, None, None, None].expand(-1, -1, frames, height, width)

        parts = [time_emb, pose_emb, ray_emb, pos_emb, diffusion_emb]
        if self.text_proj is not None:
            if text_context is None:
                text_emb = torch.zeros(
                    batch_size, self.diffusion_time_dim, frames, height, width, device=device, dtype=dtype
                )
            else:
                pooled = text_context.to(device=device, dtype=dtype).mean(dim=1).float()
                text_emb = self.text_proj(pooled).to(dtype=dtype)
                text_emb = text_emb[:, :, None, None, None].expand(-1, -1, frames, height, width)
            parts.append(text_emb)

        parts = [part.to(dtype=dtype, device=device) for part in parts]
        return self.out(torch.cat(parts, dim=1))


class SourceViewConditionEncoder(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        time_dim: int = 128,
        pose_dim: int = 128,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.pose_dim = pose_dim
        self.pose_mlp = torch.nn.Sequential(
            torch.nn.Linear(25, pose_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(pose_dim, pose_dim),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Conv3d(time_dim + pose_dim, out_channels, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=1),
        )

    def _source_time(
        self,
        source_times: torch.Tensor | None,
        bsz: int,
        views: int,
        device,
        dtype,
        time_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if source_times is None:
            return torch.linspace(0, 1, views, device=device, dtype=dtype)[None].expand(bsz, -1)
        values = source_times.to(device=device, dtype=dtype)
        if values.ndim == 1:
            values = values[None].expand(bsz, -1)
        if values.shape[1] != views:
            values = _interp_sequence(values[..., None], views).squeeze(-1)
        denom = time_scale.to(device=device, dtype=dtype) if time_scale is not None else values.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        return values / denom

    def _source_pose(
        self,
        source_poses: torch.Tensor | None,
        source_intrs: torch.Tensor | None,
        bsz: int,
        views: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if source_poses is None:
            pose = torch.eye(4, device=device, dtype=dtype).reshape(1, 1, 16).expand(bsz, views, -1)
        else:
            pose = source_poses.to(device=device, dtype=dtype).reshape(source_poses.shape[0], source_poses.shape[1], 16)
            if pose.shape[1] != views:
                pose = _interp_sequence(pose, views)
        if source_intrs is None:
            intr = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 9).expand(bsz, views, -1)
        else:
            intr = source_intrs.to(device=device, dtype=dtype).reshape(source_intrs.shape[0], source_intrs.shape[1], 9)
            if intr.shape[1] != views:
                intr = _interp_sequence(intr, views)
        pose_intr = torch.cat([pose, intr], dim=-1).to(dtype=self.pose_mlp[0].weight.dtype)
        return self.pose_mlp(pose_intr).to(dtype=dtype)

    def forward(
        self,
        *,
        batch_size: int,
        views: int,
        height: int,
        width: int,
        device,
        dtype,
        source_times: torch.Tensor | None = None,
        source_poses: torch.Tensor | None = None,
        source_intrs: torch.Tensor | None = None,
        time_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_values = self._source_time(source_times, batch_size, views, device, dtype, time_scale)
        time_emb = sinusoidal_embedding(time_values, self.time_dim)
        pose_emb = self._source_pose(source_poses, source_intrs, batch_size, views, device, dtype)
        cond = torch.cat([time_emb, pose_emb], dim=-1).transpose(1, 2)
        cond = cond[:, :, :, None, None].expand(-1, -1, -1, height, width)
        return self.out(cond)


class ConvAdapter(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        control_layers: Iterable[int],
        hidden_dim: int = 512,
        num_res_blocks: int = 4,
        dropout: float = 0.0,
        condition_time_dim: int = 128,
        text_context_dim: int | None = None,
        use_dit_state: bool = True,
        output_mode: str = "condition_embedding",
    ):
        super().__init__()
        if output_mode not in {"condition_embedding", "hints"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}")
        self.output_mode = output_mode
        self.output_dim = int(output_dim)
        self.control_layers = tuple(int(layer) for layer in control_layers)
        self.input_channels = int(input_channels)
        self.condition = ConditionGridEncoder(
            hidden_dim,
            time_dim=condition_time_dim,
            diffusion_time_dim=condition_time_dim,
            text_context_dim=text_context_dim,
        )
        self.source_condition = SourceViewConditionEncoder(
            hidden_dim,
            time_dim=condition_time_dim,
            pose_dim=condition_time_dim,
        )
        self.input_proj = torch.nn.Conv3d(self.input_channels, hidden_dim, kernel_size=1)
        self.dit_state_proj = (
            torch.nn.Conv3d(self.output_dim, hidden_dim, kernel_size=1) if use_dit_state else None
        )
        self.blocks = torch.nn.Sequential(*[CausalResBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)])
        self.condition_head = torch.nn.Conv3d(hidden_dim, self.output_dim, kernel_size=1)
        self.heads = (
            torch.nn.ModuleDict(
                {_control_key(layer): torch.nn.Conv3d(hidden_dim, self.output_dim, kernel_size=1) for layer in self.control_layers}
            )
            if self.output_mode == "hints"
            else torch.nn.ModuleDict()
        )

    def _tokens_to_grid(self, tokens: torch.Tensor, output_grid: tuple[int, int, int]) -> torch.Tensor:
        if tokens.ndim != 6:
            raise ValueError(f"Expected tokens with shape B x N x L x h x w x C, got {tuple(tokens.shape)}")
        bsz, views, groups, token_h, token_w, channels = tokens.shape
        expected = groups * channels
        if expected != self.input_channels:
            raise ValueError(
                f"Adapter input_channels={self.input_channels}, but token groups x channels is {groups} x {channels} = {expected}."
            )
        x = tokens.permute(0, 1, 2, 5, 3, 4).reshape(bsz, views, groups * channels, token_h, token_w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return _resize_5d(x, output_grid, mode="nearest-exact")

    def _dit_state_to_grid(self, dit_tokens: torch.Tensor, output_grid: tuple[int, int, int]) -> torch.Tensor:
        frames, height, width = output_grid
        expected_seq = frames * height * width
        if dit_tokens.shape[1] != expected_seq:
            raise ValueError(
                f"DiT token sequence length mismatch: got {dit_tokens.shape[1]}, expected {expected_seq} for grid {output_grid}."
            )
        return dit_tokens.transpose(1, 2).reshape(dit_tokens.shape[0], dit_tokens.shape[-1], frames, height, width)

    def forward(
        self,
        tokens: torch.Tensor,
        output_grid: tuple[int, int, int],
        *,
        target_times: torch.Tensor | None = None,
        target_poses: torch.Tensor | None = None,
        target_intrs: torch.Tensor | None = None,
        target_plucker: torch.Tensor | None = None,
        source_times: torch.Tensor | None = None,
        source_poses: torch.Tensor | None = None,
        source_intrs: torch.Tensor | None = None,
        dit_tokens: torch.Tensor | None = None,
        diffusion_timestep: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
    ) -> OrderedDict[str, torch.Tensor]:
        tokens = tokens.to(dtype=self.input_proj.weight.dtype)
        bsz, views, _, token_h, token_w, _ = tokens.shape
        device, dtype = tokens.device, tokens.dtype
        time_scale = _shared_time_scale(bsz, device, dtype, target_times, source_times)
        x = self.input_proj(self._tokens_to_grid(tokens, output_grid))
        source_cond = self.source_condition(
            batch_size=bsz,
            views=views,
            height=token_h,
            width=token_w,
            device=device,
            dtype=dtype,
            source_times=source_times,
            source_poses=source_poses,
            source_intrs=source_intrs,
            time_scale=time_scale,
        )
        x = x + _resize_5d(source_cond, output_grid, mode="nearest-exact")
        x = x + self.condition(
            output_grid,
            batch_size=bsz,
            device=device,
            dtype=dtype,
            target_times=target_times,
            target_poses=target_poses,
            target_intrs=target_intrs,
            target_plucker=target_plucker,
            diffusion_timestep=diffusion_timestep,
            text_context=text_context,
            time_scale=time_scale,
        )
        if self.dit_state_proj is not None and dit_tokens is not None:
            x = x + self.dit_state_proj(self._dit_state_to_grid(dit_tokens.to(dtype=dtype), output_grid))
        x = self.blocks(x)

        if self.output_mode == "condition_embedding":
            return self.condition_head(x).flatten(2).transpose(1, 2).contiguous()

        out: OrderedDict[str, torch.Tensor] = OrderedDict()
        for layer in self.control_layers:
            hint = self.heads[_control_key(layer)](x)
            out[_control_key(layer)] = hint.flatten(2).transpose(1, 2).contiguous()
        return out


def _time_positions(
    times: torch.Tensor | None,
    batch_size: int,
    length: int,
    device,
    dtype,
    time_scale: torch.Tensor | None,
) -> torch.Tensor:
    values = _time_values(times, batch_size, length, device, dtype)
    if times is None:
        return values
    denom = time_scale.to(device=device, dtype=dtype) if time_scale is not None else values.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    return values / denom


def _time_values(
    times: torch.Tensor | None,
    batch_size: int,
    length: int,
    device,
    dtype,
) -> torch.Tensor:
    if times is None:
        return torch.linspace(0, 1, length, device=device, dtype=dtype)[None].expand(batch_size, -1)
    values = times.to(device=device, dtype=dtype)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values[None]
    if values.shape[0] == 1 and batch_size != 1:
        values = values.expand(batch_size, -1)
    if values.shape[1] != length:
        values = _interp_sequence(values[..., None], length).squeeze(-1)
    return values


def _reroped_time_positions(
    query_times: torch.Tensor | None,
    source_times: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    source_length: int,
    device,
    dtype,
    interval: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = _time_values(query_times, batch_size, query_length, device, dtype)
    source = _time_values(source_times, batch_size, source_length, device, dtype)
    if source_length <= 1:
        return torch.zeros_like(query), torch.zeros_like(source)

    interval = float(interval)
    query_positions = []
    source_positions = []
    for batch_idx in range(batch_size):
        src = source[batch_idx].float()
        qry = query[batch_idx].float()
        order = torch.argsort(src)
        sorted_src = src[order].contiguous()

        ranks = torch.arange(source_length, device=device, dtype=torch.float32) * interval
        src_pos = torch.empty_like(sorted_src)
        src_pos.scatter_(0, order, ranks)

        right = torch.searchsorted(sorted_src, qry.contiguous(), right=False).clamp(1, source_length - 1)
        left = right - 1
        left_t = sorted_src[left]
        right_t = sorted_src[right]
        frac = ((qry - left_t) / (right_t - left_t).clamp_min(1e-6)).clamp(0.0, 1.0)
        qry_pos = (left.float() + frac) * interval

        query_positions.append(qry_pos.to(dtype=dtype))
        source_positions.append(src_pos.to(dtype=dtype))
    return torch.stack(query_positions, dim=0), torch.stack(source_positions, dim=0)


def _time_interpolate_5d(
    x: torch.Tensor,
    target_frames: int,
    source_times: torch.Tensor | None,
    target_times: torch.Tensor | None,
) -> torch.Tensor:
    if source_times is None or target_times is None:
        return _resize_5d(x, (target_frames, x.shape[-2], x.shape[-1]), mode="trilinear")
    batch_size, channels, views, height, width = x.shape
    if views == target_frames:
        source_values = _time_values(source_times, batch_size, views, x.device, x.dtype)
        target_values = _time_values(target_times, batch_size, target_frames, x.device, x.dtype)
        if torch.allclose(source_values, target_values, atol=1e-6, rtol=1e-5):
            return x
    if views <= 1:
        return x.expand(-1, -1, target_frames, -1, -1)

    source_values = _time_values(source_times, batch_size, views, x.device, x.dtype)
    target_values = _time_values(target_times, batch_size, target_frames, x.device, x.dtype)
    outputs = []
    for batch_idx in range(batch_size):
        src = source_values[batch_idx].float()
        tgt = target_values[batch_idx].float()
        order = torch.argsort(src)
        sorted_src = src[order].contiguous()
        sorted_x = x[batch_idx, :, order]

        right = torch.searchsorted(sorted_src, tgt.contiguous(), right=False).clamp(1, views - 1)
        left = right - 1
        left_t = sorted_src[left]
        right_t = sorted_src[right]
        weight = ((tgt - left_t) / (right_t - left_t).clamp_min(1e-6)).clamp(0.0, 1.0)
        left_grid = sorted_x[:, left]
        right_grid = sorted_x[:, right]
        weight = weight[None, :, None, None].to(dtype=x.dtype)
        outputs.append(left_grid * (1 - weight) + right_grid * weight)
    return torch.stack(outputs, dim=0).reshape(batch_size, channels, target_frames, height, width)


def _camera_centers_for_positions(
    poses: torch.Tensor | None,
    batch_size: int,
    length: int,
    device,
    dtype,
) -> torch.Tensor:
    if poses is None:
        return torch.zeros(batch_size, length, 3, device=device, dtype=dtype)
    centers = poses.to(device=device, dtype=dtype)[..., :3, 3]
    if centers.ndim == 2:
        centers = centers[None].expand(batch_size, -1, -1)
    if centers.shape[1] != length:
        centers = _interp_sequence(centers, length)
    centers = centers - centers[:, :1]
    scale = centers.detach().norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1.0)
    return centers / scale[..., None]


def _query_positions_for_grid(
    batch_size: int,
    frames: int,
    height: int,
    width: int,
    device,
    dtype,
    target_times: torch.Tensor | None,
    target_poses: torch.Tensor | None,
    time_scale: torch.Tensor | None,
    source_times: torch.Tensor | None = None,
    source_length: int | None = None,
    time_position_mode: str = "normalized",
    rerope_interval: float = 4.0,
) -> torch.Tensor:
    if time_position_mode == "reroped" and source_times is not None:
        if source_length is None:
            source_length = source_times.shape[-1] if source_times.ndim > 0 else 1
        t, _ = _reroped_time_positions(
            target_times,
            source_times,
            batch_size,
            frames,
            int(source_length),
            device,
            dtype,
            rerope_interval,
        )
    else:
        t = _time_positions(target_times, batch_size, frames, device, dtype, time_scale)
    cam = _camera_centers_for_positions(target_poses, batch_size, frames, device, dtype)
    y = torch.linspace(0, 1, height, device=device, dtype=dtype)
    x = torch.linspace(0, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    t = t[:, :, None, None, None].expand(-1, -1, height, width, -1)
    cam = cam[:, :, None, None, :].expand(-1, -1, height, width, -1)
    yy = yy[None, None, :, :, None].expand(batch_size, frames, -1, -1, -1)
    xx = xx[None, None, :, :, None].expand(batch_size, frames, -1, -1, -1)
    return torch.cat([t, cam, yy, xx], dim=-1).reshape(batch_size, frames * height * width, 6)


def _apply_rope_nd(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    # x: B x heads x seq x dim, positions: B x seq x axes
    axes = positions.shape[-1]
    dim = x.shape[-1]
    axis_width = (dim // max(axes, 1)) // 2 * 2
    if axis_width <= 0:
        return x
    parts = []
    offset = 0
    positions = positions.to(device=x.device, dtype=torch.float32)
    if positions.shape[0] == 1 and x.shape[0] != 1:
        positions = positions.expand(x.shape[0], -1, -1)
    for axis in range(axes):
        if offset + axis_width > dim:
            break
        part = x[..., offset : offset + axis_width]
        inv_freq = torch.exp(
            -math.log(base)
            * torch.arange(0, axis_width, 2, device=x.device, dtype=torch.float32)
            / max(axis_width, 1)
        )
        angles = positions[:, None, :, axis, None] * inv_freq[None, None, None, :]
        sin = torch.sin(angles).to(dtype=x.dtype)
        cos = torch.cos(angles).to(dtype=x.dtype)
        even = part[..., 0::2]
        odd = part[..., 1::2]
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)
        parts.append(rotated)
        offset += axis_width
    if offset < dim:
        parts.append(x[..., offset:])
    return torch.cat(parts, dim=-1)


class TemporalFiLM(torch.nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int = 128):
        super().__init__()
        self.time_dim = int(time_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.time_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, times: torch.Tensor | None) -> torch.Tensor:
        if times is None:
            return x
        emb = sinusoidal_embedding(times.to(device=x.device, dtype=x.dtype), self.time_dim)
        emb = emb.to(dtype=self.net[0].weight.dtype)
        scale, shift = self.net(emb).to(dtype=x.dtype).chunk(2, dim=-1)
        return x * (1 + scale) + shift


@dataclass
class SourceTokenPack:
    tokens: torch.Tensor
    positions: torch.Tensor
    times: torch.Tensor


class MinTTimePosition:
    def __init__(self, mode: str = "reroped", rerope_interval: float = 4.0):
        if mode not in {"normalized", "reroped"}:
            raise ValueError(f"Unsupported time_position_mode: {mode}")
        self.mode = mode
        self.rerope_interval = float(rerope_interval)

    def normalized(
        self,
        times: torch.Tensor | None,
        batch_size: int,
        length: int,
        device,
        dtype,
        time_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        return _time_positions(times, batch_size, length, device, dtype, time_scale)

    def frame_positions(
        self,
        times: torch.Tensor | None,
        batch_size: int,
        length: int,
        device,
        dtype,
        time_scale: torch.Tensor | None,
        *,
        anchor_times: torch.Tensor | None = None,
        anchor_length: int | None = None,
    ) -> torch.Tensor:
        if self.mode == "reroped" and anchor_times is not None:
            if anchor_length is None:
                anchor_length = anchor_times.shape[-1] if anchor_times.ndim > 0 else 1
            positions, _ = _reroped_time_positions(
                times,
                anchor_times,
                batch_size,
                length,
                int(anchor_length),
                device,
                dtype,
                self.rerope_interval,
            )
            return positions
        return self.normalized(times, batch_size, length, device, dtype, time_scale)

    def query_grid_positions(
        self,
        batch_size: int,
        output_grid: tuple[int, int, int],
        device,
        dtype,
        target_times: torch.Tensor | None,
        target_poses: torch.Tensor | None,
        source_times: torch.Tensor | None,
        source_length: int,
        time_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        frames, height, width = output_grid
        return _query_positions_for_grid(
            batch_size,
            frames,
            height,
            width,
            device,
            dtype,
            target_times,
            target_poses,
            time_scale,
            source_times=source_times,
            source_length=source_length,
            time_position_mode=self.mode,
            rerope_interval=self.rerope_interval,
        )

    def query_token_times(
        self,
        target_times: torch.Tensor | None,
        batch_size: int,
        output_grid: tuple[int, int, int],
        device,
        dtype,
        time_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        frames, height, width = output_grid
        values = self.normalized(target_times, batch_size, frames, device, dtype, time_scale)
        return values[:, :, None, None].expand(-1, -1, height, width).reshape(batch_size, -1)

    def source_token_geometry(
        self,
        source_times: torch.Tensor | None,
        source_poses: torch.Tensor | None,
        batch_size: int,
        views: int,
        groups: int,
        token_h: int,
        token_w: int,
        device,
        dtype,
        time_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_times = self.normalized(source_times, batch_size, views, device, dtype, time_scale)
        if self.mode == "reroped" and source_times is not None:
            _, rope_times = _reroped_time_positions(
                source_times,
                source_times,
                batch_size,
                views,
                views,
                device,
                dtype,
                self.rerope_interval,
            )
        else:
            rope_times = token_times

        cam = _camera_centers_for_positions(source_poses, batch_size, views, device, dtype)
        y = torch.linspace(0, 1, token_h, device=device, dtype=dtype)
        x = torch.linspace(0, 1, token_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        rope_times = rope_times[:, :, None, None, None, None].expand(-1, -1, groups, token_h, token_w, -1)
        token_times = token_times[:, :, None, None, None].expand(-1, -1, groups, token_h, token_w)
        cam = cam[:, :, None, None, None, :].expand(-1, -1, groups, token_h, token_w, -1)
        yy = yy[None, None, None, :, :, None].expand(batch_size, views, groups, -1, -1, -1)
        xx = xx[None, None, None, :, :, None].expand(batch_size, views, groups, -1, -1, -1)
        positions = torch.cat([rope_times, cam, yy, xx], dim=-1)
        return (
            positions.reshape(batch_size, views * groups * token_h * token_w, 6),
            token_times.reshape(batch_size, views * groups * token_h * token_w),
        )

    def interpolate_source_grid(
        self,
        x: torch.Tensor,
        target_frames: int,
        source_times: torch.Tensor | None,
        target_times: torch.Tensor | None,
    ) -> torch.Tensor:
        return _time_interpolate_5d(x, target_frames, source_times, target_times)


class SourceTokenEncoder(torch.nn.Module):
    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        *,
        source_pool_hw: tuple[int, int] | None,
        max_source_tokens: int | None,
        max_token_groups: int,
        use_group_embedding: bool,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.source_pool_hw = None if source_pool_hw is None else tuple(int(v) for v in source_pool_hw)
        self.max_source_tokens = None if max_source_tokens is None else int(max_source_tokens)
        self.max_token_groups = int(max_token_groups)
        self.source_proj = torch.nn.Linear(self.token_dim, self.hidden_dim)
        self.source_condition = SourceViewConditionEncoder(self.hidden_dim)
        self.group_embed = (
            torch.nn.Embedding(self.max_token_groups, self.hidden_dim) if bool(use_group_embedding) else None
        )

    @property
    def dtype(self):
        return self.source_proj.weight.dtype

    def _check_shape(self, tokens: torch.Tensor) -> tuple[int, int, int, int, int, int]:
        if tokens.ndim != 6:
            raise ValueError(f"Expected tokens with shape B x N x L x h x w x C, got {tuple(tokens.shape)}")
        bsz, views, groups, token_h, token_w, channels = tokens.shape
        if channels != self.token_dim:
            raise ValueError(f"Adapter token_dim={self.token_dim}, but input token dim is {channels}.")
        if self.group_embed is not None and groups > self.max_token_groups:
            raise ValueError(
                f"Input has {groups} token groups, but adapter max_token_groups={self.max_token_groups}."
            )
        return bsz, views, groups, token_h, token_w, channels

    def _group_embedding(self, groups: int, device, dtype) -> torch.Tensor | None:
        if self.group_embed is None:
            return None
        group_ids = torch.arange(groups, device=device, dtype=torch.long)
        return self.group_embed(group_ids).to(dtype=dtype)

    @staticmethod
    def _source_group_ids(views: int, groups: int, height: int, width: int, device) -> torch.Tensor:
        return (
            torch.arange(groups, device=device, dtype=torch.long)[None, :, None, None]
            .expand(views, -1, height, width)
            .reshape(-1)
        )

    def _maybe_pool(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, views, groups, token_h, token_w, channels = tokens.shape
        if self.source_pool_hw is None or (token_h, token_w) == self.source_pool_hw:
            return tokens
        pooled_h, pooled_w = self.source_pool_hw
        x = tokens.permute(0, 1, 2, 5, 3, 4).reshape(bsz * views * groups, channels, token_h, token_w)
        x = F.interpolate(x.float(), size=(pooled_h, pooled_w), mode="bilinear", align_corners=False)
        x = x.to(tokens.dtype)
        return x.reshape(bsz, views, groups, channels, pooled_h, pooled_w).permute(0, 1, 2, 4, 5, 3)

    def _source_condition_grid(
        self,
        *,
        batch_size: int,
        views: int,
        height: int,
        width: int,
        device,
        dtype,
        source_times: torch.Tensor | None,
        source_poses: torch.Tensor | None,
        source_intrs: torch.Tensor | None,
        time_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.source_condition(
            batch_size=batch_size,
            views=views,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
            source_times=source_times,
            source_poses=source_poses,
            source_intrs=source_intrs,
            time_scale=time_scale,
        )

    def local_view_grid(
        self,
        tokens: torch.Tensor,
        *,
        source_times: torch.Tensor | None,
        source_poses: torch.Tensor | None,
        source_intrs: torch.Tensor | None,
        time_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, views, groups, token_h, token_w, channels = self._check_shape(tokens)
        projected = self.source_proj(tokens.reshape(bsz * views * groups * token_h * token_w, channels))
        projected = projected.reshape(bsz, views, groups, token_h, token_w, self.hidden_dim)
        group_emb = self._group_embedding(groups, tokens.device, projected.dtype)
        if group_emb is not None:
            projected = projected + group_emb[None, None, :, None, None, :]
        x = projected.mean(dim=2).permute(0, 4, 1, 2, 3).contiguous()
        return x + self._source_condition_grid(
            batch_size=bsz,
            views=views,
            height=token_h,
            width=token_w,
            device=tokens.device,
            dtype=tokens.dtype,
            source_times=source_times,
            source_poses=source_poses,
            source_intrs=source_intrs,
            time_scale=time_scale,
        )

    def attention_pack(
        self,
        tokens: torch.Tensor,
        *,
        time_position: MinTTimePosition,
        source_times: torch.Tensor | None,
        source_poses: torch.Tensor | None,
        source_intrs: torch.Tensor | None,
        time_scale: torch.Tensor | None,
    ) -> SourceTokenPack:
        bsz, views, groups, _, _, channels = self._check_shape(tokens)
        tokens = self._maybe_pool(tokens)
        _, _, groups, token_h, token_w, _ = tokens.shape

        source = self.source_proj(tokens.reshape(bsz * views * groups * token_h * token_w, channels))
        source = source.reshape(bsz, views * groups * token_h * token_w, self.hidden_dim)
        group_emb = self._group_embedding(groups, tokens.device, source.dtype)
        group_ids = self._source_group_ids(views, groups, token_h, token_w, tokens.device)
        if group_emb is not None:
            source = source + group_emb[group_ids][None]

        source_cond = self._source_condition_grid(
            batch_size=bsz,
            views=views,
            height=1,
            width=1,
            device=tokens.device,
            dtype=tokens.dtype,
            source_times=source_times,
            source_poses=source_poses,
            source_intrs=source_intrs,
            time_scale=time_scale,
        ).squeeze(-1).squeeze(-1).transpose(1, 2)
        source_cond = source_cond[:, :, None, None, None, :].expand(-1, -1, groups, token_h, token_w, -1)
        source = source + source_cond.reshape(bsz, views * groups * token_h * token_w, -1)

        positions, token_times = time_position.source_token_geometry(
            source_times,
            source_poses,
            bsz,
            views,
            groups,
            token_h,
            token_w,
            tokens.device,
            tokens.dtype,
            time_scale,
        )
        if self.max_source_tokens is not None and source.shape[1] > self.max_source_tokens:
            idx = torch.linspace(0, source.shape[1] - 1, self.max_source_tokens, device=tokens.device).long()
            source = source[:, idx]
            positions = positions[:, idx]
            token_times = token_times[:, idx]
        return SourceTokenPack(tokens=source, positions=positions, times=token_times)


class LocalGridShortcut(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.proj = torch.nn.Sequential(
            torch.nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
            torch.nn.SiLU(),
            CausalResBlock(hidden_dim),
        )

    def forward(
        self,
        source_grid: torch.Tensor,
        output_grid: tuple[int, int, int],
        *,
        time_position: MinTTimePosition,
        source_times: torch.Tensor | None,
        target_times: torch.Tensor | None,
    ) -> torch.Tensor:
        frames, height, width = output_grid
        views = source_grid.shape[2]
        if source_grid.shape[-2:] != (height, width):
            source_grid = _resize_5d(source_grid, (views, height, width), mode="trilinear")
        source_grid = time_position.interpolate_source_grid(source_grid, frames, source_times, target_times)
        x = self.proj(source_grid)
        return self.scale.to(dtype=x.dtype) * x.flatten(2).transpose(1, 2)


class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, use_rope: bool = True):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rope = use_rope
        self.norm_q = torch.nn.LayerNorm(hidden_dim)
        self.norm_kv = torch.nn.LayerNorm(hidden_dim)
        self.to_q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.to_k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.to_v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, mlp_hidden),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(mlp_hidden, hidden_dim),
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        source_pos: torch.Tensor | None = None,
        query_chunk_size: int | None = None,
    ) -> torch.Tensor:
        q = self._shape(self.to_q(self.norm_q(query)))
        k = self._shape(self.to_k(self.norm_kv(source)))
        v = self._shape(self.to_v(self.norm_kv(source)))
        if self.use_rope and query_pos is not None and source_pos is not None:
            q = _apply_rope_nd(q, query_pos)
            k = _apply_rope_nd(k, source_pos)
        if query_chunk_size is None or query.shape[1] <= query_chunk_size:
            attn = F.scaled_dot_product_attention(q, k, v)
        else:
            chunks = []
            for start in range(0, q.shape[2], query_chunk_size):
                chunks.append(F.scaled_dot_product_attention(q[:, :, start : start + query_chunk_size], k, v))
            attn = torch.cat(chunks, dim=2)
        attn = attn.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.hidden_dim)
        query = query + self.out(attn)
        query = query + self.ffn(query)
        return query


class CrossAttentionRoPEAdapter(torch.nn.Module):
    def __init__(
        self,
        token_dim: int,
        output_dim: int,
        control_layers: Iterable[int],
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 4,
        source_pool_hw: tuple[int, int] | None = (16, 16),
        max_source_tokens: int | None = 32768,
        query_chunk_size: int | None = 4096,
        text_context_dim: int | None = None,
        use_rope: bool = True,
        use_dit_state: bool = True,
        max_token_groups: int = 8,
        use_group_embedding: bool = True,
        use_local_grid: bool = True,
        use_time_film: bool = True,
        time_film_dim: int = 128,
        time_position_mode: str = "reroped",
        rerope_interval: float = 4.0,
        post_num_res_blocks: int = 2,
        output_mode: str = "condition_embedding",
    ):
        super().__init__()
        if output_mode not in {"condition_embedding", "hints"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}")
        self.output_mode = output_mode
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.control_layers = tuple(int(layer) for layer in control_layers)
        self.query_chunk_size = None if query_chunk_size is None else int(query_chunk_size)
        self.time_position = MinTTimePosition(mode=time_position_mode, rerope_interval=rerope_interval)
        self.condition = ConditionGridEncoder(hidden_dim, text_context_dim=text_context_dim)
        self.source_encoder = SourceTokenEncoder(
            token_dim,
            hidden_dim,
            source_pool_hw=source_pool_hw,
            max_source_tokens=max_source_tokens,
            max_token_groups=max_token_groups,
            use_group_embedding=use_group_embedding,
        )
        self.local_grid = LocalGridShortcut(hidden_dim) if bool(use_local_grid) else None
        self.query_time_film = TemporalFiLM(hidden_dim, time_film_dim) if bool(use_time_film) else None
        self.source_time_film = TemporalFiLM(hidden_dim, time_film_dim) if bool(use_time_film) else None
        self.dit_state_proj = torch.nn.Linear(self.output_dim, hidden_dim) if use_dit_state else None
        self.blocks = torch.nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads=num_heads, use_rope=use_rope) for _ in range(num_blocks)]
        )
        post_blocks = [torch.nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1), torch.nn.SiLU()]
        post_blocks.extend(CausalResBlock(hidden_dim) for _ in range(max(0, int(post_num_res_blocks))))
        self.post = torch.nn.Sequential(*post_blocks)
        self.condition_head = torch.nn.Conv3d(hidden_dim, self.output_dim, kernel_size=1)
        self.heads = (
            torch.nn.ModuleDict(
                {_control_key(layer): torch.nn.Conv3d(hidden_dim, self.output_dim, kernel_size=1) for layer in self.control_layers}
            )
            if self.output_mode == "hints"
            else torch.nn.ModuleDict()
        )

    def forward(
        self,
        tokens: torch.Tensor,
        output_grid: tuple[int, int, int],
        *,
        target_times: torch.Tensor | None = None,
        target_poses: torch.Tensor | None = None,
        target_intrs: torch.Tensor | None = None,
        target_plucker: torch.Tensor | None = None,
        source_times: torch.Tensor | None = None,
        source_poses: torch.Tensor | None = None,
        source_intrs: torch.Tensor | None = None,
        dit_tokens: torch.Tensor | None = None,
        diffusion_timestep: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
    ) -> OrderedDict[str, torch.Tensor]:
        tokens = tokens.to(dtype=self.source_encoder.dtype)
        bsz, views = tokens.shape[:2]
        device, dtype = tokens.device, tokens.dtype
        frames, height, width = output_grid
        time_scale = _shared_time_scale(bsz, device, dtype, target_times, source_times)
        cond = self.condition(
            output_grid,
            batch_size=bsz,
            device=device,
            dtype=dtype,
            target_times=target_times,
            target_poses=target_poses,
            target_intrs=target_intrs,
            target_plucker=target_plucker,
            diffusion_timestep=diffusion_timestep,
            text_context=text_context,
            time_scale=time_scale,
        )
        query = cond.flatten(2).transpose(1, 2)
        if self.dit_state_proj is not None and dit_tokens is not None:
            query = query + self.dit_state_proj(dit_tokens.to(dtype=query.dtype))
        if self.local_grid is not None:
            source_grid = self.source_encoder.local_view_grid(
                tokens.to(dtype=query.dtype),
                source_times=source_times,
                source_poses=source_poses,
                source_intrs=source_intrs,
                time_scale=time_scale,
            )
            query = query + self.local_grid(
                source_grid,
                output_grid,
                time_position=self.time_position,
                source_times=source_times,
                target_times=target_times,
            )
        query_time = self.time_position.query_token_times(
            target_times,
            bsz,
            output_grid,
            device,
            query.dtype,
            time_scale,
        )
        if self.query_time_film is not None:
            query = self.query_time_film(query, query_time)

        source_pack = self.source_encoder.attention_pack(
            tokens.to(dtype=query.dtype),
            time_position=self.time_position,
            source_times=source_times,
            source_poses=source_poses,
            source_intrs=source_intrs,
            time_scale=time_scale,
        )
        source = source_pack.tokens
        if self.source_time_film is not None:
            source = self.source_time_film(source, source_pack.times)
        query_pos = self.time_position.query_grid_positions(
            bsz,
            output_grid,
            device,
            query.dtype,
            target_times,
            target_poses,
            source_times,
            views,
            time_scale,
        )
        for block in self.blocks:
            query = block(query, source, query_pos, source_pack.positions, query_chunk_size=self.query_chunk_size)

        x = query.transpose(1, 2).reshape(bsz, -1, frames, height, width)
        x = self.post(x)
        if self.output_mode == "condition_embedding":
            return self.condition_head(x).flatten(2).transpose(1, 2).contiguous()

        out: OrderedDict[str, torch.Tensor] = OrderedDict()
        for layer in self.control_layers:
            hint = self.heads[_control_key(layer)](x)
            out[_control_key(layer)] = hint.flatten(2).transpose(1, 2).contiguous()
        return out


def build_student_adapter(adapter_type: str, **kwargs) -> torch.nn.Module:
    if adapter_type == "conv":
        return ConvAdapter(**kwargs)
    if adapter_type in {"cross_attention_rope", "cross_attn_rope"}:
        return CrossAttentionRoPEAdapter(**kwargs)
    raise ValueError(f"Unknown adapter_type: {adapter_type}")
