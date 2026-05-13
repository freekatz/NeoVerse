from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Iterable, Literal

import numpy as np
import torch


IncludeCamera = bool | Literal["auto"]


def compose_views_from_list(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Match the NeoVerse pipeline's list-of-views collation convention."""
    batched: dict[str, Any] = {}
    for key in batch[0].keys():
        value = batch[0][key]
        if isinstance(value, torch.Tensor):
            batched[key] = torch.stack([item[key] for item in batch], dim=1)
        elif isinstance(value, np.ndarray):
            batched[key] = np.stack([item[key] for item in batch], axis=1)
        elif isinstance(value, (int, float, str, bool, list)):
            batched[key] = [item[key] for item in batch]
    return batched


def filter_context_views(views: dict[str, Any]) -> dict[str, Any]:
    """Keep only views where `is_target` is false when that mask exists."""
    if "is_target" not in views or not isinstance(views["is_target"], torch.Tensor):
        return views
    keep = ~views["is_target"].bool()
    if keep.ndim == 2:
        # The existing SpatialVID loader uses the same context count per batch.
        count = int(keep[0].sum().item())
        return {
            key: value[:, :count] if isinstance(value, torch.Tensor) and value.ndim >= 2 else value
            for key, value in views.items()
        }
    return views


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _reshape_patch_tokens(tokens: torch.Tensor, height: int, width: int, patch_size: int) -> torch.Tensor:
    bsz, views, patches, channels = tokens.shape
    token_h, token_w = height // patch_size, width // patch_size
    expected = token_h * token_w
    if patches != expected:
        raise ValueError(
            f"Patch-token count mismatch: got {patches}, expected {expected} "
            f"from image {height}x{width} and patch size {patch_size}."
        )
    return tokens.reshape(bsz, views, token_h, token_w, channels)


@torch.no_grad()
def _extract_from_depth_anything3(
    reconstructor: torch.nn.Module,
    views: dict[str, Any],
    layer_indices: Iterable[int],
    include_camera_token: IncludeCamera,
    ref_view_strategy: str,
) -> dict[str, Any]:
    imgs = views["img"]
    bsz, num_views, _, height, width = imgs.shape
    mean = imgs.new_tensor(reconstructor.IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = imgs.new_tensor(reconstructor.IMAGENET_STD).view(1, 1, 3, 1, 1)
    imgs_normalized = (imgs - mean) / std

    da3_net = reconstructor.da3.model
    if hasattr(da3_net, "da3") and hasattr(da3_net.da3, "backbone"):
        da3_net = da3_net.da3

    cam_token = None
    if (
        getattr(da3_net, "cam_enc", None) is not None
        and "extrinsics" in views
        and "intrinsics" in views
    ):
        cam_token = da3_net.cam_enc(views["extrinsics"], views["intrinsics"], imgs.shape[-2:])

    backbone = da3_net.backbone
    outputs, _ = backbone.get_intermediate_layers(
        imgs_normalized,
        n=tuple(layer_indices),
        export_feat_layers=[],
        cam_token=cam_token,
        ref_view_strategy=ref_view_strategy,
    )
    patch_size = getattr(backbone, "patch_size", reconstructor.PATCH_SIZE)
    patch_groups = []
    camera_tokens = []
    for layer_output, camera_token in outputs:
        patch_groups.append(_reshape_patch_tokens(layer_output, height, width, patch_size))
        camera_tokens.append(camera_token)

    tokens = torch.stack(patch_groups, dim=2)
    backbone_has_camera_token = getattr(backbone, "alt_start", -1) != -1
    has_camera = include_camera_token is True or (
        include_camera_token == "auto"
        and backbone_has_camera_token
        and len(camera_tokens) > 0
        and camera_tokens[-1].shape[-1] == tokens.shape[-1]
    )
    if has_camera:
        token_h, token_w = tokens.shape[-3:-1]
        camera_grid = camera_tokens[-1][:, :, None, None, :].expand(bsz, num_views, token_h, token_w, -1)
        tokens = torch.cat([tokens, camera_grid[:, :, None]], dim=2)

    return {
        "tokens": tokens,
        "layers": tuple(layer_indices),
        "has_camera_token": bool(has_camera),
        "token_hw": tuple(tokens.shape[-3:-1]),
        "token_dim": int(tokens.shape[-1]),
        "num_token_groups": int(tokens.shape[2]),
        "backend": "depth_anything_3",
    }


@torch.no_grad()
def pack_worldmirror_token_list(
    token_list: list[torch.Tensor],
    patch_start_idx: int,
    height: int,
    width: int,
    patch_size: int,
    layer_indices: Iterable[int],
    include_camera_token: IncludeCamera,
) -> dict[str, Any]:
    patch_groups = []
    camera_tokens = []
    for token_tensor in token_list:
        camera_tokens.append(token_tensor[:, :, 0])
        patch_tokens = token_tensor[:, :, patch_start_idx:]
        patch_groups.append(_reshape_patch_tokens(patch_tokens, height, width, patch_size))

    tokens = torch.stack(patch_groups, dim=2)
    has_camera = include_camera_token is True or (
        include_camera_token == "auto" and len(camera_tokens) > 0 and camera_tokens[-1].shape[-1] == tokens.shape[-1]
    )
    if has_camera:
        token_h, token_w = tokens.shape[-3:-1]
        camera_grid = camera_tokens[-1][:, :, None, None, :].expand(
            tokens.shape[0], tokens.shape[1], token_h, token_w, -1
        )
        tokens = torch.cat([tokens, camera_grid[:, :, None]], dim=2)

    return {
        "tokens": tokens,
        "layers": tuple(layer_indices),
        "has_camera_token": bool(has_camera),
        "token_hw": tuple(tokens.shape[-3:-1]),
        "token_dim": int(tokens.shape[-1]),
        "num_token_groups": int(tokens.shape[2]),
        "backend": "worldmirror",
    }


def _extract_from_worldmirror(
    reconstructor: torch.nn.Module,
    views: dict[str, Any],
    layer_indices: Iterable[int],
    include_camera_token: IncludeCamera,
) -> dict[str, Any]:
    imgs = views["img"]
    _, _, _, height, width = imgs.shape
    vgt = reconstructor.visual_geometry_transformer
    old_indices = list(vgt.intermediate_idxs)
    try:
        vgt.intermediate_idxs = list(layer_indices)
        token_list, patch_start_idx, _, _ = vgt(imgs, use_motion=False)
    finally:
        vgt.intermediate_idxs = old_indices

    patch_size = getattr(vgt, "patch_size", 14)
    return pack_worldmirror_token_list(
        token_list,
        patch_start_idx=patch_start_idx,
        height=height,
        width=width,
        patch_size=patch_size,
        layer_indices=layer_indices,
        include_camera_token=include_camera_token,
    )


def extract_vggt_tokens(
    reconstructor: torch.nn.Module,
    source_views: dict[str, Any] | list[dict[str, Any]],
    layer_indices: Iterable[int] = (4, 11, 17, 23),
    include_camera_token: IncludeCamera = "auto",
    context_only: bool = False,
    save_path: str | None = None,
    ref_view_strategy: str = "saddle_balanced",
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, Any]:
    """
    Extract Gen3R-style intermediate reconstruction tokens.

    Returns `tokens` in shape `B x N x L x h_v x w_v x C`, where `L` is 4 for
    layers `[4, 11, 17, 23]` and 5 when a camera token is broadcast and appended.
    """
    views = compose_views_from_list(source_views) if isinstance(source_views, list) else source_views
    views = filter_context_views(views) if context_only else views
    layer_indices = tuple(int(idx) for idx in layer_indices)
    if "img" not in views:
        raise KeyError("source_views must contain an `img` tensor with shape B x N x 3 x H x W.")

    device = views["img"].device
    with _autocast_context(device, autocast_dtype):
        if hasattr(reconstructor, "da3"):
            result = _extract_from_depth_anything3(
                reconstructor,
                views,
                layer_indices=layer_indices,
                include_camera_token=include_camera_token,
                ref_view_strategy=ref_view_strategy,
            )
        elif hasattr(reconstructor, "visual_geometry_transformer"):
            result = _extract_from_worldmirror(
                reconstructor,
                views,
                layer_indices=layer_indices,
                include_camera_token=include_camera_token,
            )
        else:
            raise TypeError(
                "Unsupported reconstructor type. Expected DepthAnything3Reconstructor "
                "or WorldMirror-like object with `visual_geometry_transformer`."
            )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(
            {
                "tokens": result["tokens"].detach().cpu(),
                "layers": result["layers"],
                "has_camera_token": result["has_camera_token"],
                "token_hw": result["token_hw"],
                "token_dim": result["token_dim"],
                "num_token_groups": result["num_token_groups"],
                "backend": result["backend"],
            },
            save_path,
        )
    return result
