import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_layer_weight(weights, layer_key: str) -> float:
    if weights is None:
        return 1.0
    if layer_key in weights:
        return float(weights[layer_key])
    raw = layer_key.replace("layer_", "")
    return float(weights.get(raw, 1.0))


def latent_stats(x: torch.Tensor):
    xf = x.detach().float()
    return {
        "mean": xf.mean(),
        "std": xf.std(unbiased=False),
        "min": xf.amin(),
        "max": xf.amax(),
    }


def compute_distill_loss(student, teacher, cfg):
    if isinstance(student, torch.Tensor):
        student = OrderedDict(condition=student)
    if isinstance(teacher, torch.Tensor):
        teacher = OrderedDict(condition=teacher)
    total = None
    metrics = {}
    for key, teacher_lat in teacher.items():
        student_lat = student[key]
        teacher_lat = teacher_lat.detach()
        diff = student_lat.float() - teacher_lat.float()
        l1 = diff.abs().mean()
        l2 = diff.square().mean()
        cosine = F.cosine_similarity(student_lat.float().flatten(1), teacher_lat.float().flatten(1), dim=1).mean()
        stat = (
            (student_lat.float().mean() - teacher_lat.float().mean()).abs()
            + (student_lat.float().std(unbiased=False) - teacher_lat.float().std(unbiased=False)).abs()
        )
        layer_loss = (
            float(cfg.loss.l1_weight) * l1
            + float(cfg.loss.l2_weight) * l2
            + float(cfg.loss.cosine_weight) * (1 - cosine)
            + float(cfg.loss.stat_weight) * stat
        )
        layer_loss = layer_loss * get_layer_weight(cfg.loss.layer_weights, key)
        total = layer_loss if total is None else total + layer_loss
        metrics[f"{key}/l1"] = l1.detach()
        metrics[f"{key}/l2"] = l2.detach()
        metrics[f"{key}/cosine"] = cosine.detach()
        metrics[f"{key}/stat"] = stat.detach()
        for name, value in latent_stats(teacher_lat).items():
            metrics[f"{key}/teacher_{name}"] = value
        for name, value in latent_stats(student_lat).items():
            metrics[f"{key}/student_{name}"] = value
    metrics["loss"] = total.detach()
    return total, metrics


def normalize_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float()
    x = x - x.amin()
    x = x / x.amax().clamp_min(1e-6)
    return (x.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def save_heatmaps(output_dir: str, step: int, teacher, student, output_grid):
    os.makedirs(output_dir, exist_ok=True)
    key = next(iter(teacher.keys()))
    frames, height, width = output_grid
    teacher_map = teacher[key][0].float().reshape(frames, height, width, -1).mean(dim=-1)
    student_map = student[key][0].float().reshape(frames, height, width, -1).mean(dim=-1)
    error_map = (student_map - teacher_map).abs()
    frame = frames // 2
    for name, tensor in {
        "teacher": teacher_map[frame],
        "student": student_map[frame],
        "abs_error": error_map[frame],
    }.items():
        Image.fromarray(normalize_image(tensor)).save(os.path.join(output_dir, f"step_{step:08d}_{key}_{name}.png"))

