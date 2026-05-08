import argparse
import csv
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def read_video(path, size=None, max_frames=None):
    frames = []
    reader = imageio.get_reader(path)
    try:
        for idx, frame in enumerate(reader):
            if max_frames is not None and idx >= max_frames:
                break
            image = Image.fromarray(frame).convert("RGB")
            if size is not None and image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            arr = np.asarray(image, dtype=np.float32) / 255.0
            frames.append(arr)
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def align_pair(reference, predicted):
    length = min(len(reference), len(predicted))
    return reference[:length], predicted[:length]


def psnr(reference, predicted, mask=None):
    diff = (reference - predicted) ** 2
    if mask is not None:
        denom = np.maximum(mask.sum() * reference.shape[-1], 1.0)
        mse = float((diff * mask[..., None]).sum() / denom)
    else:
        mse = float(diff.mean())
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def l1(reference, predicted, mask=None):
    diff = np.abs(reference - predicted)
    if mask is not None:
        denom = np.maximum(mask.sum() * reference.shape[-1], 1.0)
        return float((diff * mask[..., None]).sum() / denom)
    return float(diff.mean())


def motion_mask(reference, threshold=0.06):
    if len(reference) <= 1:
        return np.ones(reference.shape[:3], dtype=np.float32)
    gray = reference.mean(axis=-1)
    delta = np.zeros_like(gray)
    delta[1:] = np.abs(gray[1:] - gray[:-1])
    mask = (delta > threshold).astype(np.float32)
    if mask.mean() < 0.01:
        cutoff = np.quantile(delta.reshape(-1), 0.90)
        mask = (delta >= max(cutoff, 1e-6)).astype(np.float32)
    return mask


def tensor_video(frames, device):
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    return tensor


@torch.no_grad()
def ssim_metric(reference, predicted, device):
    ref = tensor_video(reference, device)
    pred = tensor_video(predicted, device)
    vals = []
    for start in range(0, len(ref), 16):
        vals.append(structural_similarity_index_measure(pred[start : start + 16], ref[start : start + 16], data_range=1.0))
    return float(torch.stack(vals).mean().detach().cpu())


@torch.no_grad()
def lpips_metric(reference, predicted, metric, device):
    ref = tensor_video(reference, device) * 2 - 1
    pred = tensor_video(predicted, device) * 2 - 1
    vals = []
    for start in range(0, len(ref), 8):
        vals.append(metric(pred[start : start + 8], ref[start : start + 8]))
    return float(torch.stack(vals).mean().detach().cpu())


def safe_float(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe_float(row.get(key, "")) for key in keys})


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["group"], row["method"])
        groups.setdefault(key, []).append(row)
    out = []
    for (group, method), items in sorted(groups.items()):
        summary = {"group": group, "method": method, "n": len(items)}
        for metric in ("psnr", "ssim", "lpips", "l1", "motion_psnr", "motion_l1"):
            vals = [float(item[metric]) for item in items if item.get(metric) not in ("", None) and math.isfinite(float(item[metric]))]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = math.sqrt(var)
        out.append(summary)
    return out


def write_report(output_root, rows, summary):
    best_by_group = {}
    for group in sorted({row["group"] for row in rows}):
        candidates = [row for row in rows if row["group"] == group and row.get("method") in {"student", "teacher", "zero_control", "teacher_scale_0_5"}]
        candidates = [row for row in candidates if row.get("psnr") not in ("", None)]
        if candidates:
            best_by_group[group] = max(candidates, key=lambda row: float(row["psnr"]))
    lines = [
        "# Midterm Generation Experiments",
        "",
        "## What Was Measured",
        "",
        "- Full-frame PSNR / SSIM / LPIPS / L1 against `gt_81_sorted.mp4`.",
        "- Motion-region PSNR / L1 using an automatic GT temporal-difference mask.",
        "- Rendered degraded RGB is included as a non-generative reference.",
        "",
        "## Group Summary",
        "",
        "| group | method | n | PSNR | SSIM | LPIPS | motion PSNR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {group} | {method} | {n} | {psnr:.3f} | {ssim:.4f} | {lpips:.4f} | {mpsnr:.3f} |".format(
                group=row["group"],
                method=row["method"],
                n=row["n"],
                psnr=row.get("psnr_mean", float("nan")),
                ssim=row.get("ssim_mean", float("nan")),
                lpips=row.get("lpips_mean", float("nan")),
                mpsnr=row.get("motion_psnr_mean", float("nan")),
            )
        )
    lines.extend(["", "## Key Takeaways", ""])
    for group, row in best_by_group.items():
        lines.append(
            f"- `{group}` best PSNR: `{row['method']}` in `{row['run_id']}` with PSNR={float(row['psnr']):.3f}, SSIM={float(row['ssim']):.4f}, LPIPS={float(row['lpips']):.4f}."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics_by_video.csv`: per-run, per-method metrics.",
            "- `metrics_summary.csv`: grouped averages.",
            "- each run directory contains GT/render/teacher/student videos and comparison grids where available.",
        ]
    )
    (Path(output_root) / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--motion_threshold", type=float, default=0.06)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device).eval()

    rows = []
    for run in manifest["runs"]:
        run_dir = Path(run["output_dir"])
        if not run_dir.exists():
            continue
        gt_path = run_dir / "gt_81_sorted.mp4"
        if not gt_path.exists():
            continue
        gt = read_video(gt_path, max_frames=args.max_frames)
        size = (gt.shape[2], gt.shape[1])
        mask = motion_mask(gt, threshold=args.motion_threshold)
        method_files = {
            "render": "rendered_degraded_rgb.mp4",
            "teacher": "teacher.mp4",
            "student": "student.mp4",
        }
        alias = run.get("alias", {})
        for method, filename in method_files.items():
            path = run_dir / filename
            if not path.exists():
                continue
            effective_method = alias.get(method, method)
            pred = read_video(path, size=size, max_frames=args.max_frames)
            ref, pred = align_pair(gt, pred)
            mask_aligned = mask[: len(ref)]
            row = {
                "run_id": run["id"],
                "group": run["group"],
                "method": effective_method,
                "dataset_index": run["dataset_index"],
                "control_scale": run["control_scale"],
                "config": run["config"],
                "video": str(path),
                "frames": len(ref),
                "psnr": psnr(ref, pred),
                "ssim": ssim_metric(ref, pred, device),
                "lpips": lpips_metric(ref, pred, lpips_model, device),
                "l1": l1(ref, pred),
                "motion_pixels_ratio": float(mask_aligned.mean()),
                "motion_psnr": psnr(ref, pred, mask=mask_aligned),
                "motion_l1": l1(ref, pred, mask=mask_aligned),
            }
            rows.append(row)
            print(f"[metrics] {run['id']} {effective_method} PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f}")
    summary = summarize(rows)
    write_csv(output_root / "metrics_by_video.csv", rows)
    write_csv(output_root / "metrics_summary.csv", summary)
    with open(output_root / "metrics_by_video.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    write_report(output_root, rows, summary)
    print(f"[complete] wrote metrics to {output_root}")


if __name__ == "__main__":
    main()
