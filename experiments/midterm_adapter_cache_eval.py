import argparse
import csv
import importlib.util
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

STUDENT_ADAPTERS_PATH = CODE_DIR / "diffsynth" / "models" / "student_adapters.py"
_spec = importlib.util.spec_from_file_location("midterm_student_adapters", STUDENT_ADAPTERS_PATH)
_student_adapters = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_student_adapters)
build_student_adapter = _student_adapters.build_student_adapter


CONTROL_LAYERS_14B = (0, 5, 10, 15, 20, 25, 30, 35)


def load_jsonable_config(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    return dict(config)


def build_adapter_from_checkpoint_config(config, sample_cache, device, dtype):
    config = load_jsonable_config(config)
    adapter_cfg = dict(config.get("adapter", {}))
    adapter_type = adapter_cfg.pop("type")
    use_text_context = bool(adapter_cfg.pop("use_text_context", False))

    token_dim = int(config.get("token", {}).get("token_dim", sample_cache.get("token_dim", 2048)))
    token_groups = int(config.get("token", {}).get("token_groups", sample_cache.get("num_token_groups", 5)))
    output_dim = int(config.get("control_dim", sample_cache["teacher"].shape[-1]))

    common = {
        "output_dim": output_dim,
        "control_layers": CONTROL_LAYERS_14B,
        "text_context_dim": output_dim if use_text_context else None,
    }
    if adapter_type == "conv":
        if adapter_cfg.get("input_channels") is None:
            adapter_cfg["input_channels"] = token_dim * token_groups
        adapter = build_student_adapter(adapter_type, **common, **adapter_cfg)
    elif adapter_type in {"cross_attention_rope", "cross_attn_rope"}:
        for conv_only_key in ("input_channels", "num_res_blocks", "dropout", "condition_time_dim"):
            adapter_cfg.pop(conv_only_key, None)
        if adapter_cfg.get("token_dim") is None:
            adapter_cfg["token_dim"] = token_dim
        if adapter_cfg.get("max_token_groups") is None:
            adapter_cfg["max_token_groups"] = max(token_groups, 8)
        adapter = build_student_adapter(adapter_type, **common, **adapter_cfg)
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
    return adapter.to(device=device, dtype=dtype)


def tensor_to_device(value, device, dtype):
    if not isinstance(value, torch.Tensor):
        return value
    if torch.is_floating_point(value):
        return value.to(device=device, dtype=dtype, non_blocking=True)
    return value.to(device=device, non_blocking=True)


def cache_to_device(cache, device, dtype):
    return {key: tensor_to_device(value, device, dtype) for key, value in cache.items()}


def distill_metrics(student, teacher):
    student_f = student.float()
    teacher_f = teacher.float()
    diff = student_f - teacher_f
    l1 = diff.abs().mean()
    l2 = diff.square().mean()
    cosine = torch.nn.functional.cosine_similarity(student_f.flatten(1), teacher_f.flatten(1), dim=1).mean()
    teacher_abs = teacher_f.abs().mean()
    student_abs = student_f.abs().mean()
    mean_gap = (student_f.mean() - teacher_f.mean()).abs()
    std_gap = (student_f.std(unbiased=False) - teacher_f.std(unbiased=False)).abs()
    return {
        "l1": float(l1.detach().cpu()),
        "l2": float(l2.detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
        "teacher_abs_mean": float(teacher_abs.detach().cpu()),
        "student_abs_mean": float(student_abs.detach().cpu()),
        "mean_gap": float(mean_gap.detach().cpu()),
        "std_gap": float(std_gap.detach().cpu()),
    }


def summarize_metrics(rows):
    keys = ["l1", "l2", "cosine", "teacher_abs_mean", "student_abs_mean", "mean_gap", "std_gap", "seconds"]
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and row[key] is not None and math.isfinite(float(row[key]))]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = math.sqrt(var)
    return out


def count_parameters(module):
    return sum(param.numel() for param in module.parameters())


def read_event_final_loss(run_dir):
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))
    if not event_files:
        return None, None, None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None, None, None
    best = None
    for event_file in event_files:
        try:
            ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
            ea.Reload()
            if "loss" not in ea.Tags().get("scalars", []):
                continue
            vals = ea.Scalars("loss")
            if not vals:
                continue
            candidate = vals[-1]
            if best is None or candidate.step > best.step:
                best = candidate
        except Exception:
            continue
    if best is None:
        return None, None, None
    return int(best.step), float(best.value), len(vals)


def read_run_config(run_dir):
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def checkpoint_metadata(path):
    run_dir = os.path.dirname(path)
    config = load_jsonable_config(read_run_config(run_dir))
    adapter_cfg = config.get("adapter", {})
    return {
        "path": path,
        "run_dir": run_dir,
        "step": int(config.get("max_steps", 0) or 0),
        "adapter_type": str(adapter_cfg.get("type", "unknown")),
        "hidden_dim": adapter_cfg.get("hidden_dim"),
        "num_blocks": adapter_cfg.get("num_blocks", adapter_cfg.get("num_res_blocks")),
        "source_pool_hw": adapter_cfg.get("source_pool_hw"),
        "use_rope": adapter_cfg.get("use_rope"),
        "event_step": None,
        "event_final_loss": None,
        "event_count": None,
        "config": config,
    }


def select_diverse_checkpoints(paths, max_checkpoints):
    metas = []
    for path in paths:
        try:
            metas.append(checkpoint_metadata(path))
        except Exception as exc:
            print(f"[warn] skip {path}: {exc}", flush=True)
    groups = defaultdict(list)
    for meta in metas:
        groups[meta["adapter_type"]].append(meta)
    for group in groups.values():
        group.sort(key=lambda m: (m["step"], m.get("event_final_loss") is not None), reverse=True)

    selected = []
    per_type = max(1, max_checkpoints // max(1, len(groups)))
    for adapter_type in sorted(groups):
        selected.extend(groups[adapter_type][:per_type])
    remaining = [m for m in metas if m not in selected]
    remaining.sort(key=lambda m: m["step"], reverse=True)
    selected.extend(remaining[: max(0, max_checkpoints - len(selected))])
    selected = selected[:max_checkpoints]
    selected.sort(key=lambda m: (m["adapter_type"], -m["step"], m["path"]))
    return selected


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary_rows, eval_paths):
    ordered = sorted(summary_rows, key=lambda row: row.get("l1_mean", float("inf")))
    lines = [
        "# Midterm Adapter Cache Evaluation",
        "",
        "Common evaluation set:",
    ]
    for idx, eval_path in enumerate(eval_paths, start=1):
        lines.append(f"- E{idx:02d}: `{os.path.basename(eval_path)}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| rank | experiment | type | step | params_M | L1 mean | cosine mean | seconds/sample | event final loss |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            "| {rank} | `{name}` | {typ} | {step} | {params:.2f} | {l1:.6f} | {cos:.6f} | {sec:.3f} | {event} |".format(
                rank=rank,
                name=row["experiment_id"],
                typ=row["adapter_type"],
                step=row["step"],
                params=row.get("params_m", 0.0),
                l1=row.get("l1_mean", float("nan")),
                cos=row.get("cosine_mean", float("nan")),
                sec=row.get("seconds_mean", float("nan")),
                event=(
                    "NA"
                    if row.get("event_final_loss") is None
                    else f"{float(row['event_final_loss']):.6f}"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This evaluates the adapter-only first-stage objective on frozen teacher/token caches.",
            "- Lower L1 is better; higher cosine is better.",
            "- `zero_condition` is an untrained lower-bound baseline.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


@torch.no_grad()
def evaluate_zero(eval_paths, device, dtype):
    rows = []
    for path in eval_paths:
        cache = cache_to_device(torch.load(path, map_location="cpu"), device, dtype)
        teacher = cache["teacher"]
        start = time.perf_counter()
        student = torch.zeros_like(teacher)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        metric = distill_metrics(student, teacher)
        metric.update({"cache": os.path.basename(path), "seconds": elapsed})
        rows.append(metric)
        del cache, teacher, student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_metrics(rows)
    summary.update(
        {
            "experiment_id": "zero_condition",
            "adapter_type": "baseline",
            "step": 0,
            "params": 0,
            "params_m": 0.0,
            "checkpoint": "",
            "event_final_loss": None,
        }
    )
    return summary, rows


@torch.no_grad()
def evaluate_checkpoint(meta, sample_cache, eval_paths, device, dtype):
    ckpt = torch.load(meta["path"], map_location="cpu")
    if not meta.get("config"):
        meta["config"] = load_jsonable_config(ckpt.get("config", {}))
    state_dict = ckpt["adapter"]
    adapter_cfg = dict(meta["config"].get("adapter", {}))
    if "input_proj.weight" in state_dict:
        adapter_cfg["type"] = "conv"
    elif any(key.startswith("source_encoder.") for key in state_dict):
        adapter_cfg["type"] = "cross_attention_rope"
    meta["config"]["adapter"] = adapter_cfg
    meta["adapter_type"] = str(adapter_cfg.get("type", meta.get("adapter_type", "unknown")))
    meta["step"] = int(ckpt.get("step", meta.get("step", 0) or 0))
    event_step, event_loss, event_count = read_event_final_loss(meta["run_dir"])
    meta["event_step"] = event_step
    meta["event_final_loss"] = event_loss
    meta["event_count"] = event_count
    adapter = build_adapter_from_checkpoint_config(meta["config"], sample_cache, device, dtype)
    missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State mismatch for {meta['path']}: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    adapter.eval()
    params = count_parameters(adapter)
    rows = []
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for path in eval_paths:
        cache = cache_to_device(torch.load(path, map_location="cpu"), device, dtype)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        student = adapter(
            cache["tokens"],
            cache["output_grid"],
            target_times=cache.get("target_times"),
            target_poses=cache.get("target_poses"),
            target_intrs=cache.get("target_intrs"),
            target_plucker=cache.get("target_plucker"),
            source_times=cache.get("source_times"),
            source_poses=cache.get("source_poses"),
            source_intrs=cache.get("source_intrs"),
        )
        if isinstance(student, dict):
            student = student.get("condition", next(iter(student.values())))
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        metric = distill_metrics(student, cache["teacher"])
        metric.update({"cache": os.path.basename(path), "seconds": elapsed})
        rows.append(metric)
        del cache, student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_metrics(rows)
    peak_mem = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    run_dir_abs = Path(meta["run_dir"]).resolve()
    root_abs = (CODE_DIR / "outputs" / "NeoVerseControlLatentDistill").resolve()
    try:
        experiment_id = run_dir_abs.relative_to(root_abs)
    except ValueError:
        experiment_id = Path(os.path.basename(meta["run_dir"]))
    summary.update(
        {
            "experiment_id": str(experiment_id),
            "adapter_type": meta["adapter_type"],
            "step": meta["step"],
            "params": params,
            "params_m": params / 1e6,
            "checkpoint": meta["path"],
            "hidden_dim": meta.get("hidden_dim"),
            "num_blocks": meta.get("num_blocks"),
            "source_pool_hw": json.dumps(meta.get("source_pool_hw")),
            "use_rope": meta.get("use_rope"),
            "event_step": meta.get("event_step"),
            "event_final_loss": meta.get("event_final_loss"),
            "event_count": meta.get("event_count"),
            "peak_mem_gb": peak_mem,
        }
    )
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="outputs/NeoVerseControlLatentDistill/frozen_cache")
    parser.add_argument("--checkpoint_glob", default="outputs/NeoVerseControlLatentDistill/**/adapter_last.pt")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_checkpoints", type=int, default=10)
    parser.add_argument("--num_eval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--include_zero", action="store_true")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    if device == "cuda":
        device = "cuda:0"

    output_dir = args.output_dir or os.path.join(
        "outputs",
        "midterm_experiments",
        time.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    cache_paths = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
    if len(cache_paths) < args.num_eval:
        raise RuntimeError(f"Need at least {args.num_eval} caches, found {len(cache_paths)}")
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(cache_paths), generator=generator).tolist()
    eval_paths = [cache_paths[i] for i in perm[: args.num_eval]]
    sample_cache = torch.load(eval_paths[0], map_location="cpu")

    checkpoint_paths = sorted(glob.glob(args.checkpoint_glob, recursive=True))
    selected = select_diverse_checkpoints(checkpoint_paths, args.max_checkpoints)
    if len(selected) < args.max_checkpoints:
        print(f"[warn] selected only {len(selected)} checkpoints", flush=True)

    manifest = {
        "cache_dir": args.cache_dir,
        "eval_paths": eval_paths,
        "device": device,
        "dtype": args.dtype,
        "selected_checkpoints": [
            {key: value for key, value in meta.items() if key not in {"state_dict", "config"}}
            for meta in selected
        ],
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    detail_rows = []
    summary_rows = []
    if args.include_zero:
        summary, rows = evaluate_zero(eval_paths, device, dtype)
        summary_rows.append(summary)
        for row in rows:
            row.update({"experiment_id": summary["experiment_id"], "checkpoint": ""})
            detail_rows.append(row)
        print(f"[done] zero_condition l1={summary['l1_mean']:.6f}", flush=True)

    for idx, meta in enumerate(selected, start=1):
        print(f"[eval {idx}/{len(selected)}] {meta['run_dir']} step={meta['step']} type={meta['adapter_type']}", flush=True)
        try:
            summary, rows = evaluate_checkpoint(meta, sample_cache, eval_paths, device, dtype)
        except Exception as exc:
            print(f"[warn] failed {meta['run_dir']}: {exc}", flush=True)
            summary_rows.append(
                {
                    "experiment_id": str(Path(meta["run_dir"]).name),
                    "adapter_type": meta.get("adapter_type", "unknown"),
                    "step": meta.get("step", 0),
                    "checkpoint": meta["path"],
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue
        summary["status"] = "ok"
        summary_rows.append(summary)
        for row in rows:
            row.update({"experiment_id": summary["experiment_id"], "checkpoint": summary["checkpoint"]})
            detail_rows.append(row)
        print(
            f"[done] {summary['experiment_id']} l1={summary['l1_mean']:.6f} "
            f"cos={summary['cosine_mean']:.6f} sec={summary['seconds_mean']:.3f}",
            flush=True,
        )

    write_csv(os.path.join(output_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(output_dir, "details.csv"), detail_rows)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    write_markdown(os.path.join(output_dir, "REPORT.md"), summary_rows, eval_paths)
    print(f"[complete] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
