import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from omegaconf import OmegaConf

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.control_latent.cache import (
    FrozenForwardCacheDataset,
    batch_preloaded_frozen_cache,
    cache_float_dtype,
    cache_target_is_cpu,
    cache_to_target_device,
    collate_frozen_cache_entries,
    looks_like_frozen_cache,
    make_dataloader_kwargs,
    optional_int,
    validation_enabled,
)
from models.control_latent.camera import compact_distill_log_metrics
from models.control_latent.loss import save_heatmaps
from models.control_latent.module import ControlLatentDistillModule
from utils.config import build_spatialvid_dataset
from utils.swanlab import init_swanlab_logger


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(accelerator, model, optimizer, cfg, output_path, step, epoch=None, name=None, include_optimizer=True):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    os.makedirs(output_path, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    ckpt_name = name or "adapter_last.pt"
    payload = {
        "adapter": unwrapped.adapter.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "step": step,
        "epoch": epoch,
        "shapes": getattr(unwrapped, "last_shapes", None),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    ckpt_path = os.path.join(output_path, ckpt_name)
    tmp_path = f"{ckpt_path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)
    OmegaConf.save(cfg, os.path.join(output_path, "config.yaml"))


def resolve_resume_path(cfg):
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        return str(resume_from)
    if bool(cfg.get("auto_resume", False)):
        candidate = os.path.join(str(cfg.output_path), "adapter_last.pt")
        if os.path.exists(candidate):
            return candidate
        if os.path.isdir(str(cfg.output_path)):
            latest_step = -1
            latest_path = None
            for name in os.listdir(str(cfg.output_path)):
                if not (name.startswith("adapter_step_") and name.endswith(".pt")):
                    continue
                try:
                    step = int(name.removeprefix("adapter_step_").removesuffix(".pt"))
                except ValueError:
                    continue
                if step > latest_step:
                    latest_step = step
                    latest_path = os.path.join(str(cfg.output_path), name)
            if latest_path is not None:
                return latest_path
    return None


def load_checkpoint(accelerator, model, optimizer, resume_path, load_optimizer=True):
    checkpoint = torch.load(resume_path, map_location=accelerator.device)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.adapter.load_state_dict(checkpoint["adapter"])
    if load_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = int(checkpoint.get("step", 0))
    if accelerator.is_main_process:
        print(f"Resumed adapter checkpoint from {resume_path} at step={step}")
        if load_optimizer and "optimizer" not in checkpoint:
            print("WARNING: checkpoint has no optimizer state; continuing with a fresh optimizer.")
    return step


def run_validation(accelerator, model, eval_dataloader, cfg):
    max_eval_steps = int(cfg.get("eval_steps", 0))
    if eval_dataloader is None or max_eval_steps <= 0:
        return {}

    model.eval()
    metric_totals = {}
    local_batches = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    try:
        with torch.no_grad():
            for eval_idx, batch in enumerate(eval_dataloader, start=1):
                loss, metrics = model(batch)
                local_batches += 1.0
                metrics = dict(metrics)
                metrics["loss"] = loss.detach()
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        value = value.detach().float().to(device=accelerator.device)
                        metric_totals[key] = metric_totals.get(key, torch.zeros_like(value)) + value
                if eval_idx >= max_eval_steps:
                    break
    finally:
        model.train()

    total_batches = accelerator.gather(local_batches).sum()
    if float(total_batches.detach().cpu()) <= 0:
        return {}
    averaged = {}
    for key, value in metric_totals.items():
        total = accelerator.gather(value).sum()
        averaged[key] = float((total / total_batches).detach().cpu())
    return averaged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. output_path=outputs/run")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    os.makedirs(cfg.output_path, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=6000))],
    )
    seed_everything(int(cfg.seed) + accelerator.process_index)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(cfg.output_path, "config.yaml"))

    train_from_frozen_cache = bool(cfg.get("train_from_frozen_cache", False))
    eval_dataloader = None
    eval_dataset = None
    if train_from_frozen_cache:
        if not cfg.get("frozen_cache_dir", None):
            raise ValueError("train_from_frozen_cache=true requires frozen_cache_dir.")
        frozen_cache_split = str(cfg.get("frozen_cache_split", "train") or "train").strip().lower()
        dataset = FrozenForwardCacheDataset(
            cfg.frozen_cache_dir,
            pattern=cfg.get("frozen_cache_pattern", "*.pt"),
            split=frozen_cache_split,
            eval_ratio=float(cfg.get("frozen_cache_eval_ratio", 0.05)),
            split_seed=int(cfg.get("frozen_cache_split_seed", 0)),
            split_mode=cfg.get("frozen_cache_split_mode", "hash"),
        )
        if validation_enabled(cfg, train_from_frozen_cache, frozen_cache_split):
            eval_dataset = FrozenForwardCacheDataset(
                cfg.frozen_cache_dir,
                pattern=cfg.get("frozen_cache_pattern", "*.pt"),
                split="eval",
                eval_ratio=float(cfg.get("frozen_cache_eval_ratio", 0.05)),
                split_seed=int(cfg.get("frozen_cache_split_seed", 0)),
                split_mode=cfg.get("frozen_cache_split_mode", "hash"),
                allow_empty=True,
            )
            if len(eval_dataset) == 0:
                eval_dataset = None
        if accelerator.is_main_process:
            counts = dataset.split_counts
            print(
                f"Using frozen forward cache dataset: split={frozen_cache_split} selected={len(dataset)} "
                f"total={counts['all']} train={counts['train']} eval={counts['eval']} "
                f"dir={cfg.frozen_cache_dir} pattern={cfg.get('frozen_cache_pattern', '*.pt')!r} "
                f"eval_ratio={float(cfg.get('frozen_cache_eval_ratio', 0.05)):.4g} "
                f"split_seed={int(cfg.get('frozen_cache_split_seed', 0))}"
            )
            if eval_dataset is None:
                print("Frozen cache validation is disabled.")
            else:
                print(
                    f"Using frozen forward eval dataset: selected={len(eval_dataset)} "
                    f"eval_freq={int(cfg.get('eval_freq', 0))} eval_steps={int(cfg.get('eval_steps', 0))}"
                )
    else:
        dataset = build_spatialvid_dataset(cfg)
    collate_fn = collate_frozen_cache_entries if train_from_frozen_cache else None
    dataloader_kwargs = make_dataloader_kwargs(
        cfg,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_fn,
    )
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    if train_from_frozen_cache and eval_dataset is not None:
        eval_batch_size = optional_int(cfg.get("eval_batch_size", None), int(cfg.batch_size))
        eval_num_workers = optional_int(cfg.get("eval_num_workers", None), int(cfg.num_workers))
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            **make_dataloader_kwargs(
                cfg,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=eval_num_workers,
                collate_fn=collate_frozen_cache_entries,
            ),
        )
    model = ControlLatentDistillModule(cfg)
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    if eval_dataloader is None:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    else:
        model, optimizer, dataloader, eval_dataloader = accelerator.prepare(model, optimizer, dataloader, eval_dataloader)
    resume_path = resolve_resume_path(cfg)
    step = 0
    if resume_path is not None:
        step = load_checkpoint(
            accelerator,
            model,
            optimizer,
            resume_path,
            load_optimizer=bool(cfg.get("resume_optimizer", True)),
        )
    if cfg.get("max_steps", None) is not None and step >= int(cfg.max_steps):
        if accelerator.is_main_process:
            print(f"Checkpoint step={step} already reached max_steps={int(cfg.max_steps)}. Exiting.")
        accelerator.end_training()
        return
    experiment_logger = (
        init_swanlab_logger(
            cfg,
            default_project="NeoVerseControlLatentDistill",
            default_experiment_name=os.path.basename(os.path.normpath(str(cfg.output_path))),
            output_path=str(cfg.output_path),
        )
        if accelerator.is_main_process
        else None
    )

    cached_train_batch = None
    if bool(cfg.get("cache_train_batch", False)):
        cached_train_batch = next(iter(dataloader))
        if accelerator.is_main_process:
            print("Cached one DataLoader batch for repeated overfit training.")

    preloaded_frozen_batches = None
    if bool(cfg.get("preload_frozen_cache", False)):
        unwrapped = accelerator.unwrap_model(model)
        preloaded_frozen_batches = []
        preload_device = str(cfg.get("preload_frozen_cache_device", "cuda"))
        preload_cache_device = "cpu" if preload_device == "cpu" else accelerator.device
        preload_total = len(dataloader) if hasattr(dataloader, "__len__") else None
        preload_log_freq = int(cfg.get("preload_frozen_cache_log_freq", 20))
        preload_start = time.time()
        if accelerator.is_main_process:
            total_text = "unknown" if preload_total is None else str(preload_total)
            print(f"Preloading frozen cache from DataLoader to {preload_device} ({total_text} batches on this process).")
        for preload_idx, preload_batch in enumerate(dataloader, start=1):
            if looks_like_frozen_cache(preload_batch):
                cache = cache_to_target_device(
                    preload_batch,
                    preload_cache_device,
                    dtype=cache_float_dtype(cfg) if cache_target_is_cpu(preload_cache_device) else None,
                )
            else:
                cache = unwrapped.get_forward_cache(preload_batch, device=preload_cache_device)
            preloaded_frozen_batches.append(cache)
            if accelerator.is_main_process and preload_log_freq > 0 and (
                preload_idx == 1 or preload_idx % preload_log_freq == 0
            ):
                elapsed = time.time() - preload_start
                if preload_total is None:
                    print(f"Preloaded frozen cache batches: {preload_idx} elapsed={elapsed:.1f}s")
                else:
                    print(
                        f"Preloaded frozen cache batches: {preload_idx}/{preload_total} "
                        f"elapsed={elapsed:.1f}s"
                    )
        preloaded_frozen_batches = batch_preloaded_frozen_cache(
            preloaded_frozen_batches,
            int(cfg.get("preload_frozen_cache_batch_size", 1)),
        )
        if accelerator.is_main_process:
            print(
                f"Preloaded {len(preloaded_frozen_batches)} frozen cache batches on this process "
                f"in {time.time() - preload_start:.1f}s."
            )
        if len(preloaded_frozen_batches) == 0:
            raise RuntimeError("preload_frozen_cache=true but no frozen cache entries were loaded.")

    last_log = time.time()
    for epoch in range(int(cfg.num_epochs)):
        if preloaded_frozen_batches is not None:
            if bool(cfg.get("shuffle_preloaded_frozen_cache", True)):
                random.shuffle(preloaded_frozen_batches)
            batch_iter = preloaded_frozen_batches
        elif cached_train_batch is not None:
            batch_iter = (cached_train_batch,)
        else:
            batch_iter = dataloader
        for batch in batch_iter:
            with accelerator.accumulate(model):
                loss, metrics = model(batch)
                accelerator.backward(loss)
                if cfg.clip_grad is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(cfg.clip_grad))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                if step % int(cfg.print_freq) == 0 and accelerator.is_main_process:
                    elapsed = time.time() - last_log
                    last_log = time.time()
                    metrics_float = {k: float(v.detach().float().cpu()) for k, v in metrics.items()}
                    summary = " ".join(
                        f"{k}={v:.4g}" for k, v in metrics_float.items() if k == "loss" or k.endswith("/l1")
                    )
                    print(f"epoch={epoch} step={step} dt={elapsed:.2f}s {summary}")
                    if experiment_logger is not None:
                        lr = optimizer.param_groups[0]["lr"]
                        experiment_logger.log(compact_distill_log_metrics(metrics_float, "train", lr=lr), step=step)
                    shapes = accelerator.unwrap_model(model).last_shapes
                    print(f"token_shape={shapes['tokens']} output_grid={shapes['output_grid']}")
                    first_key = next(iter(shapes["teacher"]))
                    print(f"teacher[{first_key}]={shapes['teacher'][first_key]} student[{first_key}]={shapes['student'][first_key]}")

                if eval_dataloader is not None and int(cfg.get("eval_freq", 0)) > 0 and step % int(cfg.eval_freq) == 0:
                    eval_start = time.time()
                    eval_metrics = run_validation(accelerator, model, eval_dataloader, cfg)
                    if accelerator.is_main_process:
                        if eval_metrics:
                            summary = " ".join(
                                f"eval/{k}={v:.4g}"
                                for k, v in eval_metrics.items()
                                if k == "loss" or k.endswith("/l1")
                            )
                            print(f"eval step={step} dt={time.time() - eval_start:.2f}s {summary}")
                            if experiment_logger is not None:
                                experiment_logger.log(compact_distill_log_metrics(eval_metrics, "eval"), step=step)
                        else:
                            print(f"eval step={step} skipped: no eval batches")

                if int(cfg.vis_freq) > 0 and step % int(cfg.vis_freq) == 0 and accelerator.is_main_process:
                    teacher, student, output_grid = accelerator.unwrap_model(model).last_visuals
                    save_heatmaps(os.path.join(cfg.output_path, "visuals"), step, teacher, student, output_grid)

                if int(cfg.save_freq) > 0 and step % int(cfg.save_freq) == 0:
                    include_optimizer = bool(cfg.get("resume_optimizer", True)) or bool(
                        cfg.get("save_optimizer_intermediate", False)
                    )
                    save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        cfg,
                        cfg.output_path,
                        step,
                        epoch=epoch,
                        name="adapter_last.pt",
                        include_optimizer=include_optimizer,
                    )

                if cfg.get("max_steps", None) is not None and step >= int(cfg.max_steps):
                    save_checkpoint(accelerator, model, optimizer, cfg, cfg.output_path, step, epoch=epoch, name="adapter_last.pt")
                    if experiment_logger is not None:
                        experiment_logger.finish()
                    accelerator.end_training()
                    return

    save_checkpoint(accelerator, model, optimizer, cfg, cfg.output_path, step, epoch=int(cfg.num_epochs) - 1, name="adapter_last.pt")
    if experiment_logger is not None:
        experiment_logger.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
