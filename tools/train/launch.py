import datetime
import logging
import os
import random
import re
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from omegaconf import OmegaConf

from utils.swanlab import init_swanlab_logger
from utils.training import (
    MetricLogger,
    SmoothedValue,
    get_worker_init_fn,
    printer,
    save_current_code,
)
from utils.training_module import DiffusionTrainingModule


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args,
):
    accumu_steps = args.gradient_accumulation_steps
    accelerator = Accelerator(
        gradient_accumulation_steps=accumu_steps,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=6000)),
        ],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_main_process:
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
        dst_dir = save_current_code(outdir=args.output_path)
        OmegaConf.save(args, os.path.join(args.output_path, "config.yaml"))
        printer.info(f"Saving current code to {dst_dir}")

    seed = args.seed + accelerator.process_index
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
        worker_init_fn=get_worker_init_fn(
            seed=seed,
            num_workers=args.num_workers,
        ),
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if args.resume is not None:
        printer.info(f"Resuming from {args.resume}")
        pattern = r"checkpoint-epoch-(\d+)(?:-iter-(\d+))?"
        match = re.search(pattern, args.resume)
        if match:
            start_epoch = int(match.group(1))
            resume_step = None if match.group(2) is None else int(match.group(2))
        else:
            start_epoch = int(getattr(args, "start_epoch", 0))
            rs = getattr(args, "resume_step", None)
            resume_step = None if rs is None else int(rs)
        if accelerator.distributed_type == "DEEPSPEED":
            accelerator.load_state(args.resume, load_module_strict=False)
        else:
            accelerator.load_state(args.resume, strict=False)
    else:
        start_epoch = args.start_epoch
        resume_step = args.resume_step

    experiment_logger = (
        init_swanlab_logger(
            args,
            default_project="DiffSynth",
            default_experiment_name=os.path.basename(os.path.normpath(str(args.output_path))),
            output_path=args.output_path,
        )
        if accelerator.is_main_process
        else None
    )

    printer.info("Start training")
    for epoch_id in range(start_epoch, args.num_epochs):
        metric_logger = MetricLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = "Epoch: [{}]".format(epoch_id)
        if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(epoch_id)

        if epoch_id == start_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(dataloader, resume_step)
        else:
            active_dataloader = dataloader
            resume_step = 0

        for iter_step, data in enumerate(
            metric_logger.log_every(active_dataloader, args.print_freq, accelerator, header)
        ):
            data_iter_step = iter_step + resume_step
            epoch_f = epoch_id + data_iter_step / len(dataloader)
            step = int(epoch_f * len(dataloader))
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                loss_value = float(loss)
                accelerator.backward(loss)
                if args.clip_grad is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(epoch=epoch_f)
                metric_logger.update(lr=lr)
                metric_logger.update(step=step)
                metric_logger.update(loss=loss_value)
                scheduler.step()
                if (data_iter_step + 1) % accumu_steps == 0 and (
                    (data_iter_step + 1) % (accumu_steps * args.print_freq)
                ) == 0:
                    loss_value_reduce = accelerator.gather(torch.tensor(loss_value).to(accelerator.device)).mean()

                    if experiment_logger is not None:
                        epoch_1000x = int(epoch_f * 1000)
                        experiment_logger.log(
                            {
                                "train/loss": loss_value_reduce,
                                "train/lr": lr,
                                "train/epoch_1000x": epoch_1000x,
                            },
                            step=step,
                        )
            save_period = int(args.save_freq * len(dataloader))
            if (
                save_period > 0
                and data_iter_step % save_period == 0
                and iter_step != 0
                and iter_step != len(active_dataloader) - 1
            ):
                print("saving at step", data_iter_step)
                metric_logger.save(accelerator, model, epoch_id, data_iter_step)
        metric_logger.save(accelerator, model, epoch_id + 1)
    if experiment_logger is not None:
        experiment_logger.finish()
