import datetime
import logging
import os
import random
import re
import shutil
import time
from collections import defaultdict, deque
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from accelerate.logging import get_logger
from omegaconf import OmegaConf

from utils.swanlab import init_swanlab_logger
from utils.training_module import DiffusionTrainingModule

printer = get_logger(__name__)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, accelerator: Accelerator):
        if accelerator.num_processes == 1:
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=accelerator.device)
        accelerator.wait_for_everyone()
        accelerator.reduce(t, reduction="sum")
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.ndim > 0:
                    continue
                v = v.item()
            if isinstance(v, list):
                continue
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, accelerator):
        for meter in self.meters.values():
            meter.synchronize_between_processes(accelerator)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, accelerator: Accelerator, header=None, max_iter=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        len_iterable = min(len(iterable), max_iter) if max_iter else len(iterable)
        space_fmt = ":" + str(len(str(len_iterable))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for it, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    if accelerator.is_main_process:
                        printer.info(
                            log_msg.format(
                                i,
                                len_iterable,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                                memory=torch.cuda.max_memory_allocated() / MB,
                            )
                        )
                else:
                    if accelerator.is_main_process:
                        printer.info(
                            log_msg.format(
                                i,
                                len_iterable,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                            )
                        )
            i += 1
            end = time.time()
            if max_iter and it >= max_iter:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if accelerator.is_main_process:
            printer.info("{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len_iterable))

    def save(self, accelerator, model, epoch_id, iter_id=None):
        if iter_id is not None:
            name = f"checkpoint-epoch-{epoch_id}-iter-{iter_id}"
        else:
            name = f"checkpoint-epoch-{epoch_id}"
        checkpoint_path = os.path.join(self.output_path, name)
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            accelerator.save(state_dict, checkpoint_path + ".safetensors", safe_serialization=True)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def on_step_end(self, loss):
        pass

    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


def save_current_code(outdir):
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d-%H:%M:%S")
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    ignore_pattern = shutil.ignore_patterns(
        "debug*",
        ".vscode*",
        "assets*",
        "example*",
        "checkpoints*",
        "OLD*",
        "logs*",
        "out*",
        "runs*",
        "*.png",
        "*.mp4",
        "*__pycache__*",
        "*.git*",
        "*.idea*",
        "*.zip",
        "*.jpg",
    )
    for src_dir in ["utils", "models", "tools", "neoverse", "wan"]:
        src_path = src_dir
        if os.path.isdir(src_path):
            shutil.copytree(
                src_path,
                os.path.join(dst_dir, src_dir),
                ignore=ignore_pattern,
                dirs_exist_ok=True,
            )
    return dst_dir


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def default_worker_init_fn(worker_id, num_workers, epoch, seed=0):
    rank = get_rank()
    world_size = get_world_size()
    RANK_MULTIPLIER = 1
    WORKER_MULTIPLIER = 1
    WORLD_MULTIPLIER = 1
    EPOCH_MULTIPLIER = 12345
    worker_seed = (
        rank * num_workers * RANK_MULTIPLIER
        + worker_id * WORKER_MULTIPLIER
        + seed
        + world_size * WORLD_MULTIPLIER
        + epoch * EPOCH_MULTIPLIER
    )
    torch.random.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def get_worker_init_fn(seed, num_workers, epoch=0, worker_init_fn=None):
    if worker_init_fn is not None:
        return worker_init_fn
    return partial(
        default_worker_init_fn,
        num_workers=num_workers,
        epoch=epoch,
        seed=seed,
    )
