import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch

from models.control_latent.cache import json_default, looks_like_frozen_cache


def frozen_cache_batch_key(cache):
    key = []
    for name in sorted(cache.keys()):
        value = cache[name]
        if isinstance(value, torch.Tensor):
            key.append((name, tuple(value.shape[1:]), str(value.dtype)))
        else:
            key.append((name, json.dumps(value, sort_keys=True, default=json_default)))
    return tuple(key)


def merge_frozen_cache_entries(entries):
    if len(entries) == 1:
        return entries[0]
    merged = {}
    first = entries[0]
    for name, value in first.items():
        if isinstance(value, torch.Tensor):
            merged[name] = torch.cat([entry[name] for entry in entries], dim=0)
        else:
            merged[name] = value
    return merged


def batch_preloaded_frozen_cache(caches, batch_size: int):
    if batch_size <= 1:
        return caches
    buckets = OrderedDict()
    for cache in caches:
        buckets.setdefault(frozen_cache_batch_key(cache), []).append(cache)
    batches = []
    for entries in buckets.values():
        for start in range(0, len(entries), batch_size):
            batches.append(merge_frozen_cache_entries(entries[start : start + batch_size]))
    return batches


def frozen_cache_split_score(path, seed: int) -> float:
    name = Path(path).name
    digest = hashlib.sha256(f"{int(seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def split_frozen_cache_paths(paths, eval_ratio: float, seed: int, mode: str = "hash"):
    mode = str(mode or "hash").strip().lower()
    if mode != "hash":
        raise ValueError(f"Unsupported frozen_cache_split_mode={mode!r}; expected 'hash'.")
    eval_ratio = float(eval_ratio)
    if eval_ratio < 0.0 or eval_ratio >= 1.0:
        raise ValueError(f"frozen_cache_eval_ratio must be in [0, 1), got {eval_ratio}.")
    paths = sorted(str(path) for path in paths)
    if eval_ratio <= 0.0 or len(paths) <= 1:
        return paths, []

    scored = [(frozen_cache_split_score(path, seed), path) for path in paths]
    train_paths = [path for score, path in scored if score >= eval_ratio]
    eval_paths = [path for score, path in scored if score < eval_ratio]
    if len(eval_paths) == 0:
        score, path = min(scored, key=lambda item: item[0])
        eval_paths = [path]
        train_paths = [candidate for _, candidate in scored if candidate != path]
    elif len(train_paths) == 0:
        score, path = max(scored, key=lambda item: item[0])
        train_paths = [path]
        eval_paths = [candidate for _, candidate in scored if candidate != path]
    return sorted(train_paths), sorted(eval_paths)


class FrozenForwardCacheDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir,
        pattern="*.pt",
        split="all",
        eval_ratio=0.05,
        split_seed=0,
        split_mode="hash",
        allow_empty=False,
    ):
        self.cache_dir = str(cache_dir)
        self.pattern = str(pattern or "*.pt")
        self.split = str(split or "all").strip().lower()
        self.eval_ratio = float(eval_ratio)
        self.split_seed = int(split_seed)
        self.split_mode = str(split_mode or "hash")
        root = Path(self.cache_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Frozen cache directory does not exist: {self.cache_dir}")
        all_paths = sorted(str(path) for path in root.glob(self.pattern) if path.is_file())
        if len(all_paths) == 0:
            raise FileNotFoundError(f"No frozen cache files matched {self.pattern!r} in {self.cache_dir}")
        train_paths, eval_paths = split_frozen_cache_paths(
            all_paths,
            eval_ratio=self.eval_ratio,
            seed=self.split_seed,
            mode=self.split_mode,
        )
        if self.split == "all":
            self.paths = all_paths
        elif self.split == "train":
            self.paths = train_paths
        elif self.split == "eval":
            self.paths = eval_paths
        else:
            raise ValueError(f"frozen_cache_split must be one of train/eval/all, got {self.split!r}.")
        self.total_paths = len(all_paths)
        self.split_counts = {"all": len(all_paths), "train": len(train_paths), "eval": len(eval_paths)}
        if len(self.paths) == 0 and not allow_empty:
            raise FileNotFoundError(
                f"No frozen cache files selected for split={self.split!r} from {self.cache_dir} "
                f"(total={len(all_paths)}, eval_ratio={self.eval_ratio})."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[int(idx)]
        cache = torch.load(path, map_location="cpu")
        if not looks_like_frozen_cache(cache):
            raise ValueError(f"Invalid frozen forward cache file: {path}")
        cache = dict(cache)
        cache["cache_path"] = path
        return cache


def collate_frozen_cache_entries(entries):
    return merge_frozen_cache_entries(entries)


def optional_int(value, default=None):
    if value in {None, "", "null", "None"}:
        return default
    return int(value)


def make_dataloader_kwargs(cfg, batch_size, shuffle, num_workers, collate_fn=None):
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(cfg.pin_memory),
    }
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        if cfg.get("prefetch_factor", None) is not None:
            kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return kwargs


def validation_enabled(cfg, train_from_frozen_cache, train_split):
    return (
        bool(train_from_frozen_cache)
        and str(train_split or "train").strip().lower() == "train"
        and int(cfg.get("eval_freq", 0)) > 0
        and int(cfg.get("eval_steps", 0)) > 0
        and float(cfg.get("frozen_cache_eval_ratio", 0.0)) > 0.0
    )
