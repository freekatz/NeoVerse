#!/usr/bin/env python3
import argparse
import hashlib
import os
from pathlib import Path


def human_size(num_bytes):
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def split_score(path, seed):
    digest = hashlib.sha256(f"{int(seed)}:{Path(path).name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def split_paths(paths, eval_ratio, seed, mode):
    mode = str(mode or "hash").strip().lower()
    if mode != "hash":
        raise ValueError(f"Unsupported split mode: {mode!r}")
    eval_ratio = float(eval_ratio)
    if eval_ratio < 0.0 or eval_ratio >= 1.0:
        raise ValueError(f"eval_ratio must be in [0, 1), got {eval_ratio}")
    paths = sorted(str(path) for path in paths)
    if eval_ratio <= 0.0 or len(paths) <= 1:
        return paths, []

    scored = [(split_score(path, seed), path) for path in paths]
    train = [path for score, path in scored if score >= eval_ratio]
    eval_ = [path for score, path in scored if score < eval_ratio]
    if not eval_:
        _, path = min(scored, key=lambda item: item[0])
        eval_ = [path]
        train = [candidate for _, candidate in scored if candidate != path]
    elif not train:
        _, path = max(scored, key=lambda item: item[0])
        train = [path]
        eval_ = [candidate for _, candidate in scored if candidate != path]
    return sorted(train), sorted(eval_)


def describe_value(value):
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return f"shape={tuple(shape)} dtype={dtype}"
    return repr(value)


def print_samples(paths, sample_count):
    if sample_count <= 0:
        return
    import torch

    print("")
    print("samples:")
    for path in paths[:sample_count]:
        cache = torch.load(path, map_location="cpu")
        print(f"- {Path(path).name}")
        for key in sorted(cache.keys()):
            print(f"  {key}: {describe_value(cache[key])}")


def main():
    parser = argparse.ArgumentParser(description="Inspect NeoVerse frozen cache directories.")
    parser.add_argument("--cache_dir", default="data/frozen_cache")
    parser.add_argument("--pattern", default="*.pt")
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--split_mode", default="hash")
    parser.add_argument("--sample", type=int, default=0, help="Load and print this many sample cache files.")
    parser.add_argument("--sample_split", choices=["all", "train", "eval"], default="all")
    args = parser.parse_args()

    root = Path(args.cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory does not exist: {root}")
    paths = sorted(str(path) for path in root.glob(args.pattern) if path.is_file())
    train, eval_ = split_paths(paths, args.eval_ratio, args.split_seed, args.split_mode)
    total_size = sum(os.path.getsize(path) for path in paths)

    print(f"cache_dir={root.resolve()}")
    print(f"pattern={args.pattern}")
    print(f"total_files={len(paths)}")
    print(f"total_size={human_size(total_size)}")
    print(f"split_mode={args.split_mode}")
    print(f"split_seed={args.split_seed}")
    print(f"eval_ratio={args.eval_ratio:.6g}")
    print(f"train_files={len(train)}")
    print(f"eval_files={len(eval_)}")

    sample_paths = {"all": paths, "train": train, "eval": eval_}[args.sample_split]
    print_samples(sample_paths, int(args.sample))


if __name__ == "__main__":
    main()
