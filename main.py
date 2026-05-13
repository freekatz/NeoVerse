#!/usr/bin/env python3
"""NeoVerse CLI router. Run from repository root."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
os.chdir(CODE_DIR)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

PYTHON = os.environ.get("PYTHON", sys.executable)


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def _run_sh(script_rel: str, args: list[str], env_extra: dict | None = None) -> int:
    script = CODE_DIR / script_rel
    if not script.is_file():
        _die(f"Missing {script_rel}")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(["bash", str(script), *args], cwd=str(CODE_DIR), env=env).returncode


def _common_online_env() -> dict[str, str]:
    e = {"GPU_LIST": os.environ.get("GPU_LIST", os.environ.get("CUDA_VISIBLE_DEVICES", "") or "")}
    for k, d in (
        ("CAMERA_CONDITION_NORMALIZE", "true"),
        ("CAMERA_CONDITION_MIN_TRANSLATION_SCALE", "1.0"),
        ("USE_CAMERA_ANNOTATIONS", "false"),
        ("ENABLE_VRAM_MANAGEMENT", "false"),
        ("NUM_WORKERS", "4"),
        ("PIN_MEMORY", "true"),
        ("PERSISTENT_WORKERS", "true"),
        ("PREFETCH_FACTOR", "2"),
        ("REUSE_RECONSTRUCTOR_TOKENS", "true"),
        ("SKIP_UNUSED_RECONSTRUCTOR_HEADS", "true"),
        ("REUSE_CONTEXT_FORWARD_TOKENS", "true"),
    ):
        e.setdefault(k, os.environ.get(k, d))
    return e


def _disable_frozen() -> dict[str, str]:
    return {
        k: "false"
        for k in (
            "TRAIN_FROM_FROZEN_CACHE",
            "FROZEN_CACHE_READ",
            "FROZEN_CACHE_WRITE",
            "FROZEN_CACHE_REQUIRED",
            "PRELOAD_FROZEN_CACHE",
            "CACHE_FROZEN_OUTPUTS",
        )
    }


def _train_online(rest: list[str]) -> int:
    env = {**_common_online_env(), **_disable_frozen()}
    env.setdefault("RUN_NAME", os.environ.get("RUN_NAME", "online_time_forward_pause"))
    env.setdefault("CONTEXT_SAMPLING_STRATEGY", os.environ.get("CONTEXT_SAMPLING_STRATEGY", "mixed"))
    env["TEMPORAL_AUGMENTATION"] = "true"
    env.setdefault("TEMPORAL_TRAJECTORY_PROFILE", os.environ.get("TEMPORAL_TRAJECTORY_PROFILE", "forward_pause"))
    env.setdefault(
        "TEMPORAL_VARIANT_PROFILE_WEIGHTS",
        os.environ.get(
            "TEMPORAL_VARIANT_PROFILE_WEIGHTS",
            "forward_pause:0.5,mixed_fzr:0.4,backward:0.1",
        ),
    )
    env.setdefault("TEMPORAL_ORDER", os.environ.get("TEMPORAL_ORDER", "trajectory"))
    env.setdefault("TRAJECTORIES_PER_CLIP", os.environ.get("TRAJECTORIES_PER_CLIP", "8"))
    env.setdefault("DATASET_SEED", os.environ.get("DATASET_SEED", "0"))
    return _run_sh("scripts/launch/train_distill.sh", rest, env)


def _train_online_noaug(rest: list[str]) -> int:
    env = {**_common_online_env(), **_disable_frozen()}
    env.setdefault("RUN_NAME", os.environ.get("RUN_NAME", "online_noaug"))
    env.setdefault("CONTEXT_SAMPLING_STRATEGY", os.environ.get("CONTEXT_SAMPLING_STRATEGY", "mixed"))
    env["TEMPORAL_AUGMENTATION"] = "false"
    env.setdefault("DATASET_SEED", os.environ.get("DATASET_SEED", "null"))
    return _run_sh("scripts/launch/train_distill.sh", rest, env)


def _train_cache(rest: list[str]) -> int:
    fc = os.environ.get(
        "FROZEN_CACHE_DIR",
        str(CODE_DIR / "outputs/NeoVerseControlLatentDistill/frozen_cache"),
    )
    env = {**_common_online_env()}
    env.setdefault("RUN_NAME", os.environ.get("RUN_NAME", "train_cache"))
    env.setdefault("CONTEXT_SAMPLING_STRATEGY", os.environ.get("CONTEXT_SAMPLING_STRATEGY", "mixed"))
    env.setdefault("DATASET_SEED", os.environ.get("DATASET_SEED", "null"))
    env["FROZEN_CACHE_DIR"] = fc
    env.setdefault("TRAIN_FROM_FROZEN_CACHE", "true")
    env.setdefault("FROZEN_CACHE_SPLIT", os.environ.get("FROZEN_CACHE_SPLIT", "train"))
    env.setdefault("FROZEN_CACHE_READ", "true")
    env.setdefault("FROZEN_CACHE_WRITE", "false")
    env.setdefault("FROZEN_CACHE_REQUIRED", "true")
    env.setdefault("PRELOAD_FROZEN_CACHE", os.environ.get("PRELOAD_FROZEN_CACHE", "false"))
    env.setdefault("CACHE_FROZEN_OUTPUTS", os.environ.get("CACHE_FROZEN_OUTPUTS", "false"))
    return _run_sh("scripts/launch/train_distill.sh", rest, env)


def usage() -> None:
    print(
        """Usage:
  python main.py train cache|online|online-noaug [launcher args]
  python main.py infer neoverse [...]
  python main.py cache build-camera|build-frozen|inspect [...]
  python main.py eval last CONFIG CKPT [indices...]
  python main.py legacy train-base CONFIG [...]
  python main.py dev data prepare-spatialvid-hq [...]
Optional: PYTHON=python3  FROZEN_CACHE_DIR=...  GPU_LIST=0,1
"""
    )


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        usage()
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "train":
        if not rest:
            _die("train requires cache|online|online-noaug")
        mode, *trest = rest
        if mode == "online":
            return _train_online(trest)
        if mode == "online-noaug":
            return _train_online_noaug(trest)
        if mode == "cache":
            return _train_cache(trest)
        _die(f"unknown train mode: {mode}")
        return 2

    if cmd == "infer":
        if not rest or rest[0] != "neoverse":
            _die("infer neoverse [...]")
        return subprocess.run([PYTHON, "tools/inference/neoverse_inference.py", *rest[1:]], cwd=str(CODE_DIR)).returncode

    if cmd == "app":
        if not rest or rest[0] != "neoverse":
            _die("app neoverse [...]")
        app_py = CODE_DIR / "tools/apps/neoverse_app.py"
        if not app_py.is_file():
            _die("tools/apps/neoverse_app.py missing")
        return subprocess.run([PYTHON, str(app_py), *rest[1:]], cwd=str(CODE_DIR)).returncode

    if cmd == "cache":
        if not rest:
            _die("cache build-camera|build-frozen|inspect")
        sub, *crest = rest
        if sub == "build-camera":
            return _run_sh("scripts/launch/build_camera_cache.sh", crest)
        if sub == "build-frozen":
            return _run_sh("scripts/launch/build_frozen_cache.sh", crest)
        if sub == "inspect":
            return subprocess.run([PYTHON, "tools/cache/inspect_cache.py", *crest], cwd=str(CODE_DIR)).returncode
        _die(f"unknown cache: {sub}")
        return 2

    if cmd == "eval":
        if len(rest) < 3 or rest[0] != "last":
            _die("eval last CONFIG CHECKPOINT [indices...]")
        cfg, ckpt = rest[1], rest[2]
        idx = rest[3:] if len(rest) > 3 else ["0", "17", "33", "81", "159", "240", "319"]
        base = Path(ckpt).resolve().parent
        for i in idx:
            r = subprocess.run(
                [
                    PYTHON,
                    "tools/eval/render_student_comparison.py",
                    cfg,
                    ckpt,
                    "--output_dir",
                    str(base / f"eval_idx{i}"),
                    "--dataset_index",
                    str(i),
                    "--modes",
                    "teacher,student",
                    "--enable_vram_management",
                ],
                cwd=str(CODE_DIR),
            )
            if r.returncode != 0:
                return r.returncode
        return 0

    if cmd == "legacy":
        if len(rest) < 2 or rest[0] != "train-base":
            _die("legacy train-base CONFIG [...]")
        return subprocess.run([PYTHON, "tools/train/base_train.py", *rest[1:]], cwd=str(CODE_DIR)).returncode

    if cmd == "dev":
        if len(rest) < 3 or rest[0] != "data" or rest[1] != "prepare-spatialvid-hq":
            _die("dev data prepare-spatialvid-hq [...]")
        return _run_sh("scripts/data/prepare_spatialvid_hq.sh", rest[2:])

    usage()
    _die(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
