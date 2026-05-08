import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parents[3]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def resolve_code_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (CODE_DIR / path).resolve()


def make_config(base, config_dir, name, **updates):
    cfg = dict(base)
    cfg.update(updates)
    cfg.setdefault("data_root", "data/SpatialVID")
    cfg.setdefault("temporal_augmentation", False)
    cfg.setdefault("temporal_trajectory_profile", "forward_pause")
    cfg.setdefault("temporal_order", "trajectory")
    cfg.setdefault("temporal_max_condition_frames", 8)
    cfg.setdefault("context_sampling_weights", None)
    path = config_dir / f"{name}.yaml"
    dump_yaml(cfg, path)
    return path


def output_complete(run_dir, modes):
    required = ["gt_81_sorted.mp4", "rendered_degraded_rgb.mp4"]
    if "teacher" in modes:
        required.append("teacher.mp4")
    if "student" in modes:
        required.append("student.mp4")
    if "teacher" in modes and "student" in modes:
        required.append("comparison_grid_gt.mp4")
    elif "teacher" in modes:
        required.append("comparison_grid_gt_context_render_neoversepred.mp4")
    return all((run_dir / item).exists() and (run_dir / item).stat().st_size > 0 for item in required)


def run_command(cmd, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--config", required=True, help="Base distillation config or saved run config.")
    parser.add_argument("--checkpoint", required=True, help="Adapter checkpoint used for student runs.")
    parser.add_argument("--python", default="/root/vepfs/envs/neoverse/bin/python")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root or CODE_DIR / "outputs/main_experiments/generation_suite").resolve()
    config_dir = output_root / "configs"
    run_root = output_root / "runs"
    log_dir = output_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_config = resolve_code_path(args.config)
    checkpoint = resolve_code_path(args.checkpoint)
    base = load_yaml(base_config)
    base["data_root"] = "data/SpatialVID"
    base["dataset_seed"] = int(base.get("seed", 2))
    base.setdefault("use_camera_annotations", False)
    base.setdefault("continuous_target_frames", True)
    base.setdefault("force_first_context", True)
    base.setdefault("timestamp_unit", "seconds")

    configs = {
        "base": make_config(base, config_dir, "base"),
        "context4": make_config(
            base,
            config_dir,
            "context4",
            min_num_context_views=4,
            max_num_context_views=4,
            context_sampling_strategy="uniform_first",
        ),
        "context8": make_config(
            base,
            config_dir,
            "context8",
            min_num_context_views=8,
            max_num_context_views=8,
            context_sampling_strategy="uniform_first",
        ),
        "context16": make_config(
            base,
            config_dir,
            "context16",
            min_num_context_views=16,
            max_num_context_views=16,
            context_sampling_strategy="uniform_first",
        ),
        "temporal_backward": make_config(
            base,
            config_dir,
            "temporal_backward",
            temporal_augmentation=True,
            temporal_trajectory_profile="backward",
            temporal_order="trajectory",
            temporal_max_condition_frames=8,
            variants_per_scene=4,
        ),
        "temporal_mixed": make_config(
            base,
            config_dir,
            "temporal_mixed",
            temporal_augmentation=True,
            temporal_trajectory_profile="mixed_fzr",
            temporal_order="trajectory",
            temporal_max_condition_frames=8,
            variants_per_scene=4,
        ),
    }

    runs = [
        {"id": "E01_main_case0", "group": "main_teacher_student", "config": "base", "dataset_index": 0, "modes": "teacher,student", "control_scale": 1.0},
        {"id": "E02_main_case1", "group": "main_teacher_student", "config": "base", "dataset_index": 1, "modes": "teacher,student", "control_scale": 1.0},
        {"id": "E03_main_case2", "group": "main_teacher_student", "config": "base", "dataset_index": 2, "modes": "teacher,student", "control_scale": 1.0},
        {"id": "E04_context4_case0", "group": "context_count", "config": "context4", "dataset_index": 0, "modes": "teacher,student", "control_scale": 1.0},
        {"id": "E05_context8_case0", "group": "context_count", "config": "context8", "dataset_index": 0, "modes": "teacher,student", "control_scale": 1.0},
        {"id": "E06_context16_case0", "group": "context_count", "config": "context16", "dataset_index": 0, "modes": "teacher,student", "control_scale": 1.0},
        {"id": "E07_control0_case0", "group": "control_scale", "config": "base", "dataset_index": 0, "modes": "teacher", "control_scale": 0.0, "alias": {"teacher": "zero_control"}},
        {"id": "E08_control05_case0", "group": "control_scale", "config": "base", "dataset_index": 0, "modes": "teacher", "control_scale": 0.5, "alias": {"teacher": "teacher_scale_0_5"}},
        {"id": "E09_temporal_backward_case0", "group": "temporal_trajectory", "config": "temporal_backward", "dataset_index": 0, "modes": "teacher,student", "control_scale": 1.0},
        {"id": "E10_temporal_mixed_case0", "group": "temporal_trajectory", "config": "temporal_mixed", "dataset_index": 0, "modes": "teacher,student", "control_scale": 1.0},
    ]
    if args.limit is not None:
        runs = runs[: int(args.limit)]

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_config": str(base_config),
        "checkpoint": str(checkpoint),
        "steps": args.steps,
        "seed": args.seed,
        "runs": [],
    }

    for idx, run in enumerate(runs, start=1):
        run_dir = run_root / run["id"]
        modes = [item.strip() for item in run["modes"].split(",") if item.strip()]
        if args.resume and output_complete(run_dir, modes):
            print(f"[skip {idx}/{len(runs)}] {run['id']} already complete")
            status = "skipped_complete"
        else:
            if run_dir.exists() and not args.resume:
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                args.python,
                "tools/eval/render_student_comparison.py",
                str(configs[run["config"]]),
                str(checkpoint),
                "--output_dir",
                str(run_dir),
                "--dataset_index",
                str(run["dataset_index"]),
                "--modes",
                run["modes"],
                "--num_inference_steps",
                str(args.steps),
                "--cfg_scale",
                "1.0",
                "--control_scale",
                str(run["control_scale"]),
                "--seed",
                str(args.seed),
                "--enable_vram_management",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"[run {idx}/{len(runs)}] {run['id']}")
            code = run_command(["env", f"CUDA_VISIBLE_DEVICES={args.gpu}", *cmd], CODE_DIR, log_dir / f"{run['id']}.log")
            status = "ok" if code == 0 and output_complete(run_dir, modes) else f"failed_code_{code}"
            if status != "ok":
                print(f"[warn] {run['id']} status={status}")
        entry = dict(run)
        entry.update(
            {
                "output_dir": str(run_dir),
                "config_path": str(configs[run["config"]]),
                "status": status,
            }
        )
        manifest["runs"].append(entry)
        with open(output_root / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"[complete] manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
