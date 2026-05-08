import argparse
import os
import shutil
import subprocess
from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_JOB_ROOT = CODE_DIR / "deploy" / "volc" / "generated"
DEFAULT_MLP_VOLC = "/root/.volc/bin/volc"
PRIVATE_ENV_NAMES = {"SWANLAB_API_KEY"}

PRESET_COMMANDS = {
    "train-cache": "./cli train cache",
    "build-camera": "./cli cache build-camera",
    "build-frozen": "./cli cache build-frozen",
}


def env_value(name, default=None):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def bool_env(name, default=False):
    value = env_value(name)
    if value is None:
        return default
    return str_to_bool(value)


def parse_env(items):
    result = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if not name:
            raise ValueError(f"--env expects non-empty KEY, got {item!r}")
        result.append({"Name": name, "Value": value, "IsPrivate": name in PRIVATE_ENV_NAMES})
    return result


def optional_env(name):
    value = env_value(name)
    if value is None:
        return []
    return [{"Name": name, "Value": value, "IsPrivate": False}]


def private_optional_env(name):
    value = env_value(name)
    if value is None:
        return []
    return [{"Name": name, "Value": value, "IsPrivate": True}]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else (CODE_DIR / path)


def build_envs(args):
    envs = []
    for name in (
        "RUN_NAME",
        "DATA_ROOT",
        "MAX_STEPS",
        "FROZEN_CACHE_DIR",
        "CAMERA_CACHE_DIR",
        "FIXED_CLIPS_PER_SCENE",
        "TRAJECTORIES_PER_CLIP",
        "FROZEN_CACHE_EVAL_RATIO",
        "EVAL_FREQ",
        "PRELOAD_FROZEN_CACHE",
        "CACHE_NNODES",
        "CACHE_NODE_RANK",
        "CACHE_GLOBAL_NUM_SHARDS",
        "CACHE_GLOBAL_SHARD_OFFSET",
        "SWANLAB_ENABLED",
        "SWANLAB_PROJECT",
        "SWANLAB_EXPERIMENT_NAME",
        "SWANLAB_WORKSPACE",
        "SWANLAB_MODE",
        "SWANLAB_LOGDIR",
        "SWANLAB_TAGS",
        "SWANLAB_HOST",
        "SWANLAB_WEB_HOST",
    ):
        envs.extend(optional_env(name))
    envs.extend(private_optional_env("SWANLAB_API_KEY"))
    envs.extend(parse_env(args.env))
    return envs


def storage_config(args):
    if not args.vepfs_id:
        return []
    storage = {
        "Type": "Vepfs",
        "MountPath": args.vepfs_mount_path,
        "VepfsId": args.vepfs_id,
    }
    if args.vepfs_sub_path:
        storage["SubPath"] = args.vepfs_sub_path
    return [storage]


def build_conf(args):
    command = args.command or PRESET_COMMANDS[args.preset]
    entrypoint = f"cd {args.code_dir} && {command}"
    conf = {
        "TaskName": args.name,
        "Description": args.description,
        "Tags": ["neoverse", args.preset],
        "Entrypoint": entrypoint,
        "ImageUrl": args.image_url,
        "Framework": args.framework,
        "TaskRoleSpecs": [
            {
                "RoleName": args.role_name,
                "RoleReplicas": args.replicas,
                "Flavor": args.flavor,
            }
        ],
        "AccessType": args.access_type,
        "Preemptible": args.preemptible,
    }
    if args.resource_queue_name:
        conf["ResourceQueueName"] = args.resource_queue_name
    else:
        conf["ResourceQueueID"] = args.resource_queue_id
    if args.priority is not None:
        conf["Priority"] = args.priority
    if args.active_deadline_seconds:
        conf["ActiveDeadlineSeconds"] = args.active_deadline_seconds
    envs = build_envs(args)
    if envs:
        conf["Envs"] = envs
    storages = storage_config(args)
    if storages:
        conf["Storages"] = storages
    if args.user_code_path:
        conf["UserCodePath"] = args.user_code_path
    if args.remote_mount_code_path:
        conf["RemoteMountCodePath"] = args.remote_mount_code_path
    return conf


def validate_for_submit(args):
    missing = []
    for field, value in (
        ("VOLC_IMAGE_URL/--image-url", args.image_url),
        (
            "VOLC_RESOURCE_QUEUE_NAME/--resource-queue-name or "
            "VOLC_RESOURCE_QUEUE_ID/--resource-queue-id",
            args.resource_queue_name or args.resource_queue_id,
        ),
        ("VOLC_FLAVOR/--flavor", args.flavor),
    ):
        if not value:
            missing.append(field)
    if missing:
        raise SystemExit("Missing required submit fields: " + ", ".join(missing))


def submit(args, output):
    validate_for_submit(args)
    if shutil.which(args.volc_bin) is None and not Path(args.volc_bin).exists():
        raise SystemExit(f"Cannot find volc binary: {args.volc_bin}")
    env = os.environ.copy()
    if Path(args.volc_bin).exists():
        mlp_bin_dir = str(Path(args.volc_bin).resolve().parent)
    else:
        mlp_bin_dir = "/root/.volc/bin"
    env["PATH"] = f"{mlp_bin_dir}:{env.get('PATH', '')}"
    command = [args.volc_bin, "ml_task", "submit", "--conf", str(output), "--local_diff", args.local_diff]
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def main():
    parser = argparse.ArgumentParser(description="Submit NeoVerse jobs with Volcengine MLP CLI.")
    parser.add_argument("preset", choices=sorted(PRESET_COMMANDS))
    parser.add_argument("--name", default=None)
    parser.add_argument("--description", default="NeoVerse job submitted from repo CLI.")
    parser.add_argument("--resource-queue-name", default=env_value("VOLC_RESOURCE_QUEUE_NAME", ""))
    parser.add_argument("--resource-queue-id", default=env_value("VOLC_RESOURCE_QUEUE_ID", ""))
    parser.add_argument("--image-url", default=env_value("VOLC_IMAGE_URL", ""))
    parser.add_argument("--framework", default=env_value("VOLC_FRAMEWORK", "PyTorchDDP"))
    parser.add_argument("--role-name", default=env_value("VOLC_ROLE_NAME", "worker"))
    parser.add_argument("--replicas", type=int, default=int(env_value("VOLC_REPLICAS", env_value("CACHE_NNODES", "1"))))
    parser.add_argument("--flavor", default=env_value("VOLC_FLAVOR", env_value("VOLC_INSTANCE_TYPE_ID", "")))
    parser.add_argument("--priority", type=int, default=None if env_value("VOLC_PRIORITY") is None else int(env_value("VOLC_PRIORITY")))
    parser.add_argument("--preemptible", nargs="?", const=True, type=str_to_bool, default=bool_env("VOLC_PREEMPTIBLE", False))
    parser.add_argument("--active-deadline-seconds", default=env_value("VOLC_ACTIVE_DEADLINE_SECONDS", ""))
    parser.add_argument("--access-type", default=env_value("VOLC_ACCESS_TYPE", "Queue"))
    parser.add_argument("--vepfs-id", default=env_value("VOLC_VEPFS_ID", ""))
    parser.add_argument("--vepfs-sub-path", default=env_value("VOLC_VEPFS_SUB_PATH", ""))
    parser.add_argument("--vepfs-mount-path", default=env_value("VOLC_VEPFS_MOUNT_PATH", "/root/vepfs"))
    parser.add_argument("--code-dir", default=env_value("VOLC_CODE_DIR", "/root/vepfs/diffsynth-dev/papers/neoverse/code"))
    parser.add_argument("--user-code-path", default=env_value("VOLC_USER_CODE_PATH", ""))
    parser.add_argument("--remote-mount-code-path", default=env_value("VOLC_REMOTE_MOUNT_CODE_PATH", ""))
    parser.add_argument("--command", default=None, help="Override preset command after cd CODE_DIR.")
    parser.add_argument("--env", action="append", default=[], help="Extra runtime env in KEY=VALUE form.")
    parser.add_argument("--output", default=None, help="Where to write generated MLP task YAML.")
    parser.add_argument("--volc-bin", default=env_value("VOLC_BIN", DEFAULT_MLP_VOLC))
    parser.add_argument("--local-diff", default=env_value("VOLC_LOCAL_DIFF", "off"))
    parser.add_argument("--submit", action="store_true", help="Call volc ml_task submit.")
    parser.add_argument("--dry-run", action="store_true", default=bool_env("DRY_RUN", False), help="Generate YAML and print submit command only.")
    args = parser.parse_args()

    if args.name is None:
        args.name = f"neoverse-{args.preset.replace('_', '-')}"

    conf = build_conf(args)
    output = resolve_path(args.output) if args.output else DEFAULT_JOB_ROOT / f"{args.preset}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(conf, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"[volc] wrote {output}")
    print("[volc] submit command:")
    print(f"PATH=/root/.volc/bin:$PATH {args.volc_bin} ml_task submit --conf {output} --local_diff {args.local_diff}")

    if args.dry_run or not args.submit:
        return
    submit(args, output)


if __name__ == "__main__":
    main()
