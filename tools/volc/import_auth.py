import argparse
import os
import re
import tempfile
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ACCESS_KEY_FILE = CODE_DIR.parent / "AccessKey.txt"
DEFAULT_VOLC_DIR = Path.home() / ".volc"


def parse_access_key_file(path):
    text = Path(path).read_text(encoding="utf-8")
    values = {}
    aliases = {
        "accesskeyid": "access-key",
        "accesskey": "access-key",
        "access_key_id": "access-key",
        "access key id": "access-key",
        "secretaccesskey": "secret-key",
        "secretkey": "secret-key",
        "secret_access_key": "secret-key",
        "secret access key": "secret-key",
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            name, value = line.split(":", 1)
        elif "=" in line:
            name, value = line.split("=", 1)
        else:
            continue
        key = aliases.get(re.sub(r"[\s_-]+", " ", name.strip()).lower())
        if key is None:
            key = aliases.get(re.sub(r"[\s_-]+", "", name.strip()).lower())
        if key is None:
            continue
        values[key] = re.sub(r"\s+", "", value.strip())

    missing = [name for name in ("access-key", "secret-key") if not values.get(name)]
    if missing:
        raise SystemExit(f"Cannot parse {', '.join(missing)} from {path}")
    return values["access-key"], values["secret-key"]


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def import_profile(args):
    access_key, secret_key = parse_access_key_file(args.access_key_file)
    volc_dir = Path(args.volc_dir)
    credentials_file = volc_dir / "credentials"
    config_file = volc_dir / "config"
    write_text_atomic(
        credentials_file,
        f"[{args.profile}]\naccess_key_id={access_key}\nsecret_access_key={secret_key}\n",
    )
    write_text_atomic(config_file, f"[{args.profile}]\nregion={args.region}\n")
    print(
        f"[volc-auth] imported profile={args.profile} region={args.region} "
        f"credentials={credentials_file} config={config_file}"
    )


def main():
    parser = argparse.ArgumentParser(description="Import Volcengine AK/SK into local MLP CLI config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="Import AccessKey.txt into ~/.volc/credentials.")
    import_parser.add_argument("access_key_file", nargs="?", default=str(DEFAULT_ACCESS_KEY_FILE))
    import_parser.add_argument("--profile", default=os.environ.get("VOLC_PROFILE", "default"))
    import_parser.add_argument("--region", default=os.environ.get("VOLC_REGION", "cn-beijing"))
    import_parser.add_argument("--volc-dir", default=os.environ.get("VOLC_DIR", str(DEFAULT_VOLC_DIR)))
    import_parser.set_defaults(func=import_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
