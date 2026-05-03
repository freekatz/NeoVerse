#!/usr/bin/env python3
"""Robust downloader for DL3DV-10K subsets.

Official references:
  - Project: https://dl3dv-10k.github.io/DL3DV-10K/
  - Code/data instructions: https://github.com/DL3DV-10K/Dataset
  - Example repo: https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P

DL3DV repositories are gated on Hugging Face. Accept the terms on the dataset
card first, then pass a token via --token or HF_TOKEN.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


META_URL = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv"
META_URLS = (
    META_URL,
    "https://cdn.jsdelivr.net/gh/DL3DV-10K/Dataset@main/cache/DL3DV-valid.csv",
    "https://gh-proxy.com/https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv",
)
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
SUBSETS = tuple(f"{idx}K" for idx in range(1, 12))
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)

RESOLUTION_REPOS = {
    "480P": "DL3DV/DL3DV-ALL-480P",
    "960P": "DL3DV/DL3DV-ALL-960P",
    "2K": "DL3DV/DL3DV-ALL-2K",
    "4K": "DL3DV/DL3DV-ALL-4K",
}
VIDEO_REPO = "DL3DV/DL3DV-ALL-video"
COLMAP_REPO = "DL3DV/DL3DV-ALL-ColmapCache"


@dataclass(frozen=True)
class DownloadItem:
    repo: str
    rel_path: str
    scene_hash: str
    batch: str
    size: int | None = None


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None or num_bytes <= 0:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def clear_proxy_env() -> list[str]:
    removed = []
    for key in PROXY_ENV_KEYS:
        if key in os.environ:
            removed.append(key)
            os.environ.pop(key, None)
    return removed


def import_hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency huggingface_hub. Install with:\n"
            "  python -m pip install -U huggingface_hub hf_xet"
        ) from exc
    return HfApi, hf_hub_download


def resolve_token(args: argparse.Namespace) -> str | None:
    if args.token:
        return args.token
    for env_name in args.token_env.split(","):
        env_name = env_name.strip()
        if env_name and os.environ.get(env_name):
            return os.environ[env_name]
    return None


def repo_for(file_type: str, resolution: str) -> str:
    if file_type == "images+poses":
        return RESOLUTION_REPOS[resolution]
    if file_type == "video":
        return VIDEO_REPO
    if file_type == "colmap_cache":
        return COLMAP_REPO
    raise ValueError(f"Unsupported file_type: {file_type}")


def rel_path_for(file_type: str, resolution: str, batch: str, scene_hash: str) -> tuple[str, str]:
    repo = repo_for(file_type, resolution)
    if file_type == "video":
        return repo, f"{batch}/{scene_hash}/video.mp4"
    return repo, f"{batch}/{scene_hash}.zip"


def download_text(urls: tuple[str, ...], output: Path, retries: int = 2) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=45) as response:
                    output.write_bytes(response.read())
                return
            except Exception as exc:  # noqa: BLE001 - preserve original network error text.
                last_error = exc
                print(f"[metadata] retry {attempt}/{retries} from {url}: {exc}", flush=True)
                time.sleep(attempt * 2)
    raise RuntimeError(f"failed to download metadata: {last_error}")


def load_metadata(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv)
    else:
        metadata_path = Path(args.out_dir) / ".metadata" / "DL3DV-valid.csv"
        if not metadata_path.exists() or args.refresh_metadata:
            print(f"[metadata] downloading {META_URL}")
            download_text(META_URLS, metadata_path)

    rows: list[dict[str, str]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "hash" not in (reader.fieldnames or []) or "batch" not in (reader.fieldnames or []):
            raise ValueError(f"{metadata_path} must contain hash and batch columns")
        for row in reader:
            scene_hash = (row.get("hash") or "").strip()
            batch = (row.get("batch") or "").strip()
            if scene_hash and batch:
                rows.append({"hash": scene_hash, "batch": batch})
    if not rows:
        raise ValueError(f"no usable rows found in {metadata_path}")
    return rows


def select_rows(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if args.scene_hash:
        matches = [row for row in rows if row["hash"] == args.scene_hash]
        if not matches:
            raise ValueError(f"scene hash not found in DL3DV metadata: {args.scene_hash}")
        selected = matches
    elif args.subset == "all":
        selected = rows
    else:
        selected = [row for row in rows if row["batch"] == args.subset]
        if not selected:
            raise ValueError(f"subset {args.subset} has no rows in DL3DV metadata")

    if args.max_files is not None:
        selected = selected[: int(args.max_files)]
    return selected


def attach_sizes(
    args: argparse.Namespace,
    items: list[DownloadItem],
    token: str | None,
) -> list[DownloadItem]:
    if args.skip_remote_size:
        return items

    HfApi, _ = import_hf()
    api = HfApi(endpoint=args.hf_endpoint)
    by_repo: dict[str, list[DownloadItem]] = {}
    for item in items:
        by_repo.setdefault(item.repo, []).append(item)

    sized: dict[tuple[str, str], int] = {}
    for repo, repo_items in by_repo.items():
        print(f"[plan] listing remote metadata for {repo}", flush=True)
        try:
            info = api.repo_info(repo, repo_type="dataset", token=token, files_metadata=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[plan] warning: failed to list {repo}: {exc}", flush=True)
            continue
        wanted = {item.rel_path for item in repo_items}
        for sibling in info.siblings:
            if sibling.rfilename in wanted:
                sized[(repo, sibling.rfilename)] = int(sibling.size or 0)

    return [
        DownloadItem(
            repo=item.repo,
            rel_path=item.rel_path,
            scene_hash=item.scene_hash,
            batch=item.batch,
            size=sized.get((item.repo, item.rel_path)),
        )
        for item in items
    ]


def build_items(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    token: str | None,
) -> list[DownloadItem]:
    selected = select_rows(args, rows)
    items = []
    for row in selected:
        repo, rel_path = rel_path_for(args.file_type, args.resolution, row["batch"], row["hash"])
        items.append(DownloadItem(repo=repo, rel_path=rel_path, scene_hash=row["hash"], batch=row["batch"]))
    return attach_sizes(args, items, token)


def plan_payload(args: argparse.Namespace, items: list[DownloadItem]) -> dict:
    known_size = sum(item.size or 0 for item in items)
    unknown_size_count = sum(1 for item in items if not item.size)
    return {
        "file_type": args.file_type,
        "resolution": args.resolution,
        "subset": args.subset,
        "scene_hash": args.scene_hash,
        "count": len(items),
        "known_total_bytes": known_size,
        "known_total_human": human_size(known_size),
        "unknown_size_count": unknown_size_count,
        "largest_human": human_size(max((item.size or 0 for item in items), default=0)),
        "repos": sorted({item.repo for item in items}),
        "items": [
            {
                "repo": item.repo,
                "path": item.rel_path,
                "batch": item.batch,
                "hash": item.scene_hash,
                "size": item.size,
                "size_human": human_size(item.size),
            }
            for item in items
        ],
    }


def print_plan(payload: dict) -> None:
    print("")
    print("DL3DV download plan")
    print("-" * 80)
    print(f"file_type:       {payload['file_type']}")
    print(f"resolution:      {payload['resolution']}")
    print(f"subset:          {payload['subset']}")
    if payload["scene_hash"]:
        print(f"scene_hash:      {payload['scene_hash']}")
    print(f"repos:           {', '.join(payload['repos'])}")
    print(f"files:           {payload['count']}")
    print(f"known total:     {payload['known_total_human']}")
    print(f"largest file:    {payload['largest_human']}")
    print(f"unknown sizes:   {payload['unknown_size_count']}")
    print("-" * 80)
    for item in payload["items"][:20]:
        print(f"{item['size_human']:>10}  {item['repo']}/{item['path']}")
    if payload["count"] > 20:
        print(f"... {payload['count'] - 20} more files")
    print("")


def local_file_for(args: argparse.Namespace, item: DownloadItem) -> Path:
    return Path(args.out_dir) / item.rel_path


def zip_extract_marker(archive: Path) -> Path:
    return archive.with_suffix("")


def local_complete(args: argparse.Namespace, item: DownloadItem) -> bool:
    target = local_file_for(args, item)
    if args.extract and target.suffix == ".zip" and zip_extract_marker(target).exists():
        return True
    if not target.exists():
        return False
    if item.size and target.stat().st_size != item.size:
        return False
    if args.verify_zip and target.suffix == ".zip":
        return zip_is_valid(target)
    return True


def zip_is_valid(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as handle:
            return handle.testzip() is None
    except zipfile.BadZipFile:
        return False


def verify_download(args: argparse.Namespace, item: DownloadItem) -> None:
    target = local_file_for(args, item)
    if not target.exists():
        raise RuntimeError(f"downloaded file missing: {target}")
    if item.size and target.stat().st_size != item.size:
        raise RuntimeError(f"size mismatch for {target}: {target.stat().st_size} != {item.size}")
    if args.verify_zip and target.suffix == ".zip" and not zip_is_valid(target):
        raise RuntimeError(f"zip verification failed: {target}")


def extract_archive(args: argparse.Namespace, item: DownloadItem) -> None:
    archive = local_file_for(args, item)
    if archive.suffix != ".zip":
        return
    marker = zip_extract_marker(archive)
    if marker.exists() and not args.force_extract:
        print(f"[extract] skip existing {marker}")
        if args.delete_archive_after_extract and archive.exists():
            archive.unlink()
        return
    extract_root = archive.parent if args.extract_dir is None else Path(args.extract_dir) / archive.parent.relative_to(args.out_dir)
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive} -> {extract_root}", flush=True)
    with zipfile.ZipFile(archive, "r") as handle:
        handle.extractall(extract_root)
    if args.delete_archive_after_extract:
        archive.unlink()


def cleanup_cache(args: argparse.Namespace) -> None:
    candidates = [Path(args.out_dir) / ".cache"]
    if args.cache_dir:
        candidates.append(Path(args.cache_dir))
    for candidate in candidates:
        if candidate.exists():
            print(f"[cleanup] removing {candidate}", flush=True)
            shutil.rmtree(candidate)


def download_one(args: argparse.Namespace, item: DownloadItem, token: str | None) -> None:
    _, hf_hub_download = import_hf()
    target = local_file_for(args, item)
    target.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, args.max_retries + 1):
        try:
            print(f"[download] {item.repo}/{item.rel_path} ({human_size(item.size)})", flush=True)
            hf_hub_download(
                repo_id=item.repo,
                repo_type="dataset",
                filename=item.rel_path,
                token=token,
                revision=args.revision,
                endpoint=args.hf_endpoint,
                local_dir=args.out_dir,
                cache_dir=args.cache_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            verify_download(args, item)
            if args.extract:
                extract_archive(args, item)
            if args.clean_cache:
                cleanup_cache(args)
            return
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt >= args.max_retries:
                raise
            print(f"[download] retry {attempt}/{args.max_retries}: {exc}", flush=True)
            time.sleep(5 * attempt)


def run_download(args: argparse.Namespace, items: list[DownloadItem], token: str | None) -> int:
    failed: list[str] = []
    skipped = 0
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item.rel_path}", flush=True)
        try:
            if local_complete(args, item) and not args.force:
                print(f"[skip] already complete: {local_file_for(args, item)}", flush=True)
                skipped += 1
                continue
            download_one(args, item, token)
        except Exception as exc:  # noqa: BLE001
            print(f"[failed] {item.rel_path}: {exc}", flush=True)
            failed.append(item.rel_path)
            if args.stop_on_error:
                break

    summary = {
        "total": len(items),
        "skipped": skipped,
        "failed": failed,
        "success": len(failed) == 0,
    }
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.artifacts_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failed:
        (Path(args.artifacts_dir) / "failed.txt").write_text("\n".join(failed) + "\n", encoding="utf-8")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DL3DV-10K files from the official gated Hugging Face repos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default="data/DL3DV-10K", help="download root")
    parser.add_argument("--artifacts-dir", default="outputs/download_dl3dv", help="plan/summary output")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache dir; defaults to local_dir cache")
    parser.add_argument("--resolution", choices=tuple(RESOLUTION_REPOS), default="480P")
    parser.add_argument("--file-type", choices=("images+poses", "video", "colmap_cache"), default="images+poses")
    parser.add_argument("--subset", default="1K", help="1K..11K or all")
    parser.add_argument("--scene-hash", default="", help="download one scene hash from DL3DV-valid.csv")
    parser.add_argument("--max-files", type=int, default=None, help="limit file count for smoke tests")
    parser.add_argument("--metadata-csv", default="", help="local DL3DV-valid.csv; otherwise downloaded")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--token", default="", help="Hugging Face token; prefer env vars")
    parser.add_argument("--token-env", default="HF_TOKEN,HUGGINGFACE_TOKEN,MODEL_DOWNLOAD_TOKEN")
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--use-official-hf", action="store_true", help="use https://huggingface.co instead of hf-mirror")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--keep-proxy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-json", default="", help="optional explicit plan json path")
    parser.add_argument("--skip-remote-size", action="store_true", help="do not query HF file sizes")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--verify-zip", action="store_true")
    parser.add_argument("--extract", action="store_true", help="extract downloaded zip files")
    parser.add_argument("--extract-dir", default=None, help="alternate extraction root")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--delete-archive-after-extract", action="store_true")
    parser.add_argument("--clean-cache", action="store_true")
    parser.add_argument("--force", action="store_true", help="redownload even if local file looks complete")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if args.use_official_hf:
        args.hf_endpoint = "https://huggingface.co"
    if args.subset != "all" and args.subset not in SUBSETS:
        raise SystemExit(f"--subset must be one of {', '.join(SUBSETS)} or all")
    return args


def main() -> int:
    args = parse_args()
    if not args.keep_proxy:
        removed = clear_proxy_env()
        if removed:
            print("[env] disabled proxy vars: " + ", ".join(sorted(removed)))
    os.environ["HF_ENDPOINT"] = args.hf_endpoint

    token = resolve_token(args)
    if token is None and not args.dry_run:
        print(
            "warning: no HF token found. DL3DV is gated; run `huggingface-cli login` "
            "or set HF_TOKEN after accepting the dataset terms.",
            file=sys.stderr,
        )

    rows = load_metadata(args)
    items = build_items(args, rows, token)
    payload = plan_payload(args, items)
    print_plan(payload)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.plan_json) if args.plan_json else artifacts_dir / "plan.json"
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[plan] wrote {plan_path}")

    if args.dry_run:
        return 0
    return run_download(args, items, token)


if __name__ == "__main__":
    raise SystemExit(main())
