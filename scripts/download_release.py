"""Download immutable release snapshots and verify every selected payload file."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def selected_files(siblings, patterns):
    return [item for item in siblings if not patterns or
            any(fnmatch.fnmatchcase(item.rfilename, pattern) for pattern in patterns)]


def verify_snapshot(directory, siblings):
    """Compare local bytes with LFS SHA-256 or Git blob SHA-1 at the pinned commit."""
    directory = Path(directory)
    count = total = 0
    errors = []
    for item in siblings:
        name = PurePosixPath(item.rfilename)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe repository path: {name}")
        path = directory / str(name)
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        size = path.stat().st_size
        if size != item.size:
            errors.append(f"size mismatch: {name}")
            continue
        sha256 = hashlib.sha256()
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                sha256.update(chunk)
                blob.update(chunk)
        expected = item.lfs.sha256 if item.lfs else item.blob_id
        actual = sha256.hexdigest() if item.lfs else blob.hexdigest()
        if not expected or actual != expected:
            errors.append(f"hash mismatch: {name}")
        count += 1
        total += size
    if errors:
        raise RuntimeError(f"Release verification failed ({len(errors)} files): " +
                           "; ".join(errors[:10]) + ". Re-run the download with --force-download.")
    if not count:
        raise RuntimeError("No release payload files matched this snapshot")
    return count, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("model", "dataset"))
    parser.add_argument("repo_id", nargs="?", help="Alternate repo; requires an explicit --revision")
    parser.add_argument("--revision", help="Immutable 40-character Hub commit SHA")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing files against pinned Hub metadata")
    parser.add_argument("--force-download", action="store_true", help="Repair corrupted local files by downloading again")
    parser.add_argument("--root", default=".", help="Release checkout root (default: current directory)")
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parents[1] / "release/hub-revisions.json"
    config = json.loads(config_path.read_text())[args.kind]
    repo_id = args.repo_id or config["repo_id"]
    if repo_id != config["repo_id"] and not args.revision:
        parser.error("An alternate repository requires --revision COMMIT_SHA")
    revision = args.revision or config["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("--revision must be an immutable 40-character lowercase commit SHA")
    if args.verify_only and args.force_download:
        parser.error("--verify-only and --force-download are mutually exclusive")
    # Imported lazily so metadata/hash tests have no Hub dependency or credentials.
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().repo_info(repo_id, repo_type=args.kind, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Hub returned unexpected revision {info.sha}, expected {revision}")
    directory = Path(args.root) / config["local_dir"]
    print(f"{repo_id}@{revision} -> {directory}", flush=True)
    if not args.verify_only:
        snapshot_download(repo_id, repo_type=args.kind, revision=revision,
                          local_dir=str(directory), allow_patterns=config["allow_patterns"],
                          force_download=args.force_download)
    count, total = verify_snapshot(directory, selected_files(info.siblings, config["allow_patterns"]))
    print(f"Verified {count} files ({total:,} bytes) at {revision}", flush=True)


if __name__ == "__main__":
    main()
