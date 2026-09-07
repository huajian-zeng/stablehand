"""Content-addressed, crash-safe per-clip evaluation caches (no ML dependencies)."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

CACHE_SCHEMA = 1


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    if not path:
        return None
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}


def source_identity(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256_file(path)
            for folder in ("data_loaders", "diffusion", "eval", "model", "sample", "utils")
            for path in sorted((root / folder).rglob("*.py"))}


def clip_input_identity(clip, config, rgb_dir=None):
    files = {}
    for key in ("data_dir", "wilor_dir", "visual_feat_dir", "scene_feat_dir",
                "depth_signal_dir", "da3_train_dir"):
        directory = config.get(key)
        files[key] = file_identity(Path(directory) / f"{clip}.npz") if directory else None
    files["rgb"] = file_identity(Path(rgb_dir) / f"{clip}.mp4") if rgb_dir else None
    return files


def _valid_metrics(metrics):
    if not isinstance(metrics, dict) or not {"left", "right", "MRRPE", "MRRPE_n_frames"} <= metrics.keys():
        return False
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    for side in ("left", "right"):
        hand = metrics[side]
        if hand is None:
            continue
        if not isinstance(hand, dict) or type(hand.get("n_frames")) is not int or hand["n_frames"] < 2:
            return False
        if not all(finite(hand.get(key)) for key in ("W-MPJPE", "WA-MPJPE", "PA-MPJPE")):
            return False
        if "AccErr" not in hand or (hand["AccErr"] is not None and not finite(hand["AccErr"])):
            return False
    return (type(metrics["MRRPE_n_frames"]) is int and metrics["MRRPE_n_frames"] >= 0
            and (metrics["MRRPE"] is None or finite(metrics["MRRPE"])))


def read_cached_metrics(path, clip, identity, artifacts):
    """Return (metrics, reason); any stale/incomplete cache is a miss, never a score."""
    try:
        with open(path) as handle:
            cached = json.load(handle)
        if not isinstance(cached, dict) or cached.get("schema_version") != CACHE_SCHEMA:
            return None, "legacy/unknown cache schema"
        if cached.get("clip") != clip or cached.get("fingerprint") != identity:
            return None, "run configuration or input content changed"
        metrics = cached.get("metrics")
        if not _valid_metrics(metrics) or cached.get("metrics_sha256") != fingerprint(metrics):
            return None, "invalid or modified metrics"
        recorded = cached.get("artifacts")
        if not isinstance(recorded, dict):
            return None, "missing output fingerprints"
        for name, output in artifacts.items():
            if not Path(output).is_file() or recorded.get(name) != sha256_file(output):
                return None, f"missing or modified {name} output"
        return metrics, "matching fingerprint"
    except (OSError, ValueError, TypeError):
        return None, "missing or unreadable cache"


def write_cached_metrics(path, clip, identity, metrics, artifacts, provenance):
    if not _valid_metrics(metrics):
        raise ValueError(f"Refusing to cache invalid metrics for {clip}")
    record = {**provenance, "schema_version": CACHE_SCHEMA, "clip": clip,
              "fingerprint": identity, "metrics": metrics, "metrics_sha256": fingerprint(metrics),
              "artifacts": {key: sha256_file(value) for key, value in artifacts.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
