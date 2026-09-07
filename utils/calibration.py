"""Select calibration clips without including benchmark test or inference clips."""

import json
from pathlib import Path
import re


def clip_names(values, label="clips"):
    """Normalize numeric IDs or clip-prefixed IDs, rejecting empty or duplicate lists."""
    names = []
    for value in values:
        match = re.fullmatch(r"(?:clip-)?([0-9]{1,6})", str(value).strip())
        if match is None:
            raise ValueError(f"Invalid {label} ID: {value!r}")
        names.append(f"clip-{int(match.group(1)):06d}")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{label} must be nonempty and contain unique clip IDs")
    return names


def read_clip_list(path):
    """Read newline-separated clip IDs, ignoring blank lines and full-line comments."""
    return clip_names([line.strip() for line in Path(path).read_text().splitlines()
                       if line.strip() and not line.strip().startswith("#")], str(path))


def select_calibration_clips(data_dir, inference_clips, *, explicit=None, count=30,
                             test_split_file=None):
    """Return selected IDs and the complete test-split path used to exclude test data."""
    split_path = (Path(test_split_file) if test_split_file is not None
                  else Path(data_dir).parent / "splits.json")
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Calibration requires a complete test split: {split_path}. "
            "Pass --test_split_file, or use the documented --qn_sigma_override.")
    spec = json.loads(split_path.read_text())
    if not isinstance(spec, dict) or not isinstance(spec.get("test"), list):
        raise ValueError(f"{split_path} must contain a test clip list")
    test_clips = set(clip_names(spec["test"], "test split"))
    excluded = test_clips | set(clip_names(inference_clips, "inference clips"))
    if explicit is not None:
        selected = clip_names(explicit, "calibration clips")
        overlap = sorted(excluded.intersection(selected))
        if overlap:
            raise ValueError(f"Calibration clips overlap test/inference clips: {', '.join(overlap)}")
    else:
        if count <= 0:
            raise ValueError("Automatic calibration requires a positive clip count")
        candidates = sorted(path.stem for path in Path(data_dir).glob("clip-*.npz"))
        candidates = clip_names(candidates, "cached clips") if candidates else []
        available = [name for name in candidates if name not in excluded]
        if len(available) < count:
            raise ValueError(
                f"Need {count} non-test calibration clips, found {len(available)}. "
                "The release package contains test clips only. Use the documented "
                "--qn_sigma_override or provide separate calibration caches and a "
                "complete --test_split_file.")
        selected = clip_names(available[:count], "calibration clips")
    for name in selected:
        path = Path(data_dir) / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing calibration clip: {path}")
    return selected, str(split_path)
