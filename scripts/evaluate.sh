#!/usr/bin/env bash
# Evaluate every clip in the released HOT3D and/or ARCTIC test split.
# Usage: bash scripts/evaluate.sh [hot3d|arctic|all]
set -euo pipefail

DATASET=${1:-all}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
OUT_ROOT=${OUT_ROOT:-outputs}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if (( $# > 1 )); then
    echo "Usage: bash scripts/evaluate.sh [hot3d|arctic|all]" >&2
    exit 2
fi
case "$DATASET" in
    hot3d|arctic) datasets=("$DATASET") ;;
    all)
        if [[ -n ${OUT_DIR:-} || -n ${SIGMA:-} ]]; then
            echo "OUT_DIR and SIGMA apply to one benchmark; select hot3d or arctic, or use OUT_ROOT for both." >&2
            exit 2
        fi
        datasets=(hot3d arctic)
        ;;
    *)
        echo "Usage: bash scripts/evaluate.sh [hot3d|arctic|all]" >&2
        exit 2
        ;;
esac

for dataset in "${datasets[@]}"; do
    case "$dataset" in
        hot3d) count=244; sigma=26.18328668456495,3.9939303136949373 ;;
        arctic) count=155; sigma=29.102446586693446,11.377515771894934 ;;
    esac
    ckpt="save/dit_$dataset/model.pt"
    split="data/$dataset/splits.json"
    out="${OUT_DIR:-$OUT_ROOT/benchmark_${dataset}${count}}"
    for file in "$ckpt" "${ckpt%/*}/args.json" save/qn/model.pt save/qn/args.json; do
        if [[ ! -f "$file" ]]; then
            echo "Missing $file. Run: bash scripts/download_pretrained.sh" >&2
            exit 1
        fi
    done
    for file in data_loaders/mano_models/MANO_LEFT.pkl data_loaders/mano_models/MANO_RIGHT.pkl; do
        if [[ ! -f "$file" ]]; then
            echo "Missing $file. Download the MANO models as described in README.md." >&2
            exit 1
        fi
    done
    "$PYTHON" - "$split" "$count" "${ckpt%/*}/args.json" "$out/_test_clips.txt" <<'PY'
import json
from pathlib import Path
import sys

split_path, expected, config_path, output_path = sys.argv[1:]
split_path = Path(split_path)
if not split_path.is_file():
    raise SystemExit(f"Missing {split_path}. Run: bash scripts/download_data.sh")
ids = json.loads(split_path.read_text())["test"]
if len(ids) != int(expected) or len(set(ids)) != len(ids):
    raise SystemExit(f"{split_path} must contain {expected} unique test clip IDs")
clips = [f"clip-{int(c):06d}" for c in ids]
config = json.loads(Path(config_path).read_text())
missing = []
for key in ("data_dir", "wilor_dir", "visual_feat_dir", "scene_feat_dir",
            "da3_train_dir", "depth_signal_dir"):
    directory = Path(config[key])
    missing.extend(directory / f"{c}.npz" for c in clips if not (directory / f"{c}.npz").is_file())
for path in [Path(config["normalizer_dir"]) / "mean.pt",
             Path(config["normalizer_dir"]) / "std.pt"]:
    if not path.is_file():
        missing.append(path)
if missing:
    raise SystemExit(f"Missing benchmark cache: {missing[0]} ({len(missing)} missing files). "
                     "Run: bash scripts/download_data.sh")
output = Path(output_path)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("".join(f"{c}\n" for c in clips))
print(f"{len(clips)} test clips from {split_path}")
PY
    "$PYTHON" -m sample.infer_clips \
        --ckpt "$ckpt" --qn_ckpt save/qn/model.pt \
        --qn_sigma_override "${SIGMA:-$sigma}" \
        --depth_signal_dir "data/$dataset/depth_signal" \
        --rgb_dir "data/$dataset/rgb" \
        --clips_file "$out/_test_clips.txt" \
        --seed 42 --n_steps 20 --device "$DEVICE" --no_rrd \
        --out_dir "$out"
    echo "Metrics: $out/metrics.json"
done
