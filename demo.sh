#!/bin/bash
# StableHand quick-start demo: benchmark inference on HOT3D-Clips.
#
# Prerequisites (see README):
#   save/        checkpoints        -> bash scripts/download_pretrained.sh
#   data/        benchmark caches   -> bash scripts/download_data.sh
#   MANO models  data_loaders/mano_models/MANO_{LEFT,RIGHT}.pkl (manual, gated)
#
# Usage:
#   bash demo.sh              # one demo clip: metrics, a world-frame video, a Rerun (.rrd) recording
#   CLIP=clip-002026 bash demo.sh   # any other packaged example clip
#   BENCH=full bash demo.sh   # reproduce the full 244-clip benchmark (metrics only)
set -e

BENCH=${BENCH:-demo}
OUT=${OUT:-outputs/demo}
DEVICE=${DEVICE:-cuda:0}
# Deployment sigma for q = exp(-error/sigma), calibrated once with the released
# Quality Network on 30 held-out (non-test) HOT3D clips. Pinning it keeps the
# benchmark deterministic. Recalibration needs separate non-test conditioning
# caches and the complete test split; the released test-only caches are insufficient.
SIGMA=${SIGMA:-26.18328668456495,3.9939303136949373}

for f in save/dit_hot3d/model.pt save/qn/model.pt; do
    [[ -f "$f" ]] || { echo "Missing checkpoint: $f"; echo "Run: bash scripts/download_pretrained.sh"; exit 1; }
done
[[ -d data/hot3d ]] || { echo "Missing data/hot3d."; echo "Run: bash scripts/download_data.sh"; exit 1; }
for f in data_loaders/mano_models/MANO_LEFT.pkl data_loaders/mano_models/MANO_RIGHT.pkl; do
    [[ -f "$f" ]] || { echo "Missing $f."; echo "Download MANO from https://mano.is.tue.mpg.de (see README)."; exit 1; }
done

if [[ "$BENCH" == "full" ]]; then
    OUT_DIR="${OUT_FULL:-outputs/benchmark_hot3d244}" DEVICE="$DEVICE" SIGMA="$SIGMA" \
        bash scripts/evaluate.sh hot3d
    exit 0
else
    CLIP=${CLIP:-clip-002736}
    python -m sample.infer_clips \
        --ckpt save/dit_hot3d/model.pt --qn_ckpt save/qn/model.pt \
        --qn_sigma_override "$SIGMA" \
        --depth_signal_dir data/hot3d/depth_signal \
        --clips "$CLIP" \
        --seed 42 --n_steps 20 --device "$DEVICE" \
        --out_dir "$OUT"
    # World-frame 3-D video: ground truth next to the prediction, from one fixed
    # viewpoint, which is where world-space drift is actually visible.
    python -m visualize.render_world3d \
        --pred "$OUT/$CLIP.npz" --gt_dir data/hot3d/clips_gt \
        --ckpt save/dit_hot3d/model.pt \
        --out "$OUT/${CLIP}_world3d.mp4"
    echo
    echo "== Done."
    echo "   World-frame video:  $OUT/${CLIP}_world3d.mp4"
    echo "   Interactive viewer: rerun $OUT/$CLIP.rrd"
fi
echo "Metrics: $OUT/metrics.json"
