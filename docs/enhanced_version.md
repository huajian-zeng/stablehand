# Enhanced Version

This release extends the StableHand method described in the
[paper](https://arxiv.org/abs/2605.18553), retaining its quality-aware flow-matching
framework for world-space dual-hand motion estimation. The main updates are:

- **Camera and scene geometry.** VGGT-Omega replaces Depth Anything 3 to provide
  camera poses and scene features that condition world-space motion estimation.
- **Residual wrist motion.** Instead of directly predicting wrist translation,
  the motion prior learns corrections to a smoothed trajectory derived from WiLoR
  in the gravity-aligned world frame.
- **Visual conditioning.** Each hand's WiLoR visual features are projected into a
  compact representation and fused with its motion observation.
- **Temporal motion supervision.** Joint-velocity and joint-acceleration losses
  replace the auxiliary wrist-translation loss, encouraging coherent motion
  across the hand while retaining the existing smoothness objective.
- **Depth-aware quality estimation.** The Quality Network additionally uses local
  depth cues around each hand from VGGT-Omega to assess the reliability of the
  hand proposals.

## Reported results

| Dataset | Version | PA-MPJPE (mm) | W-MPJPE (mm) | WA-MPJPE (mm) | AccErr (m/s²) |
| --- | --- | ---: | ---: | ---: | ---: |
| HOT3D | Paper | 4.02 | 57.83 | 21.02 | 3.83 |
| HOT3D | Enhanced | 3.39 | 45.19 | 18.57 | 2.82 |
| ARCTIC | Paper | 8.07 | 124.59 | 32.66 | 4.67 |
| ARCTIC | Enhanced | 7.57 | 92.52 | 30.43 | 3.36 |
