"""Window-relative camera poses for motion inference."""

import torch

from data_loaders.geometry import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)


def camera_to_window_relative(cam_feats):
    """Convert (B, T, 9) device-to-world pose to window-relative pose."""
    R_t = rotation_6d_to_matrix(cam_feats[..., :6])
    t_t = cam_feats[..., 6:9]
    R0 = R_t[:, 0]
    t0 = t_t[:, 0]
    R0T = R0.transpose(-2, -1)

    R_rel = torch.matmul(R0T.unsqueeze(1), R_t)
    rot6d_rel = matrix_to_rotation_6d(R_rel)

    delta = t_t - t0.unsqueeze(1)
    t_rel = torch.matmul(
        R0T.unsqueeze(1), delta.unsqueeze(-1)
    ).squeeze(-1)

    return torch.cat([rot6d_rel, t_rel], dim=-1)
