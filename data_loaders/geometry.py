"""6D rotation conversions using the first two matrix rows.

Adapted from PyTorch3D's transforms/rotation_conversions.py:
https://github.com/facebookresearch/pytorch3d
Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
See ../utils/PYTORCH3D_LICENSE for the BSD license and disclaimer.
"""

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert Zhou et al.'s 6D representation to a rotation matrix, using rows."""

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Return the first two matrix rows as Zhou et al.'s 6D rotation representation."""
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)
