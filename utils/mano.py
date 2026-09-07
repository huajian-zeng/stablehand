"""Consistent MANO layers for inference scoring, NPZ evaluation and rendering."""

import os

import smplx
import torch


def create_mano_layers(mano_dir, left_convention):
    """Return (left, right) scoring layers using the requested shapedirs convention."""
    if left_convention not in ("official", "patched"):
        raise ValueError(f"Unknown left MANO convention: {left_convention!r}")
    right = smplx.create(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "mano",
                         use_pca=False, is_rhand=True, num_pca_comps=45,
                         flat_hand_mean=True)
    left = smplx.create(os.path.join(mano_dir, "MANO_LEFT.pkl"), "mano",
                        use_pca=False, is_rhand=False, num_pca_comps=45,
                        flat_hand_mean=True)
    if (left_convention == "patched"
            and torch.sum(torch.abs(left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])) < 1):
        left.shapedirs[:, 0, :] *= -1
    return left, right
