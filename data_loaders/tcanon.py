"""Transform benchmark hand and camera motion into a gravity-aligned frame."""
from __future__ import annotations

import numpy as np

GRAVITY_TABLE = {
    "hot3d": np.array([0.0, -1.0, 0.0]),
    "arctic": np.array([0.0, 0.0, -1.0]),
}


def gravity_down(name):
    """Gravity-down vector for a dataset key."""
    if name not in GRAVITY_TABLE:
        raise KeyError(
            f"no gravity convention for dataset {name!r}; known: "
            f"{sorted(GRAVITY_TABLE)}. T_canon needs the world-frame gravity of "
            "the deploy dataset — add it to GRAVITY_TABLE from that dataset's "
            "own world convention, never from a camera-down heuristic.")
    return GRAVITY_TABLE[name]


def decode_r6(v):
    """(...,6) row-convention rot6d -> (...,3,3) R (first two ROWS + cross)."""
    a1, a2 = v[..., :3], v[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def encode_r6(R):
    """(...,3,3) R -> (...,6) row-convention rot6d (first two rows flattened)."""
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


HAND_CONVENTIONS = ("legacy", "fixed", "cols_unified")
_STATE_ROT_COLUMN = {"legacy": True, "cols_unified": True, "fixed": False}


def state_rot_is_column(hand_convention: str) -> bool:
    """True when the column decode of a state rot6d slice is the true rotation."""
    try:
        return _STATE_ROT_COLUMN[hand_convention]
    except KeyError:
        raise ValueError(
            f"unknown hand_convention {hand_convention!r}; expected one of "
            f"{HAND_CONVENTIONS}") from None


def gravity_align_rotation(g_down, up_plus_y=True):
    """Rotation R s.t. R @ g_down = (0,-1,0) [up=+y, ours] or (0,+1,0) [flipped]."""
    g = g_down / (np.linalg.norm(g_down) + 1e-9)
    fwd = np.array([0.0, 0.0, 1.0])
    if abs(fwd @ g) > 0.95:
        fwd = np.array([1.0, 0.0, 0.0])
    f = fwd - (fwd @ g) * g
    f = f / (np.linalg.norm(f) + 1e-9)
    y = (-g) if up_plus_y else g
    z = f
    x = np.cross(y, z); x = x / (np.linalg.norm(x) + 1e-9)
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=0)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, "gravity_align not a proper rotation"
    return R


def j0_rest(mano_layer, betas):
    """Return the rest wrist position for these betas; wrist = J0_rest + trans."""
    import torch
    b = torch.as_tensor(np.asarray(betas)[None]).float()
    j = mano_layer(betas=b, return_verts=True).joints[0].detach().numpy()
    return j[0].astype(np.float64)


def build_R_cfw(cam_rot_6d0, cam_trans0, gravity_down, up_plus_y=True):
    """World->canonical rotation R_cfw and origin c0 from frame-0 camera + gravity."""
    R_wc0 = decode_r6(np.asarray(cam_rot_6d0, dtype=np.float64))
    R_cw0 = R_wc0.T
    R_grav = gravity_align_rotation(R_cw0 @ np.asarray(gravity_down, float), up_plus_y)
    R_cfw = R_grav @ R_cw0
    c0 = np.asarray(cam_trans0, dtype=np.float64).copy()
    return R_cfw, c0


def tcanon_forward(d, gravity_down, mano, up_plus_y=True,
                   hand_convention="legacy"):
    """Transform an NPZ-like dict into the canonical frame. Returns (out, params)."""
    R_cfw, c0 = build_R_cfw(d["cam_rot_6d"][0], d["cam_trans"][0], gravity_down, up_plus_y)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    J0 = {}

    Rwc = decode_r6(np.asarray(d["cam_rot_6d"], np.float64))
    out["cam_rot_6d"] = encode_r6(np.einsum("ij,tjk->tik", R_cfw, Rwc)).astype(np.float32)
    out["cam_trans"] = ((np.asarray(d["cam_trans"], np.float64) - c0) @ R_cfw.T).astype(np.float32)

    for hand in ("left", "right"):
        j0 = j0_rest(mano[hand], d[f"{hand}_betas"]); J0[hand] = j0
        rot6d_in = np.asarray(d[f"{hand}_rot_6d"], np.float64)
        absent = np.linalg.norm(rot6d_in, axis=-1) < 1e-8
        Rh = decode_r6(rot6d_in)
        if hand_convention == "fixed":
            out[f"{hand}_rot_6d"] = encode_r6(np.einsum("tij,jk->tik", Rh, R_cfw.T)).astype(np.float32)
        else:
            out[f"{hand}_rot_6d"] = encode_r6(np.einsum("ij,tjk->tik", R_cfw, Rh)).astype(np.float32)
        tr = np.asarray(d[f"{hand}_trans"], np.float64)
        wrist_c = (j0[None] + tr - c0) @ R_cfw.T
        out[f"{hand}_trans"] = (wrist_c - j0[None]).astype(np.float32)
        out[f"{hand}_rot_6d"][absent] = 0.0
        out[f"{hand}_trans"][absent] = 0.0
    return out, {"R_cfw": R_cfw, "c0": c0, "J0": J0,
                 "hand_convention": hand_convention}
