"""Clip dataset loader for the released HOT3D and ARCTIC benchmark caches."""

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from utils.normalizer import Normalizer

SINGLE_HAND_DIM = 54
DUAL_HAND_DIM = 108
CAM_FEAT_DIM = 9


def collate_dual_hand(batch: List[Dict]) -> Dict:
    """Pad clip inputs to the longest sequence in a batch."""
    if not batch:
        raise ValueError("Cannot collate an empty clip batch")
    lengths = torch.tensor([sample["length"] for sample in batch], dtype=torch.long)
    max_len = int(lengths.max())
    result = {
        "length": lengths,
        "mask": torch.arange(max_len)[None, :] < lengths[:, None],
        "betas": torch.stack([sample["betas"] for sample in batch]),
        "crop_info": torch.stack([sample["crop_info"] for sample in batch]),
    }
    sequence_keys = (
        "x", "cam_feats", "proposal", "proposal_cam", "proposal_conf",
        "hand_valid", "visual_feats_perhand", "scene_feats", "da3_depth_signal",
        "gt_mano_world_raw", "proposal_world_raw", "trans_base",
    )
    for key in sequence_keys:
        values = [sample.get(key, sample["proposal"] if key == "proposal_cam" else None)
                  for sample in batch]
        template = next((value for value in values if value is not None), None)
        if template is None:
            continue
        padded = template.new_zeros((len(batch), max_len) + tuple(template.shape[1:]))
        for i, value in enumerate(values):
            if value is not None:
                padded[i, :int(lengths[i])] = value
        result[key] = padded
    result["x"] = result["x"].transpose(1, 2)
    return result


_R_W2D_DA3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)


def _build_da3_cam_feats(clip_id: str, da3_train_dir: str,
                          gt_cam_rot6d_fix: np.ndarray,
                          gt_cam_trans: np.ndarray,
                          apply_r_w2d: bool = True) -> np.ndarray:
    """Build cam_feats from an estimated (VGGT-Omega) rebased pose and GT frame-0 anchor."""
    da3_path = os.path.join(da3_train_dir, f"{clip_id}.npz")
    if not os.path.exists(da3_path):
        raise FileNotFoundError(
            f"estimated-camera npz not found: {da3_path}")

    W2C = np.load(da3_path)["camera_extrinsics"].astype(np.float64)
    if W2C.shape[1] == 3:
        T = W2C.shape[0]
        pad = np.zeros((T, 4, 4), dtype=np.float64); pad[:, :3, :] = W2C; pad[:, 3, 3] = 1.0
        W2C = pad
    if not np.allclose(W2C[0], np.eye(4), atol=1e-6):
        W2C = np.einsum("tij,jk->tik", W2C, np.linalg.inv(W2C[0]))
    R_w2c = W2C[:, :3, :3]; t_w2c = W2C[:, :3, 3]
    C2W = np.zeros_like(W2C)
    C2W[:, :3, :3] = R_w2c.transpose(0, 2, 1)
    C2W[:, :3, 3] = -np.einsum("tij,tj->ti", C2W[:, :3, :3], t_w2c)
    C2W[:, 3, 3] = 1.0

    a1 = gt_cam_rot6d_fix[:, :3].astype(np.float64)
    a2 = gt_cam_rot6d_fix[:, 3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    R_gt_wc = np.stack([b1, b2, b3], axis=-1)
    t_gt_wc = gt_cam_trans.astype(np.float64)

    T0_wc = np.eye(4); T0_wc[:3, :3] = R_gt_wc[0]; T0_wc[:3, 3] = t_gt_wc[0]

    if apply_r_w2d:
        R_W2D_h  = np.eye(4); R_W2D_h[:3, :3]  = _R_W2D_DA3
        R_W2D_Th = np.eye(4); R_W2D_Th[:3, :3] = _R_W2D_DA3.T
        T_wc_da3 = np.einsum("ij,tjk,kl->til", R_W2D_Th, C2W, R_W2D_h)
    else:
        T_wc_da3 = C2W
    T_wc_da3 = T0_wc @ T_wc_da3

    R_wc_da3 = T_wc_da3[:, :3, :3]
    t_wc_da3 = T_wc_da3[:, :3, 3]

    cam_rot6d = np.concatenate([R_wc_da3[:, :, 0], R_wc_da3[:, :, 1]], axis=-1)
    cam_feats = np.concatenate([cam_rot6d, t_wc_da3], axis=-1).astype(np.float32)
    return cam_feats


def _fix_rot6d_convention(rot6d: np.ndarray) -> np.ndarray:
    """Transpose the rotation decoded from each cached 6D pair, preserving its shape."""
    r = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    a1, a2 = r[:, :3], r[:, 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    R = np.stack([b1, b2, b3], axis=-1)
    R_fixed = R.transpose(0, 2, 1)
    fixed = np.concatenate([R_fixed[:, :, 0], R_fixed[:, :, 1]], axis=-1)
    return fixed.reshape(rot6d.shape).astype(np.float32)


FLIP_GUARD_RATIO = 0.5
FLIP_GUARD_MIN_JUMP_M = 0.10
FLIP_GUARD_MIN_SEP_M = 0.06
FLIP_GUARD_STICKY_FRAMES = 5
FLIP_GUARD_MAX_REJECT_FRAC = 0.5


def _reject_handedness_flips(tr, det):
    """tr: {0,1} -> (T,3) world wrist; det: {0,1} -> (T,) bool. Returns (acc, n_rej)."""
    acc = {h: det[h].copy() for h in (0, 1)}
    last = {0: None, 1: None}
    rej = {0: None, 1: None}
    n_rej = {0: 0, 1: 0}
    T = len(det[0])
    for t in range(T):
        if det[0][t] and det[1][t]:
            if float(np.linalg.norm(tr[0][t] - tr[1][t])) < FLIP_GUARD_MIN_SEP_M:
                jump = {h: (float(np.linalg.norm(tr[h][t] - last[h]))
                            if last[h] is not None else None) for h in (0, 1)}
                if jump[0] is not None and jump[1] is not None:
                    bad = 0 if jump[0] > jump[1] else 1
                    acc[bad][t] = False
                    n_rej[bad] += 1
                    rej[bad] = (t, tr[bad][t])
        for h in (0, 1):
            if not det[h][t] or not acc[h][t]:
                continue
            o = 1 - h
            if rej[h] is not None and t - rej[h][0] <= FLIP_GUARD_STICKY_FRAMES \
                    and last[h] is not None:
                d_rej = float(np.linalg.norm(tr[h][t] - rej[h][1]))
                if d_rej < float(np.linalg.norm(tr[h][t] - last[h])):
                    acc[h][t] = False
                    n_rej[h] += 1
                    rej[h] = (t, tr[h][t])
                    continue
            if (not det[o][t]) and last[h] is not None and last[o] is not None:
                d_own = float(np.linalg.norm(tr[h][t] - last[h]))
                d_other = float(np.linalg.norm(tr[h][t] - last[o]))
                if d_own > FLIP_GUARD_MIN_JUMP_M and d_other < FLIP_GUARD_RATIO * d_own:
                    acc[h][t] = False
                    n_rej[h] += 1
                    rej[h] = (t, tr[h][t])
                    continue
            last[h] = tr[h][t]
    for h in (0, 1):
        n_det = int(det[h].sum())
        if n_det and n_rej[h] > FLIP_GUARD_MAX_REJECT_FRAC * n_det:
            acc[h] = det[h].copy()
            n_rej[h] = -n_rej[h]
    return acc, n_rej


def anchor_accepted_mask(proposal_world, proposal_conf, flip_guard: bool,
                         anchor_conf_min: float = 0.0) -> np.ndarray:
    """Return the per-hand anchor-support acceptance mask."""
    world_shape = tuple(np.shape(proposal_world))
    conf_shape = tuple(np.shape(proposal_conf))
    if len(world_shape) != 2 or world_shape[1] != DUAL_HAND_DIM:
        raise ValueError(
            f"proposal_world must have shape (T, {DUAL_HAND_DIM}), "
            f"got {world_shape}")
    T = world_shape[0]
    if conf_shape != (T, 2):
        raise ValueError(
            f"proposal_conf must have shape ({T}, 2), got {conf_shape}")

    conf = proposal_conf.numpy() if hasattr(proposal_conf, "numpy") \
        else np.asarray(proposal_conf)
    accepted = conf > 0

    if flip_guard:
        world = proposal_world.numpy() if hasattr(proposal_world, "numpy") \
            else np.asarray(proposal_world)
        flip_ok, _ = _reject_handedness_flips(
            {0: world[:, 51:54], 1: world[:, SINGLE_HAND_DIM + 51:
                                         SINGLE_HAND_DIM + 54]},
            {0: accepted[:, 0], 1: accepted[:, 1]})
        for hi in (0, 1):
            if flip_ok[hi].any():
                accepted[:, hi] &= flip_ok[hi]

    if anchor_conf_min > 0:
        for hi in (0, 1):
            picky = conf[:, hi] >= anchor_conf_min
            filtered = accepted[:, hi] & picky
            if filtered.any():
                accepted[:, hi] = filtered

    return accepted


def recompose_residual_world_np(arr, trans_base):
    """Add the translation residual base in place."""
    if arr.ndim != 2 or arr.shape[1] != DUAL_HAND_DIM:
        raise ValueError(
            f"arr must have shape (T, {DUAL_HAND_DIM}), got {arr.shape}")
    T = arr.shape[0]
    if trans_base is None:
        raise ValueError(
            "trans_residual=True requires trans_base with shape (T, 6)")
    if np.shape(trans_base) != (T, 6):
        raise ValueError(
            f"trans_base must have shape {(T, 6)}, got {np.shape(trans_base)}")

    for si, off in ((0, 0), (1, SINGLE_HAND_DIM)):
        trans_slice = slice(off + 51, off + 54)
        anchor_trans = trans_base[:, si * 3:(si + 1) * 3]
        arr[:, trans_slice] += anchor_trans
    return arr


def _build_trans_base(proposal_world, proposal_conf, smooth_win: int = 0,
                      anchor_conf_min: float = 0.0, flip_guard: bool = False):
    """Interpolate and smooth reliable observations for residual wrist translation."""
    accepted = anchor_accepted_mask(proposal_world, proposal_conf, flip_guard,
                                    anchor_conf_min)
    T = proposal_world.shape[0]
    base = np.zeros((T, 6), dtype=np.float32)
    allt = np.arange(T)
    for hi, off in ((0, 51), (1, SINGLE_HAND_DIM + 51)):
        idx = np.where(accepted[:, hi])[0]
        if len(idx) == 0:
            continue
        tr = proposal_world[:, off:off + 3]
        tr = tr.numpy() if hasattr(tr, "numpy") else tr
        for d in range(3):
            base[:, hi * 3 + d] = np.interp(allt, idx, tr[idx, d])
    if smooth_win and smooth_win > 2:
        from scipy.signal import savgol_filter, medfilt
        win = min(int(smooth_win) | 1, T if T % 2 == 1 else T - 1)
        if win > 2:
            for d in range(6):
                base[:, d] = medfilt(base[:, d], kernel_size=min(5, win))
            base = savgol_filter(base, win, polyorder=2, axis=0,
                                 mode="interp").astype(np.float32)
    return torch.from_numpy(base)


def load_depth_signal(path, start, end, length):
    """Load complete finite (T, 2, 3) depth features; missing files are not missing detections."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required depth signal not found: {path}. Check --depth_signal_dir.")
    with np.load(path, allow_pickle=False) as artifact:
        if "depth_signal" not in artifact.files:
            raise ValueError(f"{path} has no depth_signal field")
        values = np.asarray(artifact["depth_signal"], dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (2, 3) or values.shape[0] < end:
        raise ValueError(f"{path}: expected depth_signal (T, 2, 3) covering {end} frames, got {values.shape}")
    cropped = values[start:end]
    if len(cropped) != length or not np.isfinite(cropped).all():
        raise ValueError(f"{path}: depth_signal must contain {length} finite frames")
    return torch.from_numpy(cropped.copy())


class ClipMotionLoader:
    """Load one HOT3D or ARCTIC clip into the inference and collation sample schema."""

    def __init__(
        self,
        base_dir: str,
        normalizer_dir: Optional[str] = None,
        wilor_dir: Optional[str] = None,
        visual_feat_dir: Optional[str] = None,
        scene_feat_dir: Optional[str] = None,
        scene_feat_key: str = "scene_pooled",
        depth_signal_dir: Optional[str] = None,
        use_da3_train: bool = False,
        da3_train_dir: Optional[str] = None,
        da3_apply_r_w2d: bool = True,
        gravity_canonical=None,
        tcanon_hand_convention="legacy",
        tcanon_mano_left="official",
        proposal_orient_flip=False,
        mano_dir: Optional[str] = None,
        trans_residual: bool = False,
        trans_base_smooth_win: int = 0,
        anchor_conf_min: float = 0.0,
        anchor_flip_guard: bool = False,
        trans_base_edge_mode: str = "clamp",
    ):
        self.base_dir = base_dir
        self.wilor_dir = wilor_dir
        self.visual_feat_dir = visual_feat_dir
        self.scene_feat_dir = scene_feat_dir
        self.scene_feat_key = scene_feat_key
        self.depth_signal_dir = depth_signal_dir
        # Checkpoints retain the original names for their VGGT-Omega camera settings.
        self.use_da3_train = use_da3_train
        self.da3_train_dir = da3_train_dir
        self.da3_apply_r_w2d = da3_apply_r_w2d
        self.trans_residual = bool(trans_residual)
        self.trans_base_smooth_win = int(trans_base_smooth_win or 0)
        self.anchor_conf_min = float(anchor_conf_min or 0.0)
        self.anchor_flip_guard = bool(anchor_flip_guard)
        if trans_base_edge_mode != "clamp":
            raise ValueError("Released checkpoints use trans_base_edge_mode='clamp'")
        self.gravity_canonical = gravity_canonical
        self.tcanon_hand_convention = tcanon_hand_convention
        self.proposal_orient_flip = proposal_orient_flip
        self._tcanon = None
        self._tcanon_grav = None
        self._mano = None
        if gravity_canonical is not None:
            from data_loaders import tcanon as _tc
            import smplx
            self._tcanon = _tc
            self._tcanon_grav = (_tc.gravity_down(gravity_canonical)
                                 if isinstance(gravity_canonical, str)
                                 else np.asarray(gravity_canonical, dtype=np.float64))
            md = mano_dir or os.path.join(os.path.dirname(__file__), "mano_models")
            self.tcanon_mano_left = tcanon_mano_left
            if tcanon_mano_left not in ("official", "patched"):
                raise ValueError(f"tcanon_mano_left must be official|patched, got {tcanon_mano_left!r}")
            self._mano = {
                "left":  smplx.create(os.path.join(md, "MANO_LEFT.pkl"), "mano",
                                      use_pca=False, is_rhand=False, flat_hand_mean=False),
                "right": smplx.create(os.path.join(md, "MANO_RIGHT.pkl"), "mano",
                                      use_pca=False, is_rhand=True, flat_hand_mean=False)}
            if tcanon_mano_left == "patched":
                import torch as _t
                lL, lR = self._mano["left"], self._mano["right"]
                if _t.sum(_t.abs(lL.shapedirs[:, 0, :] - lR.shapedirs[:, 0, :])) < 1:
                    lL.shapedirs[:, 0, :] *= -1
                else:
                    raise RuntimeError(
                        "tcanon_mano_left=patched requested but MANO_LEFT.shapedirs[:,0,:] "
                        "already differs from MANO_RIGHT's — the smplx quirk this patch "
                        "targets is absent; refusing to double-apply the sign flip.")
        if use_da3_train and (not da3_train_dir or not os.path.isdir(da3_train_dir)):
            raise ValueError(
                f"use_da3_train=True but da3_train_dir is missing: {da3_train_dir!r}")

        self.normalizer = None
        if normalizer_dir:
            if not os.path.exists(os.path.join(normalizer_dir, "mean.pt")):
                raise FileNotFoundError(
                    f"Normalizer not found: {os.path.join(normalizer_dir, 'mean.pt')}. "
                    "Pass the correct directory with --normalizer_dir."
                )
            self.normalizer = Normalizer(base_dir=normalizer_dir)

    def __call__(self, clip_id: str) -> dict:
        npz_path = os.path.join(self.base_dir, clip_id + ".npz")
        data = np.load(npz_path)

        if self.gravity_canonical is not None:
            data = {k: data[k] for k in data.files}
            data, _ = self._tcanon.tcanon_forward(data, self._tcanon_grav, self._mano,
                                                  hand_convention=self.tcanon_hand_convention)

        left_rot_6d_fix = _fix_rot6d_convention(data["left_rot_6d"])
        right_rot_6d_fix = _fix_rot6d_convention(data["right_rot_6d"])
        cam_rot_6d_fix = _fix_rot6d_convention(data["cam_rot_6d"])

        left_feats = np.concatenate([
            left_rot_6d_fix, data["left_aa"], data["left_trans"]
        ], axis=-1).astype(np.float32)
        right_feats = np.concatenate([
            right_rot_6d_fix, data["right_aa"], data["right_trans"]
        ], axis=-1).astype(np.float32)
        cam_feats = np.concatenate([
            cam_rot_6d_fix, data["cam_trans"]
        ], axis=-1).astype(np.float32)

        if self.use_da3_train:
            cam_feats = _build_da3_cam_feats(
                clip_id, self.da3_train_dir,
                cam_rot_6d_fix, data["cam_trans"],
                apply_r_w2d=self.da3_apply_r_w2d)
        betas = np.concatenate([
            data["left_betas"], data["right_betas"]
        ]).astype(np.float32)

        left_valid = data["left_valid"].astype(bool)
        right_valid = data["right_valid"].astype(bool)
        T_total = left_feats.shape[0]

        crop_len = T_total
        start = 0
        s = slice(start, start + crop_len)

        left_feats = left_feats[s]
        right_feats = right_feats[s]
        cam_feats_crop = cam_feats[s]
        left_valid = left_valid[s]
        right_valid = right_valid[s]

        dual_feats = np.concatenate([left_feats, right_feats], axis=-1)
        hand_valid = np.stack([left_valid, right_valid], axis=-1)

        dual_t = torch.from_numpy(dual_feats).float()
        cam_t = torch.from_numpy(cam_feats_crop).float()
        T = dual_t.shape[0]

        proposal, proposal_cam, proposal_conf = self._load_wilor(
            clip_id, start, start + crop_len, T, cam_feats)
        gt_mano_world_raw = dual_t.clone()
        proposal_world_raw = proposal.clone()

        trans_base = None
        if self.trans_residual:
            trans_base = _build_trans_base(proposal, proposal_conf,
                                           self.trans_base_smooth_win,
                                           self.anchor_conf_min,
                                           flip_guard=self.anchor_flip_guard)
            dual_t[:, 51:54] -= trans_base[:, 0:3]
            dual_t[:, SINGLE_HAND_DIM + 51:SINGLE_HAND_DIM + 54] -= trans_base[:, 3:6]
            pl = proposal_conf[:, 0] > 0
            pr = proposal_conf[:, 1] > 0
            proposal[pl, 51:54] -= trans_base[pl, 0:3]
            proposal[pr, SINGLE_HAND_DIM + 51:SINGLE_HAND_DIM + 54] -= trans_base[pr, 3:6]

        if self.normalizer is not None:
            dual_t = self.normalizer(dual_t)
            det_mask = (proposal_conf.sum(dim=-1) > 0).unsqueeze(-1)
            proposal = torch.where(det_mask, self.normalizer(proposal), torch.zeros_like(proposal))

        crop_info = torch.tensor([
            start / max(T_total, 1),
            crop_len / max(T_total, 1),
            float(start),
        ], dtype=torch.float32)

        result = {
            "x": dual_t,
            "cam_feats": cam_t,
            "betas": torch.from_numpy(betas),
            "proposal": proposal,
            "proposal_cam": proposal_cam,
            "proposal_conf": proposal_conf,
            "hand_valid": torch.from_numpy(hand_valid).float(),
            "crop_info": crop_info,
            "length": T,
            "gt_mano_world_raw": gt_mano_world_raw,
            "proposal_world_raw": proposal_world_raw,
        }
        if trans_base is not None:
            result["trans_base"] = trans_base

        visual_feats_perhand = self._load_visual_feats_perhand(
            clip_id, start, start + crop_len, T)
        if visual_feats_perhand is not None:
            result["visual_feats_perhand"] = visual_feats_perhand

        scene_feats = self._load_scene_feats(clip_id, start, start + crop_len, T)
        if scene_feats is not None:
            result["scene_feats"] = scene_feats

        if self.depth_signal_dir is not None:
            dpath = os.path.join(self.depth_signal_dir, f"{clip_id}.npz")
            dsig = load_depth_signal(dpath, start, start + crop_len, T)
            result["da3_depth_signal"] = dsig

        return result

    def _load_wilor(self, clip_id, start, end, T, cam_feats_full):
        """Load WiLoR proposals and lift them with the selected camera track."""
        if self.wilor_dir is None:
            raise ValueError("WiLoR proposal directory is required; pass --wilor_dir")
        path = os.path.join(self.wilor_dir, clip_id + ".npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"WiLoR proposal file not found: {path}")

        with np.load(path, allow_pickle=True) as w:
            L_present = np.asarray(w["L_present"])[start:end]
            R_present = np.asarray(w["R_present"])[start:end]
            L_t = np.asarray(w["L_t"])[start:end].astype(np.float64)
            R_t = np.asarray(w["R_t"])[start:end].astype(np.float64)
            L_q = np.asarray(w["L_q"])[start:end]
            R_q = np.asarray(w["R_q"])[start:end]
            L_poses = np.asarray(w["L_poses"])[start:end].reshape(end - start, -1).astype(np.float32)
            R_poses = np.asarray(w["R_poses"])[start:end].reshape(end - start, -1).astype(np.float32)
            L_conf = np.asarray(w["L_conf"])[start:end].astype(np.float32) if "L_conf" in w.files else None
            R_conf = np.asarray(w["R_conf"])[start:end].astype(np.float32) if "R_conf" in w.files else None

        cr6d = cam_feats_full[start:end, :6].astype(np.float64)
        ct = cam_feats_full[start:end, 6:9].astype(np.float64)
        a1 = cr6d[:, :3]; a2 = cr6d[:, 3:6]
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
        b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
        b3 = np.cross(b1, b2, axis=-1)
        R_wc = np.stack([b1, b2, b3], axis=-1)

        proposal_np = np.zeros((T, DUAL_HAND_DIM), dtype=np.float32)
        proposal_cam_np = np.zeros((T, DUAL_HAND_DIM), dtype=np.float32)
        conf_np = np.zeros((T, 2), dtype=np.float32)

        for si, (side, offset, bundle) in enumerate([
            ("L", 0, (L_present, L_t, L_q, L_poses, L_conf)),
            ("R", SINGLE_HAND_DIM, (R_present, R_t, R_q, R_poses, R_conf)),
        ]):
            present, cam_t_T, cam_q_T, poses_T, conf_T = bundle
            mask = present.astype(np.float32) > 0.5
            if not mask.any():
                continue

            q = cam_q_T.astype(np.float64)
            w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            R_cam = np.empty((cam_q_T.shape[0], 3, 3), dtype=np.float64)
            R_cam[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_)
            R_cam[:, 0, 1] = 2 * (x_ * y_ - w_ * z_)
            R_cam[:, 0, 2] = 2 * (x_ * z_ + w_ * y_)
            R_cam[:, 1, 0] = 2 * (x_ * y_ + w_ * z_)
            R_cam[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_)
            R_cam[:, 1, 2] = 2 * (y_ * z_ - w_ * x_)
            R_cam[:, 2, 0] = 2 * (x_ * z_ - w_ * y_)
            R_cam[:, 2, 1] = 2 * (y_ * z_ + w_ * x_)
            R_cam[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)

            rot6d_cam = np.concatenate([R_cam[:, :, 0], R_cam[:, :, 1]], axis=-1).astype(np.float32)

            wrist_world = np.einsum("tij,tj->ti", R_wc, cam_t_T) + ct

            R_world = np.einsum("tij,tjk->tik", R_wc, R_cam)
            if self.proposal_orient_flip:
                rot6d_world = np.concatenate([R_world[:, 0, :], R_world[:, 1, :]], axis=-1).astype(np.float32)
            else:
                rot6d_world = np.concatenate([R_world[:, :, 0], R_world[:, :, 1]], axis=-1).astype(np.float32)

            proposal_np[:, offset:offset + 6] = np.where(mask[:, None], rot6d_world, 0)
            proposal_np[:, offset + 6:offset + 51] = np.where(mask[:, None], poses_T, 0)
            proposal_np[:, offset + 51:offset + 54] = np.where(mask[:, None], wrist_world.astype(np.float32), 0)

            proposal_cam_np[:, offset:offset + 6] = np.where(mask[:, None], rot6d_cam, 0)
            proposal_cam_np[:, offset + 6:offset + 51] = np.where(mask[:, None], poses_T, 0)
            proposal_cam_np[:, offset + 51:offset + 54] = np.where(mask[:, None], cam_t_T.astype(np.float32), 0)

            conf_np[:, si] = np.where(mask, conf_T, 0.0) if conf_T is not None else mask.astype(np.float32)

        return (torch.from_numpy(proposal_np),
                torch.from_numpy(proposal_cam_np),
                torch.from_numpy(conf_np))


    def _load_scene_feats(self, clip_id, start, end, T):
        if self.scene_feat_dir is None:
            return None
        path = os.path.join(self.scene_feat_dir, f"{clip_id}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Scene feature file not found: {path}. "
                "Pass the correct directory with --scene_feat_dir."
            )
        s = np.load(path)
        key = self.scene_feat_key
        if key not in s.files:
            raise KeyError(
                f"Scene feature file {path} has no field {key!r}; "
                f"available={list(s.files)}. Pass the correct key with "
                "--scene_feat_key."
            )
        feats = s[key][start:end].astype(np.float32)
        if key == "scene_tokens" and feats.ndim == 3 and "camera_token" in s.files:
            cam_tok = s["camera_token"][start:end].astype(np.float32)[:, None, :]
            feats = np.concatenate([feats, cam_tok], axis=1)
        if feats.shape[0] < T:
            pad = np.zeros((T - feats.shape[0],) + feats.shape[1:], dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        return torch.from_numpy(feats[:T])


    def _load_visual_feats_perhand(self, clip_id, start, end, T):
        if self.visual_feat_dir is None:
            return None
        path = os.path.join(self.visual_feat_dir, clip_id + ".npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Visual feature file not found: {path}")
        v = np.load(path)
        L = v["L_visual"][start:end].astype(np.float32)
        R = v["R_visual"][start:end].astype(np.float32)
        feats = np.stack([L, R], axis=1)
        if feats.shape[0] < T:
            pad = np.zeros((T - feats.shape[0],) + feats.shape[1:], dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        return torch.from_numpy(feats[:T])
