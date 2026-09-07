"""Run StableHand inference, evaluation and Rerun recording on HOT3D or ARCTIC."""

import argparse
import hashlib
import json
import os
import pickle
import sys
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from data_loaders.clip_dataset import ClipMotionLoader, recompose_residual_world_np
from data_loaders.tcanon import state_rot_is_column
from utils.window_util import camera_to_window_relative
from utils.model_util import create_model_and_scheduler
from utils.normalizer import Normalizer
from utils.mano import create_mano_layers
from utils.calibration import clip_names, read_clip_list, select_calibration_clips
from utils.validation import (positive_float, positive_int,
                              random_seed, sigma_vector,
                              validate_positive_values)
from utils.inference_cache import (
    CACHE_SCHEMA, clip_input_identity, file_identity, fingerprint,
    read_cached_metrics, source_identity, write_cached_metrics,
)
from eval.metrics import (
    compute_hand_metrics, compute_mrrpe, summarize_metrics,
)
from diffusion.flow_matching import FlowMatchingScheduler


IMAGE_ROLL = np.array([[0.0, -1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]])
RRD_IMAGE_SIZE = 704
RRD_IMAGE_QUALITY = 88

HAND_COLORS = {
    "gt":   {"left": (0, 200, 100),   "right": (0, 160, 80)},
    "pred": {"left": (100, 160, 255), "right": (60, 120, 220)},
}

DEFAULT_CLIPS = [
    "clip-002223", "clip-002945", "clip-003052",
    "clip-002026", "clip-002736", "clip-002904",
]


def rot6d_to_mat_np(r6d: np.ndarray) -> np.ndarray:
    """(N, 6) → (N, 3, 3), with the two input vectors treated as columns."""
    a1 = r6d[..., :3].astype(np.float64)
    a2 = r6d[..., 3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) → (N, 4) quaternion in (w, x, y, z) order via Shepperd."""
    N = R.shape[0]
    q = np.empty((N, 4), dtype=np.float64)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    mask_tr = trace > 0
    t = trace[mask_tr] + 1.0
    s = 2.0 * np.sqrt(t)
    q[mask_tr, 0] = 0.25 * s
    q[mask_tr, 1] = (R[mask_tr, 2, 1] - R[mask_tr, 1, 2]) / s
    q[mask_tr, 2] = (R[mask_tr, 0, 2] - R[mask_tr, 2, 0]) / s
    q[mask_tr, 3] = (R[mask_tr, 1, 0] - R[mask_tr, 0, 1]) / s
    rest = ~mask_tr
    if rest.any():
        Rr = R[rest]
        diag = np.stack([Rr[:, 0, 0], Rr[:, 1, 1], Rr[:, 2, 2]], axis=-1)
        ix = np.argmax(diag, axis=-1)
        for i, k in enumerate(ix):
            r_ = Rr[i]
            if k == 0:
                s = 2.0 * np.sqrt(max(1e-12, 1.0 + r_[0, 0] - r_[1, 1] - r_[2, 2]))
                w = (r_[2, 1] - r_[1, 2]) / s
                x = 0.25 * s
                y = (r_[0, 1] + r_[1, 0]) / s
                z = (r_[0, 2] + r_[2, 0]) / s
            elif k == 1:
                s = 2.0 * np.sqrt(max(1e-12, 1.0 + r_[1, 1] - r_[0, 0] - r_[2, 2]))
                w = (r_[0, 2] - r_[2, 0]) / s
                x = (r_[0, 1] + r_[1, 0]) / s
                y = 0.25 * s
                z = (r_[1, 2] + r_[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(max(1e-12, 1.0 + r_[2, 2] - r_[0, 0] - r_[1, 1]))
                w = (r_[1, 0] - r_[0, 1]) / s
                x = (r_[0, 2] + r_[2, 0]) / s
                y = (r_[1, 2] + r_[2, 1]) / s
                z = 0.25 * s
            idx = np.where(rest)[0][i]
            q[idx] = (w, x, y, z)
    q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    q[q[:, 0] < 0] *= -1.0
    return q.astype(np.float32)


def load_mano_faces(mano_dir: str) -> np.ndarray:
    with open(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return np.array(d["f"], dtype=np.int32)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(ckpt_path: str, device: torch.device, normalizer_dir=None):
    ckpt_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(ckpt_dir, "args.json")) as f:
        margs = argparse.Namespace(**json.load(f))
    if (hasattr(margs, "trans_residual_frame")
            and margs.trans_residual_frame != "world"):
        raise NotImplementedError(
            "wrist-frame translation residual is not supported by the release")
    hash_before = _sha256_file(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hash_after = _sha256_file(ckpt_path)
    if hash_before != hash_after:
        raise RuntimeError(
            f"checkpoint changed while being loaded: {ckpt_path} "
            f"({hash_before} -> {hash_after})")
    model_state = state["model"]
    model, quality_net, quality_gate, _ = create_model_and_scheduler(margs)
    model.load_state_dict(model_state, strict=True)
    quality_net.load_state_dict(state.get("quality_net", {}), strict=True)
    quality_gate.load_state_dict(state.get("quality_gate", {}), strict=True)
    for m in (model, quality_net, quality_gate):
        m.to(device).eval()
    scheduler = FlowMatchingScheduler()
    # Apply the override before attempting to open the original path. Keep margs
    # unchanged here so provenance can still describe the checkpoint's settings.
    normalizer = Normalizer(base_dir=normalizer_dir or margs.normalizer_dir)
    checkpoint_identity = {
        "checkpoint_sha256": hash_after,
        "checkpoint_step": int(state["step"]) if "step" in state else None,
    }
    return (model, quality_net, quality_gate, scheduler, normalizer, margs,
            checkpoint_identity)


def load_qn_only(qn_ckpt_path: str, device: torch.device):
    """Load the standalone Quality Network used to predict quality at inference."""
    qn_dir = os.path.dirname(qn_ckpt_path)
    with open(os.path.join(qn_dir, "args.json")) as f:
        qn_args = argparse.Namespace(**json.load(f))
    if (getattr(qn_args, "qn_output_mode", "q") != "error"
            or int(getattr(qn_args, "q_granularity", 2)) != 2):
        raise ValueError("Use the released Quality Network checkpoint with wrist/finger error outputs")
    _, qn_model, _, _ = create_model_and_scheduler(qn_args)
    state = torch.load(qn_ckpt_path, map_location="cpu", weights_only=False)
    qn_state = state.get("quality_net") or state
    qn_model.load_state_dict(qn_state, strict=True)
    qn_model.to(device).eval()
    return qn_model, qn_args


def qn_variant_inputs(qn_args, sample, device):
    """Depth-at-hand observations required by the released Quality Network."""
    if int(getattr(qn_args, "qn_da3_depth_dim", 0) or 0) == 0:
        return {}
    if "da3_depth_signal" not in sample:
        raise RuntimeError("Quality Network requires --depth_signal_dir")
    return {"da3_depth_signal": sample["da3_depth_signal"].unsqueeze(0).to(device)}


@torch.no_grad()
def calibrate_qn_sigma(qn_model, qn_args, loader, calibration_clips, device):
    """Compute positive σ_deploy from QN predictions on validated calibration clips."""
    if getattr(qn_args, "qn_output_mode", "q") != "error":
        raise ValueError("Calibration requires the released error-output Quality Network")
    granularity = int(getattr(qn_args, "q_granularity", 2))
    group_errors = [[] for _ in range(granularity)]
    for cn in calibration_clips:
        try:
            sample = loader(cn)
        except Exception as exc:
            raise RuntimeError(f"Cannot load calibration clip {cn}: {exc}") from exc
        proposal_cam = sample["proposal_cam"].unsqueeze(0).to(device)
        proposal_conf = sample["proposal_conf"].unsqueeze(0).to(device)
        cam_feats = sample["cam_feats"].unsqueeze(0).to(device)
        out_log = qn_model(mano_proposal=proposal_cam,
                           raw_confidence=proposal_conf,
                           cam_feats=cam_feats,
                           **qn_variant_inputs(qn_args, sample, device)
                           ).squeeze(0).cpu().numpy()
        e_hat = np.expm1(out_log).clip(min=0.0)
        if not np.isfinite(e_hat).all():
            raise ValueError(f"Non-finite QN calibration predictions for {cn}")
        det = (proposal_conf.squeeze(0).cpu().numpy() > 0.1)
        if e_hat.shape[-1] != 2 * granularity:
            raise RuntimeError(
                f"QN calibration expected {2 * granularity} outputs, got "
                f"{e_hat.shape[-1]}")
        for group in range(granularity):
            group_errors[group].extend(
                e_hat[det[:, 0], group].tolist())
            group_errors[group].extend(
                e_hat[det[:, 1], granularity + group].tolist())
    if any(not values for values in group_errors):
        raise RuntimeError("QN calibration produced no valid frames.")
    sigma = np.asarray([
        np.percentile(values, 80) / np.log(10)
        for values in group_errors
    ], dtype=np.float64)
    validate_positive_values(sigma, "Calibrated sigma")
    return sigma


def fk_frame(world_feats: torch.Tensor, offset: int, betas: torch.Tensor,
             mano_layer, rot_column: bool):
    """Single-frame MANO FK from a 108-D world-space feature vector."""
    from data_loaders.geometry import rotation_6d_to_matrix
    R = rotation_6d_to_matrix(world_feats[offset:offset + 6])
    R = (R.transpose(-1, -2) if rot_column else R).numpy()
    go_aa = cv2.Rodrigues(R)[0].reshape(3).astype(np.float32)
    trans = world_feats[offset + 51:offset + 54].numpy()
    with torch.no_grad():
        out = mano_layer(
            betas=betas.unsqueeze(0),
            global_orient=torch.tensor(go_aa[None]),
            hand_pose=world_feats[offset + 6:offset + 51].unsqueeze(0),
            transl=torch.tensor(trans[None]),
            return_verts=True,
        )
    return out.vertices[0].numpy(), out.joints[0].numpy()


@torch.no_grad()
def infer_clip(clip_name: str, loader, model, qg, scheduler, normalizer, train_args,
               device: torch.device, n_steps: int,
               *, qn_predict_model, qn_predict_args, qn_sigma_vec,
               sample_override=None):
    """Return pred_world (T, 108) numpy + T + raw sample dict."""
    sample = loader(clip_name) if sample_override is None else sample_override
    rot_column = state_rot_is_column(loader.tcanon_hand_convention)
    T = sample["length"]
    q_granularity = int(getattr(train_args, "q_granularity", 2))
    if q_granularity != 2:
        raise ValueError("The released checkpoints use wrist/finger quality groups (q_granularity=2)")
    if (int(getattr(train_args, "scene_feat_dim", 0) or 0) > 0
            and "scene_feats" not in sample):
        raise RuntimeError(
            "Checkpoint requires scene features, but the clip sample has none. "
            "Pass the correct cache with --scene_feat_dir."
        )

    x_gt = sample["x"].unsqueeze(0).to(device)
    proposal = sample["proposal"].unsqueeze(0).to(device)
    proposal_conf = sample["proposal_conf"].unsqueeze(0).to(device)
    cam_feats = sample["cam_feats"].unsqueeze(0).to(device)
    betas = sample["betas"].unsqueeze(0).to(device)
    mask = torch.ones(1, T, dtype=torch.bool, device=device)
    crop_info = sample["crop_info"].unsqueeze(0).to(device)

    proposal_cam_dev = sample["proposal_cam"].unsqueeze(0).to(device)
    qn_kwargs = dict(
        mano_proposal=proposal_cam_dev,
        raw_confidence=proposal_conf,
        cam_feats=cam_feats,
    )
    qn_kwargs.update(qn_variant_inputs(qn_predict_args, sample, device))
    q_pred = qn_predict_model(**qn_kwargs)
    if not torch.isfinite(q_pred).all():
        raise ValueError(f"Non-finite QN predictions for {clip_name}")
    expected_q_dim = 2 * q_granularity
    if q_pred.shape[-1] != expected_q_dim:
        raise ValueError(
            f"QN ckpt must output grouped q ({expected_q_dim} dims). Got "
            f"{q_pred.shape[-1]}; use the released Quality Network checkpoint.")

    e_hat = torch.expm1(q_pred).clamp(min=0.0)
    sigma_vec = np.asarray(qn_sigma_vec, dtype=np.float64).reshape(-1)
    if len(sigma_vec) != q_granularity:
        raise ValueError(
            f"sigma vector has length {len(sigma_vec)}, expected "
            f"{q_granularity}")
    validate_positive_values(sigma_vec, "QN sigma")
    sig_one = torch.as_tensor(
        sigma_vec, device=e_hat.device, dtype=e_hat.dtype)
    if not torch.isfinite(sig_one).all() or not (sig_one > 0).all():
        raise ValueError("QN sigma must remain finite and positive in the model dtype")
    sig = torch.cat([sig_one, sig_one], dim=0)
    q_pred = torch.exp(-e_hat / sig)
    q_percomp = q_pred.squeeze(0).cpu()
    qpc = q_percomp.unsqueeze(0).to(device)
    qpc = torch.where(torch.isnan(qpc), torch.zeros_like(qpc), qpc)
    q_L = qpc[:, :, :q_granularity].mean(-1)
    q_R = qpc[:, :, q_granularity:].mean(-1)
    q_for_gate = torch.stack([q_L, q_R], dim=-1)

    scene_feats_batch = sample.get("scene_feats")
    if scene_feats_batch is not None:
        scene_feats_batch = scene_feats_batch.unsqueeze(0).to(device)

    vit_feats_perhand = None
    if getattr(train_args, "vit_obs_mode", "") == "perhand":
        vit_feats_perhand = sample.get("visual_feats_perhand")
        if vit_feats_perhand is None:
            raise ValueError(
                "vit_obs_mode='perhand' requires sample['visual_feats_perhand']; "
                "pass --visual_feat_dir")
        vit_feats_perhand = vit_feats_perhand.unsqueeze(0).to(device)
    obs_gated, obs_bias = qg(proposal, q_for_gate,
                             vit_feats_perhand=vit_feats_perhand)
    head_feats = camera_to_window_relative(cam_feats)

    model_kwargs = {
        "head_motion_feats": head_feats,
        "obs_gated_tokens": obs_gated,
        "obs_attn_bias": obs_bias,
        "proposal": proposal,
        "proposal_conf": proposal_conf,
        "scene_feats": scene_feats_batch,
        "betas": betas,
        "attention_mask": mask,
        "crop_info": crop_info,
    }

    qpc_inject = q_percomp.unsqueeze(0).to(device)
    qpc_inject = torch.where(torch.isnan(qpc_inject),
                             torch.zeros_like(qpc_inject), qpc_inject)
    model_kwargs["q_per_frame"] = qpc_inject

    qpc_pd = q_percomp.unsqueeze(0).to(device)
    qpc_pd = torch.where(torch.isnan(qpc_pd), torch.zeros_like(qpc_pd), qpc_pd)
    q_per_dim = torch.zeros(1, 108, T, device=device)
    for hand_idx, state_off in enumerate((0, 54)):
        qoff = hand_idx * 2
        q_wrist = qpc_pd[:, :, qoff].unsqueeze(1)
        q_fingers = qpc_pd[:, :, qoff + 1].unsqueeze(1)
        q_per_dim[:, state_off:state_off + 6, :] = q_wrist
        q_per_dim[:, state_off + 51:state_off + 54, :] = q_wrist
        q_per_dim[:, state_off + 6:state_off + 51, :] = q_fingers

    q_left = q_for_gate[:, :, 0]
    q_right = q_for_gate[:, :, 1]
    x_pred_channels = scheduler.sample_with_quality_schedule(
        model,
        shape=(1, 108, T),
        quality_left=q_left,
        quality_right=q_right,
        x_proposal=proposal.transpose(1, 2),
        model_kwargs=model_kwargs,
        num_steps=n_steps,
        device=device,
        q_per_dim=q_per_dim,
    )

    x_pred = x_pred_channels.transpose(1, 2)

    pred_world = normalizer.inverse(x_pred.squeeze(0).cpu()).unsqueeze(0)
    gt_world = normalizer.inverse(x_gt.squeeze(0).cpu()).unsqueeze(0)
    proposal_world = normalizer.inverse(sample["proposal"]).numpy()
    trans_residual = bool(getattr(train_args, "trans_residual", False))
    if trans_residual:
        trans_base = sample.get("trans_base")
        tb = trans_base.numpy() if trans_base is not None else None
        pred_world_np = pred_world[0].numpy()
        gt_world_np = gt_world[0].numpy()
        for arr in (pred_world_np, gt_world_np, proposal_world):
            recompose_residual_world_np(arr, tb)
        pred_world = torch.from_numpy(pred_world_np).unsqueeze(0)
        gt_world = torch.from_numpy(gt_world_np).unsqueeze(0)
    pred_np = pred_world[0].numpy()
    if not np.isfinite(pred_np).all():
        raise ValueError(f"Non-finite prediction for {clip_name}; refusing to save it")
    sample["_state_rot_column"] = rot_column
    return pred_np, gt_world[0].numpy(), T, sample, proposal_world


def save_pred_npz(clip_name: str, pred_world: np.ndarray, betas_L: np.ndarray,
                  betas_R: np.ndarray, out_path: str):
    """pred_world: (T, 108) DiT output, already denormalised."""
    T = pred_world.shape[0]
    L_rot6d = pred_world[:, 0:6]
    L_aa    = pred_world[:, 6:51]
    L_trans = pred_world[:, 51:54]
    R_rot6d = pred_world[:, 54:60]
    R_aa    = pred_world[:, 60:105]
    R_trans = pred_world[:, 105:108]

    L_R = rot6d_to_mat_np(L_rot6d)
    R_R = rot6d_to_mat_np(R_rot6d)
    L_q = mat_to_quat_wxyz(L_R)
    R_q = mat_to_quat_wxyz(R_R)

    L_poses = L_aa.reshape(T, 15, 3).astype(np.float32)
    R_poses = R_aa.reshape(T, 15, 3).astype(np.float32)

    indices = np.arange(T, dtype=np.int64)

    payload = dict(
        sequence=np.array(clip_name),
        n_frames=np.int64(T),
        indices=indices,
        L_poses=L_poses,
        R_poses=R_poses,
        L_t=L_trans.astype(np.float32),
        R_t=R_trans.astype(np.float32),
        L_q=L_q,
        R_q=R_q,
        L_beta=betas_L.astype(np.float32),
        R_beta=betas_R.astype(np.float32),
        L_present=np.ones(T, dtype=np.uint8),
        R_present=np.ones(T, dtype=np.uint8),
    )
    for key, value in payload.items():
        if np.asarray(value).dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Non-finite {key} for {clip_name}; refusing to save predictions")
    np.savez_compressed(out_path, **payload)


def canonical_camera(gt_raw, train_args, mano_l, mano_r, clip_name):
    """Camera track in the canonical frame: the estimated one the model consumed."""
    from data_loaders import tcanon as _tc
    gravity = getattr(train_args, "gravity_canonical", None)
    if gravity is not None:
        gt_raw, _ = _tc.tcanon_forward(
            dict(gt_raw), _tc.gravity_down(gravity),
            {"left": mano_l, "right": mano_r},
            hand_convention=getattr(train_args, "tcanon_hand_convention", "legacy"))
    from data_loaders.clip_dataset import _fix_rot6d_convention
    from data_loaders.geometry import rotation_6d_to_matrix
    fixed = _fix_rot6d_convention(np.asarray(gt_raw["cam_rot_6d"], dtype=np.float32))
    trans = np.asarray(gt_raw["cam_trans"], dtype=np.float64)
    if getattr(train_args, "use_da3_train", False):
        from data_loaders.clip_dataset import _build_da3_cam_feats
        cam = _build_da3_cam_feats(
            clip_name, train_args.da3_train_dir, fixed, trans,
            apply_r_w2d=not bool(getattr(train_args, "da3_no_r_w2d", False)))
        fixed, trans = cam[:, :6], cam[:, 6:9].astype(np.float64)
        R = rotation_6d_to_matrix(torch.from_numpy(fixed).float()).numpy()
        return R.transpose(0, 2, 1), trans
    R = rotation_6d_to_matrix(torch.from_numpy(fixed).float()).numpy()
    return R.transpose(0, 2, 1), trans


def read_rgb_frames(rgb_path, T, size):
    """JPEG frames for the recording, or None when the clip has no video."""
    if rgb_path is None or not os.path.exists(rgb_path):
        print(f"  no RGB video at {rgb_path} — recording the camera without images")
        return None, None
    cap = cv2.VideoCapture(rgb_path)
    frames, shape = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shape = frame.shape[:2]
        scale = size / max(shape)
        small = cv2.resize(frame, (round(shape[1] * scale), round(shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small,
                               [int(cv2.IMWRITE_JPEG_QUALITY), RRD_IMAGE_QUALITY])
        if not ok:
            raise RuntimeError(f"failed to JPEG-encode a frame of {rgb_path}")
        frames.append(buf.tobytes())
    cap.release()
    if len(frames) < T:
        raise ValueError(
            f"{rgb_path} has {len(frames)} frames but the clip is {T} frames "
            "long; the RGB video must cover the whole clip.")
    return frames[:T], shape


def rrd_and_metrics(
    clip_name: str, pred_world: np.ndarray, gt_world: np.ndarray, T: int,
    train_args, rrd_path: str, mano_dir: str, *, fps: float = 30.0,
    write_rrd: bool = True,
    mano_left_convention: str = "patched",
    rot_column: bool,
    rgb_path: Optional[str] = None,
):
    import rerun as rr
    mano_l, mano_r = create_mano_layers(mano_dir, mano_left_convention)

    gt_raw = np.load(os.path.join(train_args.data_dir, f"{clip_name}.npz"))
    betas_l = torch.tensor(gt_raw["left_betas"], dtype=torch.float32)
    betas_r = torch.tensor(gt_raw["right_betas"], dtype=torch.float32)

    mano_faces = load_mano_faces(mano_dir)
    if write_rrd:
        os.makedirs(os.path.dirname(rrd_path) or ".", exist_ok=True)
        rr.init(f"StableHand {clip_name}", spawn=False)
        rr.save(rrd_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    def _log_mesh(path, verts, color):
        if not write_rrd:
            return
        rr.log(path, rr.Mesh3D(
            vertex_positions=verts, triangle_indices=mano_faces,
            vertex_colors=np.tile(np.array(color, dtype=np.uint8), (verts.shape[0], 1))))

    pred_world_t = torch.from_numpy(pred_world)
    gt_world_t = torch.from_numpy(gt_world)

    metrics = {h: {"pred_j_full": np.zeros((T, 16, 3), dtype=np.float32),
                   "gt_j_full": np.zeros((T, 16, 3), dtype=np.float32),
                   "valid_mask": np.zeros(T, dtype=bool)} for h in ["left", "right"]}
    hands_info = [
        ("left",  0, mano_l, betas_l),
        ("right", 54, mano_r, betas_r),
    ]
    rgb_frames = None
    if write_rrd:
        cam_R, cam_t = canonical_camera(gt_raw, train_args, mano_l, mano_r, clip_name)
        rolled = not bool(getattr(train_args, "da3_no_r_w2d", False))
        rgb_frames, source_shape = read_rgb_frames(rgb_path, T, RRD_IMAGE_SIZE)
        if rgb_frames is not None:
            src_h, src_w = source_shape
            fx, fy, cx, cy = [float(v) for v in gt_raw["image_intrinsics"]]
            scale = RRD_IMAGE_SIZE / max(src_h, src_w)
            w, h = round(src_w * scale), round(src_h * scale)
            if rolled:
                K = np.array([[fy * scale, 0.0, (src_h - 1 - cy) * scale],
                              [0.0, fx * scale, cx * scale],
                              [0.0, 0.0, 1.0]])
                w, h = h, w
            else:
                K = np.array([[fx * scale, 0.0, cx * scale],
                              [0.0, fy * scale, cy * scale],
                              [0.0, 0.0, 1.0]])
            rr.log("world/camera/image",
                   rr.Pinhole(image_from_camera=K, width=w, height=h),
                   static=True)

    for fi in range(T):
        if write_rrd:
            rr.set_time("frame", sequence=fi)
            rr.log("world/camera", rr.Transform3D(
                translation=cam_t[fi],
                mat3x3=cam_R[fi] @ IMAGE_ROLL.T if rolled else cam_R[fi]))
            if rgb_frames is not None:
                rr.log("world/camera/image",
                       rr.EncodedImage(contents=rgb_frames[fi],
                                       media_type="image/jpeg"))
        for hand, offset, mano_m, betas_h in hands_info:
            valid = gt_raw[f"{hand}_valid"][fi]
            pr_v = None
            if valid:
                gt_v, gt_j = fk_frame(
                    gt_world_t[fi], offset, betas_h, mano_m, rot_column)
                pr_v, pr_j = fk_frame(
                    pred_world_t[fi], offset, betas_h, mano_m, rot_column)
                metrics[hand]["pred_j_full"][fi] = pr_j
                metrics[hand]["gt_j_full"][fi] = gt_j
                metrics[hand]["valid_mask"][fi] = True
                _log_mesh(f"world/gt_{hand}/mesh", gt_v, HAND_COLORS["gt"][hand])
            else:
                if write_rrd:
                    rr.log(f"world/gt_{hand}", rr.Clear.recursive())

            if pr_v is None:
                pr_v, _ = fk_frame(
                    pred_world_t[fi], offset, betas_h, mano_m, rot_column)
            _log_mesh(f"world/pred_{hand}/mesh", pr_v, HAND_COLORS["pred"][hand])


    out = {}
    for hand in ["left", "right"]:
        out[hand] = compute_hand_metrics(
            torch.from_numpy(metrics[hand]["pred_j_full"]).float(),
            torch.from_numpy(metrics[hand]["gt_j_full"]).float(),
            metrics[hand]["valid_mask"], fps=fps)

    L_valid = metrics["left"]["valid_mask"]
    R_valid = metrics["right"]["valid_mask"]
    both_valid = L_valid & R_valid
    if both_valid.sum() >= 2:
        pL = torch.from_numpy(metrics["left"]["pred_j_full"][both_valid, 0]).float()
        pR = torch.from_numpy(metrics["right"]["pred_j_full"][both_valid, 0]).float()
        gL = torch.from_numpy(metrics["left"]["gt_j_full"][both_valid, 0]).float()
        gR = torch.from_numpy(metrics["right"]["gt_j_full"][both_valid, 0]).float()
        out["MRRPE"] = float(compute_mrrpe(pL, pR, gL, gR))
        out["MRRPE_n_frames"] = int(both_valid.sum())
    else:
        out["MRRPE"] = None
        out["MRRPE_n_frames"] = int(both_valid.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clips", default=None,
                    help="comma-separated clip names; default = the packaged HOT3D examples")
    ap.add_argument("--clips_file", default=None,
                    help="newline-separated clip names; blank and # lines ignored")
    ap.add_argument("--no_rrd", action="store_true",
                    help="Skip writing per-clip Rerun recordings.")
    ap.add_argument("--per_clip_dir", default=None,
                    help="Write crash-safe per-clip metrics JSON files here.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Reuse valid per-clip metrics JSON files instead of inference.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_steps", type=positive_int, default=20)
    ap.add_argument("--seed", type=random_seed, default=42,
                    help="Quality-scheduled initialization seed.")
    ap.add_argument("--mano_dir", default="data_loaders/mano_models")
    ap.add_argument("--fps", type=positive_float, default=30.0,
                    help="Native clip frame rate used to report AccErr in m/s^2.")
    ap.add_argument("--normalizer_dir", default=None,
                    help="Directory containing the checkpoint's normalization statistics.")
    ap.add_argument("--data_dir", default=None, help="Ground-truth clip NPZ directory.")
    ap.add_argument("--wilor_dir", default=None, help="WiLoR proposal cache directory.")
    ap.add_argument("--visual_feat_dir", default=None, help="Per-hand visual feature directory.")
    ap.add_argument("--scene_feat_dir", default=None, help="VGGT-Omega scene feature directory.")
    ap.add_argument("--scene_feat_key",  default=None,
                    help="Scene feature field in each cache NPZ (released models use scene_pooled).")
    ap.add_argument("--depth_signal_dir", default=None,
                    help="VGGT-Omega per-clip depth-at-hand signal directory; "
                         "required by checkpoints with qn_da3_depth_dim>0.")
    ap.add_argument("--rgb_dir", default="data/hot3d/rgb",
                    help="Per-clip input videos (<clip>.mp4) shown in the Rerun "
                         "recording. Shipped for the example clips only; without "
                         "one the recording keeps the camera but no image.")
    ap.add_argument("--vggt_cam_dir", default=None,
                    help="Override the VGGT-Omega estimated per-frame camera cache "
                         "directory.")
    ap.add_argument("--qn_ckpt", required=True,
                    help="Released Quality Network checkpoint for predicting proposal quality.")
    ap.add_argument("--qn_calibration_clips", default=None,
                    help="Comma-separated non-test IDs for σ=p80(predicted error)/ln(10). "
                         "Requires complete calibration caches and a test split. "
                         "Only used with an error-output QN; overlap is rejected.")
    ap.add_argument("--qn_calibration_clips_file", default=None,
                    help="Newline-separated non-test calibration IDs; alternative "
                         "to --qn_calibration_clips.")
    ap.add_argument("--test_split_file", default=None,
                    help="Complete benchmark splits.json used to exclude every test "
                         "clip from calibration; defaults to data_dir/../splits.json.")
    ap.add_argument("--qn_n_calibration", type=positive_int, default=30,
                    help="Number of non-test calibration clips. The released test-only "
                         "caches require the documented --qn_sigma_override.")
    ap.add_argument("--qn_sigma_override", type=sigma_vector, default=None,
                    help="Comma-separated positive finite σ_deploy values in mm, "
                         "one per QN quality group. Bypasses calibration.")
    args = ap.parse_args()

    if args.qn_calibration_clips is not None and args.qn_calibration_clips_file is not None:
        ap.error("Specify only one calibration list")
    has_calibration_list = (args.qn_calibration_clips is not None
                            or args.qn_calibration_clips_file is not None)
    if has_calibration_list and args.qn_sigma_override is not None:
        ap.error("A calibration list and --qn_sigma_override are mutually exclusive")
    if args.clips is not None and args.clips_file is not None:
        raise SystemExit("--clips and --clips_file are mutually exclusive")
    if args.skip_existing and args.per_clip_dir is None:
        raise SystemExit("--skip_existing requires --per_clip_dir")

    cpu_threads = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(min(2, cpu_threads))

    if args.clips_file is not None:
        clips = read_clip_list(args.clips_file)
    else:
        clips = clip_names(args.clips.split(",") if args.clips is not None else DEFAULT_CLIPS)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.per_clip_dir is not None:
        os.makedirs(args.per_clip_dir, exist_ok=True)

    print(f"Loading model from {args.ckpt}...")
    (model, qn, qg, scheduler, normalizer, train_args,
     checkpoint_identity) = load_model(args.ckpt, device, normalizer_dir=args.normalizer_dir)
    checkpoint_provenance = {
        "checkpoint_use_est_cam": bool(getattr(train_args, "use_est_cam", False)),
        "checkpoint_use_vggt_cam": bool(getattr(train_args, "use_da3_train", False)),
        "checkpoint_vggt_cam_dir": getattr(train_args, "da3_train_dir", None),
        "checkpoint_vggt_no_r_w2d": bool(getattr(train_args, "da3_no_r_w2d", False)),
        "checkpoint_normalizer_dir": getattr(train_args, "normalizer_dir", None),
        "checkpoint_trans_residual": bool(getattr(train_args, "trans_residual", False)),
        "checkpoint_trans_base_smooth_win": int(
            getattr(train_args, "trans_base_smooth_win", 0) or 0
        ),
        "checkpoint_trans_base_edge_mode": getattr(
            train_args, "trans_base_edge_mode", "clamp"
        ),
        "checkpoint_anchor_conf_min": float(
            getattr(train_args, "anchor_conf_min", 0.0) or 0.0
        ),
        "checkpoint_gravity_canonical": getattr(train_args, "gravity_canonical", None),
        "checkpoint_scene_feat_dir": getattr(train_args, "scene_feat_dir", None),
        "checkpoint_scene_feat_key": getattr(train_args, "scene_feat_key", None),
    }

    qn_sigma_w = None
    qn_sigma_f = None
    qn_sigma_vec = None
    qn_sigma_source = None
    calib_clips = None
    calibration_test_split = None
    print(f"Loading standalone QN from {args.qn_ckpt}...")
    qn_predict_model, qn_predict_args = load_qn_only(args.qn_ckpt, device)
    print(f"  QN arch: {qn_predict_args.qn_arch}, "
          f"output_percomp={getattr(qn_predict_args, 'qn_output_percomp', False)}, "
          f"output_mode={getattr(qn_predict_args, 'qn_output_mode', 'q')}, "
          f"cam_feat_dim={getattr(qn_predict_args, 'qn_cam_feat_dim', 0)}, "
          f"params: {sum(p.numel() for p in qn_predict_model.parameters()):,}")
    dit_granularity = int(getattr(train_args, "q_granularity", 2))
    qn_granularity = int(
        getattr(qn_predict_args, "q_granularity", 2))
    if qn_granularity != dit_granularity:
        raise ValueError(
            "QN/DiT q_granularity mismatch: "
            f"QN={qn_granularity}, DiT={dit_granularity}")

    if args.data_dir:        train_args.data_dir        = args.data_dir
    if args.wilor_dir:       train_args.wilor_dir       = args.wilor_dir
    if args.visual_feat_dir: train_args.visual_feat_dir = args.visual_feat_dir
    if args.scene_feat_dir:  train_args.scene_feat_dir  = args.scene_feat_dir
    if args.scene_feat_key:  train_args.scene_feat_key  = args.scene_feat_key
    if args.depth_signal_dir: train_args.depth_signal_dir = args.depth_signal_dir
    if args.vggt_cam_dir: train_args.da3_train_dir = args.vggt_cam_dir
    ckpt_left = getattr(train_args, "tcanon_mano_left", None)
    if ckpt_left not in ("patched", "official"):
        raise ValueError(
            "checkpoint args.json has no valid tcanon_mano_left (got "
            f"{ckpt_left!r}); cannot pick the left MANO model for scoring")
    if args.normalizer_dir:
        train_args.normalizer_dir = args.normalizer_dir
    _rec_mano = getattr(train_args, "mano_dir", None)
    if getattr(train_args, "gravity_canonical", None) and (
            not _rec_mano or not os.path.isdir(_rec_mano)):
        train_args.mano_dir = args.mano_dir

    loader = ClipMotionLoader(
        base_dir=train_args.data_dir,
        normalizer_dir=train_args.normalizer_dir,
        wilor_dir=getattr(train_args, "wilor_dir", None),
        visual_feat_dir=getattr(train_args, "visual_feat_dir", None),
        scene_feat_dir=getattr(train_args, "scene_feat_dir", None),
        scene_feat_key=getattr(train_args, "scene_feat_key", "geo_feats_cls"),
        depth_signal_dir=getattr(train_args, "depth_signal_dir", None),
        use_da3_train=getattr(train_args, "use_da3_train", False),
        da3_train_dir=getattr(train_args, "da3_train_dir", None),
        da3_apply_r_w2d=not getattr(train_args, "da3_no_r_w2d", False),
        gravity_canonical=getattr(train_args, "gravity_canonical", None),
        tcanon_hand_convention=getattr(train_args, "tcanon_hand_convention", "legacy"),
        tcanon_mano_left=getattr(train_args, "tcanon_mano_left", "official"),
        proposal_orient_flip=bool(getattr(train_args, "proposal_orient_flip", False)),
        mano_dir=getattr(train_args, "mano_dir", None),
        trans_residual=getattr(train_args, "trans_residual", False),
        trans_base_smooth_win=getattr(train_args, "trans_base_smooth_win", 0),
        trans_base_edge_mode=getattr(
            train_args, "trans_base_edge_mode", "clamp"),
        anchor_conf_min=getattr(train_args, "anchor_conf_min", 0.0),
        anchor_flip_guard=getattr(train_args, "anchor_flip_guard", False),
    )
    if args.qn_sigma_override is not None:
        sigma_values = args.qn_sigma_override
        expected_groups = int(getattr(qn_predict_args, "q_granularity", 2))
        if len(sigma_values) != expected_groups:
            raise ValueError(
                f"--qn_sigma_override expected {expected_groups} values for "
                f"{expected_groups} QN quality groups "
                f"({2 * expected_groups} outputs), got {len(sigma_values)}")
        qn_sigma_vec = np.asarray(sigma_values, dtype=np.float64)
        if len(qn_sigma_vec) == 2:
            qn_sigma_w = float(qn_sigma_vec[0])
            qn_sigma_f = float(qn_sigma_vec[1])
        qn_sigma_source = "override"
        print(
            "Using σ_deploy override: "
            f"{np.array2string(qn_sigma_vec, precision=3)} mm")
    else:
        explicit = None
        if args.qn_calibration_clips is not None:
            explicit = args.qn_calibration_clips.split(",")
        elif args.qn_calibration_clips_file is not None:
            explicit = read_clip_list(args.qn_calibration_clips_file)
        calib_clips, calibration_test_split = select_calibration_clips(
            train_args.data_dir, clips, explicit=explicit, count=args.qn_n_calibration,
            test_split_file=args.test_split_file)
        print(f"Calibrating σ_deploy on {len(calib_clips)} clips...")
        qn_sigma_vec = calibrate_qn_sigma(
            qn_predict_model, qn_predict_args, loader, calib_clips, device)
        if len(qn_sigma_vec) == 2:
            qn_sigma_w = float(qn_sigma_vec[0])
            qn_sigma_f = float(qn_sigma_vec[1])
        qn_sigma_source = "calibrated"
        print(
            "  σ_deploy: "
            f"{np.array2string(qn_sigma_vec, precision=3)} mm (p80(QN_e)/ln10)")
    summary = {
        "ckpt": args.ckpt,
        **checkpoint_identity,
        "qn_ckpt": args.qn_ckpt,
        "qn_sigma_w_calibrated": qn_sigma_w,
        "qn_sigma_f_calibrated": qn_sigma_f,
        "qn_sigma_vec_calibrated": (
            None if qn_sigma_vec is None else qn_sigma_vec.tolist()),
        "qn_sigma_source": qn_sigma_source,
        "qn_calibration_clips": calib_clips,
        "qn_calibration_test_split": calibration_test_split,
        "q_granularity": int(getattr(train_args, "q_granularity", 2)),
        "seed": args.seed,
        "n_steps": args.n_steps,
        "fps": args.fps,
        "mano_left_convention": ckpt_left,
        "effective_use_estimated_camera": bool(getattr(train_args, "use_da3_train", False)),
        "data_dir": train_args.data_dir,
        "wilor_dir": getattr(train_args, "wilor_dir", None),
        "scene_feat_dir": getattr(train_args, "scene_feat_dir", None),
        "scene_feat_key": getattr(train_args, "scene_feat_key", None),
        "depth_signal_dir": getattr(train_args, "depth_signal_dir", None),
        "rgb_dir": args.rgb_dir,
        "normalizer_dir": getattr(train_args, "normalizer_dir", None),
        "accerr_convention": "contiguous_v2",
        "anchor_flip_guard": bool(getattr(train_args, "anchor_flip_guard", False)),
        "trans_residual": bool(getattr(train_args, "trans_residual", False)),
        "trans_base_smooth_win": int(
            getattr(train_args, "trans_base_smooth_win", 0) or 0
        ),
        "trans_base_edge_mode": getattr(
            train_args, "trans_base_edge_mode", "clamp"
        ),
        "anchor_conf_min": float(
            getattr(train_args, "anchor_conf_min", 0.0) or 0.0
        ),
        **checkpoint_provenance,
        "estimated_camera_dir": getattr(train_args, "da3_train_dir", None),
        "estimated_camera_apply_r_w2d": not bool(getattr(train_args, "da3_no_r_w2d", False)),
        "gravity_canonical": getattr(train_args, "gravity_canonical", None),
        "tcanon_hand_convention": getattr(train_args, "tcanon_hand_convention", "legacy"),
        "tcanon_mano_left": getattr(train_args, "tcanon_mano_left", "official"),
        "proposal_orient_flip": bool(getattr(train_args, "proposal_orient_flip", False)),
        "cpu_thread_limits": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "clips": {},
    }

    cache_run = None
    if args.per_clip_dir is not None:
        # Hash content, not just paths or mtimes: replacing weights, normalization
        # statistics or conditioning caches in place must invalidate old results.
        import scipy
        ignored = {"out_dir", "per_clip_dir", "skip_existing", "clips", "clips_file"}
        files = [args.ckpt, os.path.join(os.path.dirname(args.ckpt), "args.json"),
                 os.path.join(train_args.normalizer_dir, "mean.pt"),
                 os.path.join(train_args.normalizer_dir, "std.pt")]
        if calibration_test_split is not None:
            files.append(calibration_test_split)
        if args.qn_ckpt:
            files += [args.qn_ckpt, os.path.join(os.path.dirname(args.qn_ckpt), "args.json")]
        for directory in sorted({args.mano_dir, getattr(train_args, "mano_dir", args.mano_dir)} - {None}):
            files += [os.path.join(directory, f"MANO_{side}.pkl") for side in ("LEFT", "RIGHT")]
        cache_run = fingerprint({
            "schema": CACHE_SCHEMA,
            "cli": {key: value for key, value in vars(args).items() if key not in ignored},
            "effective_config": vars(train_args), "qn_config": vars(qn_predict_args) if qn_predict_args else None,
            "summary": {key: value for key, value in summary.items() if key != "clips"},
            "files": [file_identity(path) for path in files],
            "source": source_identity(Path(__file__).resolve().parents[1]),
            "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                        "scipy": scipy.__version__, "opencv": cv2.__version__,
                        "python": sys.version, "smplx": package_version("smplx"),
                        "chumpy": package_version("chumpy"), "rerun": package_version("rerun-sdk"),
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "cudnn": torch.backends.cudnn.version(),
                        "matmul_precision": torch.get_float32_matmul_precision(),
                        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "deterministic": torch.are_deterministic_algorithms_enabled()},
        })
        summary["cache_schema_version"] = CACHE_SCHEMA
        summary["run_fingerprint"] = cache_run

    for c in clips:
        artifacts = {"prediction": os.path.join(args.out_dir, f"{c}.npz")}
        if not args.no_rrd:
            artifacts["rrd"] = os.path.join(args.out_dir, f"{c}.rrd")
        cache_identity = None
        if cache_run is not None:
            cache_identity = fingerprint({"run": cache_run, "clip": c,
                "inputs": clip_input_identity(c, vars(train_args), args.rgb_dir)})
        if args.skip_existing:
            per_clip_path = os.path.join(args.per_clip_dir, f"{c}.json")
            cached, reason = read_cached_metrics(per_clip_path, c, cache_identity, artifacts)
            if cached is not None:
                summary["clips"][c] = cached
                print(f"{c}: skip (cached, fingerprint verified)")
                continue
            print(f"{c}: recompute ({reason})")
        print(f"\n=== {c} ===")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        pred_world, gt_world, T, sample, proposal_world = infer_clip(
            c, loader, model, qg, scheduler, normalizer, train_args,
            device, args.n_steps,
            qn_predict_model=qn_predict_model,
            qn_predict_args=qn_predict_args,
            qn_sigma_vec=qn_sigma_vec)

        gt_raw = np.load(os.path.join(train_args.data_dir, f"{c}.npz"))
        betas_L = np.asarray(gt_raw["left_betas"], dtype=np.float32)
        betas_R = np.asarray(gt_raw["right_betas"], dtype=np.float32)
        npz_path = os.path.join(args.out_dir, f"{c}.npz")
        save_pred_npz(c, pred_world, betas_L, betas_R, npz_path)
        print(f"  saved NPZ → {npz_path}  ({T} frames)")

        rrd_path = os.path.join(args.out_dir, f"{c}.rrd")
        m = rrd_and_metrics(
            c, pred_world, gt_world, T, train_args, rrd_path, args.mano_dir,
            fps=args.fps, write_rrd=not args.no_rrd,
            mano_left_convention=ckpt_left,
            rgb_path=os.path.join(args.rgb_dir, f"{c}.mp4") if args.rgb_dir else None,
            rot_column=sample["_state_rot_column"])
        if not args.no_rrd:
            print(f"  saved RRD → {rrd_path}")
        for hand in ("left", "right"):
            if m[hand] is None:
                print(f"  {hand}: (insufficient valid frames)")
            else:
                acceleration = m[hand]["AccErr"]
                acc_text = "n/a" if acceleration is None else f"{acceleration:.2f}"
                print(f"  {hand}: W={m[hand]['W-MPJPE']:.2f}  "
                      f"WA={m[hand]['WA-MPJPE']:.2f}  "
                      f"PA={m[hand]['PA-MPJPE']:.2f}  "
                      f"Acc={acc_text}")
        summary["clips"][c] = m
        if args.per_clip_dir is not None:
            per_clip_path = os.path.join(args.per_clip_dir, f"{c}.json")
            if cache_identity != fingerprint({"run": cache_run, "clip": c,
                    "inputs": clip_input_identity(c, vars(train_args), args.rgb_dir)}):
                raise RuntimeError(f"Input files changed while evaluating {c}; refusing to cache its metrics")
            write_cached_metrics(per_clip_path, c, cache_identity, m, artifacts,
                {"ckpt": args.ckpt, "seed": args.seed, "n_steps": args.n_steps,
                 "run_fingerprint": cache_run})

    summary.update(summarize_metrics(summary["clips"]))

    ranked = []
    for c, m in summary["clips"].items():
        if not m:
            continue
        hw = [m[h]["W-MPJPE"] for h in ("left", "right") if m.get(h) is not None]
        if hw:
            ranked.append({"clip": c, "mean_W-MPJPE": float(np.mean(hw)),
                           "L": m["left"]["W-MPJPE"] if m.get("left") else None,
                           "R": m["right"]["W-MPJPE"] if m.get("right") else None})
    ranked.sort(key=lambda x: x["mean_W-MPJPE"])
    summary["clips_ranked_by_W"] = ranked

    summary_path = os.path.join(args.out_dir, "metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"\n=== Summary ===")
    for k in ("mean_W-MPJPE", "mean_WA-MPJPE", "mean_PA-MPJPE", "mean_AccErr", "n_hands"):
        print(f"  {k}: {summary[k]}")
    print(f"\n=== Per-clip ranked (best → worst) ===")
    for r in ranked:
        lr = f"L={r['L']:.2f}  R={r['R']:.2f}" if r['L'] is not None and r['R'] is not None else "partial"
        print(f"  {r['clip']:<15}  mean_W={r['mean_W-MPJPE']:6.2f}   ({lr})")
    print(f"\nmetrics.json → {summary_path}")


if __name__ == "__main__":
    main()
