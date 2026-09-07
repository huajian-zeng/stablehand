"""Evaluation metrics for the released HOT3D and ARCTIC predictions."""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRotation
from tqdm import tqdm

from data_loaders.tcanon import state_rot_is_column
from utils.mano import create_mano_layers
from utils.validation import positive_float


def compute_w_mpjpe(pred_joints, gt_joints):
    """W-MPJPE: World-space Mean Per-Joint Position Error in mm (no alignment)."""
    return (torch.norm(pred_joints - gt_joints, dim=-1).mean() * 1000).item()


def compute_wa_mpjpe(pred_joints, gt_joints):
    """Wrist-aligned MPJPE in mm after subtracting each frame's root joint."""
    pred_rel = pred_joints - pred_joints[:, 0:1, :]
    gt_rel = gt_joints - gt_joints[:, 0:1, :]
    return (torch.norm(pred_rel - gt_rel, dim=-1).mean() * 1000).item()


def compute_pa_mpjpe(pred_joints, gt_joints):
    """Procrustes-Aligned MPJPE in mm."""
    T, J, _ = pred_joints.shape
    pred_np = pred_joints.cpu().numpy()
    gt_np = gt_joints.cpu().numpy()
    errors = []
    for t in range(T):
        P = pred_np[t]
        Q = gt_np[t]
        mu_p = P.mean(0)
        mu_q = Q.mean(0)
        Pc = P - mu_p
        Qc = Q - mu_q
        s_p = np.linalg.norm(Pc) + 1e-8
        s_q = np.linalg.norm(Qc) + 1e-8
        Pc_n = Pc / s_p
        Qc_n = Qc / s_q
        U, sv, Vt = np.linalg.svd(Pc_n.T @ Qc_n)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        s = float((sv * np.array([1.0, 1.0, d])).sum())
        aligned = Pc_n @ R * s * s_q + mu_q
        errors.append(np.sqrt(((aligned - Q) ** 2).sum(axis=-1)).mean())
    return float(np.mean(errors) * 1000)


def _contiguous_triples(valid, n_frames):
    """(T-2,) bool mask of triples whose three frames are all GT-valid."""
    if valid is None:
        return None
    v = valid.cpu().numpy() if torch.is_tensor(valid) else np.asarray(valid)
    v = v.reshape(-1).astype(bool)
    if v.shape[0] != n_frames:
        raise ValueError(
            f"valid has length {v.shape[0]} but the joint array has {n_frames} "
            "frames — the two must be on the same (uncompacted) time axis")
    return torch.from_numpy(v[:-2] & v[1:-1] & v[2:])


def _accel_terms(pred_joints, gt_joints, valid):
    """Per-joint acceleration differences in metres/frame² on contiguous valid triples."""
    pred_acc = pred_joints[2:] + pred_joints[:-2] - 2 * pred_joints[1:-1]
    gt_acc = gt_joints[2:] + gt_joints[:-2] - 2 * gt_joints[1:-1]
    err = torch.norm(pred_acc - gt_acc, dim=-1)
    keep = _contiguous_triples(valid, pred_joints.shape[0])
    if keep is None:
        return err
    if not bool(keep.any()):
        raise ValueError(
            "no GT-valid contiguous frame triple — acceleration is not "
            "measurable for this hand; exclude it instead of scoring it")
    return err[keep]


def compute_acceleration_error(pred_joints, gt_joints, fps=30.0, valid=None):
    """Acceleration error in m/s² over contiguous valid frames."""
    if pred_joints.shape[0] < 3:
        raise ValueError("no GT-valid contiguous frame triple — acceleration is not measurable")
    dt = 1.0 / fps
    return (_accel_terms(pred_joints, gt_joints, valid).mean() / (dt ** 2)).item()


def compute_hand_metrics(pred_joints, gt_joints, valid, fps=30.0):
    """Score a hand, leaving acceleration undefined without three valid neighbours."""
    valid = np.asarray(valid, dtype=bool)
    if valid.sum() < 2:
        return None
    pred_valid, gt_valid = pred_joints[valid], gt_joints[valid]
    triples = valid[:-2] & valid[1:-1] & valid[2:]
    return {
        "n_frames": int(valid.sum()),
        "W-MPJPE": float(compute_w_mpjpe(pred_valid, gt_valid)),
        "WA-MPJPE": float(compute_wa_mpjpe(pred_valid, gt_valid)),
        "PA-MPJPE": float(compute_pa_mpjpe(pred_valid, gt_valid)),
        "AccErr": (float(compute_acceleration_error(
            pred_joints, gt_joints, fps=fps, valid=valid)) if triples.any() else None),
    }


def summarize_metrics(clips):
    """Average each metric over the hands or clips for which it is defined."""
    hands = [metrics[hand] for metrics in clips.values() if metrics
             for hand in ("left", "right") if metrics.get(hand) is not None]
    summary = {"n_hands": len(hands)}
    for key in ("W-MPJPE", "WA-MPJPE", "PA-MPJPE", "AccErr"):
        values = [hand[key] for hand in hands if hand[key] is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
        summary[f"n_{key}"] = len(values)
    values = [metrics["MRRPE"] for metrics in clips.values()
              if metrics and metrics.get("MRRPE") is not None]
    summary["mean_MRRPE"] = float(np.mean(values)) if values else None
    summary["n_clips_MRRPE"] = len(values)
    return summary


def compute_mrrpe(pred_wrist_L, pred_wrist_R, gt_wrist_L, gt_wrist_R):
    """Mean Relative Root Position Error in mm (ARCTIC protocol)."""
    pred_rel = pred_wrist_L - pred_wrist_R
    gt_rel = gt_wrist_L - gt_wrist_R
    return (torch.norm(pred_rel - gt_rel, dim=-1).mean() * 1000).item()


def _matrix_to_rotvec(R):
    """Convert a matrix batch with the same OpenCV path used by `fk_frame`."""
    return np.stack([
        cv2.Rodrigues(matrix)[0].reshape(3) for matrix in R
    ]).astype(np.float32)


def _quat_wxyz_to_rotvec(q, rot_column):
    """(T, 4) wxyz quaternion → (T, 3) axis-angle, matching the pipeline's decode."""
    q = np.asarray(q, dtype=np.float64)
    R = SciRotation.from_quat(np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=-1)).as_matrix()
    if not rot_column:
        R = R.transpose(0, 2, 1)
    return _matrix_to_rotvec(R)


def _gt_rot6d_to_rotvec(rot6d, rot_column):
    """(T, 6) GT clip rot6d → (T, 3) axis-angle."""
    from data_loaders.clip_dataset import _fix_rot6d_convention
    from data_loaders.geometry import rotation_6d_to_matrix
    fixed = _fix_rot6d_convention(np.asarray(rot6d, dtype=np.float32))
    R = rotation_6d_to_matrix(torch.from_numpy(fixed).float()).numpy()
    if rot_column:
        R = R.transpose(0, 2, 1)
    return _matrix_to_rotvec(R)


def _mano_joints(layer, global_orient_aa, pose_aa_45, transl, betas):
    """MANO FK → (T, 16, 3) kinematic joints, the joint set the paper reports."""
    T = global_orient_aa.shape[0]
    betas_t = torch.as_tensor(betas, dtype=torch.float32).reshape(1, -1)
    orient_t = torch.as_tensor(global_orient_aa, dtype=torch.float32)
    pose_t = torch.as_tensor(pose_aa_45, dtype=torch.float32)
    transl_t = torch.as_tensor(transl, dtype=torch.float32)
    joints = []
    with torch.no_grad():
        for frame in range(T):
            out = layer(
                betas=betas_t,
                global_orient=orient_t[frame:frame + 1],
                hand_pose=pose_t[frame:frame + 1],
                transl=transl_t[frame:frame + 1],
                return_verts=True,
            )
            joints.append(out.joints)
    return torch.cat(joints, dim=0)


def evaluate_clip(pred_data, gt_data, mano_r, mano_l, fps=30.0,
                  gravity_canonical="hot3d", hand_convention="legacy",
                  mano_l_tcanon=None):
    """Evaluate one predicted clip against its ground-truth clip."""
    rot_column = state_rot_is_column(hand_convention)
    if gravity_canonical is not None:
        from data_loaders import tcanon as _tc
        gt_data, _ = _tc.tcanon_forward(
            dict(gt_data), _tc.gravity_down(gravity_canonical),
            {"left": mano_l_tcanon if mano_l_tcanon is not None else mano_l,
             "right": mano_r}, hand_convention=hand_convention)
    T = int(gt_data["left_rot_6d"].shape[0])
    if T == 0:
        raise ValueError("Cannot evaluate an empty clip")
    for hand, prefix in (("left", "L"), ("right", "R")):
        for suffix in ("poses", "q", "t"):
            key = f"{prefix}_{suffix}"
            if len(pred_data[key]) != T:
                raise ValueError(
                    f"Prediction {key} has {len(pred_data[key])} frames; "
                    f"expected the complete {T}-frame ground-truth clip")
        for suffix in ("rot_6d", "aa", "trans", "valid"):
            key = f"{hand}_{suffix}"
            if len(gt_data[key]) != T:
                raise ValueError(f"Ground-truth {key} must contain {T} frames")
    if "n_frames" in pred_data and int(pred_data["n_frames"]) != T:
        raise ValueError(f"Prediction n_frames must equal the ground-truth length {T}")

    out = {}
    joints = {}
    valids = {}
    for hand, layer, P, G in (("left", mano_l, "L", "left"),
                              ("right", mano_r, "R", "right")):
        valid = np.asarray(gt_data[f"{G}_valid"][:T]).astype(bool)
        pred_j = _mano_joints(
            layer,
            _quat_wxyz_to_rotvec(pred_data[f"{P}_q"][:T], rot_column),
            np.asarray(pred_data[f"{P}_poses"][:T]).reshape(T, 45),
            np.asarray(pred_data[f"{P}_t"][:T]),
            np.asarray(pred_data[f"{P}_beta"]).reshape(-1)[:10])
        gt_j = _mano_joints(
            layer,
            _gt_rot6d_to_rotvec(gt_data[f"{G}_rot_6d"][:T], rot_column),
            np.asarray(gt_data[f"{G}_aa"][:T]),
            np.asarray(gt_data[f"{G}_trans"][:T]),
            np.asarray(gt_data[f"{G}_betas"]).reshape(-1)[:10])
        joints[hand] = (pred_j, gt_j)
        valids[hand] = valid

        out[hand] = compute_hand_metrics(pred_j, gt_j, valid, fps=fps)

    both_valid = valids["left"] & valids["right"]
    if both_valid.sum() >= 2:
        out["MRRPE"] = float(compute_mrrpe(
            joints["left"][0][both_valid, 0], joints["right"][0][both_valid, 0],
            joints["left"][1][both_valid, 0], joints["right"][1][both_valid, 0]))
    else:
        out["MRRPE"] = None
    out["MRRPE_n_frames"] = int(both_valid.sum())
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Recompute release metrics from prediction NPZs written "
                    "by sample/infer_clips.py")
    parser.add_argument("--pred_dir", required=True,
                        help="Directory with prediction .npz files (clip-*.npz)")
    parser.add_argument("--gt_dir", required=True,
                        help="Directory with the matching ground-truth clip .npz files")
    parser.add_argument("--mano_model_folder", default="data_loaders/mano_models")
    parser.add_argument("--fps", type=positive_float, default=30.0,
                        help="Frame rate used by AccErr")
    parser.add_argument("--gravity_canonical", default="hot3d",
                        help="Checkpoint's gravity_canonical (dataset key). The "
                             "prediction NPZ lives in that canonical frame, so GT "
                             "is mapped into it before comparing. Pass 'none' for "
                             "a checkpoint trained without T_canon.")
    parser.add_argument("--tcanon_hand_convention", default="legacy",
                        choices=["legacy", "fixed"],
                        help="Checkpoint's tcanon_hand_convention")
    parser.add_argument("--tcanon_mano_left", default="patched",
                        choices=["official", "patched"],
                        help="Checkpoint's left MANO convention for both T_canon "
                             "rest-wrist correction and metric scoring")
    parser.add_argument("--out_json", default=None,
                        help="Optional path to write the per-clip metrics to")
    args = parser.parse_args()

    mano_l, mano_r = create_mano_layers(args.mano_model_folder, args.tcanon_mano_left)
    mano_l_tcanon = mano_l

    grav = None if args.gravity_canonical.lower() == "none" else args.gravity_canonical
    pred_files = sorted(f for f in os.listdir(args.pred_dir) if f.endswith(".npz"))
    if not pred_files:
        raise FileNotFoundError(
            f"no prediction NPZ files found in {args.pred_dir!r}; "
            "--pred_dir must contain the clip-*.npz files written by "
            "sample/infer_clips.py")
    print(f"Evaluating {len(pred_files)} predictions (fps={args.fps})...")

    clips = {}
    for pf in tqdm(pred_files):
        gt_path = os.path.join(args.gt_dir, pf)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(
                f"missing ground-truth NPZ {gt_path!r} for prediction {pf!r}; "
                "--gt_dir must contain a matching file for every prediction")
        pred_data = dict(np.load(os.path.join(args.pred_dir, pf)).items())
        gt_data = dict(np.load(gt_path).items())
        clips[os.path.splitext(pf)[0]] = evaluate_clip(
            pred_data, gt_data, mano_r, mano_l, args.fps,
            gravity_canonical=grav, hand_convention=args.tcanon_hand_convention,
            mano_l_tcanon=mano_l_tcanon)

    summary = {"fps": args.fps, "gravity_canonical": grav,
               "tcanon_hand_convention": args.tcanon_hand_convention,
               "tcanon_mano_left": args.tcanon_mano_left,
               "clips": clips}
    summary.update(summarize_metrics(clips))

    print(f"\n{'Metric':<12} {'Mean':>12} {'n':>6}")
    print("-" * 32)
    for key in ("W-MPJPE", "WA-MPJPE", "PA-MPJPE", "AccErr"):
        value = summary['mean_' + key]
        shown = "n/a" if value is None else f"{value:.4f}"
        print(f"{key:<12} {shown:>12} {summary['n_' + key]:6d}")
    if summary["mean_MRRPE"] is not None:
        print(f"{'MRRPE':<12} {summary['mean_MRRPE']:12.4f} "
              f"{summary['n_clips_MRRPE']:6d}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
