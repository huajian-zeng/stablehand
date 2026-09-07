"""World-frame 3-D comparison video: input video | ground truth | StableHand."""

import argparse
import json
import os
import pickle

import cv2
import numpy as np
import torch

from data_loaders import tcanon as _tc
from data_loaders.tcanon import state_rot_is_column
from eval.metrics import _gt_rot6d_to_rotvec, _quat_wxyz_to_rotvec
from utils.mano import create_mano_layers
from utils.validation import positive_float, positive_int

SIDE_COLOR = {"L": (235, 140, 40), "R": (60, 90, 235)}
BACKGROUND = 24
GRID_COLOR = (44, 44, 48)
GT_TRAIL = (58, 96, 58)
CAMERA = (150, 150, 150)

ELEVATION_DEG = 35.0


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError(f"cannot normalise a degenerate vector: {v}")
    return v / n


def look_at(centre, up, gaze_h, radius, size, extra=None,
            elevation_deg=ELEVATION_DEG, fill=1.35):
    """Fixed virtual camera, elevated behind the scene, looking along `gaze_h`."""
    up = _unit(np.asarray(up, dtype=np.float64))
    gaze_h = np.asarray(gaze_h, dtype=np.float64)
    gaze_h = _unit(gaze_h - up * float(gaze_h @ up))
    elev = np.radians(elevation_deg)

    eye_dir = -np.cos(elev) * gaze_h + np.sin(elev) * up
    forward = _unit(-eye_dir)
    right = _unit(np.cross(forward, up))
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])

    f = 0.9 * min(size)
    K = (f, f, size[0] / 2.0, size[1] / 2.0)
    dist = f * radius / (fill * 0.5 * min(size)) + radius
    if extra is not None:
        # Pull back until the head-camera track is on screen as well.
        for _ in range(40):
            uv, ok, _ = project(extra, R, centre + eye_dir * dist, K)
            if ok.all() and np.abs(uv - np.array(K[2:])).max() <= 0.47 * min(size):
                break
            dist *= 1.08
    return R, centre + eye_dir * dist, K


def camera_heading(cam_rot6d_fixed, up, points, centre):
    """Mean horizontal heading of the head camera, so left/right match the video."""
    from data_loaders.geometry import rotation_6d_to_matrix
    R = rotation_6d_to_matrix(torch.from_numpy(
        np.asarray(cam_rot6d_fixed, np.float32)).float()).numpy().transpose(0, 2, 1)
    forward = R[:, :, 2].mean(axis=0)
    horizontal = forward - up * float(forward @ up)
    if np.linalg.norm(horizontal) > 1e-3:
        return horizontal
    spread = (points - centre) - np.outer((points - centre) @ up, up)
    _, _, vt = np.linalg.svd(spread, full_matrices=False)
    return vt[0]


def image_left_right(gt_npz, rolled):
    """Sign of (left - right) along the displayed image's horizontal axis, per frame."""
    from data_loaders.clip_dataset import _fix_rot6d_convention
    from data_loaders.geometry import rotation_6d_to_matrix
    fixed = _fix_rot6d_convention(np.asarray(gt_npz["cam_rot_6d"], dtype=np.float32))
    R = rotation_6d_to_matrix(torch.from_numpy(fixed).float()).numpy().transpose(0, 2, 1)
    t = np.asarray(gt_npz["cam_trans"], dtype=np.float64)
    fx, fy, cx, cy = [float(v) for v in gt_npz["image_intrinsics"]]
    out = np.zeros(len(t))
    for i in range(len(t)):
        if not (gt_npz["left_valid"][i] and gt_npz["right_valid"][i]):
            continue
        xs = []
        for hand in ("left_trans", "right_trans"):
            p = R[i].T @ (np.asarray(gt_npz[hand][i], dtype=np.float64) - t[i])
            if p[2] < 1e-3:
                break
            xs.append(-fy * p[1] / p[2] if rolled else fx * p[0] / p[2])
        if len(xs) == 2:
            out[i] = np.sign(xs[0] - xs[1])
    return out


IMAGE_ROLL_T = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def estimated_camera(clip, cam_rot6d, cam_t, ck):
    """The estimated camera track the model consumed, in the canonical frame."""
    from data_loaders.clip_dataset import _build_da3_cam_feats, _fix_rot6d_convention
    fixed = _fix_rot6d_convention(np.asarray(cam_rot6d, np.float32))
    if not getattr(ck, "use_est_cam", False):
        return fixed, cam_t
    cam = _build_da3_cam_feats(clip, ck.cam_dir, fixed, cam_t,
                               apply_r_w2d=not bool(getattr(ck, "no_r_w2d", False)))
    return cam[:, :6], cam[:, 6:9].astype(np.float64)


def camera_frustum(cam_rot6d_fixed, cam_t, intrinsics, shape, rolled, depth):
    """Camera frustum corners per frame, in the canonical frame."""
    from data_loaders.geometry import rotation_6d_to_matrix
    R = rotation_6d_to_matrix(torch.from_numpy(
        np.asarray(cam_rot6d_fixed, np.float32)).float()).numpy().transpose(0, 2, 1)
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    h, w = shape
    xs = np.array([-cx, w - cx]) / fx * depth
    ys = np.array([-cy, h - cy]) / fy * depth
    corners = np.array([[xs[0], ys[0], depth], [xs[1], ys[0], depth],
                        [xs[1], ys[1], depth], [xs[0], ys[1], depth]])
    if rolled:
        corners = corners @ IMAGE_ROLL_T
    return np.einsum("tij,kj->tki", R, corners) + cam_t[:, None, :]


def project(points, R, eye, K):
    """Project (..., 3) world points. Returns (uv, valid mask, camera-frame points)."""
    cam = np.einsum("ij,...j->...i", R, np.asarray(points, np.float64) - eye)
    z = cam[..., 2]
    ok = z > 1e-3
    zs = np.where(ok, z, 1.0)
    uv = np.stack([K[0] * cam[..., 0] / zs + K[2],
                   K[1] * cam[..., 1] / zs + K[3]], axis=-1)
    return uv, ok & np.isfinite(uv).all(axis=-1), cam


def ground_grid(points, up, n=11):
    """Gravity-aligned grid below the scene, as a list of (2, 3) segments."""
    centre = points.mean(0)
    height = (points - centre) @ up
    base = centre + up * (height.min() - 0.10)
    horizontal = (points - centre) - np.outer((points - centre) @ up, up)
    _, _, vt = np.linalg.svd(horizontal, full_matrices=False)
    e1 = _unit(vt[0])
    e2 = np.cross(up, e1)
    span = float(np.abs(horizontal @ e1).max() + np.abs(horizontal @ e2).max()) + 0.15
    segments = []
    for t in np.linspace(-span, span, n):
        segments.append(np.stack([base + e1 * t - e2 * span, base + e1 * t + e2 * span]))
        segments.append(np.stack([base + e2 * t - e1 * span, base + e2 * t + e1 * span]))
    return segments


def draw_mesh(image, uv, ok, faces, depths, colour, clip_margin=64):
    """Painter's-algorithm fill of (V, 2) `uv` into `image`, shaded by face depth."""
    h, w = image.shape[:2]
    points = np.rint(uv).astype(np.int32)
    base = np.asarray(colour, dtype=np.float32)
    finite = depths[np.isfinite(depths)]
    d_min = float(finite.min()) if finite.size else 0.0
    d_range = max(float(finite.max()) - d_min, 1e-6) if finite.size else 1.0
    for face_idx in np.argsort(depths)[::-1]:
        i0, i1, i2 = faces[face_idx]
        if not (ok[i0] and ok[i1] and ok[i2]):
            continue
        tri = points[[i0, i1, i2]]
        if np.all(tri[:, 0] < -clip_margin) or np.all(tri[:, 0] > w + clip_margin):
            continue
        if np.all(tri[:, 1] < -clip_margin) or np.all(tri[:, 1] > h + clip_margin):
            continue
        shade = 1.0 - 0.6 * (float(depths[face_idx]) - d_min) / d_range
        cv2.fillConvexPoly(image, tri,
                           tuple(int(c) for c in np.clip(base * shade, 0, 255)),
                           lineType=cv2.LINE_AA)


def polyline(image, uv, mask, colour):
    """Draw the masked part of a 2-D track."""
    if mask.sum() < 2:
        return
    cv2.polylines(image, [np.rint(uv[mask]).astype(np.int32).reshape(-1, 1, 2)],
                  False, colour, 1, cv2.LINE_AA)


def read_rgb(path, T, size):
    """Read T frames of <clip>.mp4, resized to `size` square."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no RGB video for this clip: {path}. The data package ships "
            "RGB videos for the HOT3D and ARCTIC example clips only; pass "
            "--rgb_dir to point at another source.")
    cap = cv2.VideoCapture(path)
    frames, shape = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shape = h, w = frame.shape[:2]
        scale = size / max(h, w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        panel = np.full((size, size, 3), BACKGROUND, np.uint8)
        y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
        panel[y:y + small.shape[0], x:x + small.shape[1]] = small
        frames.append(panel)
    cap.release()
    if len(frames) < T:
        raise ValueError(
            f"{path} has {len(frames)} frames but the clip is {T} frames long; "
            "the RGB video must cover the whole clip.")
    return frames[:T], shape


def mano_layers(mano_dir, left_convention):
    """MANO layers (left shapedirs per `left_convention`) and the shared faces."""
    left, right = create_mano_layers(mano_dir, left_convention)
    with open(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "rb") as f:
        faces = np.asarray(pickle.load(f, encoding="latin1")["f"],
                           dtype=np.int64).reshape(-1, 3)
    return left, right, faces


def mano_vertices(layer, orient_aa, pose_aa, transl, betas):
    """MANO FK, frame by frame as the metrics do."""
    T = orient_aa.shape[0]
    betas_t = torch.as_tensor(betas, dtype=torch.float32).reshape(1, -1)
    orient_t = torch.as_tensor(orient_aa, dtype=torch.float32)
    pose_t = torch.as_tensor(pose_aa, dtype=torch.float32)
    transl_t = torch.as_tensor(transl, dtype=torch.float32)
    verts = []
    with torch.no_grad():
        for frame in range(T):
            out = layer(betas=betas_t,
                        global_orient=orient_t[frame:frame + 1],
                        hand_pose=pose_t[frame:frame + 1],
                        transl=transl_t[frame:frame + 1],
                        return_verts=True)
            verts.append(out.vertices[0].numpy())
    return np.stack(verts).astype(np.float64)


def checkpoint_settings(ckpt_path):
    """Read (gravity, hand_convention, left_convention) from the checkpoint args.json."""
    args_path = os.path.join(os.path.dirname(ckpt_path), "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(
            f"no args.json next to the checkpoint: {args_path}. The renderer "
            "reads gravity_canonical / tcanon_hand_convention / tcanon_mano_left "
            "from it; pass --ckpt <dir>/model.pt of a released checkpoint.")
    with open(args_path) as f:
        args = json.load(f)
    gravity = args.get("gravity_canonical")
    _tc.gravity_down(gravity)
    left = args.get("tcanon_mano_left")
    if left not in ("patched", "official"):
        raise ValueError(
            f"checkpoint args.json has tcanon_mano_left={left!r}; expected "
            "'patched' or 'official'.")
    return (gravity, args.get("tcanon_hand_convention", "legacy"), left,
            argparse.Namespace(no_r_w2d=args.get("da3_no_r_w2d", False),
                               use_est_cam=args.get("use_da3_train", False),
                               cam_dir=args.get("da3_train_dir")))


def load_layers(pred_npz, gt_npz, gravity, hand_convention, left_layer,
                right_layer, mano_left_tcanon):
    """Ground-truth and predicted vertices in the per-clip canonical frame."""
    rot_column = state_rot_is_column(hand_convention)
    gt, _ = _tc.tcanon_forward(
        dict(gt_npz), _tc.gravity_down(gravity),
        {"left": mano_left_tcanon, "right": right_layer},
        hand_convention=hand_convention)

    T = min(int(pred_npz["L_poses"].shape[0]), int(gt["left_rot_6d"].shape[0]))
    layers = {}
    for side, hand, layer in (("L", "left", left_layer),
                              ("R", "right", right_layer)):
        layers[("gt", side)] = (
            mano_vertices(layer,
                          _gt_rot6d_to_rotvec(gt[f"{hand}_rot_6d"][:T], rot_column),
                          np.asarray(gt[f"{hand}_aa"][:T]).reshape(T, 45),
                          np.asarray(gt[f"{hand}_trans"][:T]),
                          np.asarray(gt[f"{hand}_betas"]).reshape(-1)[:10]),
            np.asarray(gt[f"{hand}_valid"][:T]).astype(bool))
        layers[("pred", side)] = (
            mano_vertices(layer,
                          _quat_wxyz_to_rotvec(pred_npz[f"{side}_q"][:T], rot_column),
                          np.asarray(pred_npz[f"{side}_poses"][:T]).reshape(T, 45),
                          np.asarray(pred_npz[f"{side}_t"][:T]),
                          np.asarray(pred_npz[f"{side}_beta"]).reshape(-1)[:10]),
            np.ones(T, dtype=bool))
    return (layers, np.asarray(gt["cam_rot_6d"][:T], np.float64),
            np.asarray(gt["cam_trans"][:T], np.float64), T)


def render(pred_path, gt_dir, rgb_dir, ckpt, out_path, mano_dir, panel, fps):
    """Write the three-panel comparison video. Returns (out_path, frames)."""
    clip = os.path.splitext(os.path.basename(pred_path))[0]
    gt_path = os.path.join(gt_dir, f"{clip}.npz")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"no ground-truth clip for {clip}: {gt_path}. --gt_dir must point at "
            "the clips_gt directory of the data package.")
    gravity, hand_convention, left_convention, _ck = checkpoint_settings(ckpt)
    left_layer, right_layer, faces = mano_layers(mano_dir, left_convention)
    mano_left_tcanon = left_layer

    pred_npz = np.load(pred_path, allow_pickle=True)
    gt_npz = np.load(gt_path, allow_pickle=True)
    layers, cam_rot6d, cam_t, T = load_layers(
        pred_npz, gt_npz, gravity, hand_convention, left_layer, right_layer,
        mano_left_tcanon)

    up = np.array([0.0, 1.0, 0.0])

    visible = np.concatenate([v[m].reshape(-1, 3)[::37]
                              for (v, m) in layers.values() if m.any()])
    centre = visible.mean(0)
    radius = float(np.percentile(np.linalg.norm(visible - centre, axis=1), 97))
    est_rot6d, est_t = estimated_camera(clip, cam_rot6d, cam_t, _ck)
    rolled = not bool(getattr(_ck, "no_r_w2d", False))
    heading = camera_heading(est_rot6d, up, visible, centre)
    R, eye, K = look_at(centre, up, heading, radius, (panel, panel), extra=est_t)
    # Two azimuths show the same scene mirrored; keep the one whose left/right
    # agrees with the input video more often.
    side = image_left_right(gt_npz, rolled)
    uL = project(layers[("gt", "L")][0][:, 0], R, eye, K)[0][:, 0]
    uR = project(layers[("gt", "R")][0][:, 0], R, eye, K)[0][:, 0]
    live = side != 0
    if live.any() and (np.sign(uL - uR)[live] == side[live]).mean() < 0.5:
        R, eye, K = look_at(centre, up, -heading, radius, (panel, panel), extra=est_t)

    grid = [project(seg, R, eye, K) for seg in ground_grid(visible, up)]
    wrist = {key: project(v[:, 0], R, eye, K)[0] for key, (v, _) in layers.items()}

    rgb, src_shape = read_rgb(os.path.join(rgb_dir, f"{clip}.mp4"), T, panel)
    frustum = camera_frustum(est_rot6d, est_t, gt_npz["image_intrinsics"],
                             src_shape, rolled, min(0.18 * radius, 0.06))
    frustum2 = np.stack([project(frustum[:, k], R, eye, K)[0] for k in range(4)], 1)
    cam2, cam_ok, _ = project(est_t, R, eye, K)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (panel * 3, panel))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open the video writer for {out_path}")
    scale = panel / 900.0
    for fi in range(T):
        rgb_panel = rgb[fi].copy()
        cv2.rectangle(rgb_panel, (0, 0), (panel, int(40 * scale)), (0, 0, 0), -1)
        cv2.putText(rgb_panel, "Input video", (int(12 * scale), int(29 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, scale * 0.85, (255, 255, 255),
                    max(1, int(2 * scale)), cv2.LINE_AA)
        cells = [rgb_panel]
        for kind, label in (("gt", "Ground truth"), ("pred", "StableHand")):
            img = np.full((panel, panel, 3), BACKGROUND, np.uint8)
            for seg2, sok, _ in grid:
                if sok.all():
                    cv2.line(img, tuple(np.rint(seg2[0]).astype(int)),
                             tuple(np.rint(seg2[1]).astype(int)),
                             GRID_COLOR, 1, cv2.LINE_AA)
            polyline(img, cam2[:fi + 1], cam_ok[:fi + 1], CAMERA)
            if cam_ok[fi]:
                apex = tuple(np.rint(cam2[fi]).astype(int))
                quad = np.rint(frustum2[fi]).astype(np.int32)
                cv2.polylines(img, [quad.reshape(-1, 1, 2)], True, CAMERA, 1, cv2.LINE_AA)
                for corner in quad:
                    cv2.line(img, apex, tuple(corner), CAMERA, 1, cv2.LINE_AA)
            for side in ("L", "R"):
                gt_mask = layers[("gt", side)][1][:fi + 1]
                polyline(img, wrist[("gt", side)][:fi + 1], gt_mask, GT_TRAIL)
                if kind == "pred":
                    trail_colour = tuple(int(0.62 * c) for c in SIDE_COLOR[side])
                    polyline(img, wrist[(kind, side)][:fi + 1],
                             layers[(kind, side)][1][:fi + 1], trail_colour)
            drawn = False
            for side in ("L", "R"):
                verts, mask = layers[(kind, side)]
                if not mask[fi]:
                    continue
                uv, ok, cam = project(verts[fi], R, eye, K)
                if ok.sum() < 100:
                    continue
                drawn = True
                draw_mesh(img, uv.astype(np.float32), ok, faces,
                          cam[faces, 2].mean(axis=1), SIDE_COLOR[side])
            cv2.rectangle(img, (0, 0), (panel, int(40 * scale)), (0, 0, 0), -1)
            cv2.putText(img, label, (int(12 * scale), int(29 * scale)),
                        cv2.FONT_HERSHEY_DUPLEX, scale * 0.85, (255, 255, 255),
                        max(1, int(2 * scale)), cv2.LINE_AA)
            if not drawn:
                cv2.putText(img, "hand not annotated on this frame",
                            (int(12 * scale), int(62 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, scale * 0.55,
                            (130, 140, 155), 1, cv2.LINE_AA)
            cells.append(img)
        canvas = np.hstack(cells)
        cv2.putText(canvas, f"{clip}  f{fi:03d}   world frame (gravity aligned)",
                    (panel + 10, panel - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    panel / 1400.0, (170, 170, 170), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    return out_path, T


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True,
                    help="Prediction NPZ written by sample/infer_clips.py.")
    ap.add_argument("--gt_dir", required=True,
                    help="Ground-truth clip directory (data/hot3d/clips_gt).")
    ap.add_argument("--rgb_dir", default="data/hot3d/rgb",
                    help="Directory with the clips' RGB videos (<clip>.mp4).")
    ap.add_argument("--ckpt", required=True,
                    help="Checkpoint whose args.json describes the frame "
                         "conventions (save/dit_hot3d/model.pt).")
    ap.add_argument("--out", default=None,
                    help="Output MP4. Default: <pred>_world3d.mp4.")
    ap.add_argument("--mano_dir", default="data_loaders/mano_models")
    ap.add_argument("--panel", type=positive_int, default=680,
                    help="Side length of one square panel, in pixels.")
    ap.add_argument("--fps", type=positive_float, default=30.0)
    a = ap.parse_args()
    out = a.out or (os.path.splitext(a.pred)[0] + "_world3d.mp4")
    path, frames = render(a.pred, a.gt_dir, a.rgb_dir, a.ckpt, out, a.mano_dir,
                          a.panel, a.fps)
    print(f"wrote {path}  ({frames} frames)")


if __name__ == "__main__":
    main()
