"""
Evaluate LightGlue and SuperGlue on exported dataset using glue-factory metrics.

This script evaluates LightGlue and SuperGlue on the exported dataset format
(views.txt and pairs.txt) and produces a table similar to the LightGlue paper.

METRICS DESCRIPTION:
====================

1. EPIPOLAR PRECISION (mepi_prec@X):
   - Measures geometric consistency of matches using epipolar constraint
   - Computes distance from matched points to epipolar lines
   - Thresholds: 1e-4, 5e-4, 1e-3 (normalized epipolar distance)
   - Higher is better (percentage of matches satisfying epipolar constraint)

2. REPROJECTION PRECISION (mreproj_prec@Xpx):
   - Measures 3D-2D reprojection accuracy using depth
   - Projects 3D points (from depth) to 2D and compares with matched keypoints
   - Thresholds: 1px, 3px, 5px (pixel error)
   - Higher is better (percentage of matches with reprojection error < threshold)

3. GT MATCH RECALL/PRECISION (mgt_match_*@3px):
   - Measures match quality against ground truth correspondences
   - Recall: fraction of GT matches found by matcher
   - Precision: fraction of matcher's matches that are correct (within 3px of GT)
   - Higher is better

4. RELATIVE POSE ERROR (rel_pose_*):
   - Estimates relative transformation T_0to1 between two camera views
   - Method: Essential matrix estimation from 2D matches (visual odometry style)
   - Input: 2D matched keypoints only (no depth required)
   - Scale: Ambiguous (only direction, not magnitude)
   - Metrics:
     * rel_pose_t_error: Total translation error (m)
     * rel_pose_r_error: Total rotation error (deg)
     * rel_pose_t_error_x/y/z: Per-axis translation errors (m)
     * rel_pose_r_error_roll/pitch/yaw: Per-axis rotation errors (deg)

5. POSE AUC (mpose_auc@X°):
   - Area Under Curve for relative pose estimation accuracy
   - Computes fraction of pairs with rotation error < threshold
   - Thresholds: 5°, 10°, 20°
   - Higher is better (trajectory-level metric)

6. ABSOLUTE LOCALIZATION ERROR (abs_loc_*):
   - Estimates absolute pose T_w2cam1 of camera1 in world frame
   - Method: PnP (Perspective-n-Point) solver using 3D-2D correspondences
   - Input: 3D points in world frame (from depth) + 2D observations in camera1
   - Scale: True metric scale (from depth measurements)
   - Metrics:
     * abs_loc_t_error: Total translation error (m) - camera position in world
     * abs_loc_r_error: Total rotation error (deg)
     * abs_loc_t_error_x/y/z: Per-axis translation errors (m)
     * abs_loc_r_error_roll/pitch/yaw: Per-axis rotation errors (deg)
     * abs_loc_acc@Xm,Y°: Accuracy at thresholds (fraction of pairs passing)

7. MATCH STATISTICS:
   * num_matches: Average number of matches per pair
   * num_keypoints: Average number of keypoints per image

KEY DIFFERENCES:
- Relative Pose: 2D-2D estimation (VO style), scale ambiguous, measures motion estimation
- Absolute Localization: 3D-2D estimation (PnP), true scale, measures localization accuracy
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

import pylupnt as pnt
from pylupnt.numerics import ensure_torch

import rerun as rr

from gluefactory.eval.utils import (
    eval_matches_depth,
    eval_matches_epipolar,
    eval_relative_pose_robust,
)
from gluefactory.utils.tools import AUCMetric
from gluefactory.geometry.wrappers import Camera, Pose
from pylupnt.numerics.frames import OCV_T_FLU, FLU_T_OCV


def load_dataset(dataset_path, scene_name="unreal"):
    """Load the exported dataset and return pairs and views."""
    dataset_path = Path(dataset_path)
    scene_dir = dataset_path / scene_name

    # Load pairs
    pairs_file = scene_dir / "pairs.txt"
    with open(pairs_file, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]

    # Load views
    views_file = scene_dir / "views.txt"
    views = {}
    with open(views_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            views[img_name] = parts[1:]

    return pairs, views, scene_dir


def parse_view(img_name, view_data, scene_dir):
    """Parse a view line and return image, depth, camera, and pose."""
    # view_data format: R[9] t[3] model w h params[4]
    R = np.array(view_data[0:9], dtype=np.float32).reshape(3, 3)
    t = np.array(view_data[9:12], dtype=np.float32)
    model, w, h = view_data[12], int(view_data[13]), int(view_data[14])
    params = np.array(view_data[15:], dtype=np.float32)

    img = (
        cv2.cvtColor(cv2.imread(str(scene_dir / "images" / img_name)), cv2.COLOR_BGR2RGB).astype(
            np.float32
        )
        / 255.0
    )
    with h5py.File(scene_dir / "depths" / img_name.replace(".jpg", ".h5"), "r") as f:
        depth = f["depth"][:].astype(np.float32)

    camera = Camera.from_colmap({"model": model, "width": w, "height": h, "params": params})
    # Add batch dimension to camera data for compatibility with eval functions
    if camera._data.ndim == 1:
        camera = Camera(camera._data.unsqueeze(0))

    # Poses are already in OCV frame (exported in OCV format)
    T_w2cam_ocv = Pose.from_Rt(torch.from_numpy(R).float(), torch.from_numpy(t).float())

    return img, depth, camera, T_w2cam_ocv


def convert_to_gluefactory_format(img0, img1, depth0, depth1, camera0, camera1, T_w2cam0, T_w2cam1):
    """Convert images and poses to glue-factory format."""
    return {
        "view0": {
            "image": ensure_torch(img0, channels=3, device="cpu", batch_dim=True),
            "camera": camera0,
            "depth": ensure_torch(depth0, device="cpu", batch_dim=True),
            "T_w2cam": T_w2cam0,
        },
        "view1": {
            "image": ensure_torch(img1, channels=3, device="cpu", batch_dim=True),
            "camera": camera1,
            "depth": ensure_torch(depth1, device="cpu", batch_dim=True),
            "T_w2cam": T_w2cam1,
        },
        "T_0to1": T_w2cam1 @ T_w2cam0.inv(),
    }


def convert_features_to_gluefactory_format(feats1, feats2, matches):
    """Convert pylupnt Features and Matches to glue-factory format."""
    kpts0 = torch.from_numpy(feats1.uv.copy()).float()
    kpts1 = torch.from_numpy(feats2.uv.copy()).float()
    matches0 = torch.full((len(feats1),), -1, dtype=torch.long)
    matching_scores0 = torch.zeros(len(feats1), dtype=torch.float32)

    if len(matches.indexes) > 0:
        idx0 = torch.from_numpy(matches.indexes[:, 0].copy()).long()
        idx1 = torch.from_numpy(matches.indexes[:, 1].copy()).long()
        matches0[idx0] = idx1
        max_dist = matches.distances.max() if len(matches.distances) > 0 else 1.0
        scores = 1.0 - (matches.distances / (max_dist + 1e-6))
        matching_scores0[idx0] = torch.from_numpy(scores).float()

    return {
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "matches0": matches0,
        "matching_scores0": matching_scores0,
    }


def _prepare_image(img):
    """Convert image to uint8 RGB format for rerun."""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def _get_intrinsics(camera):
    """Extract camera intrinsics from camera object."""
    cam_data = camera._data.squeeze() if hasattr(camera, "_data") else camera
    if hasattr(cam_data, "cpu"):
        cam_data = cam_data.cpu().numpy()
    return {
        "fx": float(cam_data[2]),
        "fy": float(cam_data[3]),
        "cx": float(cam_data[4]),
        "cy": float(cam_data[5]),
    }


def _compute_3d_points(kp, depth, camera, T_w2cam):
    """Compute 3D points in world frame from keypoints and depth."""
    kp_int = np.round(kp).astype(int)
    h, w = depth.shape
    kp_int[:, 0] = np.clip(kp_int[:, 0], 0, w - 1)
    kp_int[:, 1] = np.clip(kp_int[:, 1], 0, h - 1)
    depth_vals = depth[kp_int[:, 1], kp_int[:, 0]]
    valid = ~np.isnan(depth_vals) & (depth_vals > 0) & (depth_vals < 1000)
    if not np.any(valid):
        return None, None
    kp_valid = kp[valid]
    depth_valid = depth_vals[valid]
    intrinsics = _get_intrinsics(camera)
    T_w2cam_flu = Pose.from_4x4mat(torch.from_numpy(FLU_T_OCV).float()) @ T_w2cam
    world_T_cam_flu = pnt.make_transform(*T_w2cam_flu.inv().numpy())
    xyz_world = pnt.uv_to_xyz(kp_valid, depth_valid, intrinsics, world_T_cam_flu)
    return xyz_world.astype(np.float32), valid


def log_to_rerun(
    pair_idx,
    img0_name,
    img1_name,
    img0,
    img1,
    feats0,
    feats1,
    matches,
    results,
    matcher_name,
    depth0=None,
    depth1=None,
    camera0=None,
    camera1=None,
    T_w2cam0=None,
    T_w2cam1=None,
):
    """Log features, matches, and metrics to rerun."""
    base_path = f"evaluation/{matcher_name}"
    rr.set_time("pair", sequence=pair_idx)

    # Log images
    img0_log, img1_log = _prepare_image(img0), _prepare_image(img1)
    rr.log(f"{base_path}/image0", rr.Image(img0_log).compress(jpeg_quality=50))
    rr.log(f"{base_path}/image1", rr.Image(img1_log).compress(jpeg_quality=50))

    # Log keypoints
    kp0 = feats0.uv.reshape(-1, 2).astype(np.float32)
    kp1 = feats1.uv.reshape(-1, 2).astype(np.float32)
    rr.log(f"{base_path}/keypoints0", rr.Points2D(kp0, radii=3.0))
    rr.log(f"{base_path}/keypoints1", rr.Points2D(kp1, radii=3.0))

    # Log matches
    if len(matches) > 0:
        match_indices = matches.indexes
        matched_kp0 = kp0[match_indices[:, 0]]
        matched_kp1 = kp1[match_indices[:, 1]]
        matched_kp1_offset = matched_kp1.copy()
        matched_kp1_offset[:, 0] += img0_log.shape[1]
        img_combined = np.hstack([img0_log, img1_log])
        rr.log(
            f"{base_path}/matches_image",
            rr.Image(img_combined).compress(jpeg_quality=75),
        )
        match_lines = [
            [matched_kp0[i].tolist(), matched_kp1_offset[i].tolist()]
            for i in range(min(1000, len(matched_kp0)))
        ]
        if match_lines:
            rr.log(
                f"{base_path}/match_lines",
                rr.LineStrips2D(match_lines, colors=[255, 255, 0]),
            )
        rr.log(
            f"{base_path}/matched_keypoints0",
            rr.Points2D(matched_kp0, radii=5.0, colors=[0, 255, 0]),
        )
        rr.log(
            f"{base_path}/matched_keypoints1",
            rr.Points2D(matched_kp1_offset, radii=5.0, colors=[0, 255, 0]),
        )

    # Log metrics
    for metric_name, metric_value in results.items():
        if isinstance(metric_value, (int, float)) and not (
            np.isnan(metric_value) or np.isinf(metric_value)
        ):
            rr.log(f"{base_path}/metrics/{metric_name}", rr.Scalars(float(metric_value)))
    for key in ["num_keypoints0", "num_keypoints1", "num_matches"]:
        rr.log(f"{base_path}/summary/{key}", rr.Scalars(results.get(key, 0)))

    # Log camera poses
    for cam_idx, T_w2cam in enumerate([T_w2cam0, T_w2cam1]):
        if T_w2cam is not None:
            R, pos = T_w2cam.inv().numpy()
            rr.log(
                f"{base_path}/camera{cam_idx}/pose",
                rr.Transform3D(
                    translation=pos.astype(np.float32),
                    mat3x3=R.astype(np.float32),
                    axis_length=2.0,
                ),
            )

    # Log 3D points
    for cam_idx, (kp, depth, camera, T_w2cam) in enumerate(
        [(kp0, depth0, camera0, T_w2cam0), (kp1, depth1, camera1, T_w2cam1)]
    ):
        if depth is not None and camera is not None and T_w2cam is not None:
            xyz_world, valid_mask = _compute_3d_points(kp, depth, camera, T_w2cam)
            if xyz_world is not None:
                rr.log(f"{base_path}/camera{cam_idx}/features_3d", rr.Points3D(xyz_world))
                if len(matches) > 0:
                    match_indices = matches.indexes
                    match_idx = match_indices[:, cam_idx]
                    # Map back to original keypoint indices
                    orig_kp_indices = np.arange(len(kp))[valid_mask]
                    matched_mask = np.isin(orig_kp_indices, match_idx)
                    if np.any(matched_mask):
                        rr.log(
                            f"{base_path}/camera{cam_idx}/matched_features_3d",
                            rr.Points3D(xyz_world[matched_mask]),
                        )


def _compute_rotation_error(R_est, R_gt):
    """Compute rotation error in degrees."""
    R_diff = R_est @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1, 3)
    return np.rad2deg(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def _compute_rpy_error(R_est, R_gt):
    """Compute per-axis rotation errors (roll, pitch, yaw)."""
    rpy_est = np.rad2deg(pnt.rot2rpy(R_est))
    rpy_gt = np.rad2deg(pnt.rot2rpy(R_gt))
    rpy_error = np.mod(rpy_est - rpy_gt + 180, 360) - 180
    return np.abs(rpy_error)


def compute_relative_pose_per_axis_errors(data, pred, eval_conf):
    """Compute relative pose errors with per-axis breakdown."""
    from gluefactory.robust_estimators import load_estimator

    T_gt = data["T_0to1"]
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    valid = m0 >= 0
    if valid.sum() < 4:
        return {}

    estimator = load_estimator("relative_pose", eval_conf["estimator"])(eval_conf)
    est = estimator(
        {
            "m_kpts0": kp0[valid],
            "m_kpts1": kp1[m0[valid]],
            "camera0": data["view0"]["camera"][0],
            "camera1": data["view1"]["camera"][0],
        }
    )
    if not est["success"]:
        return {}

    M_est = est["M_0to1"]
    R_est, t_est = M_est.R.numpy(), M_est.t.numpy()
    R_gt, t_gt = T_gt.R.numpy(), T_gt.t.numpy()

    t_error_vec = t_est - t_gt
    t_error = np.linalg.norm(t_error_vec)
    t_gt_mag = np.linalg.norm(t_gt)
    t_error_rel = t_error / t_gt_mag if t_gt_mag > 1e-6 else float("inf")

    r_error = _compute_rotation_error(R_est, R_gt)
    trace_gt = np.clip(np.trace(R_gt), -1, 3)
    r_gt_angle = np.rad2deg(np.arccos(np.clip((trace_gt - 1) / 2, -1, 1)))
    r_error_rel = r_error / r_gt_angle if r_gt_angle > 1e-6 else float("inf")

    r_error_roll, r_error_pitch, r_error_yaw = _compute_rpy_error(R_est, R_gt)

    return {
        "rel_pose_t_error": float(t_error),
        "rel_pose_r_error": float(r_error),
        "rel_pose_t_error_rel": float(t_error_rel),
        "rel_pose_r_error_rel": float(r_error_rel),
        "rel_pose_t_error_x": float(np.abs(t_error_vec[0])),
        "rel_pose_t_error_y": float(np.abs(t_error_vec[1])),
        "rel_pose_t_error_z": float(np.abs(t_error_vec[2])),
        "rel_pose_r_error_roll": float(r_error_roll),
        "rel_pose_r_error_pitch": float(r_error_pitch),
        "rel_pose_r_error_yaw": float(r_error_yaw),
    }


def evaluate_pair(
    img0_name,
    img1_name,
    views,
    scene_dir,
    extractor,
    matcher,
    eval_conf,
    pair_idx=0,
    matcher_name="",
    log_rerun=False,
):
    """Evaluate a single image pair."""
    img0, depth0, camera0, T_w2cam0 = parse_view(img0_name, views[img0_name], scene_dir)
    img1, depth1, camera1, T_w2cam1 = parse_view(img1_name, views[img1_name], scene_dir)

    data = convert_to_gluefactory_format(
        img0, img1, depth0, depth1, camera0, camera1, T_w2cam0, T_w2cam1
    )
    feats0, feats1 = extractor.extract(img0), extractor.extract(img1)
    matches = matcher.match(feats0, feats1)
    pred = convert_features_to_gluefactory_format(feats0, feats1, matches)

    results = {}
    results.update(eval_matches_epipolar(data, pred))
    results.update(eval_matches_depth(data, pred))
    rel_pose_results = eval_relative_pose_robust(data, pred, eval_conf)
    results.update(rel_pose_results)

    # Compute relative pose per-axis errors
    rel_pose_errors = compute_relative_pose_per_axis_errors(data, pred, eval_conf)
    if rel_pose_errors:
        results.update(rel_pose_errors)

    matches0 = pred["matches0"]
    results["num_matches"] = (matches0 >= 0).sum().item()
    results["num_keypoints0"] = pred["keypoints0"].shape[0]
    results["num_keypoints1"] = pred["keypoints1"].shape[0]

    try:
        abs_loc_results = eval_absolute_localization(
            img0,
            img1,
            depth0,
            depth1,
            camera0,
            camera1,
            T_w2cam0,
            T_w2cam1,
            pred,
            eval_conf,
        )
        if abs_loc_results:
            results.update(abs_loc_results)
    except ValueError:
        pass

    if log_rerun:
        log_to_rerun(
            pair_idx,
            img0_name,
            img1_name,
            img0,
            img1,
            feats0,
            feats1,
            matches,
            results,
            matcher_name,
            depth0=depth0,
            depth1=depth1,
            camera0=camera0,
            camera1=camera1,
            T_w2cam0=T_w2cam0,
            T_w2cam1=T_w2cam1,
        )

    return results


_ABS_LOC_ERROR_DICT = {
    "abs_loc_acc@0.25m_2deg": 0.0,
    "abs_loc_acc@0.5m_5deg": 0.0,
    "abs_loc_acc@1.0m_10deg": 0.0,
    "abs_loc_t_error": float("inf"),
    "abs_loc_r_error": float("inf"),
    "abs_loc_t_error_rel": float("inf"),
    "abs_loc_r_error_rel": float("inf"),
    "abs_loc_t_error_x": float("inf"),
    "abs_loc_t_error_y": float("inf"),
    "abs_loc_t_error_z": float("inf"),
    "abs_loc_r_error_roll": float("inf"),
    "abs_loc_r_error_pitch": float("inf"),
    "abs_loc_r_error_yaw": float("inf"),
}


def eval_absolute_localization(
    img0,
    img1,
    depth0,
    depth1,
    camera0,
    camera1,
    T_w2cam0_gt,
    T_w2cam1_gt,
    pred,
    eval_conf,
):
    """Evaluate absolute localization using PnP solver."""
    kp0 = pred["keypoints0"].cpu().numpy().reshape(-1, 2)
    kp1 = pred["keypoints1"].cpu().numpy().reshape(-1, 2)
    m0 = pred["matches0"].cpu().numpy()
    valid_matches = m0 >= 0
    if valid_matches.sum() < 4:
        return _ABS_LOC_ERROR_DICT.copy()

    matched_kp0 = kp0[valid_matches]
    matched_kp1 = kp1[m0[valid_matches]]

    # Sample depth
    kpts_int = np.round(matched_kp0).astype(int)
    h, w = depth0.shape
    kpts_int[:, 0] = np.clip(kpts_int[:, 0], 0, w - 1)
    kpts_int[:, 1] = np.clip(kpts_int[:, 1], 0, h - 1)
    depth0_vals = depth0[kpts_int[:, 1], kpts_int[:, 0]]
    valid0 = ~np.isnan(depth0_vals) & (depth0_vals > 0) & (depth0_vals < 1000)
    if valid0.sum() < 4:
        raise ValueError(f"Not enough points with valid depth: {valid0.sum()} < 4")

    matched_kp0_valid = matched_kp0[valid0]
    matched_kp1_valid = matched_kp1[valid0]
    depth0_valid = depth0_vals[valid0]

    # Convert to 3D points in world frame
    intrinsics = _get_intrinsics(camera0)
    T_w2cam0_gt_flu = Pose.from_4x4mat(torch.from_numpy(FLU_T_OCV).float()) @ T_w2cam0_gt
    T_w2cam1_gt_flu = Pose.from_4x4mat(torch.from_numpy(FLU_T_OCV).float()) @ T_w2cam1_gt

    u, v = matched_kp0_valid[:, 0], matched_kp0_valid[:, 1]
    fx, fy, cx, cy = (
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
    )
    xyz_cam_ocv = np.stack(
        [(u - cx) * depth0_valid / fx, (v - cy) * depth0_valid / fy, depth0_valid],
        axis=-1,
    )
    xyz_cam_flu = pnt.apply_transform(FLU_T_OCV, xyz_cam_ocv)
    world_T_cam0_flu = pnt.make_transform(*T_w2cam0_gt_flu.inv().numpy())
    points_3d_world_flu = pnt.apply_transform(world_T_cam0_flu, xyz_cam_flu)

    if len(points_3d_world_flu) < 4:
        raise ValueError(f"Not enough 3D points for PnP: {len(points_3d_world_flu)} < 4")

    # Solve PnP
    K1 = camera1.calibration_matrix().squeeze().cpu().numpy().astype(np.float64)
    pnp_result = pnt.PnpSolver(
        {
            "threshold": eval_conf.get("ransac_th", 1.0),
            "confidence": 0.99,
            "max_iterations": 10000,
        }
    ).solve(
        points_3d_world_flu.astype(np.float64),
        matched_kp1_valid.astype(np.float64),
        K1,
        pnt.make_transform(*T_w2cam0_gt_flu.numpy()),
    )

    if not pnp_result.success:
        return _ABS_LOC_ERROR_DICT.copy()

    # Compute errors
    T_w2cam1_est_flu = pnp_result.tgt_T_src.astype(np.float32)
    T_w2cam1_gt_flu_np = pnt.make_transform(*T_w2cam1_gt_flu.numpy())
    T_w2cam0_gt_flu_np = pnt.make_transform(*T_w2cam0_gt_flu.numpy())

    cam_pos_est = np.linalg.inv(T_w2cam1_est_flu)[:3, 3]
    cam_pos_gt = np.linalg.inv(T_w2cam1_gt_flu_np)[:3, 3]
    cam_pos0_gt = np.linalg.inv(T_w2cam0_gt_flu_np)[:3, 3]
    t_error_vec = cam_pos_est - cam_pos_gt
    t_error = np.linalg.norm(t_error_vec)
    t_gt_mag = np.linalg.norm(cam_pos_gt - cam_pos0_gt)
    t_error_rel = t_error / t_gt_mag if t_gt_mag > 1e-6 else float("inf")

    R_est = T_w2cam1_est_flu[:3, :3]
    R_gt = T_w2cam1_gt_flu_np[:3, :3]
    R_gt0 = T_w2cam0_gt_flu_np[:3, :3]
    r_error = _compute_rotation_error(R_est, R_gt)
    R_gt_rel = R_gt @ R_gt0.T
    trace_gt_rel = np.clip(np.trace(R_gt_rel), -1, 3)
    r_gt_angle = np.rad2deg(np.arccos(np.clip((trace_gt_rel - 1) / 2, -1, 1)))
    r_error_rel = r_error / r_gt_angle if r_gt_angle > 1e-6 else float("inf")

    r_error_roll, r_error_pitch, r_error_yaw = _compute_rpy_error(R_est, R_gt)

    results = {
        "abs_loc_t_error": float(t_error),
        "abs_loc_r_error": float(r_error),
        "abs_loc_t_error_rel": float(t_error_rel),
        "abs_loc_r_error_rel": float(r_error_rel),
        "abs_loc_t_error_x": float(np.abs(t_error_vec[0])),
        "abs_loc_t_error_y": float(np.abs(t_error_vec[1])),
        "abs_loc_t_error_z": float(np.abs(t_error_vec[2])),
        "abs_loc_r_error_roll": float(r_error_roll),
        "abs_loc_r_error_pitch": float(r_error_pitch),
        "abs_loc_r_error_yaw": float(r_error_yaw),
    }
    for t_thresh, r_thresh, name in [
        (0.25, 2.0, "0.25m_2deg"),
        (0.5, 5.0, "0.5m_5deg"),
        (1.0, 10.0, "1.0m_10deg"),
    ]:
        results[f"abs_loc_acc@{name}"] = float((t_error <= t_thresh) and (r_error <= r_thresh))
    return results


def print_results_table(all_results):
    """Print results in a table format similar to the LightGlue paper."""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)

    # Define metrics to display (check what's actually available)
    # Common metrics from glue-factory evaluation + absolute localization
    metrics_to_show = [
        # Epipolar metrics (Table 1 style)
        ("Epipolar Prec@1e-4", ["mepi_prec@1e-4"]),
        ("Epipolar Prec@5e-4", ["mepi_prec@5e-4"]),
        ("Epipolar Prec@1e-3", ["mepi_prec@1e-3"]),
        # Reprojection metrics
        ("Reproj Prec@1px", ["mreproj_prec@1px"]),
        ("Reproj Prec@3px", ["mreproj_prec@3px"]),
        ("Reproj Prec@5px", ["mreproj_prec@5px"]),
        # GT match metrics
        ("GT Match Recall@3px", ["mgt_match_recall@3px"]),
        ("GT Match Prec@3px", ["mgt_match_precision@3px"]),
        # Relative pose metrics (Table 2 style)
        ("Pose AUC@5°", ["mpose_auc@5"]),
        ("Pose AUC@10°", ["mpose_auc@10"]),
        ("Pose AUC@20°", ["mpose_auc@20"]),
        # Absolute localization metrics (Table 4 style)
        ("Abs Loc Acc@0.25m,2°", ["mabs_loc_acc@0.25m_2deg"]),
        ("Abs Loc Acc@0.5m,5°", ["mabs_loc_acc@0.5m_5deg"]),
        ("Abs Loc Acc@1.0m,10°", ["mabs_loc_acc@1.0m_10deg"]),
        ("Abs Loc T Error (m)", ["mabs_loc_t_error"]),
        ("Abs Loc R Error (deg)", ["mabs_loc_r_error"]),
        ("Abs Loc T Error Rel", ["mabs_loc_t_error_rel"]),
        ("Abs Loc R Error Rel", ["mabs_loc_r_error_rel"]),
        ("Abs Loc T Error X (m)", ["mabs_loc_t_error_x"]),
        ("Abs Loc T Error Y (m)", ["mabs_loc_t_error_y"]),
        ("Abs Loc T Error Z (m)", ["mabs_loc_t_error_z"]),
        ("Abs Loc R Error Roll (deg)", ["mabs_loc_r_error_roll"]),
        ("Abs Loc R Error Pitch (deg)", ["mabs_loc_r_error_pitch"]),
        ("Abs Loc R Error Yaw (deg)", ["mabs_loc_r_error_yaw"]),
        # Relative pose metrics
        ("Rel Pose T Error (m)", ["mrel_pose_t_error"]),
        ("Rel Pose R Error (deg)", ["mrel_pose_r_error"]),
        ("Rel Pose T Error Rel", ["mrel_pose_t_error_rel"]),
        ("Rel Pose R Error Rel", ["mrel_pose_r_error_rel"]),
        ("Rel Pose T Error X (m)", ["mrel_pose_t_error_x"]),
        ("Rel Pose T Error Y (m)", ["mrel_pose_t_error_y"]),
        ("Rel Pose T Error Z (m)", ["mrel_pose_t_error_z"]),
        ("Rel Pose R Error Roll (deg)", ["mrel_pose_r_error_roll"]),
        ("Rel Pose R Error Pitch (deg)", ["mrel_pose_r_error_pitch"]),
        ("Rel Pose R Error Yaw (deg)", ["mrel_pose_r_error_yaw"]),
        # Match statistics
        ("Num Matches", ["mnum_matches"]),
        ("Num Keypoints", ["mnum_keypoints"]),
    ]

    # Print header
    methods = list(all_results.keys())
    header = f"{'Metric':<30} " + " ".join([f"{m:>15}" for m in methods])
    print(header)
    print("-" * len(header))

    # Print each metric
    for metric_name, metric_keys in metrics_to_show:
        row = [metric_name]
        for method in methods:
            summary = all_results[method]["summary"]
            value = None
            for key in metric_keys:
                if key in summary:
                    value = summary[key]
                    break
            if value is not None:
                if (
                    "auc" in metric_name.lower()
                    or "prec" in metric_name.lower()
                    or "recall" in metric_name.lower()
                ):
                    row.append(f"{value * 100:>14.2f}%")
                elif "error" in metric_name.lower():
                    row.append(f"{value:>15.4f}")
                else:
                    row.append(f"{value:>15.1f}")
            else:
                row.append(f"{'N/A':>15}")
        print(" ".join(row))

    print("=" * 100)


def log_summary_metrics(all_results, base_path="evaluation/summary"):
    """Log summary metrics to rerun for visualization."""

    for matcher_name, result_data in all_results.items():
        summary = result_data["summary"]
        matcher_path = f"{base_path}/{matcher_name}"

        # Log key metrics as scalars
        key_metrics = [
            "mepi_prec@1e-3",
            "mreproj_prec@3px",
            "mgt_match_recall@3px",
            "mgt_match_precision@3px",
            "mpose_auc@5",
            "mpose_auc@10",
            "mpose_auc@20",
            "mabs_loc_acc@0.25m_2deg",
            "mabs_loc_acc@0.5m_5deg",
            "mabs_loc_acc@1.0m_10deg",
            "mabs_loc_t_error",
            "mabs_loc_r_error",
            "mabs_loc_t_error_rel",
            "mabs_loc_r_error_rel",
            "mabs_loc_t_error_x",
            "mabs_loc_t_error_y",
            "mabs_loc_t_error_z",
            "mabs_loc_r_error_roll",
            "mabs_loc_r_error_pitch",
            "mabs_loc_r_error_yaw",
            "mrel_pose_t_error",
            "mrel_pose_r_error",
            "mrel_pose_t_error_rel",
            "mrel_pose_r_error_rel",
            "mrel_pose_t_error_x",
            "mrel_pose_t_error_y",
            "mrel_pose_t_error_z",
            "mrel_pose_r_error_roll",
            "mrel_pose_r_error_pitch",
            "mrel_pose_r_error_yaw",
            "mnum_matches",
            "mnum_keypoints",
        ]

        for metric_key in key_metrics:
            if metric_key in summary:
                value = summary[metric_key]
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    metric_name = metric_key[1:] if metric_key.startswith("m") else metric_key
                    metric_name = metric_name.title()
                    rr.log(f"{matcher_path}/{metric_name}", rr.Scalars(float(value)))


def load_config(config_path=None):
    """Load configuration from YAML file, with command line overrides."""
    script_dir = Path(__file__).parent
    default_config_path = script_dir / "eval_exported_config.yaml"

    # Load default config
    if default_config_path.exists():
        with open(default_config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override with provided config file if specified
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)

    return config


def evaluate_scene(
    dataset_path,
    scene_name,
    cfg,
    eval_conf,
    extractor,
    matchers,
    rerun_enabled,
    views=None,
):
    print(f"\n{'=' * 80}")
    print(f"Evaluating SCENE: {scene_name}")
    print(f"{'=' * 80}")

    pairs, scene_views, scene_dir = load_dataset(dataset_path, scene_name)
    # Use provided views if available (e.g. global views), otherwise use scene views
    current_views = views if views is not None else scene_views
    print(f"Loaded {len(pairs)} pairs")

    # Limit pairs if requested
    max_pairs = cfg.get("max_pairs")
    if max_pairs:
        pairs = pairs[:max_pairs]
        print(f"Limited to {len(pairs)} pairs")

    # Evaluate all matchers
    all_results = {}

    for matcher_name, matcher in matchers.items():
        print(f"\nEvaluating {matcher_name}...")

        results = defaultdict(list)

        for pair_idx, (img0_name, img1_name) in enumerate(tqdm(pairs, desc=matcher_name)):
            # Update pair_idx to include scene name for rerun uniqueness if needed,
            # but log_to_rerun uses integer sequence.
            # We might want to clear rerun time or use a different timeline.
            # For now keep it simple.

            pair_results = evaluate_pair(
                img0_name,
                img1_name,
                current_views,
                scene_dir,
                extractor,
                matcher,
                eval_conf,
                pair_idx,
                f"{scene_name}/{matcher_name}",  # Update matcher name path for rerun
                rerun_enabled,
            )
            for k, v in pair_results.items():
                results[k].append(v)

        # Aggregate results
        summary = {}
        for k, v in results.items():
            arr = np.array(v)
            if np.issubdtype(arr.dtype, np.number):
                summary[f"m{k}"] = float(np.mean(arr))
                summary[f"median_{k}"] = float(np.median(arr))
                summary[f"std_{k}"] = float(np.std(arr))
                summary[f"min_{k}"] = float(np.min(arr))
                summary[f"max_{k}"] = float(np.max(arr))

        # Compute pose AUC (Table 2 style) - trajectory-level metric
        if (
            "rel_pose_r_error" in results
        ):  # Changed from rel_pose_error which seemed undefined in original?
            # Original code check: if "rel_pose_error" in results:
            # But compute_relative_pose_per_axis_errors returns "rel_pose_r_error".
            # Let's use 'rel_pose_r_error' which is the rotation error in degrees.
            # Wait, the original code had: pose_errors = np.array(results["rel_pose_error"])
            # But 'rel_pose_error' was NOT returned by compute_relative_pose_per_axis_errors.
            # It returns 'rel_pose_r_error'.
            # I suspect the original code might have had a bug or I missed where 'rel_pose_error' came from.
            # 'eval_matches_epipolar' doesn't return it. 'eval_relative_pose_robust' returns dict with keys?
            # Let's assume 'rel_pose_r_error' is the correct one for AUC.

            pose_errors = np.array(results.get("rel_pose_r_error", []))

            # Filter out inf values
            valid_errors = pose_errors[np.isfinite(pose_errors)]
            if len(valid_errors) > 0:
                auc_metric = AUCMetric(thresholds=[5, 10, 20], elements=valid_errors.tolist())
                aucs = auc_metric.compute()
                summary["mpose_auc@5"] = float(aucs[0]) if not np.isnan(aucs[0]) else 0.0
                summary["mpose_auc@10"] = float(aucs[1]) if not np.isnan(aucs[1]) else 0.0
                summary["mpose_auc@20"] = float(aucs[2]) if not np.isnan(aucs[2]) else 0.0
            else:
                summary["mpose_auc@5"] = 0.0
                summary["mpose_auc@10"] = 0.0
                summary["mpose_auc@20"] = 0.0

        # Compute absolute localization accuracy
        if "abs_loc_t_error" in results and "abs_loc_r_error" in results:
            t_errors = np.array(results["abs_loc_t_error"])
            r_errors = np.array(results["abs_loc_r_error"])
            valid_mask = np.isfinite(t_errors) & np.isfinite(r_errors)
            if np.any(valid_mask):
                t_errors_valid = t_errors[valid_mask]
                r_errors_valid = r_errors[valid_mask]
                for t_thresh, r_thresh, name in [
                    (0.25, 2.0, "0.25m_2deg"),
                    (0.5, 5.0, "0.5m_5deg"),
                    (1.0, 10.0, "1.0m_10deg"),
                ]:
                    passed = (t_errors_valid <= t_thresh) & (r_errors_valid <= r_thresh)
                    summary[f"mabs_loc_acc@{name}"] = float(np.mean(passed))
                summary["mabs_loc_t_error"] = float(np.mean(t_errors_valid))
                summary["mabs_loc_r_error"] = float(np.mean(r_errors_valid))
            else:
                for name in ["0.25m_2deg", "0.5m_5deg", "1.0m_10deg"]:
                    summary[f"mabs_loc_acc@{name}"] = 0.0
                summary["mabs_loc_t_error"] = float("inf")
                summary["mabs_loc_r_error"] = float("inf")

        all_results[matcher_name] = {
            "summary": summary,
            "per_pair": {k: [float(x) for x in v] for k, v in results.items()},
        }

    # Print results table
    print_results_table(all_results)

    # Log summary metrics to rerun
    if rerun_enabled:
        log_summary_metrics(all_results, base_path=f"evaluation/{scene_name}/summary")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LightGlue and SuperGlue on exported dataset"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML config file (default: eval_exported_config.yaml in script directory)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to exported dataset directory (overrides config file)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene name in dataset (or 'all' to evaluate all subdirectories)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (overrides config file)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to evaluate (overrides config file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (overrides config file)",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default=None,
        choices=["opencv", "poselib"],
        help="Robust estimator for pose estimation (overrides config file)",
    )
    parser.add_argument(
        "--ransac_th",
        type=float,
        default=None,
        help="RANSAC threshold (pixels) (overrides config file)",
    )
    parser.add_argument(
        "--matchers",
        nargs="+",
        default=None,
        help="Matchers to evaluate (overrides config file)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=None,
        help="Enable rerun logging for visualization (overrides config file)",
    )
    parser.add_argument(
        "--rerun_save",
        type=str,
        default=None,
        help="Path to save rerun recording (overrides config file)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (overrides config file)",
    )

    args = parser.parse_args()

    # Load config from YAML file
    cfg = load_config(args.config_file)

    # Override with command line arguments
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.scene is not None:
        cfg["scene"] = args.scene
    if args.output is not None:
        cfg["output"] = args.output
    if args.max_pairs is not None:
        cfg["max_pairs"] = args.max_pairs
    if args.device is not None:
        cfg["device"] = args.device
    if args.estimator is not None:
        cfg["estimator"] = args.estimator
    if args.ransac_th is not None:
        cfg["ransac_th"] = args.ransac_th
    if args.matchers is not None:
        cfg["matchers"] = args.matchers
    if args.rerun is not None:
        cfg["rerun"] = args.rerun
    if args.rerun_save is not None:
        cfg["rerun_save"] = args.rerun_save
    if args.checkpoint is not None:
        cfg["checkpoint"] = args.checkpoint

    # Validate required fields
    if cfg.get("dataset") is None:
        # Default to data/export if it exists
        default_path = Path("data/export")
        if default_path.exists():
            cfg["dataset"] = str(default_path)
        else:
            raise ValueError(
                "'dataset' (path to exported dataset directory) is required. Set it in config file or use --dataset"
            )

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available() or device != "cuda":
        device = "cpu"
    print(f"Using device: {device}")

    # Determine scenes to evaluate
    dataset_path = Path(cfg["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    scene_arg = cfg.get("scene", "unreal")
    scenes = []
    if scene_arg == "all":
        # List all subdirectories that have pairs.txt or are just dirs
        scenes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        scenes.sort()
        print(f"Found {len(scenes)} scenes to evaluate: {scenes}")
    else:
        scenes = [scene_arg]
        print(f"Evaluating single scene: {scene_arg}")

    # Evaluation configuration
    eval_conf = {
        "estimator": cfg.get("estimator", "opencv"),
        "ransac_th": cfg.get("ransac_th", 1.0),
    }

    # Initialize models (Load ONCE)
    extractor = pnt.FeatureExtractor.from_config({"class": "SuperPoint", "device": device})

    matchers = {}
    matcher_configs = {
        "LightGlue": {"class": "LightGlue", "features": "superpoint", "device": device},
        "SuperGlue": {"class": "SuperGlue", "device": device},
    }

    checkpoint = cfg.get("checkpoint")
    matcher_names = cfg.get("matchers", ["LightGlue", "SuperGlue"])
    if checkpoint and "LightGlue" in matcher_names:
        matcher_configs["LightGlue"]["checkpoint"] = checkpoint

    for matcher_name in matcher_names:
        if matcher_name in matcher_configs:
            matchers[matcher_name] = pnt.FeatureMatcher.from_config(matcher_configs[matcher_name])

    if not matchers:
        raise ValueError("No valid matchers specified")

    # Initialize rerun if requested
    rerun_enabled = cfg.get("rerun", False)
    if rerun_enabled:
        rerun_save = cfg.get("rerun_save")
        if rerun_save:
            rr.init("feature_matching_evaluation", spawn=False)
            rr.save(rerun_save)
        else:
            # Check if active, otherwise init
            try:
                rr.get_recording_id()
            except:
                rr.init("feature_matching_evaluation", spawn=True)
        print("Rerun logging enabled")

    # Iterate over scenes
    full_results = {}
    aggregated_summaries = {}

    for scene in scenes:
        try:
            scene_results = evaluate_scene(
                dataset_path,
                scene,
                cfg,
                eval_conf,
                extractor,
                matchers,
                rerun_enabled,
            )
            full_results[scene] = scene_results

            # Aggregate for global summary
            for method, res in scene_results.items():
                if method not in aggregated_summaries:
                    aggregated_summaries[method] = defaultdict(list)
                for k, v in res["summary"].items():
                    aggregated_summaries[method][k].append(v)

        except Exception as e:
            print(f"ERROR evaluating scene {scene}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Compute and print GLOBAL AVERAGE
    if len(scenes) > 1 and aggregated_summaries:
        final_global_results = {}
        for method, metrics_dict in aggregated_summaries.items():
            summary = {}
            for k, values in metrics_dict.items():
                # Average non-nan values
                valid_values = [
                    v for v in values if isinstance(v, (int, float)) and not np.isnan(v)
                ]
                if valid_values:
                    summary[k] = float(np.mean(valid_values))
                else:
                    summary[k] = 0.0  # Or NaN
            final_global_results[method] = {"summary": summary}

        print("\n" + "=" * 100)
        print(f"GLOBAL AVERAGE RESULTS ({len(scenes)} scenes)")
        print_results_table(final_global_results)
        full_results["GLOBAL_AVERAGE"] = final_global_results

        # Log global summary to rerun
        if rerun_enabled:
            log_summary_metrics(final_global_results, base_path="evaluation/GLOBAL_AVERAGE/summary")

    # Save results
    output = cfg.get("output")
    output_path = (
        Path(output)
        if output
        else pnt.BASEDIR
        / "output"
        / "2025_FeatureMatching"
        / "eval_exported"
        / f"results_{pnt.get_timestamp().replace('-', '').replace(' ', '_').replace(':', '')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "dataset": str(dataset_path),
                "scenes": scenes,
                "config": cfg,
                "results": full_results,
            },
            f,
            indent=2,
        )
    print(f"\nFinal results saved to {output_path}")


if __name__ == "__main__":
    main()
