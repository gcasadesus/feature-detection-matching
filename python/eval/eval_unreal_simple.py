"""
Evaluate feature extractors and matchers on Unreal dataset using glue-factory-style metrics.

This version works directly with pylupnt datasets and includes comprehensive metrics:
- Epipolar precision
- Reprojection precision
- Relative pose errors (with per-axis breakdown)
- Pose AUC
- Absolute localization using PnP
- GT match metrics
"""

import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import pylupnt as pnt
from pylupnt.numerics import ensure_torch
from pylupnt.numerics.frames import OCV_T_FLU, FLU_T_OCV

from gluefactory.eval.utils import (
    eval_matches_depth,
    eval_matches_epipolar,
    eval_relative_pose_robust,
)
from gluefactory.utils.tools import AUCMetric
from gluefactory.geometry.wrappers import Camera, Pose


def process_config(e_name, e_cfg, m_name, m_cfg):
    """Configure extractor and matcher based on compatibility.

    This function handles the configuration logic for matching extractors
    with matchers, setting appropriate parameters based on descriptor types.
    """
    # Copy
    e_cfg_tmp = e_cfg.copy()
    m_cfg_tmp = m_cfg.copy()

    # LightGlue requires pretraining
    if m_name == "LightGlue":
        if e_name == "SuperPoint":
            m_cfg_tmp["features"] = "superpoint"
        elif e_name == "SIFT":
            m_cfg_tmp["features"] = "sift"
        else:
            return None, None

    elif m_name == "SuperGlue":
        if e_name != "SuperPoint":
            return None, None

    elif m_name in ["Flann", "BruteForce"]:
        if e_name in ["SuperPoint"]:
            return None, None

        if e_name in ["SIFT"]:  # Float extractors
            if m_name == "Flann":
                m_cfg_tmp["index_type"] = "KDTree"
            elif m_name == "BruteForce":
                m_cfg_tmp["norm_type"] = "NORM_L2"

        elif e_name in ["ORB", "BRISK", "AKAZE"]:  # Binary extractors
            if m_name == "Flann":
                m_cfg_tmp["index_type"] = "LSH"
            elif m_name == "BruteForce":
                m_cfg_tmp["norm_type"] = "NORM_HAMMING"
    return e_cfg_tmp, m_cfg_tmp


def compute_epipolar_error(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    T_0to1: np.ndarray,
    threshold: float = 1e-3,
) -> dict:
    """Compute epipolar error for matched points.

    Args:
        pts0: Points in image 0 [N, 2]
        pts1: Points in image 1 [N, 2]
        K0: Camera matrix for image 0 [3, 3]
        K1: Camera matrix for image 1 [3, 3]
        T_0to1: Relative pose from 0 to 1 [4, 4]
        threshold: Error threshold

    Returns:
        Dictionary with epipolar error metrics
    """
    # Convert to homogeneous coordinates
    pts0_h = np.concatenate([pts0, np.ones((len(pts0), 1))], axis=1)  # [N, 3]
    pts1_h = np.concatenate([pts1, np.ones((len(pts1), 1))], axis=1)  # [N, 3]

    # Normalize by camera intrinsics
    pts0_norm = (np.linalg.inv(K0) @ pts0_h.T).T  # [N, 3]
    pts1_norm = (np.linalg.inv(K1) @ pts1_h.T).T  # [N, 3]

    # Extract rotation and translation
    R = T_0to1[:3, :3]
    t = T_0to1[:3, 3]

    # Essential matrix: E = [t]_x @ R
    E = pnt.skew(t) @ R  # [3, 3]

    # Compute epipolar lines: l1 = E @ x0
    epi_lines = (E @ pts0_norm.T).T  # [N, 3]

    # Distance from points to epipolar lines
    # Distance = |x1^T @ l1| / ||l1[:2]||
    distances = np.abs((pts1_norm * epi_lines).sum(axis=1))
    line_norms = np.linalg.norm(epi_lines[:, :2], axis=1)
    line_norms = np.maximum(line_norms, 1e-6)  # Avoid division by zero
    epi_errors = distances / line_norms

    results = {
        "epi_prec": (epi_errors < threshold).mean(),
        "epi_prec@1e-4": (epi_errors < 1e-4).mean(),
        "epi_prec@5e-4": (epi_errors < 5e-4).mean(),
        "epi_prec@1e-3": (epi_errors < 1e-3).mean(),
        "mean_epi_error": epi_errors.mean(),
        "median_epi_error": np.median(epi_errors),
    }

    return results


def compute_reprojection_error(
    pts0: np.ndarray,
    pts1: np.ndarray,
    depth0: np.ndarray,
    depth1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    world_T_cam0: np.ndarray,
    world_T_cam1: np.ndarray,
    threshold: float = 3.0,
) -> dict:
    """Compute symmetric reprojection error using depth.

    Args:
        pts0: Points in image 0 [N, 2]
        pts1: Points in image 1 [N, 2]
        depth0: Depth values at pts0 [N]
        depth1: Depth values at pts1 [N]
        K0: Camera matrix for image 0 [3, 3]
        K1: Camera matrix for image 1 [3, 3]
        world_T_cam0: World to camera 0 pose [4, 4]
        world_T_cam1: World to camera 1 pose [4, 4]
        threshold: Error threshold in pixels

    Returns:
        Dictionary with reprojection error metrics
    """
    # Convert to 3D points in world frame
    xyz0_world = pnt.uv_to_xyz(pts0, depth0, pnt.camera_matrix_to_intrinsics(K0), world_T_cam0)
    xyz1_world = pnt.uv_to_xyz(pts1, depth1, pnt.camera_matrix_to_intrinsics(K1), world_T_cam1)

    # Project xyz0_world to image 1
    uv0_proj, _ = pnt.xyz_to_uv(xyz0_world, pnt.camera_matrix_to_intrinsics(K1), world_T_cam1)

    # Project xyz1_world to image 0
    uv1_proj, _ = pnt.xyz_to_uv(xyz1_world, pnt.camera_matrix_to_intrinsics(K0), world_T_cam0)

    # Compute reprojection errors
    # error0: reprojection error in image 0 (pts0 vs projection of xyz1_world)
    error0 = np.linalg.norm(pts0 - uv1_proj, axis=1)
    # error1: reprojection error in image 1 (pts1 vs projection of xyz0_world)
    error1 = np.linalg.norm(pts1 - uv0_proj, axis=1)

    # Symmetric error
    errors = (error0 + error1) / 2.0

    # Filter valid points (positive depth)
    valid = (depth0 > 0) & (depth1 > 0) & ~np.isnan(errors)
    errors_valid = errors[valid]

    results = {
        "reproj_prec": (errors_valid < threshold).mean() if len(errors_valid) > 0 else 0.0,
        "reproj_prec@1px": (errors_valid < 1.0).mean() if len(errors_valid) > 0 else 0.0,
        "reproj_prec@3px": (errors_valid < 3.0).mean() if len(errors_valid) > 0 else 0.0,
        "reproj_prec@5px": (errors_valid < 5.0).mean() if len(errors_valid) > 0 else 0.0,
        "mean_reproj_error": errors_valid.mean() if len(errors_valid) > 0 else np.nan,
        "median_reproj_error": np.median(errors_valid) if len(errors_valid) > 0 else np.nan,
        "covisible": valid.sum(),
        "covisible_percent": valid.mean() * 100.0,
    }

    return results


def compute_relative_pose_error(poses_est, poses_gt):
    """Compute relative pose error.

    Args:
        poses_est: Estimated relative poses [N, 4, 4]
        poses_gt: Ground truth relative poses [N, 4, 4]

    Returns:
        Dictionary with pose error metrics
    """
    t_errors = []
    r_errors = []

    for T_est, T_gt in zip(poses_est, poses_gt):
        # Translation error
        t_error = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
        t_errors.append(t_error)

        # Rotation error
        R_error = T_est[:3, :3] @ T_gt[:3, :3].T
        r_error = pnt.rotation_angle(R_error) * pnt.DEG
        r_errors.append(r_error)

    t_errors = np.array(t_errors)
    r_errors = np.array(r_errors)

    results = {
        "mean_t_error": t_errors.mean(),
        "median_t_error": np.median(t_errors),
        "mean_r_error": r_errors.mean(),
        "median_r_error": np.median(r_errors),
        "rel_pose_error@5deg": (r_errors < 5.0).mean(),
        "rel_pose_error@10deg": (r_errors < 10.0).mean(),
        "rel_pose_error@20deg": (r_errors < 20.0).mean(),
    }

    return results


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
    img_data0,
    img_data1,
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
    depth0 = img_data0.depth
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
    K0 = pnt.make_camera_matrix(img_data0.intrinsics)
    K1 = pnt.make_camera_matrix(img_data1.intrinsics)
    T_w2cam0_gt = img_data0.world_T_cam
    T_w2cam1_gt = img_data1.world_T_cam

    # Create Camera and Pose wrappers for glue-factory compatibility
    T_w2cam0_gt_ocv = pnt.invert_transform(T_w2cam0_gt)
    T_w2cam1_gt_ocv = pnt.invert_transform(T_w2cam1_gt)

    T_w2cam0_gt_flu = Pose.from_4x4mat(torch.from_numpy(FLU_T_OCV).float()) @ Pose.from_4x4mat(
        torch.from_numpy(T_w2cam0_gt_ocv).float()
    )
    T_w2cam1_gt_flu = Pose.from_4x4mat(torch.from_numpy(FLU_T_OCV).float()) @ Pose.from_4x4mat(
        torch.from_numpy(T_w2cam1_gt_ocv).float()
    )

    u, v = matched_kp0_valid[:, 0], matched_kp0_valid[:, 1]
    intrinsics = img_data0.intrinsics
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
    pnp_result = pnt.PnpSolver(
        {
            "threshold": eval_conf.get("ransac_th", 1.0),
            "confidence": 0.99,
            "max_iterations": 10000,
        }
    ).solve(
        points_3d_world_flu.astype(np.float64),
        matched_kp1_valid.astype(np.float64),
        K1.astype(np.float64),
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


def convert_to_gluefactory_format(img_data0, img_data1):
    """Convert pylupnt image data to glue-factory format."""
    # Convert world_T_cam to T_w2cam (invert because glue-factory uses world-to-camera)
    T_w2cam0_ocv = pnt.invert_transform(img_data0.world_T_cam)
    T_w2cam1_ocv = pnt.invert_transform(img_data1.world_T_cam)

    T_w2cam0 = Pose.from_4x4mat(torch.from_numpy(T_w2cam0_ocv).float())
    T_w2cam1 = Pose.from_4x4mat(torch.from_numpy(T_w2cam1_ocv).float())

    # Create Camera objects
    K0 = pnt.make_camera_matrix(img_data0.intrinsics)
    K1 = pnt.make_camera_matrix(img_data1.intrinsics)
    h0, w0 = img_data0.rgb.shape[:2]
    h1, w1 = img_data1.rgb.shape[:2]

    # Camera data format: [width, height, fx, fy, cx, cy]
    camera0_data = torch.tensor(
        [
            w0,
            h0,
            img_data0.intrinsics["fx"],
            img_data0.intrinsics["fy"],
            img_data0.intrinsics["cx"],
            img_data0.intrinsics["cy"],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    camera1_data = torch.tensor(
        [
            w1,
            h1,
            img_data1.intrinsics["fx"],
            img_data1.intrinsics["fy"],
            img_data1.intrinsics["cx"],
            img_data1.intrinsics["cy"],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    camera0 = Camera(camera0_data)
    camera1 = Camera(camera1_data)

    return {
        "view0": {
            "image": ensure_torch(img_data0.rgb, channels=3, device="cpu", batch_dim=True),
            "camera": camera0,
            "depth": ensure_torch(img_data0.depth, device="cpu", batch_dim=True),
            "T_w2cam": T_w2cam0,
        },
        "view1": {
            "image": ensure_torch(img_data1.rgb, channels=3, device="cpu", batch_dim=True),
            "camera": camera1,
            "depth": ensure_torch(img_data1.depth, device="cpu", batch_dim=True),
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


def evaluate_pair(img_data1, img_data2, feats1, feats2, matches, eval_conf):
    """Evaluate a single image pair.

    Args:
        img_data1: First image data
        img_data2: Second image data
        feats1: Features from first image
        feats2: Features from second image
        matches: Matches between features
        eval_conf: Evaluation configuration

    Returns:
        Dictionary of evaluation metrics
    """
    if len(matches.indexes) == 0:
        return {
            "num_matches": 0,
            "num_keypoints": (len(feats1) + len(feats2)) / 2.0,
        }

    # Convert to glue-factory format
    data = convert_to_gluefactory_format(img_data1, img_data2)
    pred = convert_features_to_gluefactory_format(feats1, feats2, matches)

    results = {}

    # Basic match statistics
    matches0 = pred["matches0"]
    results["num_matches"] = (matches0 >= 0).sum().item()
    results["num_keypoints0"] = pred["keypoints0"].shape[0]
    results["num_keypoints1"] = pred["keypoints1"].shape[0]

    # Epipolar metrics
    results.update(eval_matches_epipolar(data, pred))

    # Reprojection metrics
    results.update(eval_matches_depth(data, pred))

    # Relative pose metrics
    rel_pose_results = eval_relative_pose_robust(data, pred, eval_conf)
    results.update(rel_pose_results)

    # Compute relative pose per-axis errors
    rel_pose_errors = compute_relative_pose_per_axis_errors(data, pred, eval_conf)
    if rel_pose_errors:
        results.update(rel_pose_errors)

    # Absolute localization metrics
    try:
        abs_loc_results = eval_absolute_localization(
            img_data1,
            img_data2,
            pred,
            eval_conf,
        )
        if abs_loc_results:
            results.update(abs_loc_results)
    except ValueError:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate extractors/matchers on Unreal dataset")
    parser.add_argument("--config", type=str, required=True, help="Dataset config path")
    parser.add_argument("--output", type=str, default=None, help="Output pickle file")
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=["SuperPoint", "SIFT", "ORB", "AKAZE", "BRISK"],
        help="Extractors to evaluate",
    )
    parser.add_argument(
        "--matchers",
        nargs="+",
        default=["LightGlue", "SuperGlue", "BruteForce", "Flann"],
        help="Matchers to evaluate",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to evaluate",
    )
    parser.add_argument("--step", type=int, default=1, help="Step between consecutive frames")
    parser.add_argument("--cameras", nargs="+", default=["front_left"], help="Cameras to evaluate")
    parser.add_argument(
        "--estimator",
        type=str,
        default="opencv",
        choices=["opencv", "poselib"],
        help="Robust estimator for pose estimation",
    )
    parser.add_argument(
        "--ransac_th",
        type=float,
        default=1.0,
        help="RANSAC threshold (pixels)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="Suffix to append to method name",
    )

    args = parser.parse_args()

    # Evaluation configuration
    eval_conf = {
        "estimator": args.estimator,
        "ransac_th": args.ransac_th,
    }

    # Load dataset
    ds_config = pnt.load_config(args.config)
    dataset = pnt.datasets.Dataset.from_config(ds_config)

    # Extractors and matchers base configurations
    extractor_configs = {
        "SuperPoint": {"class": "SuperPoint"},
        "SIFT": {"class": "Sift"},
        "ORB": {"class": "Orb"},
        "AKAZE": {"class": "Akaze"},
        "BRISK": {"class": "Brisk"},
    }

    matcher_configs = {
        "LightGlue": {"class": "LightGlue"},
        "SuperGlue": {"class": "SuperGlue"},
        "BruteForce": {"class": "BruteForceMatcher"},
        "Flann": {"class": "FlannMatcher"},
    }

    if args.checkpoint:
        matcher_configs["LightGlue"]["checkpoint"] = args.checkpoint

    # Generate compatible combinations using process_config
    combinations = []
    for e_name in args.extractors:
        if e_name not in extractor_configs:
            continue
        for m_name in args.matchers:
            if m_name not in matcher_configs:
                continue

            e_cfg = extractor_configs[e_name]
            m_cfg = matcher_configs[m_name]
            e_cfg_processed, m_cfg_processed = process_config(e_name, e_cfg, m_name, m_cfg)

            if e_cfg_processed is not None and m_cfg_processed is not None:
                combinations.append((e_name, e_cfg_processed, m_name, m_cfg_processed))

    # Create extractors and matchers from processed configs
    extractors = {}
    matchers_per_combo = {}

    for e_name, e_cfg, m_name, m_cfg in combinations:
        # Create extractor if not already created
        if e_name not in extractors:
            extractors[e_name] = pnt.FeatureExtractor.from_config(e_cfg)

        # Create matcher for this specific combination (config may vary per extractor)
        combo_key = (e_name, m_name)
        if combo_key not in matchers_per_combo:
            matchers_per_combo[combo_key] = pnt.FeatureMatcher.from_config(m_cfg)

    # Generate image pairs
    n_frames = len(dataset)
    pairs = []
    for i in range(0, n_frames - args.step, args.step):
        pairs.append((i, i + args.step))
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break

    pnt.Logger.info(f"Evaluating {len(pairs)} pairs", "Eval")

    # Evaluate all combinations
    all_results = {}

    for i_combo, (e_name, e_cfg, m_name, m_cfg) in enumerate(combinations):
        # Skip if extractor or matcher creation failed
        if e_name not in extractors:
            continue
        combo_key = (e_name, m_name)
        if combo_key not in matchers_per_combo:
            continue

        extractor = extractors[e_name]
        matcher = matchers_per_combo[combo_key]

        combo_name = f"{e_name}+{m_name}{args.name_suffix}"
        pnt.Logger.info(f"Evaluating {i_combo + 1}/{len(combinations)}: {combo_name}", "Eval")

        results = defaultdict(list)

        for idx1, idx2 in tqdm(pairs, desc=combo_name):
            # Get image data
            img_data1 = dataset[idx1]["cameras"][args.cameras[0]]
            img_data2 = dataset[idx2]["cameras"][args.cameras[0]]

            # Extract features with timing
            t0 = time.time()
            feats1 = extractor.extract(img_data1.rgb)
            feats2 = extractor.extract(img_data2.rgb)
            extraction_time = time.time() - t0

            # Match features with timing
            t0 = time.time()
            matches = matcher.match(feats1, feats2)
            matching_time = time.time() - t0

            # Evaluate
            pair_results = evaluate_pair(img_data1, img_data2, feats1, feats2, matches, eval_conf)

            # Add timing metrics
            pair_results["extraction_time"] = extraction_time
            pair_results["matching_time"] = matching_time
            pair_results["total_time"] = extraction_time + matching_time

            for k, v in pair_results.items():
                results[k].append(v)

        # Aggregate results
        summary = {}
        for k, v in results.items():
            arr = np.array(v)
            if np.issubdtype(arr.dtype, np.number) and not np.isnan(arr).all():
                summary[f"m{k}"] = float(np.nanmean(arr))
                summary[f"median_{k}"] = float(np.nanmedian(arr))
                summary[f"std_{k}"] = float(np.nanstd(arr))

        # Compute pose AUC if relative pose errors available
        if "rel_pose_r_error" in results:
            pose_errors = np.array(results["rel_pose_r_error"])
            valid_errors = pose_errors[np.isfinite(pose_errors)]
            if len(valid_errors) > 0:
                auc_metric = AUCMetric(thresholds=[5, 10, 20], elements=valid_errors.tolist())
                aucs = auc_metric.compute()
                summary["mpose_auc@5"] = float(aucs[0]) if not np.isnan(aucs[0]) else 0.0
                summary["mpose_auc@10"] = float(aucs[1]) if not np.isnan(aucs[1]) else 0.0
                summary["mpose_auc@20"] = float(aucs[2]) if not np.isnan(aucs[2]) else 0.0

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

        # Compute FPS metrics from timing
        if "extraction_time" in results:
            extraction_times = np.array(results["extraction_time"])
            summary["extraction_fps"] = float(2.0 / np.mean(extraction_times))  # 2 images per pair

        if "matching_time" in results:
            matching_times = np.array(results["matching_time"])
            summary["matching_fps"] = float(1.0 / np.mean(matching_times))  # 1 matching per pair

        if "total_time" in results:
            total_times = np.array(results["total_time"])
            summary["total_fps"] = float(1.0 / np.mean(total_times))  # 1 pair per iteration

        all_results[combo_name] = {
            "summary": summary,
            "per_pair": {k: v for k, v in results.items()},
        }

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = pnt.BASEDIR / "output" / "2025_FeatureMatching" / "eval_gluefactory"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"results_{pnt.get_timestamp()}.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "config": ds_config,
                "pairs": pairs,
                "results": all_results,
            },
            f,
        )

    pnt.Logger.info(f"Results saved to {output_path}", "Eval")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for combo_name, combo_results in all_results.items():
        print(f"\n{combo_name}:")
        summary = combo_results["summary"]
        for key in sorted(summary.keys()):
            if key.startswith("m"):
                print(f"  {key}: {summary[key]:.4f}")

    return all_results


if __name__ == "__main__":
    main()
