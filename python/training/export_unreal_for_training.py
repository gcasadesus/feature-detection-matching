"""
Export Unreal dataset in format suitable for glue-factory training.

This script exports image pairs from the Unreal dataset in a format that can be
used with glue-factory's training pipeline.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from tqdm import tqdm

import pylupnt as pnt
from pylupnt.perception.pairs import find_all_overlapping_pairs
from pylupnt.numerics.frames import FLU_T_OCV
import rerun as rr


def export_dataset(
    config_or_path,
    output_dir,
    step=1,
    max_pairs=None,
    cameras=["front_left"],
    use_overlapping_pairs=False,
    extractor_config=None,
    camera="front",
    max_depth=100,
    thresh_dist=0.025,
    min_covisible=200,
    max_frames=None,
    n_jobs=1,
    max_sep=None,
    log_rerun=False,
    rerun_save=None,
    agents=None,
):
    """Export Unreal dataset for glue-factory training.

    Args:
        config_or_path: Path to dataset config file or Config object
        output_dir: Output directory
        step: Step between consecutive frames (used if use_overlapping_pairs=False and max_sep=None)
        max_pairs: Maximum number of pairs to export
        cameras: List of cameras to export
        use_overlapping_pairs: If True, use find_all_overlapping_pairs instead of step-based
        extractor_config: Config dict for feature extractor (required if use_overlapping_pairs=True)
        camera: Camera name for overlap detection (default: "front")
        max_depth: Maximum depth for features
        thresh_dist: Distance threshold for covisibility (meters)
        min_covisible: Minimum number of covisible features
        max_frames: Maximum number of frames to process (None = all frames)
        n_jobs: Number of parallel jobs for overlap detection
        max_sep: Maximum separation for consecutive pairs (e.g., max_sep=5 generates (0,1), (0,2), ..., (0,5), (1,2), ...)
        agents: List of agent names to export (None = all agents in basedir)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset config
    if isinstance(config_or_path, (str, Path)):
        ds_config = pnt.load_config(config_or_path)
    else:
        ds_config = config_or_path

    # Auto-discover agents if not specified
    if agents is None:
        basedir = Path(ds_config.get("basedir", "."))
        if basedir.exists():
            # Find all subdirectories that look like agents (contain camera data)
            agent_dirs = []
            for item in basedir.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    # Check if it has camera subdirectories
                    has_cameras = any(
                        (item / f"cam_{cam}").exists()
                        for cam in ["front", "front_left", "front_right"]
                    )
                    if has_cameras:
                        agent_dirs.append(item.name)
            if agent_dirs:
                agents = sorted(agent_dirs)
                pnt.Logger.info(f"Auto-discovered agents: {agents}", "Export")
            else:
                pnt.Logger.warning(
                    "No agents found in basedir, using default from config", "Export"
                )
                agents = [ds_config.get("agent", "free_agent_0")]
        else:
            pnt.Logger.warning(
                f"Basedir {basedir} does not exist, using default agent from config",
                "Export",
            )
            agents = [ds_config.get("agent", "free_agent_0")]

    if not isinstance(agents, list):
        agents = [agents]

    pnt.Logger.info(f"Exporting {len(agents)} agent(s): {agents}", "Export")

    # Store metadata for all agents
    all_agents_metadata = {}

    # Process each agent
    for agent_idx, agent_name in enumerate(agents):
        pnt.Logger.info(f"Processing agent {agent_idx + 1}/{len(agents)}: {agent_name}", "Export")

        # Update config for this agent
        current_config = ds_config.copy()
        current_config["agent"] = agent_name

        # Load dataset for this agent
        dataset = pnt.datasets.Dataset.from_config(current_config)

        # Create agent-specific subdirectory
        scene_dir = output_dir / "unreal" / agent_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        images_dir = scene_dir / "images"
        depths_dir = scene_dir / "depths"
        images_dir.mkdir(exist_ok=True)
        depths_dir.mkdir(exist_ok=True)

        # Generate pairs
        if use_overlapping_pairs:
            if extractor_config is None:
                raise ValueError("extractor_config is required when use_overlapping_pairs=True")

            pnt.Logger.info(f"Finding overlapping pairs for {agent_name}...", "Export")
            extractor = pnt.FeatureExtractor.from_config(extractor_config)

            overlapping_pairs = find_all_overlapping_pairs(
                dataset,
                extractor,
                camera=camera,
                max_depth=max_depth,
                thresh_dist=thresh_dist,
                min_covisible=min_covisible,
                max_frames=max_frames,
                n_jobs=n_jobs,
            )

            # Convert to list of (idx1, idx2) tuples, dropping n_covisible
            pairs = [(idx1, idx2) for idx1, idx2, _ in overlapping_pairs]

            if max_pairs and len(pairs) > max_pairs:
                pairs = pairs[:max_pairs]
        elif max_sep is not None:
            # Use consecutive pairs with max separation: (0,1), (0,2), ..., (0,max_sep), (1,2), ...
            n_frames = len(dataset)
            if max_frames is not None:
                n_frames = min(n_frames, max_frames)

            pairs = []
            for i in range(n_frames):
                for j in range(i + 1, min(i + max_sep + 1, n_frames)):
                    pairs.append((i, j))
                    if max_pairs and len(pairs) >= max_pairs:
                        break
                if max_pairs and len(pairs) >= max_pairs:
                    break
        else:
            # Use step-based pairs
            n_frames = len(dataset)
            pairs = []
            for i in range(0, n_frames - step, step):
                pairs.append((i, i + step))
                if max_pairs and len(pairs) >= max_pairs:
                    break

        pnt.Logger.info(f"Exporting {len(pairs)} pairs for {agent_name}", "Export")

        # Initialize rerun if requested (only once for first agent)
        if log_rerun and agent_idx == 0:
            if rerun_save:
                rr.init("unreal_dataset_export", spawn=False)
                rr.save(rerun_save)
                pnt.Logger.info(f"Rerun logging enabled (saving to {rerun_save})", "Export")
            else:
                rr.init("unreal_dataset_export", spawn=True)
                pnt.Logger.info("Rerun logging enabled (spawning viewer)", "Export")

        # Create extractor for rerun logging if needed
        extractor = None
        matcher = None
        if log_rerun:
            extractor = pnt.FeatureExtractor.from_config({"class": "SuperPoint"})
            matcher = pnt.FeatureMatcher.from_config(
                {"class": "LightGlue", "features": "superpoint"}
            )

        # Export images, depths, and metadata
        views_data = []
        pairs_data = []

        image_idx = 0
        image_to_idx = {}
        stored_img_data = {}  # Store image data for rerun logging

        for idx1, idx2 in tqdm(pairs, desc=f"Exporting {agent_name}"):
            for idx in [idx1, idx2]:
                if idx not in image_to_idx:
                    img_data = dataset[idx]["cameras"][cameras[0]]

                    # Create symlink to image instead of copying
                    img_name = f"{image_idx:06d}.jpg"
                    img_path = images_dir / img_name

                    # Get original image path from dataset
                    # The dataset stores data in basedir/agent/cam_*/rgb/timestamp.jpg
                    basedir = Path(current_config["basedir"])
                    cam_name = cameras[0].replace(
                        "_", ""
                    )  # e.g., "front_left" -> "frontleft" or just use as is
                    # Try different camera naming conventions
                    possible_cam_dirs = [
                        basedir / agent_name / f"cam_{cameras[0]}",
                        basedir / agent_name / f"cam_{cam_name}",
                        basedir / agent_name / cameras[0],
                    ]

                    original_img_path = None
                    for cam_dir in possible_cam_dirs:
                        rgb_dir = cam_dir / "rgb"
                        if rgb_dir.exists():
                            # Find the corresponding image by index
                            rgb_files = sorted(rgb_dir.glob("*.jpg")) + sorted(
                                rgb_dir.glob("*.png")
                            )
                            if idx < len(rgb_files):
                                original_img_path = rgb_files[idx]
                                break

                    if original_img_path and original_img_path.exists():
                        # Create symlink
                        if img_path.exists():
                            img_path.unlink()
                        os.symlink(original_img_path, img_path)
                    else:
                        # Fallback: save image if we can't find original
                        pnt.Logger.warning(
                            f"Could not find original image for idx {idx}, saving copy instead",
                            "Export",
                        )
                        rgb_img = img_data.rgb
                        if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float64:
                            rgb_img = (rgb_img * 255).astype(np.uint8)
                        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(img_path), bgr_img)

                    # Create symlink to depth or save it
                    depth_name = f"{image_idx:06d}.h5"
                    depth_path = depths_dir / depth_name

                    # Find original depth file
                    original_depth_path = None
                    for cam_dir in possible_cam_dirs:
                        depth_dir = cam_dir / "depth"
                        if depth_dir.exists():
                            depth_files = sorted(depth_dir.glob("*.h5")) + sorted(
                                depth_dir.glob("*.hdf5")
                            )
                            if idx < len(depth_files):
                                original_depth_path = depth_files[idx]
                                break

                    if original_depth_path and original_depth_path.exists():
                        # Create symlink for depth
                        if depth_path.exists():
                            depth_path.unlink()
                        os.symlink(original_depth_path, depth_path)
                    else:
                        # Fallback: save depth file
                        with h5py.File(depth_path, "w") as f:
                            f.create_dataset("depth", data=img_data.depth.astype(np.float32))

                    # Get camera intrinsics and pose
                    K = pnt.make_camera_matrix(img_data.intrinsics)
                    h, w = img_data.rgb.shape[:2]

                    # Convert pose: world_T_cam to T_w2cam (inverse)
                    # Dataset poses are in FLU frame, but glue-factory expects OCV frame
                    T_w2cam_flu = pnt.invert_transform(img_data.world_T_cam)
                    # Convert camera frame from FLU to OCV: T_w2cam_ocv = OCV_T_FLU @ T_w2cam_flu
                    # This transforms points: p_cam_ocv = OCV_T_FLU @ (T_w2cam_flu @ p_w)
                    from pylupnt.numerics.frames import OCV_T_FLU

                    T_w2cam_ocv_np = OCV_T_FLU @ T_w2cam_flu

                    # Convert to Pose object for rerun logging if needed
                    T_w2cam_ocv_pose = None
                    if log_rerun:
                        import torch
                        from gluefactory.geometry.wrappers import Pose

                        T_w2cam_ocv_pose = Pose.from_Rt(
                            torch.from_numpy(T_w2cam_ocv_np[:3, :3]).float(),
                            torch.from_numpy(T_w2cam_ocv_np[:3, 3]).float(),
                        )

                    # Format: R (9 values) + t (3 values) + camera model + width + height + params
                    R = T_w2cam_ocv_np[:3, :3].flatten()
                    t = T_w2cam_ocv_np[:3, 3]

                    # Camera params: fx, fy, cx, cy
                    params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)

                    view_line = " ".join(
                        [
                            img_name,
                            *[str(x) for x in R],
                            *[str(x) for x in t],
                            "PINHOLE",
                            str(w),
                            str(h),
                            *[str(x) for x in params],
                        ]
                    )

                    views_data.append(view_line)
                    image_to_idx[idx] = image_idx

                    # Store image data for rerun logging
                    if log_rerun:
                        stored_img_data[image_idx] = {
                            "img_data": img_data,
                            "T_w2cam_ocv": T_w2cam_ocv_pose,
                            "img_name": img_name,
                        }

                    image_idx += 1

            # Add pair
            img0_name = f"{image_to_idx[idx1]:06d}.jpg"
            img1_name = f"{image_to_idx[idx2]:06d}.jpg"
            pairs_data.append(f"{img0_name} {img1_name}")

            # Log to rerun if requested
            if log_rerun:
                img0_idx = image_to_idx[idx1]
                img1_idx = image_to_idx[idx2]
                _log_pair_to_rerun(
                    pair_idx=len(pairs_data) - 1,
                    img0_idx=img0_idx,
                    img1_idx=img1_idx,
                    stored_img_data=stored_img_data,
                    extractor=extractor,
                    matcher=matcher,
                )

        # Write views.txt
        views_file = scene_dir / "views.txt"
        with open(views_file, "w") as f:
            f.write("\n".join(views_data))

        # Write pairs.txt
        pairs_file = scene_dir / "pairs.txt"
        with open(pairs_file, "w") as f:
            f.write("\n".join(pairs_data))

        # Store metadata for this agent
        all_agents_metadata[agent_name] = {
            "n_pairs": len(pairs),
            "n_images": image_idx,
        }

        pnt.Logger.info(
            f"  Agent {agent_name} complete: {image_idx} images, {len(pairs)} pairs",
            "Export",
        )

    # Write overall metadata
    config_path_str = (
        str(config_or_path) if isinstance(config_or_path, (str, Path)) else "config_dict"
    )
    metadata = {
        "config": config_path_str,
        "agents": agents,
        "agents_metadata": all_agents_metadata,
        "cameras": cameras,
        "use_overlapping_pairs": use_overlapping_pairs,
        "uses_symlinks": True,
    }
    if use_overlapping_pairs:
        metadata.update(
            {
                "camera": camera,
                "max_depth": max_depth,
                "thresh_dist": thresh_dist,
                "min_covisible": min_covisible,
                "max_frames": max_frames,
            }
        )
    elif max_sep is not None:
        metadata["max_sep"] = max_sep
        metadata["max_frames"] = max_frames
    else:
        metadata["step"] = step
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    pnt.Logger.info(f"Export complete: {output_dir}", "Export")
    pnt.Logger.info(f"  Total agents: {len(agents)}", "Export")
    for agent_name, agent_meta in all_agents_metadata.items():
        pnt.Logger.info(
            f"    {agent_name}: {agent_meta['n_images']} images, {agent_meta['n_pairs']} pairs",
            "Export",
        )

    return output_dir


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


def _get_intrinsics(img_data):
    """Extract camera intrinsics from image data."""
    return {
        "fx": float(img_data.intrinsics["fx"]),
        "fy": float(img_data.intrinsics["fy"]),
        "cx": float(img_data.intrinsics["cx"]),
        "cy": float(img_data.intrinsics["cy"]),
    }


def _compute_3d_points(kp, depth, img_data, T_w2cam_ocv):
    """Compute 3D points in world frame from keypoints and depth."""
    import torch
    from gluefactory.geometry.wrappers import Pose

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
    intrinsics = _get_intrinsics(img_data)
    # T_w2cam_ocv is a Pose object, convert to FLU frame
    T_w2cam_flu = Pose.from_4x4mat(torch.from_numpy(FLU_T_OCV).float()) @ T_w2cam_ocv
    world_T_cam_flu = pnt.make_transform(*T_w2cam_flu.inv().numpy())
    xyz_world = pnt.uv_to_xyz(kp_valid, depth_valid, intrinsics, world_T_cam_flu)
    return xyz_world.astype(np.float32), valid


def _log_pair_to_rerun(pair_idx, img0_idx, img1_idx, stored_img_data, extractor, matcher):
    """Log a pair to rerun with images, features, matches, poses, and 3D points."""
    base_path = "export"
    rr.set_time("pair", sequence=pair_idx)

    data0 = stored_img_data[img0_idx]
    data1 = stored_img_data[img1_idx]
    img_data0 = data0["img_data"]
    img_data1 = data1["img_data"]
    T_w2cam0 = data0["T_w2cam_ocv"]
    T_w2cam1 = data1["T_w2cam_ocv"]

    # Log images
    img0_log = _prepare_image(img_data0.rgb)
    img1_log = _prepare_image(img_data1.rgb)
    rr.log(f"{base_path}/image0", rr.Image(img0_log).compress(jpeg_quality=50))
    rr.log(f"{base_path}/image1", rr.Image(img1_log).compress(jpeg_quality=50))

    # Extract features and match
    feats0 = extractor.extract(img_data0.rgb)
    feats1 = extractor.extract(img_data1.rgb)
    matches = matcher.match(feats0, feats1)

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

    # Log camera poses
    for cam_idx, (T_w2cam, img_name) in enumerate(
        [(T_w2cam0, data0["img_name"]), (T_w2cam1, data1["img_name"])]
    ):
        R, pos = T_w2cam.inv().numpy()
        rr.log(
            f"{base_path}/camera{cam_idx}/pose",
            rr.Transform3D(
                translation=pos.astype(np.float32),
                mat3x3=R.astype(np.float32),
                axis_length=2.0,
            ),
        )
        rr.log(f"{base_path}/camera{cam_idx}/name", rr.TextLog(img_name))

    # Log 3D points
    for cam_idx, (kp, img_data, T_w2cam, data) in enumerate(
        [
            (kp0, img_data0, T_w2cam0, data0),
            (kp1, img_data1, T_w2cam1, data1),
        ]
    ):
        xyz_world, valid_mask = _compute_3d_points(kp, img_data.depth, img_data, T_w2cam)
        if xyz_world is not None:
            rr.log(f"{base_path}/camera{cam_idx}/features_3d", rr.Points3D(xyz_world))
            if len(matches) > 0:
                match_indices = matches.indexes
                match_idx = match_indices[:, cam_idx]
                orig_kp_indices = np.arange(len(kp))[valid_mask]
                matched_mask = np.isin(orig_kp_indices, match_idx)
                if np.any(matched_mask):
                    rr.log(
                        f"{base_path}/camera{cam_idx}/matched_features_3d",
                        rr.Points3D(xyz_world[matched_mask]),
                    )


def load_config(config_path=None):
    """Load configuration from YAML file, with command line overrides."""
    script_dir = Path(__file__).parent
    default_config_path = script_dir / "export_unreal_config.yaml"

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


def main():
    parser = argparse.ArgumentParser(description="Export Unreal dataset for glue-factory training")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML config file (default: export_unreal_config.yaml in script directory)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset config path (overrides config file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config file)",
    )
    parser.add_argument(
        "--basedir", type=str, default=None, help="Override dataset basedir from config"
    )
    parser.add_argument("--agent", type=str, default=None, help="Override agent name from config")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step between consecutive frames (overrides config file)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs (overrides config file)",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Cameras to export (overrides config file)",
    )

    # Overlapping pairs options
    parser.add_argument(
        "--use-overlapping-pairs",
        action="store_true",
        default=None,
        help="Use overlapping pairs instead of step-based pairs (overrides config file)",
    )
    parser.add_argument(
        "--extractor-config",
        type=str,
        default=None,
        help="Path to extractor config JSON (overrides config file)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera name for overlap detection (overrides config file)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Maximum depth for features (overrides config file)",
    )
    parser.add_argument(
        "--thresh-dist",
        type=float,
        default=None,
        help="Distance threshold for covisibility (meters) (overrides config file)",
    )
    parser.add_argument(
        "--min-covisible",
        type=int,
        default=None,
        help="Minimum number of covisible features (overrides config file)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (overrides config file)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for overlap detection (overrides config file)",
    )
    parser.add_argument(
        "--max-sep",
        type=int,
        default=None,
        help="Maximum separation for consecutive pairs (overrides config file)",
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
        "--agents",
        nargs="*",
        default=None,
        help="List of agents to export (default: all agents in basedir). Use --agents with no values for all agents.",
    )

    args = parser.parse_args()

    # Load config from YAML file
    cfg = load_config(args.config_file)

    # Override with command line arguments
    if args.config is not None:
        cfg["config"] = args.config
    if args.output is not None:
        cfg["output"] = args.output
    if args.basedir is not None:
        cfg["basedir"] = args.basedir
    if args.agent is not None:
        cfg["agent"] = args.agent
    if args.step is not None:
        cfg["step"] = args.step
    if args.max_pairs is not None:
        cfg["max_pairs"] = args.max_pairs
    if args.cameras is not None:
        cfg["cameras"] = args.cameras
    if args.use_overlapping_pairs is not None:
        cfg["use_overlapping_pairs"] = args.use_overlapping_pairs
    if args.extractor_config is not None:
        cfg["extractor_config"] = args.extractor_config
    if args.camera is not None:
        cfg["camera"] = args.camera
    if args.max_depth is not None:
        cfg["max_depth"] = args.max_depth
    if args.thresh_dist is not None:
        cfg["thresh_dist"] = args.thresh_dist
    if args.min_covisible is not None:
        cfg["min_covisible"] = args.min_covisible
    if args.max_frames is not None:
        cfg["max_frames"] = args.max_frames
    if args.n_jobs is not None:
        cfg["n_jobs"] = args.n_jobs
    if args.max_sep is not None:
        cfg["max_sep"] = args.max_sep
    if args.rerun is not None:
        cfg["rerun"] = args.rerun
    if args.rerun_save is not None:
        cfg["rerun_save"] = args.rerun_save
    if args.agents is not None:
        cfg["agents"] = args.agents if args.agents else None  # Empty list means all agents

    # Validate required fields
    if cfg.get("config") is None:
        raise ValueError(
            "'config' (dataset config path) is required. Set it in config file or use --config"
        )
    if cfg.get("output") is None:
        raise ValueError(
            "'output' (output directory) is required. Set it in config file or use --output"
        )

    # Load dataset config and override if needed
    ds_config = pnt.load_config(cfg["config"])
    if cfg.get("basedir") is not None:
        ds_config["basedir"] = cfg["basedir"]
    if cfg.get("agent") is not None:
        ds_config["agent"] = cfg["agent"]
    # Override cameras from config
    ds_config["cameras"] = cfg.get("cameras", ["front_left"])

    # Load extractor config if provided
    extractor_config = None
    if cfg.get("use_overlapping_pairs", False):
        if cfg.get("extractor_config") is None:
            # Default to SuperPoint if not provided
            extractor_config = {"class": "SuperPoint"}
        else:
            extractor_config = json.loads(Path(cfg["extractor_config"]).read_text())

    export_dataset(
        ds_config,
        cfg["output"],
        step=cfg.get("step", 1),
        max_pairs=cfg.get("max_pairs"),
        cameras=cfg.get("cameras", ["front_left"]),
        use_overlapping_pairs=cfg.get("use_overlapping_pairs", False),
        extractor_config=extractor_config,
        camera=cfg.get("camera", "front"),
        max_depth=cfg.get("max_depth", 100),
        thresh_dist=cfg.get("thresh_dist", 0.025),
        min_covisible=cfg.get("min_covisible", 200),
        max_frames=cfg.get("max_frames"),
        n_jobs=cfg.get("n_jobs", 1),
        max_sep=cfg.get("max_sep"),
        log_rerun=cfg.get("rerun", False),
        rerun_save=cfg.get("rerun_save"),
        agents=cfg.get("agents"),
    )


if __name__ == "__main__":
    main()
