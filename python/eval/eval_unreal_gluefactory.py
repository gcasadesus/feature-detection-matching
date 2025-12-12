"""
Evaluate feature extractors and matchers on Unreal dataset using glue-factory metrics.

This script evaluates different extractor+matcher combinations on the Unreal dataset
using the same metrics as glue-factory benchmarks (epipolar error, depth reprojection,
relative pose error, etc.).
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import pylupnt as pnt
from pylupnt.math import ensure_torch

# Import glue-factory evaluation utilities
import sys

sys.path.insert(0, str(pnt.BASEDIR / "python" / "thirdparty" / "glue-factory"))
from gluefactory.eval.utils import (
    eval_matches_depth,
    eval_matches_epipolar,
    eval_relative_pose_robust,
)
from gluefactory.geometry.wrappers import Camera, Pose
from gluefactory.robust_estimators import load_estimator


def convert_image_data_to_gluefactory_format(img_data1, img_data2):
    """Convert pylupnt ImageData to glue-factory format.

    Args:
        img_data1: First image data (pnt.ImageData)
        img_data2: Second image data (pnt.ImageData)

    Returns:
        Dictionary in glue-factory format with keys:
        - view0, view1: dictionaries with image, camera, depth
        - T_0to1: relative pose transformation
    """
    # Convert images to torch tensors [1, C, H, W]
    img1 = ensure_torch(img_data1.rgb, channels=3, device="cpu", batch_dim=True)
    img2 = ensure_torch(img_data2.rgb, channels=3, device="cpu", batch_dim=True)

    # Convert depth maps
    depth1 = ensure_torch(img_data1.depth, device="cpu", batch_dim=True).unsqueeze(1)
    depth2 = ensure_torch(img_data2.depth, device="cpu", batch_dim=True).unsqueeze(1)

    # Convert camera intrinsics to glue-factory Camera format
    K1 = pnt.make_camera_matrix(img_data1.intrinsics)
    K2 = pnt.make_camera_matrix(img_data2.intrinsics)

    h1, w1 = img_data1.rgb.shape[:2]
    h2, w2 = img_data2.rgb.shape[:2]

    # Create Camera objects (using pinhole model)
    camera1 = Camera.from_colmap(
        {
            "model": "PINHOLE",
            "width": w1,
            "height": h1,
            "params": np.array([K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]], dtype=np.float32),
        }
    )
    camera2 = Camera.from_colmap(
        {
            "model": "PINHOLE",
            "width": w2,
            "height": h2,
            "params": np.array([K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]], dtype=np.float32),
        }
    )

    # Convert poses: world_T_cam to T_w2cam (inverse)
    T_w2cam1 = Pose.from_4x4(torch.from_numpy(pnt.invert_transform(img_data1.world_T_cam)).float())
    T_w2cam2 = Pose.from_4x4(torch.from_numpy(pnt.invert_transform(img_data2.world_T_cam)).float())

    # Relative pose: T_0to1 = T_w2cam2 @ T_cam1to_w = T_w2cam2 @ T_w2cam1.inv()
    T_0to1 = T_w2cam2 @ T_w2cam1.inv()

    data = {
        "view0": {
            "image": img1,
            "camera": camera1,
            "depth": depth1,
            "T_w2cam": T_w2cam1,
        },
        "view1": {
            "image": img2,
            "camera": camera2,
            "depth": depth2,
            "T_w2cam": T_w2cam2,
        },
        "T_0to1": T_0to1,
    }

    return data


def convert_features_to_gluefactory_format(feats1, feats2, matches):
    """Convert pylupnt Features and Matches to glue-factory format.

    Args:
        feats1: First features (pnt.Features)
        feats2: Second features (pnt.Features)
        matches: Matches (pnt.Matches)

    Returns:
        Dictionary with keypoints, matches, and matching scores
    """
    # Convert keypoints to torch [N, 2]
    kpts0 = torch.from_numpy(feats1.uv.copy()).float()
    kpts1 = torch.from_numpy(feats2.uv.copy()).float()

    # Create matches array: matches0[i] = j if keypoint i matches keypoint j, else -1
    matches0 = torch.full((len(feats1),), -1, dtype=torch.long)
    if len(matches.indexes) > 0:
        matches0[matches.indexes[:, 0]] = matches.indexes[:, 1]

    # Matching scores (convert distances to scores: 1 - normalized_distance)
    matching_scores0 = torch.zeros(len(feats1), dtype=torch.float32)
    if len(matches.indexes) > 0:
        # Normalize distances to [0, 1] and convert to scores
        max_dist = matches.distances.max() if len(matches.distances) > 0 else 1.0
        scores = 1.0 - (matches.distances / (max_dist + 1e-6))
        matching_scores0[matches.indexes[:, 0]] = torch.from_numpy(scores).float()

    pred = {
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "matches0": matches0,
        "matching_scores0": matching_scores0,
    }

    return pred


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
    # Convert to glue-factory format
    data = convert_image_data_to_gluefactory_format(img_data1, img_data2)
    pred = convert_features_to_gluefactory_format(feats1, feats2, matches)

    results = {}

    # Epipolar error metrics
    results.update(eval_matches_epipolar(data, pred))

    # Depth reprojection metrics
    results.update(eval_matches_depth(data, pred))

    # Relative pose error (using robust estimator)
    pose_results = eval_relative_pose_robust(data, pred, eval_conf)
    results.update(pose_results)

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
        "--max_pairs", type=int, default=None, help="Maximum number of pairs to evaluate"
    )
    parser.add_argument("--step", type=int, default=1, help="Step between consecutive frames")
    parser.add_argument(
        "--estimator",
        type=str,
        default="opencv",
        choices=["opencv", "poselib"],
        help="Robust estimator for pose estimation",
    )
    parser.add_argument("--ransac_th", type=float, default=1.0, help="RANSAC threshold")
    parser.add_argument("--cameras", nargs="+", default=["front_left"], help="Cameras to evaluate")

    args = parser.parse_args()

    # Load dataset
    ds_config = pnt.load_config(args.config)
    dataset = pnt.datasets.Dataset.from_config(ds_config)

    # Evaluation configuration
    eval_conf = {
        "estimator": args.estimator,
        "ransac_th": args.ransac_th,
    }

    # Create extractors and matchers
    extractor_configs = {
        "SuperPoint": {"class": "SuperPoint"},
        "SIFT": {"class": "Sift"},
        "ORB": {"class": "Orb"},
        "AKAZE": {"class": "Akaze"},
        "BRISK": {"class": "Brisk"},
    }

    matcher_configs = {
        "LightGlue": {"class": "LightGlue", "features": "superpoint"},
        "SuperGlue": {"class": "SuperGlue"},
        "BruteForce": {"class": "BruteForceMatcher"},
        "Flann": {"class": "FlannMatcher"},
    }

    extractors = {}
    for name in args.extractors:
        if name in extractor_configs:
            extractors[name] = pnt.FeatureExtractor.from_config(extractor_configs[name])

    matchers = {}
    for name in args.matchers:
        if name in matcher_configs:
            matchers[name] = pnt.FeatureMatcher.from_config(matcher_configs[name])

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

    for e_name, extractor in extractors.items():
        for m_name, matcher in matchers.items():
            # Check compatibility
            if m_name == "LightGlue" and e_name != "SuperPoint":
                continue
            if m_name == "SuperGlue" and e_name != "SuperPoint":
                continue

            combo_name = f"{e_name}+{m_name}"
            pnt.Logger.info(f"Evaluating {combo_name}", "Eval")

            results = defaultdict(list)

            for idx1, idx2 in tqdm(pairs, desc=combo_name):
                try:
                    # Get image data
                    img_data1 = dataset[idx1]["cameras"][args.cameras[0]]
                    img_data2 = dataset[idx2]["cameras"][args.cameras[0]]

                    # Extract features
                    feats1 = extractor.extract(img_data1.rgb)
                    feats2 = extractor.extract(img_data2.rgb)

                    # Match features
                    matches = matcher.match(feats1, feats2)

                    # Evaluate
                    pair_results = evaluate_pair(
                        img_data1, img_data2, feats1, feats2, matches, eval_conf
                    )

                    for k, v in pair_results.items():
                        results[k].append(v)

                except Exception as e:
                    pnt.Logger.warn(f"Error evaluating pair ({idx1}, {idx2}): {e}", "Eval")
                    continue

            # Aggregate results
            summary = {}
            for k, v in results.items():
                arr = np.array(v)
                if np.issubdtype(arr.dtype, np.number):
                    summary[f"m{k}"] = float(np.mean(arr))
                    summary[f"median_{k}"] = float(np.median(arr))
                    summary[f"std_{k}"] = float(np.std(arr))

            all_results[combo_name] = {
                "summary": summary,
                "per_pair": results,
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
                "eval_conf": eval_conf,
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
