import pylupnt as pnt
import numpy as np
from tqdm import tqdm
import pickle
from datetime import datetime
from pylupnt.features import SuperPoint  # For registering
import time

pnt.Logger.set_level(pnt.Logger.INFO)

output_dir = pnt.BASEDIR / "output" / "2025_FeatureMatching" / "pnp_vo"


# Extractors
extractor_configs = {
    # "SIFT": {"class": "Sift"},
    # "ORB": {"class": "Orb"},
    # "AKAZE": {"class": "Akaze"},
    # "BRISK": {"class": "Brisk"},
    "SuperPoint": {"class": "SuperPoint"},
}

# Matchers
matcher_configs = {
    # "Flann": {"class": "FlannMatcher"},
    # "BruteForce": {"class": "BruteForceMatcher"},
    # "SuperGlue": {"class": "SuperGlue"},
    "LightGlue": {"class": "LightGlue"},
}


def process_config(e_name, e_cfg, m_name, m_cfg):

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


def main():

    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Combinations
    combinations = []
    for e_name, e_cfg in extractor_configs.items():
        for m_name, m_cfg in matcher_configs.items():
            e_cfg_tmp, m_cfg_tmp = process_config(e_name, e_cfg, m_name, m_cfg)
            if e_cfg_tmp is not None and m_cfg_tmp is not None:
                combinations.append((e_name, m_name, e_cfg_tmp, m_cfg_tmp))

    # Dataset
    run_number = 2
    # for run_type in ["base", "higher_elevation", "motion_blur", "no_lights"][1:]:
    for run_type in ["base"]:
        run_name = f"{run_type}_{run_number}"

        cameras = ["front_left"]
        ds_config = {
            "inherit_from": "datasets/unreal.yaml",
            "step": 1,
            "basedir": f"/home/shared_ws6/local_traverse/{run_name}",
            # "end": 20,
            "preload": "cpu",
            "data_types": ["rgb", "depth"],
            "cameras": cameras,
        }
        dataset = pnt.datasets.Dataset.from_config(ds_config)
        n_frames = len(dataset)
        gt_poses = np.stack([ds_item["world_T_body"] for ds_item in dataset])

        # PnP VO
        output = {
            "gt_poses": gt_poses,
            "ds_config": ds_config,
            "runs": [],
        }
        bar_comb = pnt.Logger.get_progress_bar(len(combinations), "Combinations", "Main")
        for j, (e_name, m_name, e_cfg, m_cfg) in enumerate(combinations):
            run = {
                "extractor": e_name,
                "matcher": m_name,
                "extractor_cfg": e_cfg,
                "matcher_cfg": m_cfg,
                "poses": [],
                "features": [],
                "matches": [],
                "times": [],
            }

            pnp_vo_config = pnt.Config({"feature_extractor": e_cfg, "feature_matcher": m_cfg})
            pnp_vo = pnt.PnpVo(pnp_vo_config)

            desc = f"{j+1}/{len(combinations)}: {e_name} + {m_name}"
            bar_frame = pnt.Logger.get_progress_bar(n_frames, desc, "Main")
            for i in range(n_frames):
                ds_item = dataset[i]
                img_data = ds_item["cameras"][cameras[0]]
                tstart = time.time()
                success = pnp_vo.process_mono(img_data)
                tend = time.time()
                run["times"].append(tend - tstart)
                if not success:
                    pnt.Logger.error(f"Failed to process frame {i}", "Main")
                    break
                run["features"].append(pnp_vo.get_features())
                run["matches"].append(pnp_vo.get_matches())
                bar_frame.update()
            bar_frame.finish()

            run["poses"] = np.stack(pnp_vo.get_poses())
            output["runs"].append(run)
            bar_comb.update()
        bar_comb.finish()

        # Save results
        output_path = output_dir / f"main_pnp_vo_results_{run_name}_{timestamp}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(output, f)

        pnt.Logger.info(f"Saved results to {output_path}", "Main")
        pnt.Logger.info("Done", "Main")


if __name__ == "__main__":
    main()
