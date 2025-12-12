import os
import subprocess
import yaml
from pathlib import Path
from multiprocessing import Pool

# Configuration
datasets = [
    "short_base",
    "short_camera_effects",
    "short_higher_elevation",
    "short_no_lights",
]
rover = "rover_0"
step = 10
base_data_dir = "/home/shared_ws6/data/unreal_engine/local_traverse_fov90"
output_base_dir = "/home/guillemc/dev/LuPNT-private/output/2025_FeatureMatching/eval_results"
checkpoint_path = "/home/guillemc/dev/LuPNT-private/output/2025_FeatureMatching/training/spirals_20251211_031900/checkpoint_best.tar"
script_path = (
    "/home/guillemc/dev/LuPNT-private/projects/2025_FeatureMatching/eval/eval_unreal_simple.py"
)


def run_single_eval(dataset_name):
    # Prepare config
    config_content = {
        "class": "UnrealDataset",
        "basedir": f"{base_data_dir}/{dataset_name}",
        "agent": rover,
        "cameras": ["front_left"],
        "measurements": ["front_left"],
        "euroc_format": True,
        "data_types": ["rgb", "depth"],
    }

    # Save config to temp file
    config_dir = Path("/tmp/eval_params_configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{dataset_name}_spirals.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    log_dir = Path("/tmp/eval_params_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset_name}_spirals.log"

    print(f"Starting evaluations for {dataset_name}. Logs at {log_file}")

    # Output file structure to match what notebook expects
    output_dir = Path(output_base_dir) / dataset_name / rover
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"step_{step}_spirals.pkl"

    cmd = [
        "python",
        script_path,
        "--config",
        str(config_path),
        "--step",
        str(step),
        "--cameras",
        "front_left",
        "--output",
        str(output_file),
        "--checkpoint",
        checkpoint_path,
        "--name_suffix",
        " (spirals)",
        "--matchers",
        "LightGlue",
        "--extractors",
        "SuperPoint",
    ]

    try:
        with open(log_file, "w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        print(f"Finished evaluation for {dataset_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"Error evaluating {dataset_name}. See {log_file}")
        return False


if __name__ == "__main__":
    # We already ran short_base manually, but re-running it is fine to ensure consistency
    # or we can filter it out if desired. Let's run all requested to be safe.

    print(f"Starting parallel evaluation on {len(datasets)} datasets...")
    with Pool(processes=4) as pool:
        pool.map(run_single_eval, datasets)
