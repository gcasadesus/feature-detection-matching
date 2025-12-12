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
rovers = ["rover_0"]
steps = [1, 10]
base_data_dir = "/home/shared_ws6/data/unreal_engine/local_traverse_fov90"
output_base_dir = "/home/guillemc/dev/LuPNT-private/output/2025_FeatureMatching/eval_results"
script_path = (
    "/home/guillemc/dev/LuPNT-private/projects/2025_FeatureMatching/eval/eval_unreal_simple.py"
)


def run_single_eval(args):
    dataset_name, rover, step = args

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

    # Save config to temp file (unique name for parallel execution)
    config_dir = Path("/tmp/eval_configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{dataset_name}_{rover}_step{step}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    log_dir = Path("/tmp/eval_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset_name}_{rover}_step{step}.log"

    print(f"Starting evaluation for {dataset_name} - {rover} - step {step}. Logs at {log_file}")

    output_dir = Path(output_base_dir) / dataset_name / rover
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"step_{step}.pkl"

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
    ]

    try:
        # Redirect output to log file
        with open(log_file, "w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)

        print(f"Finished evaluation for {dataset_name} - {rover} - step {step}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {dataset_name} {rover} step {step}. See {log_file}")
        return False


def run_evaluation():
    # Generate all tasks
    tasks = []
    for dataset_name in datasets:
        for rover in rovers:
            for step in steps:
                tasks.append((dataset_name, rover, step))

    # Run in parallel with 10 workers
    print(f"Starting parallel evaluation with 10 workers for {len(tasks)} tasks...")
    with Pool(processes=10) as pool:
        pool.map(run_single_eval, tasks)


if __name__ == "__main__":
    run_evaluation()
