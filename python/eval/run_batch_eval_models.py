import os
import glob
import subprocess
import yaml
from pathlib import Path
from multiprocessing import Pool


# Configuration
BASE_DATA_DIR_FOV90 = "/home/shared_ws6/data/unreal_engine/local_traverse_fov90"
BASE_DATA_DIR_SPIRALS = "/home/shared_ws6/data/unreal_engine/spirals"
OUTPUT_BASE_DIR = "/home/guillemc/dev/LuPNT-private/output/2025_FeatureMatching/eval_results_models"
SCRIPT_PATH = (
    "/home/guillemc/dev/LuPNT-private/projects/2025_FeatureMatching/eval/eval_unreal_simple.py"
)

# Training directories to scan
TRAINING_DIRS = [
    "/home/guillemc/dev/LuPNT-private/output/2025_FeatureMatching/training_old",
    "/home/guillemc/dev/LuPNT-private/output/2025_FeatureMatching/training",
]

# Datasets to evaluate
DATASETS_FOV90 = [
    "short_base",
    "short_camera_effects",
    "short_higher_elevation",
    "short_no_lights",
    "long_base",
    "long_camera_effects",
    "long_higher_elevation",
    "long_no_lights",
]
DATASETS_SPIRALS = ["rover_0"]

# Agent to evaluate
ROVER = "rover_0"
STEPS = [10]


def find_models():
    """Find all checkpoint_best.tar files in training directories."""
    models = []

    for train_dir in TRAINING_DIRS:
        search_path = os.path.join(train_dir, "*", "checkpoint_best.tar")
        for checkpoint_path in glob.glob(search_path):
            model_name = os.path.basename(os.path.dirname(checkpoint_path))

            # Filtering Logic
            if "semantic_test" in model_name:
                continue  # Skip old semantic tests
            if "spirals_multi_agent" in model_name:
                continue  # Skip failed multi-agent runs

            models.append((model_name, checkpoint_path))

    return sorted(models, key=lambda x: x[0])


def run_single_eval(args):
    model_name, checkpoint_path, dataset_name, dataset_type, step = args

    # Determine dataset path and config params based on type
    if dataset_type == "fov90":
        basedir = f"{BASE_DATA_DIR_FOV90}/{dataset_name}"
        agent_name = ROVER
    elif dataset_type == "spirals":
        basedir = BASE_DATA_DIR_SPIRALS
        agent_name = "rover_0"

    # Generate a unique ID for this task
    task_id = f"{model_name}_{dataset_name}_{step}"

    # Determine ground truth path to verify existence
    gt_csv_path = None
    if dataset_type == "fov90":
        gt_csv_path = os.path.join(
            basedir, agent_name, f"state_groundtruth_estimate_{agent_name}", "data.csv"
        )
    elif dataset_type == "spirals":
        gt_csv_path = os.path.join(basedir, agent_name, "state_groundtruth_estimate", "data.csv")

    if gt_csv_path and not os.path.exists(gt_csv_path):
        # Silent skip or log? Silent for now to keep output clean, usually prints explicitly
        # print(f"Skipping {task_id}: Ground truth CSV not found")
        return False

    # Prepare config map
    config_content = {
        "class": "UnrealDataset",
        "basedir": basedir,
        "agent": agent_name,
        "cameras": ["front_left"] if dataset_type == "fov90" else ["front"],
        "measurements": ["front_left"] if dataset_type == "fov90" else ["front"],
        "euroc_format": True,
        "data_types": ["rgb", "depth"],
    }

    # Save config to temp file
    config_dir = Path("/tmp/eval_configs_models")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{task_id}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    # Output directory
    output_dir = Path(OUTPUT_BASE_DIR) / model_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"step_{step}.pkl"

    # Log file
    log_dir = Path("/tmp/eval_logs_models")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task_id}.log"

    # Check if already done
    if output_file.exists():
        # print(f"Skipping {task_id}, output exists.")
        return True

    cmd = [
        "python",
        SCRIPT_PATH,
        "--config",
        str(config_path),
        "--step",
        str(step),
        "--cameras",
        config_content["cameras"][0],
        "--output",
        str(output_file),
        "--checkpoint",
        checkpoint_path,
        "--matchers",
        "LightGlue",
        "--extractors",
        "SuperPoint",
    ]

    try:
        with open(log_file, "w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        print(f"Finished: {task_id}")
        return True
    except subprocess.CalledProcessError:
        print(f"FAILED: {task_id}. See {log_file}")
        return False


def run_evaluation():
    models = find_models()
    print(f"Found {len(models)} models to evaluate:")
    for m, _ in models:
        print(f" - {m}")

    tasks = []

    # Generate tasks for FOV90 datasets
    for model_name, checkpoint_path in models:
        for ds in DATASETS_FOV90:
            for step in STEPS:
                tasks.append((model_name, checkpoint_path, ds, "fov90", step))

    # Generate tasks for Spirals
    for model_name, checkpoint_path in models:
        for ds in DATASETS_SPIRALS:
            for step in STEPS:
                tasks.append((model_name, checkpoint_path, ds, "spirals", step))

    print(f"Starting parallel evaluation with 4 workers for {len(tasks)} tasks...")

    with Pool(processes=4) as pool:
        pool.map(run_single_eval, tasks)


if __name__ == "__main__":
    run_evaluation()
