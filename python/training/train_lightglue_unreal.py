"""
Fine-tune LightGlue on Unreal dataset.

This script fine-tunes LightGlue on the Unreal dataset using glue-factory's training infrastructure.
First, export the dataset using export_unreal_for_training.py, then use this script to train.
"""

import argparse
import sys
from pathlib import Path

import pylupnt as pnt
import yaml

# Add glue-factory to path
sys.path.insert(0, str(pnt.BASEDIR / "python" / "thirdparty" / "glue-factory"))

from omegaconf import OmegaConf


def load_config(config_path=None):
    """Load configuration from YAML file, with command line overrides."""
    script_dir = Path(__file__).parent
    default_config_path = script_dir / "train_lightglue_config.yaml"

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
    parser = argparse.ArgumentParser(
        description="Fine-tune LightGlue on Unreal dataset"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML config file (default: train_lightglue_config.yaml in script directory)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory with exported dataset (overrides config file)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (overrides config file)",
    )
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        default=None,
        help="Don't append timestamp to experiment name (overrides config file)",
    )
    parser.add_argument(
        "--load_experiment",
        type=str,
        default=None,
        help="Load pretrained model from experiment (overrides config file)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config file)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config file)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config file)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of data loading workers (overrides config file)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Image resize size (overrides config file)",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=None,
        help="Max keypoints (overrides config file)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=None,
        help="Start TensorBoard automatically (overrides config file)",
    )
    parser.add_argument(
        "--tensorboard_port",
        type=int,
        default=None,
        help="Port for TensorBoard (overrides config file)",
    )

    args = parser.parse_args()

    # Load config from YAML file
    cfg = load_config(args.config_file)

    # Override with command line arguments
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.experiment is not None:
        cfg["experiment"] = args.experiment
    if args.no_timestamp is not None:
        cfg["no_timestamp"] = args.no_timestamp
    if args.load_experiment is not None:
        cfg["load_experiment"] = args.load_experiment
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.resize is not None:
        cfg["resize"] = args.resize
    if args.max_keypoints is not None:
        cfg["max_keypoints"] = args.max_keypoints
    if args.tensorboard is not None:
        cfg["tensorboard"] = args.tensorboard
    if args.tensorboard_port is not None:
        cfg["tensorboard_port"] = args.tensorboard_port

    # Validate required fields
    if cfg.get("data_dir") is None:
        raise ValueError(
            "'data_dir' (directory with exported dataset) is required. Set it in config file or use --data_dir"
        )
    if cfg.get("experiment") is None:
        raise ValueError(
            "'experiment' (experiment name) is required. Set it in config file or use --experiment"
        )

    data_dir = Path(cfg["data_dir"])
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Check for multi-agent structure
    unreal_dir = data_dir / "unreal"
    scene_list = []

    if unreal_dir.exists():
        # Check if "unreal" itself is a scene (contains views.txt)
        if (unreal_dir / "views.txt").exists():
            scene_list = ["unreal"]
        else:
            # Check for agent subdirectories
            for item in unreal_dir.iterdir():
                if item.is_dir() and (item / "views.txt").exists():
                    scene_list.append(f"unreal/{item.name}")

    # Fallback/Legacy: Check root directory or reorganization
    if not scene_list:
        if (data_dir / "images").exists():
            # Reorganize: create scene subdirectory
            scene_dir = data_dir / "unreal"
            scene_dir.mkdir(exist_ok=True)

            # Move files
            import shutil

            if (data_dir / "images").exists() and not (scene_dir / "images").exists():
                shutil.move(str(data_dir / "images"), str(scene_dir / "images"))
            if (data_dir / "depths").exists() and not (scene_dir / "depths").exists():
                shutil.move(str(data_dir / "depths"), str(scene_dir / "depths"))
            if (data_dir / "views.txt").exists() and not (
                scene_dir / "views.txt"
            ).exists():
                shutil.move(str(data_dir / "views.txt"), str(scene_dir / "views.txt"))
            if (data_dir / "pairs.txt").exists() and not (
                scene_dir / "pairs.txt"
            ).exists():
                shutil.move(str(data_dir / "pairs.txt"), str(scene_dir / "pairs.txt"))

            pnt.Logger.info(f"Reorganized data directory structure", "Train")
            scene_list = ["unreal"]

    if not scene_list:
        pnt.Logger.warning(
            f"No valid scenes found in {data_dir}. Expected 'unreal' directory with agent subdirectories or 'unreal' as a scene.",
            "Train",
        )

    pnt.Logger.info(f"Found {len(scene_list)} scenes: {scene_list}", "Train")

    # Create training configuration using posed_images dataset format
    train_conf = OmegaConf.create(
        {
            "data": {
                "name": "posed_images",
                "root": str(
                    data_dir
                ),  # data_dir should contain scene folder (e.g., unreal/)
                "image_dir": "{scene}/images",
                "depth_dir": "{scene}/depths",
                "label_dir": "/home/shared_ws6/data/unreal_engine/spirals/{scene_base}/cam_front/label",
                "views": "{scene}/views.txt",
                "view_groups": "{scene}/pairs.txt",
                "depth_format": "h5",
                "scene_list": scene_list,
                "preprocessing": {
                    "resize": args.resize,
                    "side": "long",
                    "square_pad": False,  # Disable square_pad to avoid depth indexing issues
                },
                "batch_size": cfg["batch_size"],
                "num_workers": cfg["num_workers"],
                "train_batch_size": cfg["batch_size"],
                "val_batch_size": cfg["batch_size"],
                "test_batch_size": 1,
                "shuffle_training": True,
            },
            "model": {
                "name": "two_view_pipeline",
                "extractor": {
                    "name": "gluefactory_nonfree.superpoint",
                    "max_num_keypoints": cfg["max_keypoints"],
                    "force_num_keypoints": True,
                    "detection_threshold": 0.0,
                    "nms_radius": 3,
                    "trainable": False,  # Don't train extractor
                },
                "matcher": {
                    "name": "matchers.lightglue",
                    "features": "superpoint",
                    "filter_threshold": 0.1,
                    "flash": True,
                    "checkpointed": True,
                    "weights": "superpoint",
                    "num_classes": 15,
                },
                "ground_truth": {
                    "name": "matchers.depth_matcher",
                    "th_positive": 3,
                    "th_negative": 5,
                    "th_epi": 5,
                },
                "allow_no_extract": True,
            },
            "train": {
                "seed": 0,
                "epochs": cfg["epochs"],
                "log_every_iter": 100,
                "eval_every_iter": 2000,
                "save_every_iter": 5000,
                "lr": float(cfg["lr"]),
                "lr_schedule": {
                    "start": int(cfg["epochs"] * 0.6),  # Start decay at 60% of epochs
                    "type": "exp",
                    "on_epoch": True,
                    "exp_div_10": 10,
                },
                "best_key": "loss/total",
                "keep_last_checkpoints": 10,
                "load_experiment": args.load_experiment,
            },
        }
    )

    # Override with command line arguments
    if cfg.get("load_experiment"):
        train_conf.train.load_experiment = cfg["load_experiment"]

    # Add timestamp to experiment name if not disabled
    experiment_name = cfg["experiment"]
    if not cfg.get("no_timestamp", False):
        timestamp = (
            pnt.get_timestamp().replace("-", "").replace(" ", "_").replace(":", "")
        )
        experiment_name = f"{cfg['experiment']}_{timestamp}"

    pnt.Logger.info(f"Starting training experiment: {experiment_name}", "Train")
    pnt.Logger.info(f"Data directory: {data_dir}", "Train")
    pnt.Logger.info(f"Configuration:\n{OmegaConf.to_yaml(train_conf)}", "Train")

    # Set custom output directory
    custom_training_path = pnt.BASEDIR / "output" / "2025_FeatureMatching" / "training"
    custom_training_path.mkdir(parents=True, exist_ok=True)

    # Override glue-factory's TRAINING_PATH
    import gluefactory.settings

    gluefactory.settings.TRAINING_PATH = custom_training_path

    # Import and run training directly (like train.py main block)
    from gluefactory.train import main_worker, default_train_conf
    from gluefactory import logger, settings
    import torch
    import shutil
    import subprocess as sp
    import sys

    pnt.Logger.info(
        f"Running training with output dir: {custom_training_path}", "Train"
    )

    # Replicate train.py main block logic
    output_dir = Path(settings.TRAINING_PATH, experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start TensorBoard if requested
    tensorboard_process = None
    tensorboard_enabled = cfg.get("tensorboard", False)
    tensorboard_port = cfg.get("tensorboard_port", 6006)
    if tensorboard_enabled:
        tensorboard_cmd = [
            "tensorboard",
            "--logdir",
            str(output_dir),
            "--port",
            str(tensorboard_port),
            "--host",
            "0.0.0.0",
        ]
        pnt.Logger.info(
            f"Starting TensorBoard on port {tensorboard_port}: http://localhost:{tensorboard_port}",
            "Train",
        )
        tb_log_path = output_dir / "tensorboard_log.txt"
        with open(tb_log_path, "w") as tb_log:
            tensorboard_process = sp.Popen(
                tensorboard_cmd, stdout=tb_log, stderr=sp.STDOUT
            )

    # Create a simple args object for main_worker (matching train.py argparse)
    class SimpleArgs:
        def __init__(self):
            self.restore = False
            self.distributed = False
            self.compile = None
            self.mixed_precision = None
            self.overfit = False
            self.print_arch = False
            self.detect_anomaly = False
            self.profile = False
            self.log_it = False
            self.no_eval_0 = False
            self.run_benchmarks = False
            self.cleanup_interval = 120
            self.n_gpus = 1
            self.lock_file = None

    train_args = SimpleArgs()

    # Merge config with defaults
    train_conf = OmegaConf.merge(
        OmegaConf.create({"train": default_train_conf}), train_conf
    )

    if not train_args.restore:
        if train_conf.train.seed is None:
            train_conf.train.seed = torch.initial_seed() & (2**32 - 1)
        OmegaConf.save(train_conf, str(output_dir / "config.yaml"))

    # Copy gluefactory to output dir
    for module in train_conf.train.get("submodules", []) + ["gluefactory"]:
        mod_dir = Path(__import__(module).__file__).parent
        shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)

    # Run training
    try:
        main_worker(0, train_conf, output_dir, train_args)
    except Exception as e:
        pnt.Logger.error(f"Training failed: {e}", "Train")
        raise
    finally:
        # Clean up TensorBoard if we started it
        if tensorboard_process is not None:
            pnt.Logger.info("Stopping TensorBoard", "Train")
            tensorboard_process.terminate()
            tensorboard_process.wait()


if __name__ == "__main__":
    main()
