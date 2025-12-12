# Feature Extractor and Matcher Evaluation and Training

This directory contains scripts for evaluating feature extractors and matchers on the Unreal dataset using glue-factory metrics, and for fine-tuning LightGlue on the Unreal dataset.

## Evaluation

### Simple Evaluation Script

The `eval_unreal_simple.py` script evaluates different extractor+matcher combinations using glue-factory-style metrics without requiring full glue-factory integration.

**Usage:**
```bash
python eval_unreal_simple.py \
    --config configs/datasets/unreal.yaml \
    --extractors SuperPoint SIFT ORB AKAZE BRISK \
    --matchers LightGlue SuperGlue BruteForce Flann \
    --step 1 \
    --max_pairs 100 \
    --output results.pkl
```

**Metrics computed:**
- Epipolar error (precision at 1e-4, 5e-4, 1e-3)
- Reprojection error (precision at 1px, 3px, 5px)
- Number of matches and keypoints
- Coverage statistics

### Full Glue-Factory Evaluation

The `eval_unreal_gluefactory.py` script uses glue-factory's full evaluation pipeline (requires glue-factory to be properly set up).

**Usage:**
```bash
python eval_unreal_gluefactory.py \
    --config configs/datasets/unreal.yaml \
    --extractors SuperPoint \
    --matchers LightGlue \
    --estimator poselib \
    --ransac_th 1.0
```

## Training LightGlue

### Step 1: Export Dataset

First, export the Unreal dataset in a format suitable for glue-factory training:

```bash
python export_unreal_for_training.py \
    --config configs/datasets/unreal.yaml \
    --output data/unreal_exported \
    --step 1 \
    --max_pairs 1000 \
    --cameras front_left
```

This creates:
```
data/unreal_exported/
  unreal/
    images/
      000000.jpg
      000001.jpg
      ...
    depths/
      000000.h5
      000001.h5
      ...
    views.txt
    pairs.txt
  metadata.json
```

### Step 2: Train LightGlue

#### Option A: Using the Python script

```bash
python train_lightglue_unreal.py \
    --data_dir data/unreal_exported \
    --experiment sp+lg_unreal \
    --load_experiment sp+lg_homography \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4
```

#### Option B: Using glue-factory directly

First, update the config file `configs/superpoint+lightglue_unreal.yaml` with the correct data path, then:

```bash
cd python/thirdparty/glue-factory
python -m gluefactory.train sp+lg_unreal \
    --conf ../../../projects/2025_FeatureMatching/configs/superpoint+lightglue_unreal.yaml \
    train.load_experiment=sp+lg_homography
```

### Training Configuration

The training configuration includes:
- **Extractor**: SuperPoint (frozen, not trained)
- **Matcher**: LightGlue (trained)
- **Ground truth**: Depth-based matcher (uses depth maps for supervision)
- **Learning rate**: 1e-4 with exponential decay starting at epoch 30
- **Batch size**: 32 (adjust based on GPU memory)

### Evaluation During Training

The training script can run benchmarks during training. To enable:

```bash
python -m gluefactory.train sp+lg_unreal \
    --conf configs/superpoint+lightglue_unreal.yaml \
    --run_benchmarks
```

## Results

Evaluation results are saved as pickle files in:
```
output/2025_FeatureMatching/eval_gluefactory/results_<timestamp>.pkl
```

Training checkpoints are saved in glue-factory's output directory (typically `outputs/training/`).

## Notes

- The evaluation scripts work with any extractor/matcher combination that's compatible
- LightGlue requires SuperPoint features
- SuperGlue also requires SuperPoint features
- The training uses depth-based ground truth matching, which requires depth maps in the dataset
- Make sure you have sufficient GPU memory for training (batch size 32 requires ~8GB VRAM)
