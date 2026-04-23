# DiffShape

Diffusion-based brain shape prior pipeline. Given a T1 brain MRI, it produces a 3D signed-distance prior (`mu_sdt`) together with a voxel-wise confidence map (`w_conf`), which can be consumed by downstream segmentation models.

The pipeline is config-driven: adding a new dataset only requires writing one YAML config — no Python changes.

## Pipeline

```
T1 MRI  ──► prepare_data.py  ──► processed dataset  ──► train_diffusion.py  ──► checkpoint
                                                                                     │
                                                                                     ▼
                                  per-case (mu_sdt, w_conf)  ◄── samples_to_sdt.py ◄── predict.py
```

| Stage | Script | Input | Output |
|---|---|---|---|
| Data prep | `prepare_data.py` | YAML config(s) | `processed_dataset/<name>/` (registered NIfTI, centers, point clouds, splits) |
| Training | `train_diffusion.py` | processed dataset | `checkpoints/<name>.pth` + `<name>_stats.npz` |
| Prediction | `predict.py` | checkpoint + processed dataset | `samples_<dataset>_K<k>.npy` + `meta_<dataset>_K<k>.json` |
| Prior volume | `samples_to_sdt.py` | predict outputs | per-case `mu_sdt`, `w_conf` (h5 / nii.gz) |
| Visualization | `visualize_samples.py` | predict outputs | per-case 3D point-cloud plots |

## Install

```bash
pip install antspyx nibabel numpy pyyaml mcubes trimesh tqdm torch h5py scipy matplotlib
```

## 1. Prepare data

```bash
python -m diffshape.prepare_data \
  --configs diffshape/configs/cc359.yaml \
  --output-dir processed_dataset
```

Add `--skip-registration` if the input is already MNI152-aligned. Multiple configs can be passed at once.

### Adding a new dataset

Drop a YAML in `diffshape/configs/`. Two layouts are supported.

**`flat_directory`** — images and masks in separate folders:

```yaml
dataset_name: my_dataset
layout: flat_directory
data_root: "path/to/data"
image_dir: "images"
mask_dir: "masks"
image_pattern: "{case_id}.nii.gz"
mask_pattern: "{case_id}_mask.nii.gz"
case_id_regex: "^(?P<id>[A-Za-z0-9_]+)\\.nii\\.gz$"
crop_shape: [192, 192, 192]
crop_function: "legacy"
norm_method: "zs"
n_patch: 64
split:
  method: "ratio"
  train_ratio: 0.7
```

**`folder_per_case`** — one folder per subject:

```yaml
dataset_name: my_dataset
layout: folder_per_case
data_root: "path/to/subjects"
image_filename: "T1.nii.gz"
mask_filename:
  - "mask.nii.gz"
  - "brain_mask.nii.gz"
crop_shape: [192, 192, 192]
crop_function: "legacy"
norm_method: "zs"
n_patch: 64
split:
  method: "ratio"
  train_ratio: 0.8
```

## 2. Train

```bash
python -m diffshape.train_diffusion \
  --processed-dir processed_dataset \
  --datasets cc359 gbm125 nfbs \
  --variant diff_r_rm \
  --gpu 0
```

Variants:
- `diff` — plain MSE loss
- `diff_r` — radius-weighted loss
- `diff_r_rm` — radius-weighted loss + random-mask augmentation (**recommended**)

Key hyperparameters: `--timesteps 500`, `--beta-start 1e-4`, `--beta-end 2e-3`, `--n-pc 4096` (n_patch=64 → 64×64 points).

## 3. Predict

Produces K diffusion samples per case as normalized radii on a fixed direction grid.

```bash
python -m diffshape.predict \
  --checkpoint diffshape/checkpoints/checkpoints/checkpoint_final.pth \
  --dataset cc359 \
  --processed-dir processed_dataset \
  --output-dir outputs/cc359_K16 \
  --k-samples 16 \
  --ddim-steps 50 \
  --indices 0 1 2        # optional; default: all cases in the dataset
```

- `--k-samples {4, 8, 16}` (default 16)
- `--indices` accepts an arbitrary list of case indices; omit to run the whole dataset
- Outputs: `samples_<dataset>_K<k>.npy` of shape `(N, K, n_pc)` + `meta_<dataset>_K<k>.json` with fixed_center, min_r, max_r, n_patch, indices, image_paths.

## 4. Sample → SDT prior

```bash
python -m diffshape.samples_to_sdt \
  --samples  outputs/cc359_K16/samples_cc359_K16.npy \
  --meta     outputs/cc359_K16/meta_cc359_K16.json \
  --output-dir outputs/cc359_K16/priors \
  --format both          # h5 | nii | both
```

Per-case outputs (192³ volumes):
- `mu_sdt` — mean signed distance transform (negative inside, positive outside)
- `w_conf` — confidence = `exp(-var / tau)`, higher where inter-sample variance is lower

Tuning knobs: `--grid-size`, `--smooth-sigma`, `--var-blur`, `--tau`.

## 5. Visualize (optional)

```bash
python -m diffshape.visualize_samples \
  --samples  outputs/cc359_K16/samples_cc359_K16.npy \
  --meta     outputs/cc359_K16/meta_cc359_K16.json \
  --output-dir outputs/cc359_K16/viz
```

Default mode `mean` averages the K samples into a single predicted point cloud per case. `--mode grid` shows K individual samples per case; `--mode overlay` overlays them in one axes; `--mode all` produces all three.

## End-to-end validation

`scripts/validate_pipeline.sh GPU K DATASET [INDICES...]` chains predict → samples_to_sdt → visualize_samples on a small subset, writing everything to `pipeline_validation/<dataset>_K<K>/`.

```bash
bash diffshape/scripts/validate_pipeline.sh 0 16 cc359 0 1 2
```

## Reproducing the paper

The training, evaluation, and robustness scripts below let you retrain and re-evaluate every variant from scratch. Only the best-performing variant (`diff_r_rm` trained on cc359 + gbm125 + nfbs) is shipped as `checkpoint_final.pth` for inference use.

### Evaluation (clean)

`eval_diffusion.py` runs 50-step DDIM sampling (K=16) on the test split of each dataset and reports per-case RMSE (×100) and Chamfer distance.

```bash
python -m diffshape.eval_diffusion \
  --checkpoint diffshape/checkpoints/checkpoints/checkpoint_final.pth \
  --datasets cc359 gbm125 nfbs \
  --gpu 0
```

Batch-evaluate every variant after training with `scripts/eval_mixed.sh <GPU>` (3-dataset models) or `scripts/eval_cc359.sh <GPU>` (CC359-only models).

### Robustness

The same entry point, with `--mask-ratio r` applying a centered-cube zero-mask whose side length is `r^(1/3) * D` over the input image, tests how the diffusion prior behaves when part of the image is occluded.

```bash
python -m diffshape.eval_diffusion \
  --checkpoint <ckpt>.pth \
  --datasets cc359 gbm125 nfbs \
  --mask-ratio 0.4 \
  --gpu 0
```

Full sweeps at ratios `{0.2, 0.4, 0.6, 0.8}` over the `_r` and `_r_rm` variants:

```bash
bash diffshape/scripts/robust_mixed.sh 0
bash diffshape/scripts/robust_cc359.sh 0
```

### Training all variants

`scripts/train_mixed.sh <GPU>` and `scripts/train_cc359.sh <GPU>` train the three variants (`diff`, `diff_r`, `diff_r_rm`) on the two data regimes used in the paper.

## Module layout

```
diffshape/
├── configs/              # one YAML per dataset
├── data/                 # registry, registration, center_finder, gi_extractor, splits
├── checkpoints/          # released model (checkpoint_final.pth + _stats.npz)
├── prepare_data.py       # stage 1
├── train_diffusion.py    # stage 2
├── predict.py            # stage 3
├── samples_to_sdt.py     # stage 4
├── visualize_samples.py  # visualization
├── inference.py          # shared inference utilities
├── eval_diffusion.py     # paper evaluation + robustness (clean / mask-ratio)
└── scripts/              # train / eval / robust / validate launch scripts
```

## Processed dataset layout

`processed_dataset/<dataset_name>/` contains:

- `case_ids.npy`, `image_paths.npy`, `mask_paths.npy`
- `centers.npy` — `(N, 3)` brain centers in voxel coordinates
- `cGI_<name>_<n_points>rpt_preC.npy` — `(N, n_points, 3)` point clouds (n_points = n_patch²)
- `split_train_indices.npy`, `split_test_indices.npy`
- `meshes/` — exported `.obj` meshes (optional)
- `registered/` — MNI152-aligned NIfTI files (if registration was run)
- `summary.json` — run metadata

## Final checkpoint

`diffshape/checkpoints/checkpoints/checkpoint_final.pth` (+ `checkpoint_final_stats.npz`) is the released model — `diff_r_rm` trained on cc359 + gbm125 + nfbs. Loaded by default in `predict.py` via the `--checkpoint` flag.
