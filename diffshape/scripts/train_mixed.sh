#!/bin/bash
# Mixed 3-dataset training (GPU 0): 3 variants
set -e

GPU=${1:-0}
PROCESSED=processed_dataset
COMMON="--timesteps 500 --beta-start 0.0001 --beta-end 0.002 --gpu $GPU"

for VARIANT in diff diff_r diff_r_rm; do
  echo "=== Mixed training: $VARIANT ==="
  python -m diffshape.train_diffusion \
    --processed-dir $PROCESSED --datasets cc359 gbm125 nfbs \
    --variant $VARIANT --active-folds 0 1 2 3 4 $COMMON
done

echo "=== Mixed training complete ==="
