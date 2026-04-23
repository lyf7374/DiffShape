#!/bin/bash
# Train 6 experiments: 3 variants × 2 dataset configs
# beta=0.0001-0.002, T=500

set -e

GPU=${1:-0}
PROCESSED=processed_dataset
COMMON="--timesteps 500 --beta-start 0.0001 --beta-end 0.002 --gpu $GPU"

echo "=== Mixed 3-dataset training ==="

python -m diffshape.train_diffusion \
  --processed-dir $PROCESSED --datasets cc359 gbm125 nfbs \
  --variant diff --active-folds 0 1 2 3 4 $COMMON

python -m diffshape.train_diffusion \
  --processed-dir $PROCESSED --datasets cc359 gbm125 nfbs \
  --variant diff_r --active-folds 0 1 2 3 4 $COMMON

python -m diffshape.train_diffusion \
  --processed-dir $PROCESSED --datasets cc359 gbm125 nfbs \
  --variant diff_r_rm --active-folds 0 1 2 3 4 $COMMON

echo "=== CC359-only training ==="

python -m diffshape.train_diffusion \
  --processed-dir $PROCESSED --datasets cc359 \
  --variant diff $COMMON

python -m diffshape.train_diffusion \
  --processed-dir $PROCESSED --datasets cc359 \
  --variant diff_r $COMMON

python -m diffshape.train_diffusion \
  --processed-dir $PROCESSED --datasets cc359 \
  --variant diff_r_rm $COMMON

echo "=== All 6 training runs complete ==="
