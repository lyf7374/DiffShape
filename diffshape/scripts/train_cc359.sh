#!/bin/bash
# CC359-only training (GPU 1): 3 variants
set -e

GPU=${1:-1}
PROCESSED=processed_dataset
COMMON="--timesteps 500 --beta-start 0.0001 --beta-end 0.002 --gpu $GPU"

for VARIANT in diff diff_r diff_r_rm; do
  echo "=== CC359 training: $VARIANT ==="
  python -m diffshape.train_diffusion \
    --processed-dir $PROCESSED --datasets cc359 \
    --variant $VARIANT $COMMON
done

echo "=== CC359 training complete ==="
