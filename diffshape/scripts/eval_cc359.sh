#!/bin/bash
set -e

GPU=${1:-1}
PROCESSED=processed_dataset
COMMON="--timesteps 500 --beta-start 0.0001 --beta-end 0.002 --ddim-steps 50 --k-samples 16 --gpu $GPU --processed-dir $PROCESSED --batch-size 2"

MODELS=(
  "checkpoints/diff_cc359_T500_b0.002.pth"
  "checkpoints/diff_r_cc359_T500_b0.002.pth"
  "checkpoints/diff_r_rm_cc359_T500_b0.002.pth"
)

for CKPT in "${MODELS[@]}"; do
  NAME=$(basename "$CKPT" .pth)
  echo "=== Evaluating $NAME ==="
  python -m diffshape.eval_diffusion \
    --checkpoint "$CKPT" \
    --output-dir "results_diffusion/$NAME" \
    --datasets cc359 gbm125 nfbs \
    --active-folds 0 1 2 3 4 \
    $COMMON
done

echo "=== CC359-model evaluations complete ==="
