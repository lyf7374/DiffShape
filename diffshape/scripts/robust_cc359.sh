#!/bin/bash
set -e

GPU=${1:-1}
PROCESSED=processed_dataset
COMMON="--timesteps 500 --beta-start 0.0001 --beta-end 0.002 --ddim-steps 50 --k-samples 16 --gpu $GPU --processed-dir $PROCESSED --batch-size 2"
RATIOS=(0.2 0.4 0.6 0.8)

MODELS=(
  "checkpoints/diff_r_cc359_T500_b0.002.pth"
  "checkpoints/diff_r_rm_cc359_T500_b0.002.pth"
)

for CKPT in "${MODELS[@]}"; do
  NAME=$(basename "$CKPT" .pth)
  for R in "${RATIOS[@]}"; do
    TAG=$(printf "m%02d" "$(awk "BEGIN{print int($R*100)}")")
    echo "=== Robustness: $NAME @ mask_ratio=$R ==="
    python -m diffshape.eval_diffusion \
      --checkpoint "$CKPT" \
      --output-dir "results_robustness/${NAME}_${TAG}" \
      --datasets cc359 gbm125 nfbs \
      --active-folds 0 1 2 3 4 \
      --mask-ratio "$R" \
      $COMMON
  done
done

echo "=== CC359-model robustness tests complete ==="
