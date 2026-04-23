#!/bin/bash
# End-to-end pipeline validation: predict -> samples_to_sdt -> visualize
set -e

CKPT=${CKPT:-diffshape/checkpoints/checkpoints/checkpoint_final.pth}
PROCESSED=${PROCESSED:-processed_dataset}
GPU=${1:-0}
K=${2:-16}
DATASET=${3:-cc359}
shift 3 || true
INDICES="${*:-0 1}"
PY=${PY:-python}
OUT=pipeline_validation/${DATASET}_K${K}

$PY -m diffshape.predict \
  --checkpoint "$CKPT" \
  --dataset "$DATASET" \
  --processed-dir "$PROCESSED" \
  --output-dir "$OUT" \
  --k-samples "$K" \
  --indices $INDICES \
  --gpu "$GPU"

$PY -m diffshape.samples_to_sdt \
  --samples "$OUT/samples_${DATASET}_K${K}.npy" \
  --meta "$OUT/meta_${DATASET}_K${K}.json" \
  --output-dir "$OUT/priors" \
  --format both

$PY -m diffshape.visualize_samples \
  --samples "$OUT/samples_${DATASET}_K${K}.npy" \
  --meta "$OUT/meta_${DATASET}_K${K}.json" \
  --output-dir "$OUT/viz"

echo "=== Pipeline validation complete: $OUT ==="
