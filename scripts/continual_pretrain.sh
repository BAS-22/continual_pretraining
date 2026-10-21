#!/usr/bin/env bash
# Stage 1: continual pretraining of the ProFound ViT encoder with LoRA on PROMIS segmentation.
#
# Usage:
#   DATA_ROOT=/path/to/data MAE_CKPT=pretrained/profound_mae_checkpoint-799.pth \
#     bash scripts/continual_pretrain.sh [lesion|gland]
#
# Writes ${OUT}/continual/Promis/profound_vit/scratch/${SEED}/best.pth.tar
set -euo pipefail
source "$(dirname "$0")/common.sh"
MASK=${1:-gland}   # the checkpoint used for the paper's downstream results was trained on the gland masks

python -u main_segmentation.py \
  --state train --model profound_vit --train scratch --case continual \
  --dataset Promis --promis_mask "$MASK" --root "$DATA_ROOT" --pretrain "$MAE_CKPT" \
  --output_dir "${OUT}/continual" --log_dir "${OUT}/continual" \
  --batch_size 4 --epochs 100 --warmup_epochs 5 --lr 1e-3 --seed "$SEED"
