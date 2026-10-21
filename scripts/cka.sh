#!/usr/bin/env bash
# Layer-wise linear CKA (layers 3, 5, 7, 11) between the pretrained (MAE), continually
# pretrained and finetuned encoders, as in Table 6 of the paper.
# "Finetuned" is the model trained with:  bash scripts/train_downstream.sh TASK prenc
#
# Usage:
#   DATA_ROOT=/path/to/data bash scripts/cka.sh {lesion_seg|pirads_cls}
set -euo pipefail
source "$(dirname "$0")/common.sh"
TASK=${1:?task}
task_config "$TASK"

python -u "$MAIN" \
  --state sim_test --model "$MODEL" --train scratch --case test \
  --dataset "$DATASET" --root "$DATA_ROOT" \
  --pretrain "$CONT_CKPT" --mae_checkpoint "$MAE_CKPT" --continual_checkpoint "$CONT_CKPT" \
  --output_dir "${OUT}/${TASK}/prenc" --log_dir "${OUT}/${TASK}/prenc/cka" \
  --batch_size "$BS" --lr 1e-3 --seed "$SEED" --sim_batches 5 \
  $EXTRA
