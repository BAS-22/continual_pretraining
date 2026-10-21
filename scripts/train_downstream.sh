#!/usr/bin/env bash
# Stage 2: fine-tune on a downstream task from one of the three encoder initialisations.
#
# Usage:
#   DATA_ROOT=/path/to/data bash scripts/train_downstream.sh TASK INIT
#     TASK: lesion_seg | zone_seg | pirads_cls | zone_cls | volume_reg | sr
#     INIT: scratch | prenc | ours
set -euo pipefail
source "$(dirname "$0")/common.sh"
TASK=${1:?task}; INIT=${2:?init}
task_config "$TASK"; init_config "$INIT"

# The Anatomy "Ours" result in the paper was trained with the encoder weights unfrozen.
[[ "$TASK" == "zone_seg" && "$INIT" == "ours" ]] && EXTRA="$EXTRA --unfreeze_encoder"

python -u "$MAIN" \
  --state train --model "$MODEL" --train scratch --case "$CASE" \
  --dataset "$DATASET" --root "$DATA_ROOT" --pretrain "$PRETRAIN" --mae_checkpoint "$MAE_CKPT" \
  --output_dir "${OUT}/${TASK}/${INIT}" --log_dir "${OUT}/${TASK}/${INIT}" \
  --batch_size "$BS" --epochs 100 --warmup_epochs "$WARMUP" --lr 1e-3 --seed "$SEED" \
  $EXTRA
