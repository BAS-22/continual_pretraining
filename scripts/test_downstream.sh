#!/usr/bin/env bash
# Evaluate a model trained with scripts/train_downstream.sh on the test split.
# Loads ${OUT}/${TASK}/${INIT}/<dataset>/<model>/scratch/${SEED}/best.pth.tar
#
# Usage:
#   DATA_ROOT=/path/to/data bash scripts/test_downstream.sh TASK INIT
set -euo pipefail
source "$(dirname "$0")/common.sh"
TASK=${1:?task}; INIT=${2:?init}
task_config "$TASK"; init_config "$INIT"

# "ours" models are rebuilt with the LoRA-wrapped encoder (--case test); the others with a plain ViT.
[[ "$INIT" == "ours" ]] && CASE=test || CASE=scratch

python -u "$MAIN" \
  --state test --model "$MODEL" --train scratch --case "$CASE" \
  --dataset "$DATASET" --root "$DATA_ROOT" --pretrain "$PRETRAIN" --mae_checkpoint "$MAE_CKPT" \
  --output_dir "${OUT}/${TASK}/${INIT}" --log_dir "${OUT}/${TASK}/${INIT}" \
  --batch_size "$BS" --lr 1e-3 --seed "$SEED" \
  $EXTRA
