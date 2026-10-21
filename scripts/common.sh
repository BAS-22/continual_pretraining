#!/usr/bin/env bash
# Shared settings for the launch scripts. Override any of these with environment variables.
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder that the CSV paths in spilt/ are relative to}
MAE_CKPT=${MAE_CKPT:-pretrained/profound_mae_checkpoint-799.pth}          # ProFound MAE encoder
OUT=${OUT:-outputs}
SEED=${SEED:-2021}
# checkpoint written by scripts/continual_pretrain.sh
CONT_CKPT=${CONT_CKPT:-${OUT}/continual/Promis/profound_vit/scratch/${SEED}/best.pth.tar}

# task -> entry point, dataset, model, batch size, warm-up epochs, extra flags
task_config() {
  case "$1" in
    lesion_seg) MAIN=main_segmentation.py;   DATASET=Promis;  MODEL=profound_vit;    BS=4;  WARMUP=5;  EXTRA="--promis_mask lesion" ;;
    zone_seg)   MAIN=main_segmentation.py;   DATASET=anatomy; MODEL=profound_vit;    BS=4;  WARMUP=5;  EXTRA="" ;;
    pirads_cls) MAIN=main_classification.py; DATASET=risk;    MODEL=profound_vit;    BS=16; WARMUP=5;  EXTRA="" ;;
    zone_cls)   MAIN=main_classification.py; DATASET=promis;  MODEL=profound_vit;    BS=16; WARMUP=5;  EXTRA="" ;;
    volume_reg) MAIN=main_regression.py;     DATASET=UCL;     MODEL=profound_vit;    BS=16; WARMUP=5;  EXTRA="" ;;
    sr)         MAIN=main_SR.py;             DATASET=Promis;  MODEL=mae_sr_vit_base; BS=4;  WARMUP=10; EXTRA="" ;;
    *) echo "unknown task: $1 (lesion_seg|zone_seg|pirads_cls|zone_cls|volume_reg|sr)"; exit 1 ;;
  esac
}

# encoder initialisation -> --case and --pretrain
#   scratch : random weights            ("ViT Random Weights")
#   prenc   : ProFound MAE weights       ("ProFound ViT OMW")
#   ours    : continually pretrained     ("ProFound ViT Ours")
init_config() {
  case "$1" in
    scratch) CASE=scratch;  PRETRAIN=$MAE_CKPT ;;   # --pretrain must exist but is not loaded
    prenc)   CASE=prenc;    PRETRAIN=$MAE_CKPT ;;
    ours)    CASE=posprenc; PRETRAIN=$CONT_CKPT ;;
    *) echo "unknown init: $1 (scratch|prenc|ours)"; exit 1 ;;
  esac
}
