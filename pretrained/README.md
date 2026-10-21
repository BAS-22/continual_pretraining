# Pretrained weights

Put the ProFound MAE checkpoint here (it is not tracked by git):

```text
pretrained/
  profound_mae_checkpoint-799.pth
```

This is the 3D ViT-B encoder pretrained with MAE on prostate MRI ([ProFound](https://arxiv.org/abs/2603.03961)).
The continual-pretraining checkpoint is written by `scripts/continual_pretrain.sh` to
`outputs/continual/Promis/profound_vit/scratch/<seed>/best.pth.tar`.
