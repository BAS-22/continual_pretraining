# Continual Pretraining Prostate MRI ViT Encoder for Task Alignment

[![Paper](https://img.shields.io/badge/Paper-ISBI%202026-blue)](https://www.semanticscholar.org/paper/Continual-Pretraining-Prostate-MRI-ViT-Encoder-for-Sidiqi-Wang/a911e99b0e0ce89b52e253d31ba80f4b16507e71)
[![PDF](https://img.shields.io/badge/PDF-open%20access-green)](https://discovery.ucl.ac.uk/id/eprint/10220666/1/continual_pretraining%20%282%29.pdf)

Official PyTorch implementation of **continual pretraining for a prostate MRI ViT encoder**, accepted at the **IEEE 23rd International Symposium on Biomedical Imaging (ISBI 2026)**, London.

> **Continual Pretraining Prostate MRI ViT Encoder for Task Alignment**
> Bilal A. Sidiqi, Yipei Wang, Shaheer U. Saeed, Shiqi Huang, Daniel C. Alexander, Yipeng Hu
> *IEEE International Symposium on Biomedical Imaging (ISBI), 2026*
> [[Paper page]](https://www.semanticscholar.org/paper/Continual-Pretraining-Prostate-MRI-ViT-Encoder-for-Sidiqi-Wang/a911e99b0e0ce89b52e253d31ba80f4b16507e71) [[PDF]](https://discovery.ucl.ac.uk/id/eprint/10220666/1/continual_pretraining%20%282%29.pdf) [[UCL Discovery]](https://discovery.ucl.ac.uk/id/eprint/10220666)

## Overview

Foundation models for MRI are usually pretrained with a self-supervised objective, such as masked autoencoding, that is not aligned with the downstream clinical tasks. This work adds a **continual pretraining** stage between self-supervised pretraining and downstream fine-tuning to move the encoder towards a supervised prostate task.

1. **Pretrained encoder.** We start from a 3D ViT-B encoder pretrained with MAE on 5,524 prostate MRI scans ([ProFound](https://arxiv.org/abs/2603.03961)).
2. **Continual pretraining.** Low-Rank Adaptation (LoRA) adapters are added to the attention projections of every transformer block, and the original encoder weights are frozen. Hidden states from layers 3, 5, 7 and 11 are reshaped into 3D feature maps and passed to a UperNet decoder. The model is trained for segmentation on PROMIS with a Dice-based loss (MONAI `DiceCELoss`).
3. **Downstream fine-tuning.** The continually pretrained encoder is fine-tuned on six downstream tasks: PI-RADS classification, zone classification, prostate volume regression, lesion segmentation, zone segmentation and 4× T2W super-resolution.
4. **Geometric analysis.** Layer-wise linear Centered Kernel Alignment (CKA) measures how the encoder representations change between the pretrained, continually pretrained and fine-tuned encoders.

Each task compares three encoder initialisations:

| Name in the paper | `--case` | Encoder initialisation |
|---|---|---|
| ViT Random Weights | `scratch` | random |
| ProFound ViT OMW | `prenc` | ProFound MAE weights |
| ProFound ViT Ours | `posprenc` | continually pretrained encoder |

<!-- Optional: add a method figure, e.g. ![Method overview](assets/overview.png) -->

## Results

Continual pretraining improves 3 of the 6 downstream tasks and is comparable on the others.

**Classification** (AUC, %; mean ± std over bootstrap resamples of the test set)

| Model | PI-RADS AUC | PI-RADS AUC ≥3 | PI-RADS AUC ≥4 | Zone AUC |
|---|---|---|---|---|
| ViT Random Weights | 58.81 ± 2.51 | 69.98 | 62.82 | 64.41 ± 1.97 |
| ProFound ViT OMW | **61.36 ± 2.56** | 74.76 | **64.03** | **72.64 ± 1.75** |
| ProFound ViT Ours | 60.94 ± 2.63 | **79.92** | 64.00 | 70.36 ± 1.86 |

**Regression, segmentation and super-resolution**

| Model | Volume MSE ↓ | Lesion Dice ↑ | Zone Dice ↑ | SR SSIM ↑ |
|---|---|---|---|---|
| ViT Random Weights | 0.038 ± 0.028 | 0.216 ± 0.186 | 0.830 ± 0.093 | 0.900 ± 0.009 |
| ProFound ViT OMW | **0.013 ± 0.009** | 0.248 ± 0.203 | 0.835 ± 0.095 | **0.931 ± 0.005** |
| ProFound ViT Ours | 0.015 ± 0.012 | **0.253 ± 0.206** | **0.836 ± 0.095** | 0.930 ± 0.005 |

**Layer-wise CKA** between encoders (layers 3, 5, 7, 11)

| Encoder pair | L3 | L5 | L7 | L11 | Avg |
|---|---|---|---|---|---|
| Pretrained – Continual | 0.997 | 0.996 | 0.997 | 0.995 | 0.996 |
| *Lesion segmentation:* Pretrained – Finetuned | 0.918 | 0.915 | 0.893 | 0.340 | 0.767 |
| *Lesion segmentation:* Continual – Finetuned | 0.915 | 0.910 | 0.890 | 0.344 | 0.765 |
| *PI-RADS classification:* Pretrained – Finetuned | 1.000 | 1.000 | 1.000 | 0.993 | 0.998 |
| *PI-RADS classification:* Continual – Finetuned | 0.997 | 0.997 | 0.997 | 0.989 | 0.995 |

The paper also reports per-zone Dice scores for zone segmentation.

## Contents

- `main_segmentation.py`: continual pretraining (`--case continual`), lesion and zone segmentation, and the CKA analysis (`--state sim_test`).
- `main_classification.py`, `main_regression.py`, `main_SR.py`: downstream classification, volume regression and super-resolution.
- `models/LoRa.py`: LoRA adapters for the ViT attention projections (`apply_lora_to_vit`).
- `models/vit.py`, `models/patch_embed.py`: the 3D ViT-B encoder.
- `models/upernet_module.py`: UperNet decoder and ViT adapter.
- `models/build_segment.py`, `models/build_classifition.py`, `models/build_regression.py`, `models/build_SR.py`: model and optimizer builders, including the encoder initialisation for each `--case`.
- `engine/`: training, validation and test loops for each task, plus `sim_test` for CKA.
- `util/feature_sim.py`: layer-wise linear CKA between the pretrained, continual and fine-tuned encoders.
- `dataset/`: NIfTI datasets, MONAI transforms and dataloaders for each task.
- `scripts/`: launchers that don't need a scheduler (`continual_pretrain.sh`, `train_downstream.sh`, `test_downstream.sh`, `cka.sh`).
- `spilt/`: split files. Only the Anatomy split is included; see `spilt/README.md`.
- `examples/`: one template CSV per dataset format.
- `util/`: distributed and misc helpers, LR schedule, layer-wise LR decay and metrics.

## Installation

```bash
git clone https://github.com/BAS-22/continual_pretraining.git
cd continual_pretraining
conda create -n contpre python=3.10 -y
conda activate contpre
pip install -r requirements.txt
```

Tested with CUDA 11.8. Change the PyTorch build in `requirements.txt` to match your CUDA version.

## Data Format

Each split is a CSV file. The paths are relative to `--root`:

| Dataset | Split (train, val, test) | MRI input | Target | Template |
|---|---|---|---|---|
| PROMIS | 396, 57, 113 | T2W, DWI, ADC | lesion / gland mask, 20-zone labels | `examples/promis_*.csv` |
| Anatomy | 412, 60, 117 | T2W | 8 prostate and pelvic structures | `examples/anatomy_zones.csv` |
| RISK | 718, 103, 207 | T2W, DWI (high b), ADC | PI-RADS score (2–5) | `examples/risk_pirads.csv` |
| UCL | 1386, 178, 387 | T2W | prostate volume (from the gland mask) | `examples/ucl_volume.csv` |

- Volumes are `64 × 256 × 256` (D × H × W). Normalisation, centre cropping and augmentation (rotation, flips, zoom, intensity) happen inside the loaders.
- The loaders read fixed paths under `spilt/` (for example `spilt/risk/{train,val,test}.csv`). `spilt/README.md` lists the folder, columns and template for each task.
- Use de-identified filenames only.

The datasets used in the paper cannot be redistributed.

## Pretrained Encoder

Continual pretraining starts from the ViT-B encoder pretrained with MAE on prostate MRI ([ProFound](https://arxiv.org/abs/2603.03961)). Download the checkpoint and place it in `pretrained/`:

```text
pretrained/
  profound_mae_checkpoint-799.pth
```

<!-- TODO: add the download link for the ProFound weights -->

## Training

All scripts read `DATA_ROOT`, and optionally `MAE_CKPT`, `OUT` (default `outputs`) and `SEED` (default `2021`).

**Stage 1: continual pretraining** (LoRA, frozen encoder, UperNet decoder, 100 epochs):

```bash
DATA_ROOT=/path/to/data bash scripts/continual_pretrain.sh
```

This runs:

```bash
python main_segmentation.py \
  --state train --model profound_vit --train scratch --case continual \
  --dataset Promis --promis_mask gland --root /path/to/data \
  --pretrain pretrained/profound_mae_checkpoint-799.pth \
  --output_dir outputs/continual --log_dir outputs/continual \
  --batch_size 4 --epochs 100 --warmup_epochs 5 --lr 1e-3 --seed 2021
```

The checkpoint is saved to `outputs/continual/Promis/profound_vit/scratch/2021/best.pth.tar`.

**Stage 2: downstream fine-tuning.** Use `TASK` ∈ {`lesion_seg`, `zone_seg`, `pirads_cls`, `zone_cls`, `volume_reg`, `sr`} and `INIT` ∈ {`scratch`, `prenc`, `ours`}:

```bash
DATA_ROOT=/path/to/data bash scripts/train_downstream.sh lesion_seg ours
```

For example, lesion segmentation from the continually pretrained encoder:

```bash
python main_segmentation.py \
  --state train --model profound_vit --train scratch --case posprenc \
  --dataset Promis --promis_mask lesion --root /path/to/data \
  --pretrain outputs/continual/Promis/profound_vit/scratch/2021/best.pth.tar \
  --output_dir outputs/lesion_seg/ours --log_dir outputs/lesion_seg/ours \
  --batch_size 4 --epochs 100 --warmup_epochs 5 --lr 1e-3 --seed 2021
```

| Task | Entry point | `--dataset` | Batch size | Loss |
|---|---|---|---|---|
| `lesion_seg` | `main_segmentation.py` | `Promis` | 4 | Dice + cross-entropy |
| `zone_seg` | `main_segmentation.py` | `anatomy` | 4 | Dice + cross-entropy |
| `pirads_cls` | `main_classification.py` | `risk` | 16 | cross-entropy |
| `zone_cls` | `main_classification.py` | `promis` | 16 | binary cross-entropy (20 zones) |
| `volume_reg` | `main_regression.py` | `UCL` | 16 | MSE |
| `sr` | `main_SR.py` (`--model mae_sr_vit_base`) | `Promis` | 4 | L1 + 0.1 × MSE |

All tasks use 100 epochs, AdamW with layer-wise LR decay, and `--lr 1e-3`.

## Evaluation

```bash
DATA_ROOT=/path/to/data bash scripts/test_downstream.sh lesion_seg ours
```

This loads `best.pth.tar` from the training output folder and reports the test metrics (Dice, HD95 and surface distance; AUC; MSE; PSNR and SSIM). Models trained from the continually pretrained encoder are tested with `--case test`, which rebuilds the LoRA-wrapped encoder; the other models are tested with `--case scratch`.

**CKA analysis (Table 6).** First fine-tune the `prenc` model for the task, then run:

```bash
DATA_ROOT=/path/to/data bash scripts/cka.sh lesion_seg     # or pirads_cls
```

This compares the MAE encoder (`--mae_checkpoint`), the continually pretrained encoder (`--continual_checkpoint`) and the fine-tuned encoder at layers 3, 5, 7 and 11, and saves the similarity matrices under `<output_dir>/sim_figures/`.

## Notes

- An NVIDIA A40 GPU was used for all experiments.
- The checkpoint used for the downstream results in the paper was continually pretrained on the PROMIS prostate gland masks (`--promis_mask gland`). `--promis_mask lesion` trains on the lesion masks instead.
- The Anatomy "Ours" result was trained with the encoder weights unfrozen (`--unfreeze_encoder`); `scripts/train_downstream.sh` adds this flag automatically.
- `--case scratch` still needs `--pretrain` to point to an existing file, but it does not load it.
- `main_segmentation.py` and `engine/segment.py` are shared with [Contrastive_uncertainty_learning](https://github.com/BAS-22/Contrastive_uncertainty_learning). The uncertainty options in them are off by default and are not used in this paper.
- Do not commit private data, checkpoints, pretrained weights, or output folders.

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Sidiqi_ContinualPretraining_ISBI2026,
  author    = {Sidiqi, Bilal A. and Wang, Yipei and Saeed, Shaheer U. and Huang, Shiqi and Alexander, Daniel C. and Hu, Yipeng},
  title     = {Continual Pretraining Prostate MRI ViT Encoder for Task Alignment},
  booktitle = {2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026},
  address   = {London, UK},
  publisher = {IEEE}
}
```

## Acknowledgements

This codebase builds on [MAE](https://github.com/facebookresearch/mae), [DeiT](https://github.com/facebookresearch/deit), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [timm](https://github.com/huggingface/pytorch-image-models) and [MONAI](https://github.com/Project-MONAI/MONAI), and on the ProFound pretrained encoder. We thank the authors for releasing their code.

## License

<!-- TODO: choose a license. Parts of this code are derived from MAE (CC BY-NC 4.0). -->
Parts of this code are derived from [MAE](https://github.com/facebookresearch/mae), which is released under CC BY-NC 4.0.
