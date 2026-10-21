# Data splits

Each loader reads `train.csv`, `val.csv` and `test.csv` from a folder below. Paths inside the CSVs are
relative to `--root`. Only the Anatomy split is included here, using the same de-identified file names
as in [Contrastive_uncertainty_learning](https://github.com/BAS-22/Contrastive_uncertainty_learning).
For the other datasets, create the folders yourself; `examples/` has one template per format.

| Folder | Used by | Columns | Template |
|---|---|---|---|
| `spilt/promis567/` (`*_lesion.csv`) | continual pretraining / lesion segmentation (`--promis_mask lesion`) | `t2w, dwi, adc, lesion` | `examples/promis_lesion.csv` |
| `spilt/promis567/` (`train/val/test.csv`) | continual pretraining (`--promis_mask gland`), super-resolution | `t2w, dwi, adc, gland` | `examples/promis_gland.csv` |
| `spilt/promis567_hist/` | zone classification (`main_classification.py --dataset promis`) | `t2w, dwi, adc, zone_level, patient_level` | `examples/promis_zone_cls.csv` |
| `spilt/risk/` | PI-RADS classification (`--dataset risk`) | `t2w, adc, highb, pirads` | `examples/risk_pirads.csv` |
| `spilt/UCL/` | prostate volume regression (`--dataset UCL`) | `t2w, prostate` | `examples/ucl_volume.csv` |
| `spilt/anatomy/` | zone segmentation (`--dataset anatomy`) | `t2w, mask` | `examples/anatomy_zones.csv` |

- `zone_level`: 20 space-separated binary labels, one per prostate zone.
- `pirads`: PI-RADS score 2–5 (mapped to classes 0–3).
- `mask` (Anatomy): integer label map, `0` background and `1–8` the anatomical structures.
- Use de-identified file names only.
