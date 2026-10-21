import pickle
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
)

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import os
import pandas as pd


class BaseVolumeDataset(Dataset):
    def __init__(self, args, image_paths, transforms=None):
        super().__init__()
        self.img_dict = pd.read_csv(image_paths)
        self.root = args.root
        self._set_dataset_stat()
        self.transforms = transforms  # self.get_transforms()

    def _set_dataset_stat(self):
        self.spacing = (0.5, 0.5, 1.0)
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.target_class = 1

    def __len__(self):
        return len(self.img_dict)

    def read(self, path):
        vol = nib.load(os.path.join(self.root, path))
        vol = vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        vol = torch.from_numpy(vol)
        return vol

    def __getitem__(self, idx):
        return NotImplemented


class UCLSet(BaseVolumeDataset):
    def __init__(self, args, image_paths, transforms=None):
        super().__init__(args=args, image_paths=image_paths, transforms=transforms)

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"]).unsqueeze(0)
        mask = self.read(path["prostate"]).unsqueeze(0)

        if self.transforms is not None:
            trans_dict = self.transforms({"image": t2w})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]

            t2w = trans_dict["image"]
        volume = (mask > 0).sum().float()
        volume = (volume - 70000) / 500000
        return t2w, volume.unsqueeze(-1), torch.tensor(idx, dtype=torch.long)


class BpUCLSet(BaseVolumeDataset):
    def __init__(self, args, image_paths, transforms=None):
        super().__init__(args=args, image_paths=image_paths, transforms=transforms)

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path["t2w"])
        # dwi = self.read(path["dwi"])
        # adc = self.read(path["adc"])
        zero = torch.zeros_like(t2w)
        img = torch.stack([t2w, zero, zero], 0)
        mask = self.read(path["prostate"]).unsqueeze(0)

        if self.transforms is not None:
            trans_dict = self.transforms({"image": img})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            t2w = trans_dict["image"]
        volume = (mask > 0).sum().float()
        # volume = (volume - 220000) / 840000
        volume = (volume - 70000) / 500000
        return t2w, volume.unsqueeze(-1), torch.tensor(idx, dtype=torch.long)


def get_transforms(args):
    train_transforms = [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandRotated(
            keys="image",
            prob=0.3,
            range_x=30 / 180 * np.pi,
            keep_size=False,
            mode="bilinear",
        ),
        SpatialPadd(
            keys="image",
            spatial_size=[round(i * 1.2) for i in args.crop_spatial_size],
        ),
        RandSpatialCropd(
            keys="image",
            roi_size=args.crop_spatial_size,
            random_size=False,
        ),
        RandFlipd(keys="image", prob=0.5, spatial_axis=2),
    ]
    train_transforms = Compose(train_transforms)

    val_transforms = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CenterSpatialCropd(keys="image", roi_size=args.crop_spatial_size),
            SpatialPadd(keys="image", spatial_size=[i for i in args.crop_spatial_size]),
            # BinarizeLabeld(keys=["label"])
        ]
    )
    test_transforms = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CenterSpatialCropd(keys="image", roi_size=args.crop_spatial_size),
            SpatialPadd(keys="image", spatial_size=[i for i in args.crop_spatial_size]),
            # BinarizeLabeld(keys=["label"])
        ]
    )
    return train_transforms, val_transforms, test_transforms


def build_UCL_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.data20:
        train_set = UCLSet(args, "spilt/UCL/train_20.csv", train_transforms)
    else:
        train_set = UCLSet(args, "spilt/UCL/train.csv", train_transforms)
    val_set = UCLSet(args, "spilt/UCL/val.csv", val_transforms)
    test_set = UCLSet(args, "spilt/UCL/test.csv", test_transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=14,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 1
    return train_loader, val_loader, test_loader


def build_BpUCL_loader(args):
    train_transforms, val_transforms, test_transforms = get_transforms(args)
    if args.data20:
        train_set = BpUCLSet(args, "spilt/UCL/train_20.csv", train_transforms)
    else:
        train_set = BpUCLSet(args, "spilt/UCL/train.csv", train_transforms)
    val_set = BpUCLSet(args, "spilt/UCL/val.csv", val_transforms)
    test_set = BpUCLSet(args, "spilt/UCL/test.csv", test_transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=14,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=14,
        drop_last=False,
    )
    args.in_channels = 3
    return train_loader, val_loader, test_loader
