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
    RandBiasFieldd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
    LoadImaged,
    # AddChanneld,
    Orientationd,
    Spacingd,
    ToTensord,
)

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import os
import pandas as pd
from preprocess.promis_LR import MRIDegradationPipeline


class BaseVolumeDatasetSR(Dataset):
    """Base dataset class for super-resolution tasks"""
    
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__()
        self.img_dict = pd.read_csv(image_paths)
        self.args = args
        self.phase = phase
        
        if phase == 'train':
            if args.data_num > 0:
                # crop the dataset
                self.img_dict = self.img_dict.iloc[: args.data_num]
        print(f"Loading {phase} dataset with {len(self.img_dict)} samples")
        
        self.root = args.root
        self._set_dataset_stat()
        self.transforms = transforms
        
        # Initialize degradation pipeline for creating low-res images
        self.degrader = MRIDegradationPipeline(downscale_factor=getattr(args, 'upscale_factor', 2))

    def _set_dataset_stat(self):
        self.spacing = (0.5, 0.5, 1.0)
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.target_class = 1

    def __len__(self):
        return len(self.img_dict)

    def read(self, path):
        vol = nib.load(os.path.join(self.root, 'ProFound-downstream-yp', path))
        vol = vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        vol = torch.from_numpy(vol)
        return vol

    def __getitem__(self, idx):
        return NotImplemented


class UCLSetSR(BaseVolumeDatasetSR):
    """promis dataset adapted for super-resolution task using only T2 images"""
    
    def __init__(self, args, image_paths, phase, transforms=None):
        super().__init__(args=args, image_paths=image_paths, phase=phase, transforms=transforms)
        print(f"Available columns in {phase} dataset: {list(self.img_dict.columns)}")

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        
        # Load only T2 image (we only need T2 for super-resolution)
        if 't2w' in path:
            t2w = self.read(path["t2w"])
        elif 't2' in path:
            t2w = self.read(path["t2"])
        else:
            raise ValueError(f"No T2 image found in dataset. Available columns: {list(path.index)}")
        
        # Create single channel image for T2 only
        img = t2w.unsqueeze(0)  # Add channel dimension [1, D, H, W]
        
        # Create dummy label for compatibility with transforms (will be ignored)
        dummy_label = torch.zeros_like(t2w).unsqueeze(0)
        
        # Apply transformations to T2 image
        if self.transforms is not None:
            trans_dict = self.transforms({"image": img, "label": dummy_label})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            high_res_t2 = trans_dict["image"]  # [1, D, H, W]
        else:
            high_res_t2 = img
        
        # Create low-resolution version AFTER transformations
        # Add batch dimension for degradation pipeline
        high_res_batch = high_res_t2.unsqueeze(0)  # [1, 1, D, H, W]
        
        with torch.no_grad():
            low_res_batch = self.degrader.create_realistic_low_res(high_res_batch)
        
        # Remove batch dimension
        low_res_t2 = low_res_batch.squeeze(0)  # [1, D//2, H, W]
        
        return {
            'low_res': low_res_t2,       # [1, D//2, H, W] - low resolution input
            'high_res': high_res_t2,     # [1, D, H, W] - high resolution target
            'subject_id': getattr(path, 'subject_id', f'subject_{idx}'),
            'idx': torch.tensor(idx, dtype=torch.long)
        }


def get_transforms_sr(args):
    """Get transforms for super-resolution task (no labels needed)"""
    
    # Training transforms with augmentation
    train_transforms = [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandRotated(
            keys=["image", "label"],  # Keep label for compatibility
            prob=0.3,
            range_x=30 / 180 * np.pi,
            keep_size=False,
            mode=["bilinear", "nearest"],
        ),
        RandZoomd(
            keys=["image", "label"],
            prob=0.3,
            min_zoom=[1, 0.9, 0.9],
            max_zoom=[1, 1.1, 1.1],
            mode=["trilinear", "nearest"],
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=[round(i * 1.2) for i in args.crop_spatial_size],
        ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=args.crop_spatial_size,
            random_size=False,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.8),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.8),
        RandBiasFieldd(keys="image", prob=0.2),
        RandGaussianSmoothd(keys="image", prob=0.3),  # Reduced prob for SR
    ]

    train_transforms = Compose(train_transforms)
    
    # Validation transforms (no augmentation)
    val_transforms = Compose([
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CenterSpatialCropd(
            keys=["image", "label"], roi_size=args.crop_spatial_size
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=[i for i in args.crop_spatial_size],
        ),
    ])
    
    # Test transforms (same as validation)
    test_transforms = Compose([
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CenterSpatialCropd(
            keys=["image", "label"], roi_size=args.crop_spatial_size
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=[i for i in args.crop_spatial_size],
        ),
    ])
    
    return train_transforms, val_transforms, test_transforms


def build_promis_loader_sr(args):
    """Build data loaders for super-resolution task"""
    
    train_transforms, val_transforms, test_transforms = get_transforms_sr(args)
    
    # Create datasets
    if args.data20:
        train_set = UCLSetSR(args, "spilt/promis567/train_16.csv", 'train', train_transforms)
    else:
        train_set = UCLSetSR(args, "spilt/promis567/train.csv", 'train', train_transforms)
    
    val_set = UCLSetSR(args, "spilt/promis567/val.csv", 'val', val_transforms)
    test_set = UCLSetSR(args, "spilt/promis567/test.csv", 'test', test_transforms)
    
    # Custom collate function for super-resolution
    def collate_fn_sr(batch):
        """Custom collate function for super-resolution batch processing"""
        low_res_batch = torch.stack([item['low_res'] for item in batch])
        high_res_batch = torch.stack([item['high_res'] for item in batch])
        subject_ids = [item['subject_id'] for item in batch]
        indices = torch.stack([item['idx'] for item in batch])
        
        return {
            'low_res': low_res_batch,
            'high_res': high_res_batch,
            'subject_id': subject_ids,
            'idx': indices
        }
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers= 8,
        drop_last=True,
        collate_fn=collate_fn_sr,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers= 8,
        drop_last=False,
        collate_fn=collate_fn_sr,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=4,  # Keep batch size 1 for testing
        shuffle=False,
        pin_memory=True,
        num_workers= 8,
        drop_last=False,
        collate_fn=collate_fn_sr,
    )
    
    # Set model parameters for super-resolution
    args.in_channels = 1   # T2 only
    args.out_channels = 1  # T2 output
    args.num_classes = 1   # Not used in SR but kept for compatibility
    
    return train_loader, val_loader, test_loader


# # Alternative version for MONAI transforms compatibility
# def get_transforms_sr_monai(args):
#     """Alternative transforms using MONAI LoadImaged approach"""
    
#     # Training transforms
#     train_transforms = Compose([
#         LoadImaged(keys=["image"]),
#         AddChanneld(keys=["image"]),
#         Orientationd(keys=["image"], axcodes="RAS"),
#         Spacingd(
#             keys=["image"], 
#             pixdim=args.spacing if hasattr(args, 'spacing') else (0.5, 0.5, 1.0), 
#             mode=("bilinear",)
#         ),
#         NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#         CenterSpatialCropd(
#             keys=["image"], 
#             roi_size=args.crop_spatial_size
#         ),
#         SpatialPadd(
#             keys=["image"],
#             spatial_size=[i for i in args.crop_spatial_size],
#         ),
#         # Augmentations for training
#         RandRotated(
#             keys=["image"],
#             prob=0.3,
#             range_x=0.2,
#             range_y=0.2,
#             range_z=0.2,
#             mode="bilinear",
#             padding_mode="border",
#         ),
#         RandFlipd(
#             keys=["image"],
#             prob=0.5,
#             spatial_axis=[0, 1, 2],
#         ),
#         RandScaleIntensityd(
#             keys=["image"],
#             factors=0.1,
#             prob=0.3,
#         ),
#         RandShiftIntensityd(
#             keys=["image"],
#             offsets=0.1,
#             prob=0.3,
#         ),
#         ToTensord(keys=["image"]),
#     ])
    
#     # Validation/test transforms (no augmentation)
#     val_test_transforms = Compose([
#         LoadImaged(keys=["image"]),
#         AddChanneld(keys=["image"]),
#         Orientationd(keys=["image"], axcodes="RAS"),
#         Spacingd(
#             keys=["image"], 
#             pixdim=args.spacing if hasattr(args, 'spacing') else (0.5, 0.5, 1.0), 
#             mode=("bilinear",)
#         ),
#         NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#         CenterSpatialCropd(
#             keys=["image"], 
#             roi_size=args.crop_spatial_size
#         ),
#         SpatialPadd(
#             keys=["image"],
#             spatial_size=[i for i in args.crop_spatial_size],
#         ),
#         ToTensord(keys=["image"]),
#     ])
    
#     return train_transforms, val_test_transforms, val_test_transforms


# # Keep original functions for compatibility
# def get_transforms(args):
#     """Original transforms for backward compatibility"""
#     return get_transforms_sr(args)

# def build_promis_loader(args):
#     """Original loader for backward compatibility"""
#     return build_promis_loader_sr(args)



