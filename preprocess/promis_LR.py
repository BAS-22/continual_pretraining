import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MRIDegradationPipeline:
    """Realistic degradation pipeline for T2 MRI super-resolution"""
    
    def __init__(self, downscale_factor=2):
        self.downscale_factor = downscale_factor
    
    def add_noise(self, img, noise_level=0.01):
        """Add Gaussian noise to simulate MRI acquisition noise"""
        noise = torch.randn_like(img) * noise_level
        return img + noise
    
    def add_motion_blur(self, img, kernel_size=3):
        """Simulate motion artifacts in z-direction"""
        # Create 1D motion blur kernel for z-direction
        kernel = torch.ones(1, 1, kernel_size, 1, 1) / kernel_size
        kernel = kernel.to(img.device)
        
        # Apply padding and convolution
        padding = (0, 0, 0, 0, kernel_size//2, kernel_size//2)
        img_padded = F.pad(img, padding, mode='reflect')
        blurred = F.conv3d(img_padded, kernel)
        
        return blurred
    
    def simulate_partial_volume(self, img):
        """Simulate partial volume effects"""
        # Slight smoothing to simulate partial volume
        kernel = torch.ones(1, 1, 3, 3, 3) / 27  # 3x3x3 average kernel
        kernel = kernel.to(img.device)
        
        padding = (1, 1, 1, 1, 1, 1)
        img_padded = F.pad(img, padding, mode='reflect')
        smoothed = F.conv3d(img_padded, kernel)
        
        return smoothed
    
    def create_realistic_low_res(self, high_res_mri):
        """Complete degradation pipeline"""
        # 1. Add slight motion blur
        degraded = self.add_motion_blur(high_res_mri)
        
        # 2. Simulate partial volume effects
        degraded = self.simulate_partial_volume(degraded)
        
        # 3. Downsample in z-direction
        B, C, D, H, W = degraded.shape
        target_size = (D // self.downscale_factor, H, W)
        low_res = F.interpolate(
            degraded, 
            size=target_size, 
            mode='trilinear', 
            align_corners=False
        )
        
        # 4. Add acquisition noise
        low_res = self.add_noise(low_res, noise_level=0.02)
        
        return low_res

# Usage
degrader = MRIDegradationPipeline(downscale_factor=2)