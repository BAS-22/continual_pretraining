import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block
from models.patch_embed import PatchEmbed
from util.pos_embed import get_3d_sincos_pos_embed
from util.lr_decay import param_groups_lrd
from models.vit import ViT
from models.LoRa import apply_lora_to_vit 
import torch.nn.functional as F
import pytorch_msssim

class MAESuperResolutionDecoder(nn.Module):
    """Super-resolution decoder based on ESPCN and SRCNN principles"""
    
    def __init__(self, decoder_embed_dim, patch_size, upscale_factor=2, in_chans=1):
        super().__init__()
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.in_chans = in_chans
        
        # Original patch volume
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2] * in_chans
        
        # Feature extraction layers (similar to SRCNN)
        self.feature_extractor = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.ReLU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim // 2),
            nn.ReLU(),
        )
        
        # Map to spatial features for 3D processing
        self.spatial_mapper = nn.Linear(
            decoder_embed_dim // 2, 
            patch_vol * 8  # 8 feature channels
        )
        
        # 3D CNN layers for spatial refinement
        self.conv_layers = nn.Sequential(
            nn.Conv3d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Sub-pixel convolution for z-direction upsampling (inspired by ESPCN)
        self.upconv = nn.Conv3d(
            16, 
            in_chans * (upscale_factor ** 1),  # Only upscale in z-direction
            kernel_size=3, 
            padding=1
        )
        
        # Pixel shuffle for z-direction (custom implementation)
        self.pixel_shuffle_3d = PixelShuffle3D(upscale_factor, dim='z')
    
    def forward(self, x):
        """
        x: [B, L, decoder_embed_dim] - decoder features
        Returns: [B, L, upscaled_patch_volume] 
        """
        B, L, D = x.shape
        
        # Extract features
        features = self.feature_extractor(x)  # [B, L, D//2]
        
        # Map to spatial representation
        spatial = self.spatial_mapper(features)  # [B, L, patch_vol * 8]
        
        # Reshape for 3D convolution
        # Each patch becomes a small 3D volume with 8 channels
        patch_vol = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        spatial = spatial.view(B * L, 8, self.patch_size[0], self.patch_size[1], self.patch_size[2])
        
        # Apply 3D convolutions
        conv_out = self.conv_layers(spatial)  # [B*L, 16, pd, ph, pw]
        
        # Upsampling convolution
        upconv_out = self.upconv(conv_out)  # [B*L, in_chans*upscale, pd, ph, pw]
        
        # Apply pixel shuffle for z-direction upsampling
        upsampled = self.pixel_shuffle_3d(upconv_out)  # [B*L, in_chans, pd*upscale, ph, pw]
        
        # Reshape back to patch format
        upsampled_vol = upsampled.shape[2] * upsampled.shape[3] * upsampled.shape[4] * self.in_chans
        output = upsampled.view(B, L, upsampled_vol)
        
        return output

class PixelShuffle3D(nn.Module):
    """3D Pixel Shuffle for upsampling in specific dimension"""
    
    def __init__(self, upscale_factor, dim='z'):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.dim = dim
    
    def forward(self, x):
        """
        x: [B, C*r, D, H, W] where r is upscale_factor
        Returns: [B, C, D*r, H, W] if dim='z'
        """
        if self.dim == 'z':
            B, Cr, D, H, W = x.shape
            C = Cr // self.upscale_factor
            
            # Reshape and rearrange
            x = x.view(B, C, self.upscale_factor, D, H, W)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.view(B, C, D * self.upscale_factor, H, W)
            
        return x

class MAEEncoder(nn.Module):
    """MAE Encoder as a separate class"""
    
    def __init__(
        self,
        img_size=(32, 128, 128),
        patch_size=(8, 16, 16),
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        self.patch_size = patch_size
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        
        # Initialize positional embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=0.02)
    
    def random_masking(self, x, mask_ratio):
        """Perform per-sample random masking"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, x, mask_ratio=0):
        # Embed patches
        x = self.patch_embed(x)
        
        # Add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.contiguous().expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore

class MAEDecoder(nn.Module):
    """MAE Decoder with Super-Resolution capability as a separate class"""
    
    def __init__(
        self,
        grid_size,
        patch_size,
        embed_dim=768,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        upscale_factor=2,
        in_chans=1,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.in_chans = in_chans
        
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Super-resolution decoder
        self.sr_decoder = MAESuperResolutionDecoder(
            decoder_embed_dim=decoder_embed_dim,
            patch_size=patch_size,
            upscale_factor=upscale_factor,
            in_chans=in_chans
        )
        
        # Initialize positional embeddings
        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )
        
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.contiguous().repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        # Apply super-resolution decoder
        x = self.sr_decoder(x)
        
        return x

class MaskedAutoencoderSR_ori(nn.Module):
    """Masked Autoencoder with Super-Resolution capability - Refactored with separate classes"""

    def __init__(
        self,
        img_size=(32, 128, 128),
        patch_size=(8, 16, 16),
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        upscale_factor=2,
    ):
        super().__init__()
        
        self.upscale_factor = upscale_factor
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        
        # Calculate grid size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1], 
            img_size[2] // patch_size[2]
        )
        
        # Separate encoder and decoder modules
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        
        self.decoder = MAEDecoder(
            grid_size=self.grid_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            upscale_factor=upscale_factor,
            in_chans=in_chans,
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify_sr(self, x):
        """Unpatchify for super-resolution output"""
        B, L, dim = x.shape
        
        upsampled_patch_d = self.patch_size[0] * self.upscale_factor
        
        x = x.reshape(
            shape=(
                B,
                self.grid_size[0],
                self.grid_size[1], 
                self.grid_size[2],
                upsampled_patch_d,
                self.patch_size[1],
                self.patch_size[2],
                1,  # in_chans
            )
        )
        
        x = (
            x.permute(0, 7, 1, 4, 2, 5, 3, 6)
            .contiguous()
            .reshape(
                B,
                1,
                self.grid_size[0] * upsampled_patch_d,
                self.grid_size[1] * self.patch_size[1],
                self.grid_size[2] * self.patch_size[2],
            )
        )
        return x

    def forward_loss_sr(self, low_res_imgs, high_res_target, pred):
        """Compute super-resolution loss"""
        pred_img = self.unpatchify_sr(pred)
        
        l1_loss = nn.L1Loss()(pred_img, high_res_target)
        l2_loss = nn.MSELoss()(pred_img, high_res_target)
        
        total_loss = l1_loss + 0.1 * l2_loss
        return total_loss

    def forward(self, low_res_imgs, high_res_target=None, mask_ratio=0):
        """Forward pass for super-resolution"""
        # Encode
        latent, mask, ids_restore = self.encoder(low_res_imgs, mask_ratio)
        
        # Decode
        pred = self.decoder(latent, ids_restore)
        
        if high_res_target is not None:
            # Training mode
            loss = self.forward_loss_sr(low_res_imgs, high_res_target, pred)
            pred_img = self.unpatchify_sr(pred)
            return loss, pred_img, mask
        else:
            # Inference mode
            pred_img = self.unpatchify_sr(pred)
            return pred_img

class MaskedAutoencoderSR(nn.Module):
    def __init__(
        self,
        img_size=(32, 256, 256),
        patch_size=(8, 16, 16),
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        upscale_factor=2,
    ):
        super().__init__()
        
        self.upscale_factor = upscale_factor
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        
        # Calculate grid size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1], 
            img_size[2] // patch_size[2]
        )
        
        # Use ViT as encoder instead of custom MAEEncoder
        self.encoder = ViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        
        # Keep your custom decoder
        self.decoder = MAEDecoder(
            grid_size=self.grid_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            upscale_factor=upscale_factor,
            in_chans=in_chans,
        )
        
        # Don't apply _init_weights to encoder since it has its own initialization
        self.decoder.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify_sr(self, x):
        """Unpatchify for super-resolution output"""
        B, L, dim = x.shape
        
        upsampled_patch_d = self.patch_size[0] * self.upscale_factor
        
        x = x.reshape(
            shape=(
                B,
                self.grid_size[0],
                self.grid_size[1], 
                self.grid_size[2],
                upsampled_patch_d,
                self.patch_size[1],
                self.patch_size[2],
                1,  # in_chans
            )
        )
        
        x = (
            x.permute(0, 7, 1, 4, 2, 5, 3, 6)
            .contiguous()
            .reshape(
                B,
                1,
                self.grid_size[0] * upsampled_patch_d,
                self.grid_size[1] * self.patch_size[1],
                self.grid_size[2] * self.patch_size[2],
            )
        )
        return x

    def forward_loss_sr(self, pred_img, high_res_target):
        """Compute super-resolution loss"""
        # pred_img = self.unpatchify_sr(pred)
        
        l1_loss = nn.L1Loss()(pred_img, high_res_target)
        l2_loss = nn.MSELoss()(pred_img, high_res_target)
        
        total_loss = l1_loss + 0.1 * l2_loss
        return total_loss

    def forward_loss_sr_hybrid(self, pred, target, alpha=0.7, beta=0.3, gamma=0.001):
        """
        Compute the super-resolution loss using a combination of:
        - L1 loss (voxel-wise accuracy)
        - SSIM loss (structural similarity)
        - Fourier loss (frequency consistency)
        """
        # L1 loss
        l1_loss = nn.L1Loss()(pred, target)

        # SSIM loss - compute slice-wise for 3D volumes
        ssim_loss = self._compute_3d_ssim_loss(pred, target)

        # Improved Fourier loss
        fft_loss = self._compute_fourier_loss(pred, target)

        # Combine losses
        total_loss = alpha * l1_loss + beta * ssim_loss + gamma * fft_loss
        return total_loss, l1_loss, ssim_loss, fft_loss

    def _compute_fourier_loss(self, pred, target):
        """
        Compute normalized Fourier loss for frequency consistency
        """
        # Compute FFT for both prediction and target
        fft_pred = torch.fft.fftn(pred, dim=(-3, -2, -1))
        fft_target = torch.fft.fftn(target, dim=(-3, -2, -1))
        
        # Method 1: Normalize by the target's magnitude
        fft_pred_mag = torch.abs(fft_pred)
        fft_target_mag = torch.abs(fft_target)
        
        # Avoid division by zero
        eps = 1e-8
        normalized_diff = torch.abs(fft_pred_mag - fft_target_mag) / (fft_target_mag + eps)
        fft_loss = torch.mean(normalized_diff)
        
        return fft_loss
    
    def _compute_3d_ssim_loss(self, pred, target):
        """
        Compute SSIM loss for 3D volumes by averaging over depth slices
        
        Args:
            pred: [B, C, D, H, W]
            target: [B, C, D, H, W]
        """
        B, C, D, H, W = pred.shape
        
        # Initialize 2D SSIM module
        ssim_module = pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=C)
        
        ssim_values = []
        
        # Compute SSIM for each depth slice
        for d in range(D):
            pred_slice = pred[:, :, d, :, :]  # [B, C, H, W]
            target_slice = target[:, :, d, :, :]  # [B, C, H, W]
            
            # Compute SSIM for this slice
            ssim_val = ssim_module(pred_slice, target_slice)
            ssim_values.append(ssim_val)
        
        # Average SSIM across all slices
        avg_ssim = torch.stack(ssim_values).mean()
        
        # Return 1 - SSIM as loss (we want to minimize this)
        return 1 - avg_ssim

    def forward(self, low_res_imgs, high_res_target=None, mask_ratio=0):
        """Forward pass for super-resolution"""
        # Encode with ViT
        cls_token, patch_tokens = self.encoder(low_res_imgs)
        
        # # Apply masking if needed
        # if mask_ratio == -1:
        #     patch_tokens, mask, ids_restore = self.random_masking(patch_tokens, mask_ratio)
        #     # Re-add cls token
        #     latent = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        # else:
        #     latent = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        #     ids_restore = torch.arange(patch_tokens.shape[1], device=patch_tokens.device).unsqueeze(0).repeat(patch_tokens.shape[0], 1)
        #     mask = torch.zeros(patch_tokens.shape[0], patch_tokens.shape[1], device=patch_tokens.device)
        
        latent = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        ids_restore = torch.arange(patch_tokens.shape[1], device=patch_tokens.device).unsqueeze(0).repeat(patch_tokens.shape[0], 1)
        mask = torch.zeros(patch_tokens.shape[0], patch_tokens.shape[1], device=patch_tokens.device)

        # Decode
        pred = self.decoder(latent, ids_restore)
        
        if high_res_target is not None:
            # Training mode - first unpatchify the prediction
            pred_img = self.unpatchify_sr(pred)
            # Then calculate loss with unpatchified prediction and target
            # loss, l1, ss, ft = self.forward_loss_sr(pred_img, high_res_target)
            loss = self.forward_loss_sr(pred_img, high_res_target)
            return loss, pred_img, mask
        else:
            # Inference mode
            pred_img = self.unpatchify_sr(pred)
            return pred_img

    def random_masking(self, x, mask_ratio):
        """Move masking logic here since ViT doesn't have it"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # ... rest of your methods stay the same ...

# Factory functions
def mae_sr_vit_base(**kwargs):
    model = MaskedAutoencoderSR(
        img_size=(32, 256, 256),  # Match your low res input image size
        # patch_size=(4, 8, 8),  #//4
        patch_size=(16, 32, 32), # Match your pretrained model patch size
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        upscale_factor=2,
        **kwargs
    )
    return model

def mae_sr_vit_large(**kwargs):
    model = MaskedAutoencoderSR(
        img_size=(32, 256, 256),
        patch_size=(8, 16, 16),
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        upscale_factor=2,
        **kwargs
    )
    return model

def build_model(args, device):
    if args.model == "mae_sr_vit_base":
        model = mae_sr_vit_base()

        if args.case == "scratch":
            pass
        elif args.case == "prenc":
            pretrained_dict = torch.load(args.pretrain, map_location="cpu")["model"]
            model_dict = model.encoder.state_dict()
            
            # Filter out mismatched keys
            filtered_dict = {}
            
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if k == "pos_embed":
                        # Handle positional embedding size mismatch
                        pretrained_pos_embed = v
                        current_pos_embed = model_dict[k]

                        if pretrained_pos_embed.shape != current_pos_embed.shape:
                            print(f"Resizing pos_embed from {pretrained_pos_embed.shape} to {current_pos_embed.shape}")
                            # Keep the current model's pos_embed (it's properly initialized)
                            continue
                        else:
                            filtered_dict[k] = v
                            
                    elif k == "patch_embed.proj.weight":
                        # Handle input channel mismatch for patch embedding
                        pretrained_weight = v  # Shape: [embed_dim, 3, patch_d, patch_h, patch_w]
                        current_weight = model_dict[k]  # Shape: [embed_dim, 1, patch_d, patch_h, patch_w]
                        
                        if pretrained_weight.shape[1] != current_weight.shape[1]:
                            # print(f"Adapting patch_embed.proj.weight from {pretrained_weight.shape} to {current_weight.shape}")
                            # # Take only the first channel (T2) from pretrained weights
                            # adapted_weight = pretrained_weight[:, :1, :, :, :]  # Take only first channel
                            # filtered_dict[k] = adapted_weight
                            print(f"Adapting patch_embed.proj.weight from {pretrained_weight.shape} to {current_weight.shape}")
                            adapted_weight = pretrained_weight.mean(dim=1, keepdim=True)
                            filtered_dict[k] = adapted_weight
                        else:
                            filtered_dict[k] = v
                            
                    elif model_dict[k].shape == v.shape:
                        filtered_dict[k] = v
                    else:
                        print(f"Skipping {k} due to shape mismatch: pretrained {v.shape} vs current {model_dict[k].shape}")
                else:
                    print(f"Skipping {k} as it's not in current model")
            
            # Load the filtered weights
            
            msg = model.encoder.load_state_dict(filtered_dict, strict=False)
            print(f"Loaded weights. Missing keys: {msg.missing_keys}")
            print(f"Unexpected keys: {msg.unexpected_keys}")

        elif args.case == "posprenc":
            # Apply LoRA
            model.encoder, lora_params = apply_lora_to_vit(
                model.encoder,            # The ViT model
                rank=16,         # Rank of the decomposition
                lora_alpha=32,   # Scaling factor
                lora_dropout=0.1, # Dropout probability
                freeze_original=False, # Freeze original parameters
            )

            pretrained_dict = torch.load(args.pretrain, map_location="cpu")["model"]
            # model_dict = model.encoder.state_dict()
            model_dict = model.state_dict()

            # Filter out mismatched keys
            filtered_dict = {}
            
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if k == "encoder.pos_embed":
                        # Handle positional embedding size mismatch
                        pretrained_pos_embed = v
                        current_pos_embed = model_dict[k]

                        if pretrained_pos_embed.shape != current_pos_embed.shape:
                            print(f"Resizing pos_embed from {pretrained_pos_embed.shape} to {current_pos_embed.shape}")
                            # Keep the current model's pos_embed (it's properly initialized)
                            continue
                        else:
                            filtered_dict[k] = v
                            
                    elif k == "encoder.patch_embed.proj.weight":
                        # Handle input channel mismatch for patch embedding
                        pretrained_weight = v  # Shape: [embed_dim, 3, patch_d, patch_h, patch_w]
                        current_weight = model_dict[k]  # Shape: [embed_dim, 1, patch_d, patch_h, patch_w]
                        
                        if pretrained_weight.shape[1] != current_weight.shape[1]:
                            # print(f"Adapting patch_embed.proj.weight from {pretrained_weight.shape} to {current_weight.shape}")
                            # # Take only the first channel (T2) from pretrained weights
                            # adapted_weight = pretrained_weight[:, :1, :, :, :]  # Take only first channel
                            # filtered_dict[k] = adapted_weight
                            print(f"Adapting patch_embed.proj.weight from {pretrained_weight.shape} to {current_weight.shape}")
                            adapted_weight = pretrained_weight.mean(dim=1, keepdim=True)
                            filtered_dict[k] = adapted_weight
                        else:
                            filtered_dict[k] = v
                            
                    elif model_dict[k].shape == v.shape:
                        filtered_dict[k] = v
                    else:
                        print(f"Skipping {k} due to shape mismatch: pretrained {v.shape} vs current {model_dict[k].shape}")
                else:
                    print(f"Skipping {k} as it's not in current model")
            
            # Load the filtered weights
            
            # msg = model.encoder.load_state_dict(filtered_dict, strict=False)
            msg = model.load_state_dict(filtered_dict, strict=False)
            print(f"Loaded weights. Missing keys: {msg.missing_keys}")
            print(f"Unexpected keys: {msg.unexpected_keys}")


        elif args.case == "test":
            model.encoder, lora_params = apply_lora_to_vit(
                model.encoder,            # The ViT model
                rank=16,         # Rank of the decomposition
                lora_alpha=32,   # Scaling factor
                lora_dropout=0.1, # Dropout probability
                freeze_original=False, # Freeze original parameters
            )
            # print(f"LoRA applied. Number of LoRA parameters: {len(lora_params)}")
            # Load weights into entire model 
            # msg = model.load_state_dict(pretrained_dict, strict=True)
            # print(f"Loaded weights. Missing keys: {msg.missing_keys}")
            # print(f"Unexpected keys: {msg.unexpected_keys}")

        model = model.to(device)
        
        # Now you can access encoder and decoder as separate modules
        backbone_param_groups = param_groups_lrd(model.encoder, args.weight_decay)
        decoder_param_groups = decoder_parameters(model.decoder, args.weight_decay, lr=args.lr)
        
        optimizer = torch.optim.AdamW(
            backbone_param_groups + decoder_param_groups,
            lr=args.lr,
        )

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model))
        print("number of params (M): %.2f" % (n_parameters / 1.0e6))

        return model, optimizer
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")

def param_groups_weight_decay(
    model: torch.nn.Module, weight_decay=1e-5, no_weight_decay_list=(), lr=1e-3
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

def decoder_parameters(
    model: torch.nn.Module, weight_decay=1e-5, no_weight_decay_list=(), lr=1e-3
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
    ]
