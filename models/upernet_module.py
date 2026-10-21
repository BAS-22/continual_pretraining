from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from models.util import LayerNorm, GRN

# Add this class after the existing imports and before UperNetConvModule

import torch.nn.functional as F

class ContrastiveUncertaintyCalibrationLoss(nn.Module):
    """
    Novel Component 4: Contrastive Uncertainty Calibration Loss (CUCL)
    
    Key insight: Good uncertainty should satisfy:
    1. Features from low-uncertainty same-class regions should be similar (positive pairs)
    2. Features from high-uncertainty regions should be dissimilar (uncertain = ambiguous)
    3. Uncertainty should correlate with actual prediction errors (calibration)
    
    This creates a self-supervised signal to calibrate uncertainty without 
    requiring additional ground truth annotations.
    """
    
    def __init__(self, feature_dim=512, proj_dim=128, temperature=0.1, 
                 uncertainty_threshold=0.5, num_samples=256):
        super().__init__()
        self.temperature = temperature
        self.uncertainty_threshold = uncertainty_threshold
        self.num_samples = num_samples
        
        # Projection head for contrastive learning (projects decoder features)
        self.projector = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim // 2, kernel_size=1),
            nn.BatchNorm3d(feature_dim // 2),
            nn.GELU(),
            nn.Conv3d(feature_dim // 2, proj_dim, kernel_size=1),
        )
        
        # Learnable temperature (optional, can help with training stability)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
    def get_temperature(self):
        """Get current temperature value"""
        return torch.exp(self.log_temperature).clamp(min=0.01, max=1.0)
    
    def forward(self, features, predictions, ground_truth, uncertainty_map):
        """
        Compute contrastive uncertainty calibration loss.
        
        Args:
            features: (B, C, D, H, W) - Decoder features (before final classifier)
            predictions: (B, num_classes, D, H, W) - Model predictions (logits)
            ground_truth: (B, 1, D, H, W) - Ground truth labels
            uncertainty_map: (B, 1, D, H, W) - Uncertainty map from MC dropout
        
        Returns:
            contrastive_loss: Scalar - Contrastive learning loss
            calibration_loss: Scalar - Uncertainty calibration loss
            loss_info: Dict - Additional info for logging
        """
        device = features.device
        B, C, D, H, W = features.shape
        
        # === 1. Compute Calibration Loss ===
        # Uncertainty should be high where predictions are wrong
        calibration_loss, error_map = self._compute_calibration_loss(
            predictions, ground_truth, uncertainty_map
        )
        
        # === 2. Compute Contrastive Loss ===
        contrastive_loss, contrastive_info = self._compute_contrastive_loss(
            features, ground_truth, uncertainty_map, error_map
        )
        
        loss_info = {
            'calibration_loss': calibration_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'temperature': self.get_temperature().item(),
            **contrastive_info
        }
        
        return contrastive_loss, calibration_loss, loss_info
    
    def _compute_calibration_loss(self, predictions, ground_truth, uncertainty_map):
        """
        Calibration loss: uncertainty should match prediction error.
        
        - High uncertainty where prediction is wrong (encouraged)
        - Low uncertainty where prediction is correct (encouraged)
        """
        # Compute prediction correctness
        if predictions.shape[1] == 1:
            # Binary segmentation
            pred_class = (torch.sigmoid(predictions) > 0.5).float()
        else:
            # Multi-class segmentation
            pred_class = torch.argmax(predictions, dim=1, keepdim=True).float()
        
        gt_float = ground_truth.float()
        correct_mask = (pred_class == gt_float).float()  # 1 where correct
        error_map = 1.0 - correct_mask  # 1 where wrong
        
        # Normalize uncertainty to [0, 1]
        unc_min = uncertainty_map.amin(dim=(2, 3, 4), keepdim=True)
        unc_max = uncertainty_map.amax(dim=(2, 3, 4), keepdim=True)
        unc_norm = (uncertainty_map - unc_min) / (unc_max - unc_min + 1e-8)
        
        # Calibration: uncertainty should match error rate
        # MSE between normalized uncertainty and error map
        calibration_loss = F.mse_loss(unc_norm, error_map)
        
        # Additional: penalize overconfidence (low uncertainty but wrong)
        overconfidence_penalty = ((1.0 - unc_norm) * error_map).mean()
        
        # Additional: penalize underconfidence (high uncertainty but correct)
        underconfidence_penalty = (unc_norm * correct_mask).mean()
        
        # Combined calibration loss
        total_calibration = calibration_loss + 0.5 * overconfidence_penalty + 0.3 * underconfidence_penalty
        
        return total_calibration, error_map
    
    def _compute_contrastive_loss(self, features, ground_truth, uncertainty_map, error_map):
        """
        Contrastive loss: learn discriminative features based on uncertainty.
        
        - Low-uncertainty correct predictions from same class = positive pairs
        - High-uncertainty or incorrect predictions = negative pairs
        """
        device = features.device
        B, C, D, H, W = features.shape
        
        # Project features
        projected = self.projector(features)  # (B, proj_dim, D, H, W)
        proj_dim = projected.shape[1]
        
        # Normalize uncertainty
        unc_min = uncertainty_map.amin(dim=(2, 3, 4), keepdim=True)
        unc_max = uncertainty_map.amax(dim=(2, 3, 4), keepdim=True)
        unc_norm = (uncertainty_map - unc_min) / (unc_max - unc_min + 1e-8)
        
        # Create masks
        low_unc_mask = (unc_norm < self.uncertainty_threshold).float()
        high_unc_mask = 1.0 - low_unc_mask
        correct_mask = 1.0 - error_map
        
        # Anchor mask: low uncertainty AND correct prediction (confident correct)
        anchor_mask = low_unc_mask * correct_mask
        
        # Negative mask: high uncertainty OR incorrect prediction
        negative_mask = (high_unc_mask + error_map).clamp(0, 1)
        
        # Flatten for sampling
        projected_flat = projected.permute(0, 2, 3, 4, 1).reshape(-1, proj_dim)  # (B*D*H*W, proj_dim)
        anchor_mask_flat = anchor_mask.reshape(-1)
        negative_mask_flat = negative_mask.reshape(-1)
        class_flat = ground_truth.reshape(-1)
        
        # Get indices
        anchor_indices = torch.where(anchor_mask_flat > 0.5)[0]
        negative_indices = torch.where(negative_mask_flat > 0.5)[0]
        
        contrastive_loss = torch.tensor(0.0, device=device)
        info = {'num_anchors': 0, 'num_negatives': 0, 'avg_pos_sim': 0.0, 'avg_neg_sim': 0.0}
        
        # Need sufficient samples for contrastive learning
        min_samples = 32
        if len(anchor_indices) < min_samples or len(negative_indices) < min_samples:
            return contrastive_loss, info
        
        # Sample subset for efficiency
        num_samples = min(self.num_samples, len(anchor_indices), len(negative_indices))
        
        # Random sampling
        anchor_perm = torch.randperm(len(anchor_indices), device=device)[:num_samples]
        negative_perm = torch.randperm(len(negative_indices), device=device)[:num_samples]
        
        anchor_idx = anchor_indices[anchor_perm]
        negative_idx = negative_indices[negative_perm]
        
        # Get features
        anchor_features = projected_flat[anchor_idx]  # (num_samples, proj_dim)
        anchor_classes = class_flat[anchor_idx]
        negative_features = projected_flat[negative_idx]  # (num_samples, proj_dim)
        
        # Find positive pairs (same class, also confident correct)
        positive_features = self._find_positive_pairs(
            anchor_features, anchor_classes, 
            projected_flat, class_flat, anchor_mask_flat, anchor_indices
        )
        
        # L2 normalize for cosine similarity
        anchor_features = F.normalize(anchor_features, dim=1)
        positive_features = F.normalize(positive_features, dim=1)
        negative_features = F.normalize(negative_features, dim=1)
        
        # Temperature
        temperature = self.get_temperature()
        
        # Positive similarity (anchor vs positive from same class)
        pos_sim = (anchor_features * positive_features).sum(dim=1) / temperature  # (num_samples,)
        
        # Negative similarities (anchor vs all negatives)
        neg_sim = torch.mm(anchor_features, negative_features.t()) / temperature  # (num_samples, num_samples)
        
        # InfoNCE loss
        # log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = pos - log(exp(pos) + sum(exp(neg)))
        
        # Numerical stability: subtract max
        max_neg = neg_sim.max(dim=1, keepdim=True)[0].detach()
        
        # Compute log-sum-exp of negatives
        neg_exp_sum = torch.exp(neg_sim - max_neg).sum(dim=1)  # (num_samples,)
        
        # InfoNCE loss per anchor
        pos_exp = torch.exp(pos_sim - max_neg.squeeze())
        loss_per_anchor = -pos_sim + max_neg.squeeze() + torch.log(pos_exp + neg_exp_sum + 1e-8)
        
        contrastive_loss = loss_per_anchor.mean()
        
        # Compute info for logging
        with torch.no_grad():
            avg_pos_sim = (anchor_features * positive_features).sum(dim=1).mean().item()
            avg_neg_sim = torch.mm(anchor_features, negative_features.t()).mean().item()
        
        info = {
            'num_anchors': num_samples,
            'num_negatives': num_samples,
            'avg_pos_sim': avg_pos_sim,
            'avg_neg_sim': avg_neg_sim,
        }
        
        return contrastive_loss, info
    
    def _find_positive_pairs(self, anchor_features, anchor_classes, 
                              all_features, all_classes, anchor_mask, anchor_indices):
        """
        Find positive pairs for each anchor (same class, confident correct).
        Falls back to self if no other same-class sample found.
        """
        device = anchor_features.device
        num_anchors, proj_dim = anchor_features.shape
        
        positive_features = torch.zeros_like(anchor_features)
        
        for i in range(num_anchors):
            cls = anchor_classes[i]
            
            # Find other samples of same class that are also confident correct
            same_class_mask = (all_classes == cls) & (anchor_mask > 0.5)
            same_class_idx = torch.where(same_class_mask)[0]
            
            if len(same_class_idx) > 1:
                # Pick a random sample that's not the anchor itself
                # (anchor_indices are the global indices, we need to match)
                valid_idx = same_class_idx[same_class_idx != anchor_indices[i]]
                if len(valid_idx) > 0:
                    random_idx = valid_idx[torch.randint(0, len(valid_idx), (1,), device=device)]
                    positive_features[i] = all_features[random_idx]
                else:
                    # Use self (will result in similarity of 1, but still valid)
                    positive_features[i] = anchor_features[i]
            else:
                # No other same-class sample, use self
                positive_features[i] = anchor_features[i]
        
        return positive_features


class UperNetConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.batch_norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first") # nn.BatchNorm3d(out_channels)
        self.activation = nn.GELU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.activation(output)

        return output


class UperNetPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = [
            nn.AdaptiveAvgPool3d(pool_scale),
            UperNetConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class UperNetPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (`Tuple[int]`):
            Pooling scales used in Pooling Pyramid Module.
        in_channels (`int`):
            Input channels.
        channels (`int`):
            Channels after modules, before conv_seg.
        align_corners (`bool`):
            align_corners argument of F.interpolate.
    """

    def __init__(
        self,
        pool_scales: Tuple[int, ...],
        in_channels: int,
        channels: int,
        align_corners: bool,
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for i, pool_scale in enumerate(pool_scales):
            block = UperNetPyramidPoolingBlock(
                pool_scale=pool_scale, in_channels=in_channels, channels=channels
            )
            self.blocks.append(block)
            self.add_module(str(i), block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="trilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UperNetHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).
    """

    def __init__(self, in_channels, pool_scales, hidden_size, out_channels, dropout_rate=0.1):
        super().__init__()
        self.pool_scales = pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = in_channels
        self.channels = hidden_size
        self.align_corners = False
        self.classifier = nn.Conv3d(self.channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout3d(p = dropout_rate)
        # PSP Module
        self.psp_modules = UperNetPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = UperNetConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_dropouts = nn.ModuleList() #per-sclae dropout
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = UperNetConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = UperNetConvModule(
                self.channels, self.channels, kernel_size=3, padding=1
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_dropouts.append(nn.Dropout3d(p = dropout_rate))

        self.fpn_bottleneck = UperNetConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv3d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = self.dropout(output) # PSP bottleneck dropout

        return output

    def forward(self, encoder_hidden_states: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # build laterals
        laterals = [
            lateral_conv(encoder_hidden_states[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i],
                size=prev_shape,
                mode="trilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_dropouts[i](self.fpn_convs[i](laterals[i])) for i in range(used_backbone_levels - 1)
        ] #dropout after each FPN conv
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="trilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        features = self.fpn_bottleneck(fpn_outs)  # Features before classifier
        features = self.dropout(features)
        output = self.classifier(features)

        if return_features:
            return output, features
        return output



class UperNetFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        in_channels (int):
            Number of input channels.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        num_convs,
        out_channels,
        concat_input=False,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: Union[int, Tuple[int, int]] = 1,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels[in_index]
        self.channels = hidden_size
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.in_index = in_index
        self.dropout = nn.Dropout3d(p=dropout_rate)

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            UperNetConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                UperNetConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = UperNetConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        self.classifier = nn.Conv3d(self.channels, out_channels, kernel_size=1)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv3d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.dropout(output) # before classifier   
        output = self.classifier(output)
        return output


class ViTAdapter(nn.Module):
    def __init__(
        self,
        img_size=(64, 256, 256),
        patch_size=(16, 32, 32),
        embed_dim=768,
        # out_indices=[3, 5, 7, 11],
    ):
        super().__init__()
        # self.out_indices = out_indices

        self.grid_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.hidden_size = embed_dim

        if patch_size == (16, 32, 32):
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dim, embed_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
                ),
                nn.BatchNorm3d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm3d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            # 8
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dim, embed_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
                ),
                nn.BatchNorm3d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            # 16
            self.fpn3 = nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dim, embed_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
                ),
            )

            # 32
            self.fpn4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

            self.adapters = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]

    def proj_feat(self, x):
        
        new_view = (x.size(0), *self.grid_size, self.hidden_size)
        # print(f"x.shape: {x.shape}, expected: {new_view}, grid_size: {self.grid_size}")
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(
            d + 1 for d in range(len(self.grid_size))
        )
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, encoder_hidden_states):
        output = []
        # print(f"len_encoder_hidden: {len(encoder_hidden_states)}")
        for index, op in zip(range(len(encoder_hidden_states)), self.adapters):
            output.append(op(self.proj_feat(encoder_hidden_states[index])))
        return output


class BoundaryAwareAttention(nn.Module):
    """Use uncertainty to create attention that focuses on boundaries"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        
        # Channel attention branch
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention from uncertainty
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Learnable combination weights
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Balance between original and refined
        self.beta = nn.Parameter(torch.tensor(1.0))   # Uncertainty influence strength
        
    def forward(self, feat, uncertainty):
        """
        Args:
            feat: (B, C, D, H, W) encoder features
            uncertainty: (B, 1, D, H, W) uncertainty map (high at boundaries)
        """
        B, C, D, H, W = feat.shape
        
        # Channel attention
        channel_att = self.channel_attention(feat)  # (B, C, 1, 1, 1)
        
        # Spatial attention from uncertainty
        spatial_att = self.spatial_attention(uncertainty)  # (B, 1, D, H, W)
        
        # Combine: enhance features where uncertainty is high (boundaries)
        # This creates a "focus on boundaries" effect
        combined_att = channel_att * (1.0 + self.beta * spatial_att)
        
        # Apply attention
        refined_feat = feat * combined_att
        
        # Residual connection with learnable weight
        output = self.alpha * refined_feat + (1.0 - self.alpha) * feat
        
        return output


class UperNet(nn.Module):
    def __init__(
        self,
        encoder,
        in_channels,
        out_channels,
        adapter=None,
        out_indices=None,
        pool_scales=[1, 2, 3, 6],
        hidden_size=512,
        auxiliary_channels=256,
        use_auxiliary_head=True,
        dropout_rate=0.1,
        use_cucl=False,  # NEW: Enable CUCL
        cucl_feature_dim=512,  # NEW: Feature dimension for CUCL
        cucl_proj_dim=128,  # NEW: Projection dimension
        cucl_temperature=0.1,
        cucl_uncertainty_threshold=0.5,
        cucl_num_samples=256,
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.out_indices = out_indices
        self.decode_head = UperNetHead(
            in_channels=in_channels,
            pool_scales=pool_scales,
            hidden_size=hidden_size,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
        )

        # ========== IMPROVED: Boundary-Aware Attention Modules ==========
        self.boundary_attention = nn.ModuleList([
            BoundaryAwareAttention(in_channels[i], reduction=4)
            for i in range(len(in_channels))
        ])
        
        # Learnable uncertainty threshold for adaptive refinement
        self.unc_threshold = nn.Parameter(torch.tensor(0.5))


         # ========== NEW: Contrastive Uncertainty Calibration Loss ==========
        if use_cucl:
            self.cucl_module = ContrastiveUncertaintyCalibrationLoss(
                feature_dim=cucl_feature_dim,
                proj_dim=cucl_proj_dim,
                temperature=cucl_temperature,
                uncertainty_threshold=cucl_uncertainty_threshold,
                num_samples=cucl_num_samples,
            )
        else:
            self.cucl_module = None

        self.auxiliary_head = (
            UperNetFCNHead(
                in_channels=in_channels,
                hidden_size=auxiliary_channels,
                num_convs=1,
                out_channels=out_channels,
                dropout_rate=dropout_rate,
            )
            if use_auxiliary_head
            else None
        )

        self.hidden_norm = nn.ModuleList()
        for in_channel in in_channels:
            norm = LayerNorm(in_channel, eps=1e-6, data_format="channels_first")
            self.hidden_norm.append(norm)

    def compute_cucl_loss(self, decoder_features, predictions, ground_truth, uncertainty_map):
        """
        Compute CUCL loss if enabled.
        
        Args:
            decoder_features: Features from decode_head (before classifier)
            predictions: Model logits
            ground_truth: GT labels
            uncertainty_map: Uncertainty from MC dropout
        
        Returns:
            contrastive_loss, calibration_loss, loss_info (or zeros if disabled)
        """
        if self.cucl_module is None:
            device = predictions.device
            return (
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                {}
            )
        
        return self.cucl_module(
            decoder_features, predictions, ground_truth, uncertainty_map
        )

    def forward(self, x, compute_uncertainty=False, mc_samples=5, use_uncertainty_modulation=False):
        encoder_hidden_states = self.encoder(x, ret_hids=True)

        if isinstance(encoder_hidden_states, (list, tuple)):
            encoder_hidden_states = encoder_hidden_states[-1]

        if self.out_indices:
            encoder_hidden_states = [encoder_hidden_states[i] for i in self.out_indices]

        encoder_hidden_states = [norm(encoder_hidden_states[i]) for i, norm in enumerate(self.hidden_norm)]

        if self.adapter:
            encoder_hidden_states_spatial = self.adapter(encoder_hidden_states)
        else:
            encoder_hidden_states_spatial = encoder_hidden_states

        if compute_uncertainty:
            def enable_dropout(model):
                for module in model.modules():
                    if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                        module.train()

            enable_dropout(self.decode_head)
            if self.auxiliary_head is not None:
                enable_dropout(self.auxiliary_head)

            mc_batch_size = 4
            mc_logits_list = []
            eps = 1e-8

            with torch.no_grad():
                for i in range(0, mc_samples, mc_batch_size):
                    batch_end = min(i + mc_batch_size, mc_samples)
                    for _ in range(i, batch_end):
                        logits_sample = self.decode_head(encoder_hidden_states_spatial)
                        mc_logits_list.append(logits_sample.detach())
                    if (i + mc_batch_size) % 10 == 0:
                        torch.cuda.empty_cache()

                mc_logits = torch.stack(mc_logits_list, dim=0)

                if self.decode_head.classifier.out_channels == 1:
                    mc_probs = torch.sigmoid(mc_logits)
                else:
                    mc_probs = torch.softmax(mc_logits, dim=2)

                mean_probs = mc_probs.mean(dim=0)
                pred_ent = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1, keepdim=True)
                sample_ent = -(mc_probs * torch.log(mc_probs + eps)).sum(dim=2, keepdim=False)
                exp_ent = sample_ent.mean(dim=0, keepdim=True)
                exp_ent = exp_ent.permute(1, 0, 2, 3, 4).contiguous()

                mi = (pred_ent - exp_ent).clamp_min(0.0)
                per_class_var = mc_probs.var(dim=0, unbiased=False)
                var_sum = per_class_var.sum(dim=1, keepdim=True)
                raw_unc = (1.5 * mi + 1.0 * var_sum + 0.5 * pred_ent)

                umin = raw_unc.amin(dim=(2, 3, 4), keepdim=True)
                umax = raw_unc.amax(dim=(2, 3, 4), keepdim=True)
                unc_norm = (raw_unc - umin) / (umax - umin + eps)
                uncertainty_full = nn.functional.interpolate(pred_ent, size=x.shape[2:], mode="trilinear", align_corners=False)

                del mc_logits, mc_probs, mc_logits_list
                torch.cuda.empty_cache()

            if use_uncertainty_modulation:
                modulated_features = []
                for i, feat in enumerate(encoder_hidden_states_spatial):
                    unc_scale = nn.functional.interpolate(
                        unc_norm,
                        size=feat.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                    refined_feat = self.boundary_attention[i](feat, unc_scale)
                    modulated_features.append(refined_feat)

                enable_dropout(self)
                refined_logits = self.decode_head(modulated_features)
                refined_logits = nn.functional.interpolate(refined_logits, size=x.shape[2:], mode="trilinear", align_corners=False)

                if self.auxiliary_head is not None:
                    aux_logits = self.auxiliary_head(modulated_features)
                    aux_logits = nn.functional.interpolate(aux_logits, size=x.shape[2:], mode="trilinear", align_corners=False)
                    return [refined_logits, aux_logits], uncertainty_full

                return refined_logits, uncertainty_full

            logits = self.decode_head(encoder_hidden_states_spatial)
            logits = nn.functional.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)

            if self.auxiliary_head is not None:
                aux_logits = self.auxiliary_head(encoder_hidden_states_spatial)
                aux_logits = nn.functional.interpolate(aux_logits, size=x.shape[2:], mode="trilinear", align_corners=False)
                return [logits, aux_logits], uncertainty_full

            return logits, uncertainty_full

        logits = self.decode_head(encoder_hidden_states_spatial)
        logits = nn.functional.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)

        if not self.training:
            return logits

        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(encoder_hidden_states_spatial)
            auxiliary_logits = nn.functional.interpolate(auxiliary_logits, size=x.shape[2:], mode="trilinear", align_corners=False)
            return [logits, auxiliary_logits]

        return logits

class SuperResolutionModule(nn.Module):
    def __init__(self, channels, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # More sophisticated upsampling pathway
        self.upsample_path = nn.Sequential(
            nn.Conv3d(channels, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels*2),
            nn.GELU(),
            nn.ConvTranspose3d(channels*2, channels*2, kernel_size=(2,1,1), stride=(2,1,1)),
            nn.BatchNorm3d(channels*2),
            nn.GELU(),
            nn.Conv3d(channels*2, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels*2),
            nn.GELU(),
            nn.ConvTranspose3d(channels*2, channels, kernel_size=(2,1,1), stride=(2,1,1)),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        
        # Residual connection
        self.residual_upsample = nn.ConvTranspose3d(channels, channels, kernel_size=(4,1,1), stride=(4,1,1))
        
    def forward(self, x):
        residual = self.residual_upsample(x)
        main_path = self.upsample_path(x)
        return residual + main_path


class UperNetSR(nn.Module):
    def __init__(
        self,
        encoder,
        in_channels,
        out_channels,
        adapter=None,
        out_indices=None,
        pool_scales=[1, 2, 3, 6],
        hidden_size=512,
        auxiliary_channels=256,
        use_auxiliary_head=True,
        dropout_rate=0.1,
        use_super_resolution=True,  # Add this parameter
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.out_indices = out_indices
        self.decode_head = UperNetHead(
            in_channels=in_channels,
            pool_scales=pool_scales,
            hidden_size=hidden_size,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
        )
        self.auxiliary_head = (
            UperNetFCNHead(
                in_channels=in_channels,
                hidden_size=auxiliary_channels,
                num_convs=1,
                out_channels=out_channels,
            )
            if use_auxiliary_head
            else None
        )

        self.hidden_norm = nn.ModuleList()
        for in_channel in in_channels:
            norm = LayerNorm(in_channel, eps=1e-6, data_format="channels_first")
            self.hidden_norm.append(norm)
        
        # Add super-resolution module
        self.use_super_resolution = use_super_resolution
        if use_super_resolution:
            self.sr_module = SuperResolutionModule(out_channels, scale_factor=4)

    def forward(self, x):
        # Original shape for reference
        original_shape = x.shape
        target_shape = (original_shape[0], self.decode_head.classifier.out_channels, 
                       original_shape[2]*4, original_shape[3], original_shape[4])
        
        # Standard UperNet processing
        encoder_hidden_states = self.encoder(x, ret_hids=True)
        
        if isinstance(encoder_hidden_states, list) or isinstance(encoder_hidden_states, Tuple):
            encoder_hidden_states = encoder_hidden_states[-1]
            
        if self.out_indices:
            encoder_hidden_states = [
                encoder_hidden_states[i] for i in self.out_indices
            ]
        
        encoder_hidden_states = [
            norm(encoder_hidden_states[i])
            for i, norm in enumerate(self.hidden_norm)
        ]

        if self.adapter:
            encoder_hidden_states = self.adapter(encoder_hidden_states)

        logits = self.decode_head(encoder_hidden_states)
        
        # Apply super-resolution module if enabled
        if self.use_super_resolution:
            logits = self.sr_module(logits)
        else:
            # Just use interpolation
            logits = nn.functional.interpolate(
                logits, size=target_shape[2:], mode="trilinear", align_corners=False
            )
        
        if not self.training:
            return logits

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(encoder_hidden_states)
            
            if self.use_super_resolution:
                # Apply same SR technique to auxiliary logits
                auxiliary_logits = self.sr_module(auxiliary_logits)
            else:
                auxiliary_logits = nn.functional.interpolate(
                    auxiliary_logits,
                    size=target_shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
                
            return [logits, auxiliary_logits]
        return logits