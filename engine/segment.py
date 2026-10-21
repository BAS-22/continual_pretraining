# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import torch
import torch.nn.functional as F
import os
import util.misc as misc
import util.lr_sched as lr_sched
from monai.losses import DiceCELoss, DiceLoss
import numpy as np
import torch.nn as nn
from monai.metrics import DiceHelper, compute_dice, compute_hausdorff_distance, compute_average_surface_distance
from monai.networks import one_hot
import surface_distance
from surface_distance import metrics
from util.meter import DiceMeter, HausdorffMeter, SurfaceDistanceMeter
from util.feature_sim import compare_vit_encoders_with_cka_batch
import matplotlib.pyplot as plt
import matplotlib
import traceback
matplotlib.use('Agg')

from monai.inferers import sliding_window_inference


def compute_boundary_loss(pred, gt, uncertainty_map, num_classes):
    """
    Compute additional loss that penalizes errors at high-uncertainty (boundary) regions.
    
    Args:
        pred: Model predictions (B, C, D, H, W)
        gt: Ground truth (B, 1, D, H, W) 
        uncertainty_map: Uncertainty map (B, 1, D, H, W)
        num_classes: Number of output classes
    
    Returns:
        Weighted boundary loss scalar
    """
    if pred.shape[1] == 1:
        # Binary segmentation
        pred_prob = torch.sigmoid(pred)
        error = torch.abs(pred_prob - gt.float())
    else:
        # Multi-class segmentation
        pred_prob = torch.softmax(pred, dim=1)
        # Create one-hot encoding of ground truth
        gt_long = gt.long()
        if gt_long.dim() == 5 and gt_long.shape[1] == 1:
            gt_long = gt_long.squeeze(1)  # (B, D, H, W)
        gt_onehot = torch.zeros_like(pred_prob)
        gt_onehot.scatter_(1, gt_long.unsqueeze(1), 1)
        error = torch.abs(pred_prob - gt_onehot).sum(dim=1, keepdim=True)
    
    # Normalize uncertainty to [0, 1]
    unc_min = uncertainty_map.amin(dim=(2, 3, 4), keepdim=True)
    unc_max = uncertainty_map.amax(dim=(2, 3, 4), keepdim=True)
    unc_norm = (uncertainty_map - unc_min) / (unc_max - unc_min + 1e-8)
    
    # Higher weight for high-uncertainty regions (boundaries)
    # This encourages the model to focus on improving boundary predictions
    boundary_weight = 1.0 + 2.0 * unc_norm
    weighted_error = error * boundary_weight
    
    return weighted_error.mean()

def compute_confidence_calibration_loss(pred, gt, uncertainty_map, num_classes):
    """
    Penalize high uncertainty in correctly predicted regions.
    Encourage the model to be confident where it's correct.
    
    Args:
        pred: Model predictions (B, C, D, H, W)
        gt: Ground truth (B, 1, D, H, W)
        uncertainty_map: Uncertainty map (B, 1, D, H, W)
        num_classes: Number of output classes
    
    Returns:
        Confidence calibration loss scalar
    """
    if pred.shape[1] == 1:
        # Binary segmentation
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        correct = (pred_binary == gt.float()).float()
    else:
        # Multi-class segmentation
        pred_class = torch.argmax(pred, dim=1, keepdim=True)
        gt_class = gt.long() if gt.dtype != torch.long else gt
        correct = (pred_class == gt_class).float()
    
    # Normalize uncertainty
    unc_min = uncertainty_map.amin(dim=(2, 3, 4), keepdim=True)
    unc_max = uncertainty_map.amax(dim=(2, 3, 4), keepdim=True)
    unc_norm = (uncertainty_map - unc_min) / (unc_max - unc_min + 1e-8)
    
    # Penalize high uncertainty where prediction is correct (should be confident)
    confidence_loss = (unc_norm * correct).mean()
    
    # Also penalize low uncertainty where prediction is wrong (should be uncertain)
    incorrect = 1.0 - correct
    overconfidence_loss = ((1.0 - unc_norm) * incorrect).mean()
    
    return confidence_loss + overconfidence_loss

def enable_dropout(model):
    """
    Enable dropout layers during evaluation for MC Dropout.
    Keeps BatchNorm and other layers in eval mode.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def unwrap_logits(output):
    """
    Recursively unwrap model outputs to a logits tensor.
    Handles nested tuple/list outputs such as:
    - (logits, uncertainty)
    - ([logits, aux], uncertainty)
    - [logits, aux]
    """
    value = output
    while isinstance(value, (tuple, list)):
        if len(value) == 0:
            raise ValueError("Received empty model output sequence while unwrapping logits")
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor logits after unwrapping, got {type(value)}")
    return value


def normalize_uncertainty_map(uncertainty_map):
    """
    Normalize uncertainty map to [0, 1] per-sample with shape-preserving broadcast support.
    Expected shape: (B, 1, D, H, W)
    """
    unc_min = uncertainty_map.amin(dim=(2, 3, 4), keepdim=True)
    unc_max = uncertainty_map.amax(dim=(2, 3, 4), keepdim=True)
    return (uncertainty_map - unc_min) / (unc_max - unc_min + 1e-8)


def uncertainty_weighted_fusion(p_b, p_c, u_b, u_c, gamma=5.0):
    """
    Voxel-wise uncertainty-weighted fusion.

    Args:
        p_b: Baseline softmax probabilities, shape (B, C, D, H, W)
        p_c: CUCL softmax probabilities, shape (B, C, D, H, W)
        u_b: Baseline normalized uncertainty, shape (B, 1, D, H, W)
        u_c: CUCL normalized uncertainty, shape (B, 1, D, H, W)
        gamma: Uncertainty suppression strength.

    Returns:
        Fused probabilities with shape (B, C, D, H, W).
    """
    w_c = torch.exp(-gamma * u_c)
    w_b = torch.exp(-gamma * u_b)
    denom = w_c + w_b + 1e-8
    fused = (w_c * p_c + w_b * p_b) / denom
    return fused

def compute_calibration_error(pred, gt, uncertainty_map, num_classes):
    """
    Compute calibration error between uncertainty and prediction errors.
    """
    with torch.no_grad():
        if num_classes == 1 or pred.shape[1] == 1:
            pred_class = (torch.sigmoid(pred) > 0.5).float()
        else:
            pred_class = torch.argmax(pred, dim=1, keepdim=True).float()
        
        gt_float = gt.float()
        if gt_float.shape != pred_class.shape:
            if gt_float.dim() == 5 and gt_float.shape[1] > 1:
                gt_float = gt_float.argmax(dim=1, keepdim=True).float()
        
        correct = (pred_class == gt_float).float()
        error = 1.0 - correct
        
        # Normalize uncertainty to [0, 1]
        unc_min = uncertainty_map.amin()
        unc_max = uncertainty_map.amax()
        unc_norm = (uncertainty_map - unc_min) / (unc_max - unc_min + 1e-8)
        
        # Resize if needed
        if unc_norm.shape[2:] != error.shape[2:]:
            unc_norm = F.interpolate(unc_norm, size=error.shape[2:], mode='trilinear', align_corners=False)
        
        calibration_error = F.mse_loss(unc_norm, error).item()
        uncertainty_mean = unc_norm.mean().item()
        error_rate = error.mean().item()
        
        return calibration_error, uncertainty_mean, error_rate


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    use_mc_dropout=False,
    mc_samples=5,
    use_uncertainty_modulation=False,
):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
    if use_mc_dropout:
        metric_logger.add_meter("uncertainty", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
    if use_uncertainty_modulation:
        metric_logger.add_meter("boundary_loss", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("seg_loss", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("calibration_loss", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    # NEW: Add CUCL meters
    use_cucl = getattr(args, 'use_cucl', False)
    if use_cucl:
        metric_logger.add_meter("cucl_contrastive", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("cucl_calibration", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))


    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    # Loss function
    if args.out_channels == 1:
        loss_cal = DiceCELoss(sigmoid=True)
    else:
        loss_cal = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    # Uncertainty visualization directory
    if use_mc_dropout:
        unc_vis_dir = os.path.join(args.output_dir, "uncertainty_vis")
        os.makedirs(unc_vis_dir, exist_ok=True)

    # Loss weights for uncertainty-guided training
    boundary_loss_weight = getattr(args, 'boundary_loss_weight', 0.1)
    calibration_loss_weight = getattr(args, 'calibration_loss_weight', 0.05)
    cucl_contrastive_weight = getattr(args, 'cucl_contrastive_weight', 0.05)
    cucl_calibration_weight = getattr(args, 'cucl_calibration_weight', 0.1)
    aux_loss_warmup_epochs = max(0, int(getattr(args, 'aux_loss_warmup_epochs', 20)))

    for data_iter_step, (img, gt, dataidx) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        img, gt = img.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )

        epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)

        if aux_loss_warmup_epochs > 0:
            progress_epoch = epoch + (data_iter_step / len(data_loader))
            aux_loss_scale = min(1.0, (progress_epoch + 1.0) / aux_loss_warmup_epochs)
        else:
            aux_loss_scale = 1.0

        # ============ Uncertainty-Guided Training ============
        avg_uncertainty = 0.0
        uncertainty_map = None
        boundary_loss_value = 0.0
        calibration_loss_value = 0.0
        seg_loss_value = 0.0
        cucl_contrastive_value = 0.0
        cucl_calibration_value = 0.0


        if use_mc_dropout:
            # Get predictions with uncertainty
            output = model(
                img,
                compute_uncertainty=True,
                mc_samples=mc_samples,
                use_uncertainty_modulation=use_uncertainty_modulation
            )

            if isinstance(output, tuple):
                logit, uncertainty_map = output
                if uncertainty_map.shape[1] > 1:
                    avg_uncertainty = uncertainty_map.mean(dim=[0, 2, 3, 4]).mean().item()
                else:
                    avg_uncertainty = uncertainty_map.mean().item()
            else:
                logit = output

            # ========== Compute Segmentation Loss ==========
            if isinstance(logit, list):
                seg_loss = loss_cal(logit[0], gt) + 0.4 * loss_cal(logit[1], gt)
                main_logit = logit[0]
            else:
                seg_loss = loss_cal(logit, gt)
                main_logit = logit

            seg_loss_value = seg_loss.item()
            loss = seg_loss

            # ========== Uncertainty-Guided Losses ==========
            if use_uncertainty_modulation and uncertainty_map is not None:
                # Boundary-aware loss: penalize errors at high-uncertainty regions
                boundary_loss = compute_boundary_loss(
                    main_logit, gt, uncertainty_map.detach(), args.out_channels
                )
                boundary_loss_value = boundary_loss.item()

                # Confidence calibration loss
                calibration_loss = compute_confidence_calibration_loss(
                    main_logit, gt, uncertainty_map, args.out_channels
                )
                calibration_loss_value = calibration_loss.item()

                # Total loss
                boundary_weight_now = boundary_loss_weight * aux_loss_scale
                calibration_weight_now = calibration_loss_weight * aux_loss_scale
                loss = seg_loss + boundary_weight_now * boundary_loss + calibration_weight_now * calibration_loss

                       # ========== NEW: CUCL Loss ==========
            if use_cucl and uncertainty_map is not None:
                # Get model reference (handle DataParallel)
                model_ref = model.module if hasattr(model, 'module') else model
                
                if hasattr(model_ref, 'cucl_module') and model_ref.cucl_module is not None:
                    # Get decoder features for CUCL
                    # We need to do a forward pass through decode_head with return_features=True
                    # This is a bit redundant, so we'll use a more efficient approach
                    
                    # For efficiency, we'll compute CUCL on downsampled features
                    # Resize uncertainty and predictions to a smaller size
                    small_size = tuple(s // 4 for s in img.shape[2:])
                    
                    with torch.no_grad():
                        # Get encoder features (already computed in forward)
                        encoder_hidden_states = model_ref.encoder(img, ret_hids=True)
                        if isinstance(encoder_hidden_states, (list, tuple)):
                            encoder_hidden_states = encoder_hidden_states[-1]
                        if model_ref.out_indices:
                            encoder_hidden_states = [encoder_hidden_states[i] for i in model_ref.out_indices]
                        encoder_hidden_states = [
                            norm(encoder_hidden_states[i]) 
                            for i, norm in enumerate(model_ref.hidden_norm)
                        ]
                        if model_ref.adapter:
                            encoder_hidden_states = model_ref.adapter(encoder_hidden_states)
                    
                    # Get features from decode head
                    _, decoder_features = model_ref.decode_head(encoder_hidden_states, return_features=True)
                    
                    # Resize for efficiency
                    decoder_features_small = F.interpolate(
                        decoder_features, size=small_size, mode='trilinear', align_corners=False
                    )
                    main_logit_small = F.interpolate(
                        main_logit, size=small_size, mode='trilinear', align_corners=False
                    )
                    gt_small = F.interpolate(
                        gt.float(), size=small_size, mode='nearest'
                    )
                    uncertainty_small = F.interpolate(
                        uncertainty_map, size=small_size, mode='trilinear', align_corners=False
                    )
                    
                    # Compute CUCL losses
                    cucl_contrastive, cucl_calib, cucl_info = model_ref.cucl_module(
                        decoder_features_small,
                        main_logit_small,
                        gt_small,
                        uncertainty_small
                    )
                    
                    cucl_contrastive_value = cucl_contrastive.item()
                    cucl_calibration_value = cucl_calib.item()
                    
                    # Add to total loss
                    cucl_contrastive_weight_now = cucl_contrastive_weight * aux_loss_scale
                    cucl_calibration_weight_now = cucl_calibration_weight * aux_loss_scale
                    loss = loss + cucl_contrastive_weight_now * cucl_contrastive + cucl_calibration_weight_now * cucl_calib
                    
                    # Log CUCL info
                    if log_writer is not None and data_iter_step % 50 == 0:
                        log_writer.add_scalar("cucl/contrastive_loss", cucl_contrastive_value, epoch_1000x)
                        log_writer.add_scalar("cucl/calibration_loss", cucl_calibration_value, epoch_1000x)
                        log_writer.add_scalar("cucl/temperature", cucl_info.get('temperature', 0.1), epoch_1000x)
                        log_writer.add_scalar("cucl/num_anchors", cucl_info.get('num_anchors', 0), epoch_1000x)
                        log_writer.add_scalar("cucl/avg_pos_sim", cucl_info.get('avg_pos_sim', 0), epoch_1000x)
                        log_writer.add_scalar("cucl/avg_neg_sim", cucl_info.get('avg_neg_sim', 0), epoch_1000x)



            # ========== Modulation Parameter Tracking ==========
            if use_uncertainty_modulation:
                # Get uncertainty modulation parameters
                if hasattr(model, 'module'):
                    # Get boundary attention parameters if they exist
                    if hasattr(model.module, 'boundary_attention'):
                        alpha_values = [ba.alpha.item() for ba in model.module.boundary_attention]
                        beta_values = [ba.beta.item() for ba in model.module.boundary_attention]
                else:
                    if hasattr(model, 'boundary_attention'):
                        alpha_values = [ba.alpha.item() for ba in model.boundary_attention]
                        beta_values = [ba.beta.item() for ba in model.boundary_attention]

                metric_logger.update(boundary_loss=boundary_loss_value)
                metric_logger.update(seg_loss=seg_loss_value)
                metric_logger.update(calibration_loss=calibration_loss_value)

                # Log to TensorBoard
                if log_writer is not None:
                    log_writer.add_scalar("modulation/uncertainty_mean", avg_uncertainty, epoch_1000x)
                    log_writer.add_scalar("loss/aux_loss_scale", aux_loss_scale, epoch_1000x)
                    log_writer.add_scalar("loss/boundary_loss", boundary_loss_value, epoch_1000x)
                    log_writer.add_scalar("loss/calibration_loss", calibration_loss_value, epoch_1000x)
                    log_writer.add_scalar("loss/seg_loss", seg_loss_value, epoch_1000x)

                    # Log boundary attention parameters
                    if hasattr(model, 'boundary_attention') or (hasattr(model, 'module') and hasattr(model.module, 'boundary_attention')):
                        for i, (alpha, beta) in enumerate(zip(alpha_values, beta_values)):
                            log_writer.add_scalar(f"boundary_attention/alpha_scale{i}", alpha, epoch_1000x)
                            log_writer.add_scalar(f"boundary_attention/beta_scale{i}", beta, epoch_1000x)

                        # NEW: Update CUCL meters
            if use_cucl:
                metric_logger.update(cucl_contrastive=cucl_contrastive_value)
                metric_logger.update(cucl_calibration=cucl_calibration_value)

        else:
            # Standard training without uncertainty
            logit = model(img)

            if isinstance(logit, list):
                loss = loss_cal(logit[0], gt) + 0.4 * loss_cal(logit[1], gt)
            else:
                loss = loss_cal(logit, gt)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # ========== Backward Pass ==========
        optimizer.zero_grad()
        loss.backward()

        # ========== Gradient Debugging (every 50 iterations) ==========
        if use_uncertainty_modulation and data_iter_step % 50 == 0:
            if hasattr(model, 'module'):
                boundary_attention = model.module.boundary_attention
                # uncertainty_projs = model.module.uncertainty_projs
            else:
                boundary_attention = model.boundary_attention
                # uncertainty_projs = model.uncertainty_projs

            # Check boundary attention gradients
            for i, ba in enumerate(boundary_attention):
                print(f"  boundary_attention[{i}].alpha: {ba.alpha.item():.4f}, grad: {ba.alpha.grad}")
                print(f"  boundary_attention[{i}].beta: {ba.beta.item():.4f}, grad: {ba.beta.grad}")



        optimizer.step()

        # ========== Logging ==========
        metric_logger.update(loss=loss_value)
        if use_mc_dropout:
            metric_logger.update(uncertainty=avg_uncertainty)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            if use_mc_dropout:
                log_writer.add_scalar("train_uncertainty", avg_uncertainty, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

        # ========== Save Uncertainty Visualizations ==========
        if use_mc_dropout and uncertainty_map is not None:
            save_vis = False
            if data_iter_step % 100 == 0:
                save_vis = True
            if save_vis:
                save_uncertainty_visualization(
                    img, gt, logit, uncertainty_map,
                    epoch, data_iter_step, unc_vis_dir, args,
                    log_writer=log_writer, epoch_1000x=epoch_1000x,
                    model=model
                )

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validation(model, data_loader_val, device, epoch, use_mc_dropout, mc_samples, use_uncertainty_modulation, args):
    """
    Validation function.
    
    Note: uncertainty_modulation is DISABLED during validation for consistent metrics.
    MC dropout is only computed every 5 epochs for monitoring uncertainty quality.
    """
    model.eval()
    
    if args.out_channels == 1:
        dice_loss = DiceLoss(sigmoid=True)
    else:
        dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)

    # Only compute uncertainty periodically (expensive)
    if use_mc_dropout and epoch % 5 == 0:
        compute_uncertainty = True
        enable_dropout(model)  # Enable dropout for MC sampling
        print(f"Computing uncertainty during validation for epoch {epoch}")
    else:
        compute_uncertainty = False
        print(f"Skipping uncertainty computation during validation for epoch {epoch}")

    # Check if CUCL is enabled for calibration metrics
    use_cucl = getattr(args, 'use_cucl', False)
    calibration_errors = []

    with torch.no_grad():
        loss_summary = []
        for idx, (img, gt, _) in enumerate(data_loader_val):
            img, gt = img.to(device), gt.to(device)
            
            if use_mc_dropout and compute_uncertainty:
                # NOTE: use_uncertainty_modulation=False for validation!
                output = model(
                    img, 
                    compute_uncertainty=True, 
                    mc_samples=mc_samples, 
                    use_uncertainty_modulation=False  # Disabled for validation
                )
                if isinstance(output, tuple):
                    mask, uncertainty_map = output
                else:
                    mask = output
                    uncertainty_map = None
            else:
                # Standard forward pass (no uncertainty computation)
                mask = model(img)
                uncertainty_map = None

            # Handle list output (main + auxiliary)
            if isinstance(mask, list):
                mask = mask[0]

            loss = dice_loss(mask, gt)
            loss_summary.append(loss.detach().cpu().numpy())
            
            # Compute calibration error if CUCL is enabled and uncertainty is available
            if use_cucl and uncertainty_map is not None:
                calib_error, _, _ = compute_calibration_error(
                    mask, gt, uncertainty_map, args.out_channels
                )
                calibration_errors.append(calib_error)

            print(
                f"epoch: {epoch}/{args.epochs}, iter: {idx}/{len(data_loader_val)}"
                f" loss: {loss_summary[-1].flatten()[0]:.4f}"
            )
        
        avg_loss = np.mean(loss_summary)
        print(f"Averaged val stats: {avg_loss:.4f}")
        
        # Print calibration metrics if available
        if calibration_errors:
            avg_calib_error = np.mean(calibration_errors)
            print(f"Averaged calibration error: {avg_calib_error:.4f}")
    
    # Restore full eval mode
    model.eval()
        
    return avg_loss


def test_old(model, test_loader, args, use_mc_dropout, mc_samples, use_uncertainty_modulation, sliding_window=False):
    """
    Test function with uncertainty evaluation.
    
    Note: uncertainty_modulation is DISABLED during testing.
    MC dropout IS enabled to compute uncertainty metrics.
    """
    model.eval()
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    model.load_state_dict(torch.load(filepath_best)["model"])
    print("loaded for test from ", filepath_best)
    
    # Enable dropout for MC uncertainty estimation
    if use_mc_dropout:
        enable_dropout(model)
        print("Enabled dropout for MC uncertainty estimation during testing")
    
    dice_meter = DiceMeter(args)
    hausdorff_meter = HausdorffMeter(args)
    sd_meter = SurfaceDistanceMeter(args)
    
    # Uncertainty calibration metrics
    use_cucl = getattr(args, 'use_cucl', False)
    calibration_errors = []
    uncertainty_values = []
    error_rates = []
    
    log_stats = {}
    
    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            img, gt = img.to(args.device), gt.to(args.device)
            if idx == 0:
                print("image shape", img.shape)
            
            uncertainty_map = None
            
            if sliding_window:
                # Sliding window inference
                def predictor(x):
                    if use_mc_dropout:
                        # NOTE: use_uncertainty_modulation=False for testing!
                        output = model(x, compute_uncertainty=True, mc_samples=mc_samples, 
                                      use_uncertainty_modulation=False)
                        if isinstance(output, tuple):
                            logits, _ = output
                        else:
                            logits = output
                    else:
                        logits = model(x)
                    return unwrap_logits(logits)
                
                pred = sliding_window_inference(
                    img, args.crop_spatial_size, 4, predictor, overlap=0.5
                )
                
                # Get uncertainty from single forward pass (approximate)
                if use_mc_dropout:
                    output = model(img, compute_uncertainty=True, mc_samples=mc_samples,
                                  use_uncertainty_modulation=False)
                    if isinstance(output, tuple):
                        _, uncertainty_map = output
            else:
                if use_mc_dropout:
                    # NOTE: use_uncertainty_modulation=False for testing!
                    output = model(
                        img, 
                        compute_uncertainty=True, 
                        mc_samples=mc_samples, 
                        use_uncertainty_modulation=False  # Disabled for testing
                    )
                    if isinstance(output, tuple):
                        pred, uncertainty_map = output
                    else:
                        pred = output
                else:
                    pred = model(img)

            # Handle list output
            pred = unwrap_logits(pred)

            # Convert to class predictions
            if args.num_classes == 1:
                pred_binary = torch.sigmoid(pred) > 0.5
            else:
                pred_softmax = torch.softmax(pred, dim=1)
                pred_binary = torch.argmax(pred_softmax, dim=1, keepdim=True)
            
            # Update meters
            dice_meter.update(pred_binary, gt)
            hausdorff_meter.update(pred_binary, gt)
            sd_meter.update(pred_binary, gt)
            
            # Compute uncertainty calibration metrics
            if use_mc_dropout and uncertainty_map is not None:
                calib_error, unc_mean, error_rate = compute_calibration_error(
                    pred, gt, uncertainty_map, args.out_channels
                )
                calibration_errors.append(calib_error)
                uncertainty_values.append(unc_mean)
                error_rates.append(error_rate)

    # Print metrics
    print("- Test metrics Dice: ")
    dice_class_avg, dice_avg, dice_variability = dice_meter.get_average()
    print("Class wise: ", dice_class_avg)
    print("Avg.: ", dice_avg)
    print("StdDev:", dice_variability['std'])
    print(f"Percentiles: 25%={dice_variability['percentiles']['p25']}, "
          f"50%={dice_variability['percentiles']['p50']}, "
          f"75%={dice_variability['percentiles']['p75']}")

    print("- Test metrics Hausdorff95: ")
    hsd_class_avg, hsd_avg, _ = hausdorff_meter.get_average()
    print("Class wise: ", hsd_class_avg)
    print("Avg.: ", hsd_avg)

    print("- Test metrics SurfaceDistance: ")
    sd_class_avg, sd_avg, _ = sd_meter.get_average()
    print("Class wise: ", sd_class_avg)
    print("Avg.: ", sd_avg)

    # Build log stats
    log_stats = {
        "dice_class_avg": dice_class_avg.tolist() if isinstance(dice_class_avg, np.ndarray) else dice_class_avg,
        "dice_avg": dice_avg.tolist() if isinstance(dice_avg, np.ndarray) else dice_avg,
        "dice_std": dice_variability["std"].tolist() if isinstance(dice_variability["std"], np.ndarray) else dice_variability["std"],
        "dice_p25": dice_variability["percentiles"]["p25"].tolist() if isinstance(dice_variability["percentiles"]["p25"], np.ndarray) else dice_variability["percentiles"]["p25"],
        "dice_p50": dice_variability["percentiles"]["p50"].tolist() if isinstance(dice_variability["percentiles"]["p50"], np.ndarray) else dice_variability["percentiles"]["p50"],
        "dice_p75": dice_variability["percentiles"]["p75"].tolist() if isinstance(dice_variability["percentiles"]["p75"], np.ndarray) else dice_variability["percentiles"]["p75"],
        "hsd_class_avg": hsd_class_avg.tolist() if isinstance(hsd_class_avg, np.ndarray) else hsd_class_avg,
        "hsd_avg": hsd_avg.tolist() if isinstance(hsd_avg, np.ndarray) else hsd_avg,
        "sd_class_avg": sd_class_avg.tolist() if isinstance(sd_class_avg, np.ndarray) else sd_class_avg,
        "sd_avg": sd_avg.tolist() if isinstance(sd_avg, np.ndarray) else sd_avg,
    }

    # Add uncertainty calibration metrics
    if use_mc_dropout and calibration_errors:
        avg_calib_error = np.mean(calibration_errors)
        avg_uncertainty = np.mean(uncertainty_values)
        avg_error_rate = np.mean(error_rates)
        
        # Compute uncertainty-error correlation
        if len(uncertainty_values) > 2:
            try:
                from scipy.stats import spearmanr
                correlation, p_value = spearmanr(uncertainty_values, error_rates)
            except ImportError:
                correlation = np.corrcoef(uncertainty_values, error_rates)[0, 1]
                p_value = 0.0
        else:
            correlation = 0.0
            p_value = 1.0
        
        print("\n- Uncertainty Calibration Metrics:")
        print(f"  Average Calibration Error (ECE): {avg_calib_error:.4f}")
        print(f"  Average Uncertainty: {avg_uncertainty:.4f}")
        print(f"  Average Error Rate: {avg_error_rate:.4f}")
        print(f"  Uncertainty-Error Correlation: {correlation:.4f} (p={p_value:.4f})")
        
        log_stats.update({
            "calibration_error": avg_calib_error,
            "mean_uncertainty": avg_uncertainty,
            "mean_error_rate": avg_error_rate,
            "uncertainty_error_correlation": correlation,
        })

    return log_stats


def test(model, test_loader, args, use_mc_dropout, mc_samples, use_uncertainty_modulation, sliding_window=False):
    """
    Test function with uncertainty evaluation.
    
    Note: uncertainty_modulation is DISABLED during testing.
    MC dropout IS enabled to compute uncertainty metrics.
    """
    model.eval()
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    model.load_state_dict(torch.load(filepath_best)["model"])
    print("loaded for test from ", filepath_best)
    
    # Enable dropout for MC uncertainty estimation
    if use_mc_dropout:
        enable_dropout(model)
        print("Enabled dropout for MC uncertainty estimation during testing")
    
    dice_meter = DiceMeter(args)
    hausdorff_meter = HausdorffMeter(args)
    sd_meter = SurfaceDistanceMeter(args)
    
    # Uncertainty calibration metrics
    use_cucl = getattr(args, 'use_cucl', False)
    calibration_errors = []
    uncertainty_values = []
    error_rates = []
    
    log_stats = {}
    
    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            img, gt = img.to(args.device), gt.to(args.device)
            if idx == 0:
                print("image shape", img.shape)
            
            uncertainty_map = None
            
            if sliding_window:
                # Sliding window inference
                def predictor(x):
                    if use_mc_dropout:
                        # NOTE: use_uncertainty_modulation=False for testing!
                        output = model(x, compute_uncertainty=True, mc_samples=mc_samples, 
                                      use_uncertainty_modulation=False)
                        if isinstance(output, tuple):
                            logits, _ = output
                        else:
                            logits = output
                    else:
                        logits = model(x)
                    return unwrap_logits(logits)
                
                pred = sliding_window_inference(
                    img, args.crop_spatial_size, 4, predictor, overlap=0.5
                )
                
                # For sliding window, compute uncertainty using entropy of predictions
                # instead of trying to pass full image through MC dropout
                if use_mc_dropout:
                    # Compute entropy-based uncertainty from the prediction
                    if args.out_channels == 1:
                        prob = torch.sigmoid(pred)
                        # Binary entropy
                        uncertainty_map = -prob * torch.log(prob + 1e-8) - (1 - prob) * torch.log(1 - prob + 1e-8)
                    else:
                        prob = torch.softmax(pred, dim=1)
                        # Multi-class entropy
                        uncertainty_map = -(prob * torch.log(prob + 1e-8)).sum(dim=1, keepdim=True)
            else:
                if use_mc_dropout:
                    # NOTE: use_uncertainty_modulation=False for testing!
                    output = model(
                        img, 
                        compute_uncertainty=True, 
                        mc_samples=mc_samples, 
                        use_uncertainty_modulation=False  # Disabled for testing
                    )
                    if isinstance(output, tuple):
                        pred, uncertainty_map = output
                    else:
                        pred = output
                else:
                    pred = model(img)

            # Handle list output
            pred = unwrap_logits(pred)

            # Convert to class predictions
            if args.out_channels == 1:
                pred_binary = torch.sigmoid(pred) > 0.5
            else:
                pred_softmax = torch.softmax(pred, dim=1)
                pred_binary = torch.argmax(pred_softmax, dim=1, keepdim=True)
            
            # Update meters
            dice_meter.update(pred_binary, gt)
            hausdorff_meter.update(pred_binary, gt)
            sd_meter.update(pred_binary, gt)
            
            # Compute uncertainty calibration metrics
            if use_mc_dropout and uncertainty_map is not None:
                calib_error, unc_mean, error_rate = compute_calibration_error(
                    pred, gt, uncertainty_map, args.out_channels
                )
                calibration_errors.append(calib_error)
                uncertainty_values.append(unc_mean)
                error_rates.append(error_rate)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(test_loader)} samples")

    # Print metrics
    print("- Test metrics Dice: ")
    dice_class_avg, dice_avg, dice_variability = dice_meter.get_average()
    print("Class wise: ", dice_class_avg)
    print("Avg.: ", dice_avg)
    print("StdDev:", dice_variability['std'])
    print(f"Percentiles: 25%={dice_variability['percentiles']['p25']}, "
          f"50%={dice_variability['percentiles']['p50']}, "
          f"75%={dice_variability['percentiles']['p75']}")

    print("- Test metrics Hausdorff95: ")
    hsd_class_avg, hsd_avg, _ = hausdorff_meter.get_average()
    print("Class wise: ", hsd_class_avg)
    print("Avg.: ", hsd_avg)

    print("- Test metrics SurfaceDistance: ")
    sd_class_avg, sd_avg, _ = sd_meter.get_average()
    print("Class wise: ", sd_class_avg)
    print("Avg.: ", sd_avg)

    # Build log stats
    log_stats = {
        "dice_class_avg": dice_class_avg.tolist() if isinstance(dice_class_avg, np.ndarray) else dice_class_avg,
        "dice_avg": dice_avg.tolist() if isinstance(dice_avg, np.ndarray) else dice_avg,
        "dice_std": dice_variability["std"].tolist() if isinstance(dice_variability["std"], np.ndarray) else dice_variability["std"],
        "dice_p25": dice_variability["percentiles"]["p25"].tolist() if isinstance(dice_variability["percentiles"]["p25"], np.ndarray) else dice_variability["percentiles"]["p25"],
        "dice_p50": dice_variability["percentiles"]["p50"].tolist() if isinstance(dice_variability["percentiles"]["p50"], np.ndarray) else dice_variability["percentiles"]["p50"],
        "dice_p75": dice_variability["percentiles"]["p75"].tolist() if isinstance(dice_variability["percentiles"]["p75"], np.ndarray) else dice_variability["percentiles"]["p75"],
        "hsd_class_avg": hsd_class_avg.tolist() if isinstance(hsd_class_avg, np.ndarray) else hsd_class_avg,
        "hsd_avg": hsd_avg.tolist() if isinstance(hsd_avg, np.ndarray) else hsd_avg,
        "sd_class_avg": sd_class_avg.tolist() if isinstance(sd_class_avg, np.ndarray) else sd_class_avg,
        "sd_avg": sd_avg.tolist() if isinstance(sd_avg, np.ndarray) else sd_avg,
    }

    # Add uncertainty calibration metrics
    if use_mc_dropout and calibration_errors:
        avg_calib_error = np.mean(calibration_errors)
        avg_uncertainty = np.mean(uncertainty_values)
        avg_error_rate = np.mean(error_rates)
        
        # Compute uncertainty-error correlation
        if len(uncertainty_values) > 2:
            try:
                from scipy.stats import spearmanr
                correlation, p_value = spearmanr(uncertainty_values, error_rates)
            except ImportError:
                correlation = np.corrcoef(uncertainty_values, error_rates)[0, 1]
                p_value = 0.0
        else:
            correlation = 0.0
            p_value = 1.0
        
        print("\n- Uncertainty Calibration Metrics:")
        print(f"  Average Calibration Error (ECE): {avg_calib_error:.4f}")
        print(f"  Average Uncertainty: {avg_uncertainty:.4f}")
        print(f"  Average Error Rate: {avg_error_rate:.4f}")
        print(f"  Uncertainty-Error Correlation: {correlation:.4f} (p={p_value:.4f})")
        
        log_stats.update({
            "calibration_error": avg_calib_error,
            "mean_uncertainty": avg_uncertainty,
            "mean_error_rate": avg_error_rate,
            "uncertainty_error_correlation": correlation,
        })

    return log_stats

# Add this function after the existing test() function

def test_with_comparison(
    model_baseline, 
    model_uncertainty, 
    test_loader, 
    args, 
    use_mc_dropout=True,
    mc_samples=5,
    sliding_window=True,
    save_dir=None,
    threshold_hd95=0.5,  # HD95 difference threshold to consider "significant"
    threshold_sd=0.1,    # Surface distance difference threshold
    top_k=10,            # Number of top improved/degraded samples to visualize
    ensemble_weight_uncertainty=0.5,
):
    """
    Test both models and visualize samples where boundary quality differs significantly.
    
    Args:
        model_baseline: Baseline model (no uncertainty)
        model_uncertainty: Uncertainty-guided model
        test_loader: Test data loader
        args: Arguments
        threshold_hd95: Minimum HD95 difference to consider significant
        threshold_sd: Minimum surface distance difference to consider significant
        top_k: Number of best/worst samples to visualize
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    from scipy import ndimage
    
    if save_dir is None:
        save_dir = os.path.join(args.output_dir, "boundary_comparison")
    os.makedirs(save_dir, exist_ok=True)
    
    # Set models to eval mode
    model_baseline.eval()
    model_uncertainty.eval()
    
    if use_mc_dropout:
        enable_dropout(model_uncertainty)
    
    # Storage for per-sample metrics
    sample_metrics = []
    
    print("Computing per-sample boundary metrics for both models...")

    include_background = args.out_channels == 1
    spacing = tuple(args.spacing) if hasattr(args, 'spacing') else (1.0, 0.5, 0.5)

    def to_one_hot(pred_mask, gt_mask):
        if args.out_channels > 1:
            pred_oh = one_hot(pred_mask.long(), num_classes=args.out_channels + 1)
            gt_oh = one_hot(gt_mask.long(), num_classes=args.out_channels + 1)
        else:
            pred_oh = pred_mask.float()
            gt_oh = gt_mask.float()
        return pred_oh, gt_oh

    def finite_mean(metric_tensor):
        flat = metric_tensor.reshape(-1)
        finite = flat[~torch.isnan(flat)]
        if finite.numel() == 0:
            return None
        return finite.mean().item()
    
    with torch.no_grad():
        for idx, (img, gt, sample_id) in enumerate(test_loader):
            img, gt = img.to(args.device), gt.to(args.device)
            
            # Get predictions from both models
            if sliding_window:
                # Baseline prediction
                def predictor_baseline(x):
                    logits = model_baseline(x)
                    if isinstance(logits, list):
                        logits = logits[0]
                    return logits
                
                pred_baseline = sliding_window_inference(
                    img, args.crop_spatial_size, 4, predictor_baseline, overlap=0.5
                )
                
                # Uncertainty model prediction
                def predictor_uncertainty(x):
                    output = model_uncertainty(x, compute_uncertainty=True, 
                                              mc_samples=mc_samples,
                                              use_uncertainty_modulation=False)
                    if isinstance(output, tuple):
                        logits, _ = output
                    else:
                        logits = output
                    if isinstance(logits, list):
                        logits = logits[0]
                    return logits
                
                pred_uncertainty = sliding_window_inference(
                    img, args.crop_spatial_size, 4, predictor_uncertainty, overlap=0.5
                )
                
                # Compute uncertainty map using entropy
                if args.out_channels == 1:
                    prob = torch.sigmoid(pred_uncertainty)
                    uncertainty_map = -prob * torch.log(prob + 1e-8) - (1 - prob) * torch.log(1 - prob + 1e-8)
                else:
                    prob = torch.softmax(pred_uncertainty, dim=1)
                    uncertainty_map = -(prob * torch.log(prob + 1e-8)).sum(dim=1, keepdim=True)
            else:
                pred_baseline = model_baseline(img)
                if isinstance(pred_baseline, list):
                    pred_baseline = pred_baseline[0]
                
                output = model_uncertainty(img, compute_uncertainty=True,
                                          mc_samples=mc_samples,
                                          use_uncertainty_modulation=False)
                if isinstance(output, tuple):
                    pred_uncertainty, uncertainty_map = output
                else:
                    pred_uncertainty = output
                    uncertainty_map = None
                
                if isinstance(pred_uncertainty, list):
                    pred_uncertainty = pred_uncertainty[0]
            
            w_unc = float(min(max(ensemble_weight_uncertainty, 0.0), 1.0))
            w_base = 1.0 - w_unc

            # Convert to probabilities and discrete predictions
            if args.out_channels == 1:
                p_baseline = torch.sigmoid(pred_baseline)
                p_uncertainty = torch.sigmoid(pred_uncertainty)
                p_baseline_2c = torch.cat([1.0 - p_baseline, p_baseline], dim=1)
                p_uncertainty_2c = torch.cat([1.0 - p_uncertainty, p_uncertainty], dim=1)

                p_ensemble_2c = w_base * p_baseline_2c + w_unc * p_uncertainty_2c

                pred_baseline_binary = (torch.sigmoid(pred_baseline) > 0.5).float()
                pred_uncertainty_binary = (torch.sigmoid(pred_uncertainty) > 0.5).float()
                pred_ensemble_binary = (p_ensemble_2c[:, 1:2] > 0.5).float()
            else:
                p_baseline = torch.softmax(pred_baseline, dim=1)
                p_uncertainty = torch.softmax(pred_uncertainty, dim=1)

                p_ensemble = w_base * p_baseline + w_unc * p_uncertainty

                pred_baseline_binary = torch.argmax(pred_baseline, dim=1, keepdim=True).float()
                pred_uncertainty_binary = torch.argmax(pred_uncertainty, dim=1, keepdim=True).float()
                pred_ensemble_binary = torch.argmax(p_ensemble, dim=1, keepdim=True).float()

            for batch_idx in range(gt.shape[0]):
                gt_sample = gt[batch_idx:batch_idx + 1]
                pred_base_sample = pred_baseline_binary[batch_idx:batch_idx + 1]
                pred_unc_sample = pred_uncertainty_binary[batch_idx:batch_idx + 1]
                pred_ensemble_sample = pred_ensemble_binary[batch_idx:batch_idx + 1]

                pred_base_oh, gt_oh = to_one_hot(pred_base_sample, gt_sample)
                pred_unc_oh, _ = to_one_hot(pred_unc_sample, gt_sample)
                pred_ensemble_oh, _ = to_one_hot(pred_ensemble_sample, gt_sample)

                dice_base = compute_dice(
                    pred_base_oh, gt_oh, include_background=include_background, ignore_empty=True
                )
                dice_unc = compute_dice(
                    pred_unc_oh, gt_oh, include_background=include_background, ignore_empty=True
                )
                dice_ensemble = compute_dice(
                    pred_ensemble_oh, gt_oh, include_background=include_background, ignore_empty=True
                )

                hd95_base = compute_hausdorff_distance(
                    pred_base_oh, gt_oh, include_background=include_background,
                    percentile=95, spacing=spacing
                )
                hd95_unc = compute_hausdorff_distance(
                    pred_unc_oh, gt_oh, include_background=include_background,
                    percentile=95, spacing=spacing
                )
                hd95_ensemble = compute_hausdorff_distance(
                    pred_ensemble_oh, gt_oh, include_background=include_background,
                    percentile=95, spacing=spacing
                )

                sd_base = compute_average_surface_distance(
                    pred_base_oh, gt_oh, include_background=include_background, spacing=spacing
                )
                sd_unc = compute_average_surface_distance(
                    pred_unc_oh, gt_oh, include_background=include_background, spacing=spacing
                )
                sd_ensemble = compute_average_surface_distance(
                    pred_ensemble_oh, gt_oh, include_background=include_background, spacing=spacing
                )

                avg_dice_base = finite_mean(dice_base)
                avg_dice_unc = finite_mean(dice_unc)
                avg_dice_ensemble = finite_mean(dice_ensemble)
                avg_hd95_base = finite_mean(hd95_base)
                avg_hd95_unc = finite_mean(hd95_unc)
                avg_hd95_ensemble = finite_mean(hd95_ensemble)
                avg_sd_base = finite_mean(sd_base)
                avg_sd_unc = finite_mean(sd_unc)
                avg_sd_ensemble = finite_mean(sd_ensemble)

                if (
                    avg_dice_base is None or avg_dice_unc is None or
                    avg_hd95_base is None or avg_hd95_unc is None or
                    avg_sd_base is None or avg_sd_unc is None or
                    avg_dice_ensemble is None or
                    avg_hd95_ensemble is None or
                    avg_sd_ensemble is None
                ):
                    continue

                method_hd95 = {
                    'baseline': avg_hd95_base,
                    'cucl': avg_hd95_unc,
                    'ensemble': avg_hd95_ensemble,
                }
                best_method_hd95 = min(method_hd95, key=method_hd95.get)

                sample_metrics.append({
                    'idx': idx,
                    'sample_id': sample_id,
                    'img': img.cpu(),
                    'gt': gt.cpu(),
                    'pred_baseline': pred_baseline_binary.cpu(),
                    'pred_uncertainty': pred_uncertainty_binary.cpu(),
                    'uncertainty_map': uncertainty_map.cpu() if uncertainty_map is not None else None,
                    'dice_baseline': avg_dice_base,
                    'dice_uncertainty': avg_dice_unc,
                    'dice_diff': avg_dice_unc - avg_dice_base,  # Positive = uncertainty is better
                    'hd95_baseline': avg_hd95_base,
                    'hd95_uncertainty': avg_hd95_unc,
                    'hd95_diff': avg_hd95_base - avg_hd95_unc,  # Positive = uncertainty is better
                    'sd_baseline': avg_sd_base,
                    'sd_uncertainty': avg_sd_unc,
                    'sd_diff': avg_sd_base - avg_sd_unc,  # Positive = uncertainty is better
                    'dice_ensemble': avg_dice_ensemble,
                    'hd95_ensemble': avg_hd95_ensemble,
                    'sd_ensemble': avg_sd_ensemble,
                    'hd95_diff_ensemble': avg_hd95_base - avg_hd95_ensemble,
                    'dice_diff_ensemble': avg_dice_ensemble - avg_dice_base,
                    'best_method_hd95': best_method_hd95,
                })
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(test_loader)} samples")
    
    print(f"\nTotal samples with valid metrics: {len(sample_metrics)}")
    
    # Sort by HD95 improvement (CUCL model improvement)
    sorted_by_improvement = sorted(sample_metrics, key=lambda x: x['hd95_diff'], reverse=True)
    
    # Get top improved and top degraded samples
    top_improved = [s for s in sorted_by_improvement if s['hd95_diff'] > threshold_hd95][:top_k]
    top_degraded = [s for s in sorted_by_improvement if s['hd95_diff'] < -threshold_hd95][-top_k:][::-1]
    
    print(f"\nSamples with HD95 improvement > {threshold_hd95}: {len([s for s in sample_metrics if s['hd95_diff'] > threshold_hd95])}")
    print(f"Samples with HD95 degradation > {threshold_hd95}: {len([s for s in sample_metrics if s['hd95_diff'] < -threshold_hd95])}")

    mean_hd95 = lambda key: float(np.mean([s[key] for s in sample_metrics])) if sample_metrics else float('nan')
    print("\n=== Mean HD95 by method (lower is better) ===")
    print(f"Baseline: {mean_hd95('hd95_baseline'):.4f}")
    print(f"CUCL: {mean_hd95('hd95_uncertainty'):.4f}")
    print(f"Ensemble (baseline+CUCL): {mean_hd95('hd95_ensemble'):.4f}")
    
    # Visualize top improved samples
    print(f"\n=== Visualizing Top {len(top_improved)} Improved Samples ===")
    for rank, sample in enumerate(top_improved):
        visualize_boundary_comparison(
            sample, rank, "improved", save_dir, args
        )
    
    # Visualize top degraded samples
    print(f"\n=== Visualizing Top {len(top_degraded)} Degraded Samples ===")
    for rank, sample in enumerate(top_degraded):
        visualize_boundary_comparison(
            sample, rank, "degraded", save_dir, args
        )
    
    # Create summary plot
    create_comparison_summary(sample_metrics, save_dir, threshold_hd95, threshold_sd)
    
    return sample_metrics


def visualize_boundary_comparison(sample, rank, category, save_dir, args):
    """
    Visualize a single sample comparing baseline vs uncertainty model boundaries.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    from scipy import ndimage
    
    img = sample['img'][0].numpy()
    gt = sample['gt'][0].numpy()
    pred_base = sample['pred_baseline'][0].numpy()
    pred_unc = sample['pred_uncertainty'][0].numpy()
    unc_map = sample['uncertainty_map'][0].numpy() if sample['uncertainty_map'] is not None else None
    
    # Get middle slice for 3D volumes
    if img.ndim == 4:
        mid_slice = img.shape[1] // 2
        img_slice = img[0, mid_slice]
        gt_slice = gt[0, mid_slice]
        pred_base_slice = pred_base[0, mid_slice]
        pred_unc_slice = pred_unc[0, mid_slice]
        unc_slice = unc_map[0, mid_slice] if unc_map is not None else None
    else:
        img_slice = img[0]
        gt_slice = gt[0]
        pred_base_slice = pred_base[0]
        pred_unc_slice = pred_unc[0]
        unc_slice = unc_map[0] if unc_map is not None else None
    
    # Extract boundaries using edge detection
    def get_boundary(mask):
        if mask.max() == 0:
            return np.zeros_like(mask)
        eroded = ndimage.binary_erosion(mask > 0.5)
        boundary = (mask > 0.5).astype(float) - eroded.astype(float)
        return boundary
    
    gt_boundary = get_boundary(gt_slice)
    pred_base_boundary = get_boundary(pred_base_slice)
    pred_unc_boundary = get_boundary(pred_unc_slice)
    
    # Compute error maps
    error_base = np.abs(pred_base_slice - gt_slice)
    error_unc = np.abs(pred_unc_slice - gt_slice)
    error_diff = error_base - error_unc  # Positive = uncertainty is better
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Title
    improvement_str = "IMPROVED" if category == "improved" else "DEGRADED"
    hd95_change = sample['hd95_diff']
    sd_change = sample['sd_diff']
    fig.suptitle(
        f"{improvement_str} Sample (Rank #{rank+1})\n"
        f"HD95: {sample['hd95_baseline']:.2f} → {sample['hd95_uncertainty']:.2f} "
        f"(Δ={hd95_change:+.2f})\n"
        f"SD: {sample['sd_baseline']:.3f} → {sample['sd_uncertainty']:.3f} "
        f"(Δ={sd_change:+.3f})",
        fontsize=14, fontweight='bold'
    )
    
    # Row 1: Input, GT, Baseline Pred, Uncertainty Pred
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_slice, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_base_slice, cmap='gray')
    axes[0, 2].set_title(f'Baseline Pred\nHD95={sample["hd95_baseline"]:.2f}')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(pred_unc_slice, cmap='gray')
    axes[0, 3].set_title(f'Uncertainty Pred\nHD95={sample["hd95_uncertainty"]:.2f}')
    axes[0, 3].axis('off')
    
    # Row 2: Boundary overlays
    # GT boundary overlay on image
    axes[1, 0].imshow(img_slice, cmap='gray')
    axes[1, 0].contour(gt_slice, levels=[0.5], colors='green', linewidths=2)
    axes[1, 0].set_title('GT Boundary (green)')
    axes[1, 0].axis('off')
    
    # Baseline boundary vs GT
    axes[1, 1].imshow(img_slice, cmap='gray')
    axes[1, 1].contour(gt_slice, levels=[0.5], colors='green', linewidths=2)
    axes[1, 1].contour(pred_base_slice, levels=[0.5], colors='red', linewidths=2)
    axes[1, 1].set_title('Baseline (red) vs GT (green)')
    axes[1, 1].axis('off')
    
    # Uncertainty boundary vs GT
    axes[1, 2].imshow(img_slice, cmap='gray')
    axes[1, 2].contour(gt_slice, levels=[0.5], colors='green', linewidths=2)
    axes[1, 2].contour(pred_unc_slice, levels=[0.5], colors='blue', linewidths=2)
    axes[1, 2].set_title('Uncertainty (blue) vs GT (green)')
    axes[1, 2].axis('off')
    
    # All three boundaries
    axes[1, 3].imshow(img_slice, cmap='gray')
    axes[1, 3].contour(gt_slice, levels=[0.5], colors='green', linewidths=2, linestyles='solid')
    axes[1, 3].contour(pred_base_slice, levels=[0.5], colors='red', linewidths=1.5, linestyles='dashed')
    axes[1, 3].contour(pred_unc_slice, levels=[0.5], colors='blue', linewidths=1.5, linestyles='dotted')
    axes[1, 3].set_title('GT(green) vs Base(red) vs Unc(blue)')
    axes[1, 3].axis('off')
    
    # Row 3: Error maps and uncertainty
    # Baseline error
    im1 = axes[2, 0].imshow(error_base, cmap='Reds', vmin=0, vmax=1)
    axes[2, 0].set_title('Baseline Error')
    axes[2, 0].axis('off')
    plt.colorbar(im1, ax=axes[2, 0], fraction=0.046)
    
    # Uncertainty error
    im2 = axes[2, 1].imshow(error_unc, cmap='Reds', vmin=0, vmax=1)
    axes[2, 1].set_title('Uncertainty Error')
    axes[2, 1].axis('off')
    plt.colorbar(im2, ax=axes[2, 1], fraction=0.046)
    
    # Error difference (blue = uncertainty better, red = baseline better)
    cmap_diff = LinearSegmentedColormap.from_list('diff', ['red', 'white', 'blue'])
    max_diff = max(abs(error_diff.min()), abs(error_diff.max()), 0.5)
    im3 = axes[2, 2].imshow(error_diff, cmap=cmap_diff, vmin=-max_diff, vmax=max_diff)
    axes[2, 2].set_title('Error Diff (blue=unc better)')
    axes[2, 2].axis('off')
    plt.colorbar(im3, ax=axes[2, 2], fraction=0.046)
    
    # Uncertainty map
    if unc_slice is not None:
        im4 = axes[2, 3].imshow(unc_slice, cmap='hot')
        axes[2, 3].contour(gt_slice, levels=[0.5], colors='green', linewidths=1)
        axes[2, 3].set_title('Uncertainty Map + GT boundary')
        axes[2, 3].axis('off')
        plt.colorbar(im4, ax=axes[2, 3], fraction=0.046)
    else:
        axes[2, 3].axis('off')
        axes[2, 3].text(0.5, 0.5, 'No uncertainty map', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, f'{category}_rank{rank+1:02d}_sample{sample["idx"]:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_comparison_summary(sample_metrics, save_dir, threshold_hd95, threshold_sd):
    """
    Create summary visualization of all samples comparing both models.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np

    if len(sample_metrics) == 0:
        import csv
        csv_path = os.path.join(save_dir, 'sample_metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_idx',
                'dice_baseline', 'dice_uncertainty', 'dice_diff',
                'hd95_baseline', 'hd95_uncertainty', 'hd95_diff',
                'sd_baseline', 'sd_uncertainty', 'sd_diff',
                'dice_ensemble',
                'hd95_ensemble',
                'sd_ensemble',
                'hd95_diff_ensemble',
                'dice_diff_ensemble',
                'best_method_hd95',
            ])

        txt_path = os.path.join(save_dir, 'comparison_summary.txt')
        with open(txt_path, 'w') as f:
            f.write('No valid sample metrics were computed; summary plots were skipped.\n')

        print(f"No valid sample metrics found. Wrote empty metrics CSV: {csv_path}")
        print(f"Wrote placeholder summary text: {txt_path}")
        return
    
    hd95_diffs = [s['hd95_diff'] for s in sample_metrics]
    sd_diffs = [s['sd_diff'] for s in sample_metrics]
    hd95_baseline = [s['hd95_baseline'] for s in sample_metrics]
    hd95_uncertainty = [s['hd95_uncertainty'] for s in sample_metrics]
    hd95_ensemble = [s['hd95_ensemble'] for s in sample_metrics]
    sd_baseline = [s['sd_baseline'] for s in sample_metrics]
    sd_uncertainty = [s['sd_uncertainty'] for s in sample_metrics]
    sd_ensemble = [s['sd_ensemble'] for s in sample_metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. HD95 Difference Histogram
    axes[0, 0].hist(hd95_diffs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[0, 0].axvline(x=threshold_hd95, color='green', linestyle='--', linewidth=1, label=f'Threshold ±{threshold_hd95}')
    axes[0, 0].axvline(x=-threshold_hd95, color='green', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('HD95 Improvement (Baseline - Uncertainty)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'HD95 Improvement Distribution\nMean: {np.mean(hd95_diffs):.3f}, Median: {np.median(hd95_diffs):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Surface Distance Difference Histogram
    axes[0, 1].hist(sd_diffs, bins=30, color='darkorange', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[0, 1].axvline(x=threshold_sd, color='green', linestyle='--', linewidth=1, label=f'Threshold ±{threshold_sd}')
    axes[0, 1].axvline(x=-threshold_sd, color='green', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('SD Improvement (Baseline - Uncertainty)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Surface Distance Improvement Distribution\nMean: {np.mean(sd_diffs):.4f}, Median: {np.median(sd_diffs):.4f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter plot: HD95 Baseline vs Uncertainty
    axes[0, 2].scatter(hd95_baseline, hd95_uncertainty, alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
    max_val = max(max(hd95_baseline), max(hd95_uncertainty)) * 1.1
    axes[0, 2].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (equal)')
    axes[0, 2].set_xlabel('HD95 Baseline')
    axes[0, 2].set_ylabel('HD95 Uncertainty Model')
    axes[0, 2].set_title('HD95: Baseline vs Uncertainty\nBelow line = Uncertainty better')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(0, max_val)
    axes[0, 2].set_ylim(0, max_val)
    
    # 4. Scatter plot: SD Baseline vs Uncertainty
    axes[1, 0].scatter(sd_baseline, sd_uncertainty, alpha=0.6, c='darkorange', edgecolors='black', linewidth=0.5)
    max_val = max(max(sd_baseline), max(sd_uncertainty)) * 1.1
    axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (equal)')
    axes[1, 0].set_xlabel('SD Baseline')
    axes[1, 0].set_ylabel('SD Uncertainty Model')
    axes[1, 0].set_title('Surface Distance: Baseline vs Uncertainty\nBelow line = Uncertainty better')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, max_val)
    axes[1, 0].set_ylim(0, max_val)
    
    # 5. HD95 vs SD improvement correlation
    axes[1, 1].scatter(hd95_diffs, sd_diffs, alpha=0.6, c='purple', edgecolors='black', linewidth=0.5)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('HD95 Improvement')
    axes[1, 1].set_ylabel('SD Improvement')
    axes[1, 1].set_title('HD95 vs SD Improvement Correlation')
    # Add quadrant labels
    axes[1, 1].text(0.95, 0.95, 'Both Better', transform=axes[1, 1].transAxes, ha='right', va='top', fontsize=10, color='green')
    axes[1, 1].text(0.05, 0.05, 'Both Worse', transform=axes[1, 1].transAxes, ha='left', va='bottom', fontsize=10, color='red')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    n_improved_hd95 = sum(1 for d in hd95_diffs if d > threshold_hd95)
    n_degraded_hd95 = sum(1 for d in hd95_diffs if d < -threshold_hd95)
    n_improved_sd = sum(1 for d in sd_diffs if d > threshold_sd)
    n_degraded_sd = sum(1 for d in sd_diffs if d < -threshold_sd)
    
    summary_text = f"""
    BOUNDARY QUALITY COMPARISON SUMMARY
    ════════════════════════════════════════
    
    Total Samples: {len(sample_metrics)}
    
    HD95 (Hausdorff Distance 95%):
    ────────────────────────────────────────
      Baseline Mean:     {np.mean(hd95_baseline):.3f}
            CUCL Mean:         {np.mean(hd95_uncertainty):.3f}
            Ensemble Mean:     {np.mean(hd95_ensemble):.3f}
            CUCL Improvement:   {np.mean(hd95_diffs):.3f}
            Ensemble Improvement: {np.mean([s['hd95_diff_ensemble'] for s in sample_metrics]):.3f}
      
      Significantly Improved (>{threshold_hd95}): {n_improved_hd95} ({100*n_improved_hd95/len(sample_metrics):.1f}%)
      Significantly Degraded (<-{threshold_hd95}): {n_degraded_hd95} ({100*n_degraded_hd95/len(sample_metrics):.1f}%)
    
    Surface Distance:
    ────────────────────────────────────────
      Baseline Mean:     {np.mean(sd_baseline):.4f}
            CUCL Mean:         {np.mean(sd_uncertainty):.4f}
            Ensemble Mean:     {np.mean(sd_ensemble):.4f}
            CUCL Improvement:   {np.mean(sd_diffs):.4f}
            Ensemble Improvement: {np.mean([s['sd_baseline'] - s['sd_ensemble'] for s in sample_metrics]):.4f}
      
      Significantly Improved (>{threshold_sd}): {n_improved_sd} ({100*n_improved_sd/len(sample_metrics):.1f}%)
      Significantly Degraded (<-{threshold_sd}): {n_degraded_sd} ({100*n_degraded_sd/len(sample_metrics):.1f}%)
    
    Overall:
    ────────────────────────────────────────
            CUCL better than Baseline:      {sum(1 for d in hd95_diffs if d > 0)} ({100*sum(1 for d in hd95_diffs if d > 0)/len(sample_metrics):.1f}%)
            Ensemble better than Baseline:  {sum(1 for d in [s['hd95_diff_ensemble'] for s in sample_metrics] if d > 0)} ({100*sum(1 for d in [s['hd95_diff_ensemble'] for s in sample_metrics] if d > 0)/len(sample_metrics):.1f}%)
    """
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'comparison_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved summary: {save_path}")
    
    # Also save metrics to CSV
    import csv
    csv_path = os.path.join(save_dir, 'sample_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_idx',
            'dice_baseline', 'dice_uncertainty', 'dice_diff',
            'hd95_baseline', 'hd95_uncertainty', 'hd95_diff',
            'sd_baseline', 'sd_uncertainty', 'sd_diff',
            'dice_ensemble',
            'hd95_ensemble',
            'sd_ensemble',
            'hd95_diff_ensemble',
            'dice_diff_ensemble',
            'best_method_hd95',
        ])
        for s in sample_metrics:
            writer.writerow([
                s['idx'],
                s['dice_baseline'], s['dice_uncertainty'], s['dice_diff'],
                s['hd95_baseline'], s['hd95_uncertainty'], s['hd95_diff'],
                s['sd_baseline'], s['sd_uncertainty'], s['sd_diff'],
                s.get('dice_ensemble', 0.0),
                s.get('hd95_ensemble', 0.0),
                s.get('sd_ensemble', 0.0),
                s.get('hd95_diff_ensemble', 0.0),
                s.get('dice_diff_ensemble', 0.0),
                s.get('best_method_hd95', 'baseline'),
            ])
    print(f"Saved metrics CSV: {csv_path}")


def test_with_uncertainty(model, test_loader, args, num_samples=10, sliding_window=False):
    """Test with uncertainty estimation using MC Dropout"""
    dice_meter = DiceMeter(args)
    model.train()

    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            img, gt = img.to(args.device), gt.to(args.device)

            predictions = []
            for _ in range(num_samples):
                if sliding_window:
                    pred = sliding_window_inference(
                        img, args.crop_spatial_size, 4, model, overlap=0.5
                    )
                else:
                    pred = model(img)

                pred = unwrap_logits(pred)

                if args.num_classes == 1:
                    pred = torch.sigmoid(pred)
                else:
                    pred = torch.softmax(pred, dim=1)

                predictions.append(pred)

            stacked_preds = torch.stack(predictions)
            mean_pred = torch.mean(stacked_preds, dim=0)

            if args.num_classes == 1:
                uncertainty = torch.var(stacked_preds, dim=0)
            else:
                entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)
                uncertainty = entropy

            if args.num_classes == 1:
                final_pred = (mean_pred > 0.5).float()
            else:
                final_pred = torch.argmax(mean_pred, dim=1, keepdim=True)

            all_predictions.append({
                'final_pred': final_pred.cpu(),
                'uncertainty': uncertainty.cpu(),
                'mean_pred': mean_pred.cpu(),
            })
            all_ground_truth.append(gt.cpu())

    dice_scores = []
    uncertainty_vs_error = []

    for pred_data, gt in zip(all_predictions, all_ground_truth):
        pred_one_hot, gt_one_hot = dice_meter.one_hot(pred_data['final_pred'], gt)
        dice = compute_dice(pred_one_hot, gt_one_hot, dice_meter.include_background)
        dice_scores.append(dice)

        error = 1.0 - dice
        uncertainty_vs_error.append((pred_data['uncertainty'].mean().item(), error.mean().item()))

    uncertainty_values, error_values = zip(*uncertainty_vs_error)
    correlation = np.corrcoef(uncertainty_values, error_values)[0, 1]

    return {
        'dice_scores': torch.stack(dice_scores).mean().item(),
        'uncertainty_correlation': correlation,
        'mean_uncertainty': np.mean(uncertainty_values),
        'std_uncertainty': np.std(uncertainty_values),
    }


def sim_test(test_loader, args, lora=True, num_batches=None, sliding_window=False):
    """
    Compute CKA similarity between different models using multiple batches of data.
    """
    filepath_pretrained = args.mae_checkpoint
    filepath_posttrained = (args.continual_checkpoint or args.pretrain)
    filepath_finetuned = os.path.join(args.output_dir, "best.pth.tar")

    model_paths = [filepath_pretrained, filepath_posttrained, filepath_finetuned]

    all_cka_results = []
    batch_count = 0

    print(f"Computing CKA similarity across {num_batches} batches...")

    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            if batch_count >= num_batches:
                break

            print(f"Processing batch {batch_count+1}/{num_batches}")
            img, gt = img.to("cuda"), gt.to("cuda")

            batch_cka_results = compare_vit_encoders_with_cka_batch(args, model_paths, input_batch=img, sliding_window=sliding_window)

            if batch_cka_results is not None:
                all_cka_results.append(batch_cka_results)
                batch_count += 1
                print(f"Successfully processed batch {batch_count}")
            else:
                print(f"Skipping batch {idx} due to error in CKA computation")

    if not all_cka_results:
        print("No valid CKA results were computed!")
        return None

    avg_cka_results = np.mean(np.stack(all_cka_results), axis=0)

    num_models = avg_cka_results.shape[0]
    num_layers = avg_cka_results.shape[2]

    print("\nAverage CKA similarity across layers:")
    for l in range(num_layers):
        print(f"\nLayer {l}:")
        for i in range(num_models):
            row = " ".join([f"{avg_cka_results[i, j, l]:.3f}" for j in range(num_models)])
            print(f"Model {i+1}: [{row}]")

    print("\nOverall average similarity between models:")
    for i in range(num_models):
        for j in range(i+1, num_models):
            avg_sim = np.mean(avg_cka_results[i, j, :])
            print(f"Models {i+1} and {j+1}: {avg_sim:.3f}")

    return avg_cka_results


def save_uncertainty_visualization(
    img, gt, logit, uncertainty_map,
    epoch, iteration, save_dir, args,
    log_writer=None, epoch_1000x=None,
    model=None
):
    """Save uncertainty map visualizations with modulation info"""
    try:
        img_np = img[0].detach().cpu().numpy()
        gt_np = gt[0].detach().cpu().numpy()
        unc_np = uncertainty_map[0].detach().cpu().numpy()

        if isinstance(logit, list):
            logit_np = logit[0][0].detach().cpu().numpy()
        else:
            logit_np = logit[0].detach().cpu().numpy()

        if args.num_classes == 1:
            pred_binary = 1 / (1 + np.exp(-logit_np))
            pred_binary = (pred_binary > 0.5).astype(np.float32)
        else:
            exp_logits = np.exp(logit_np - np.max(logit_np, axis=0, keepdims=True))
            pred_np = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
            pred_binary = np.argmax(pred_np, axis=0, keepdims=True)

        if img_np.ndim == 4:
            mid_slice = img_np.shape[1] // 2
            img_slice = img_np[0, mid_slice]
            gt_slice = gt_np[0, mid_slice]
            pred_slice = pred_binary[0, mid_slice] if args.num_classes > 1 else pred_binary[0, mid_slice]

            if unc_np.ndim == 4:
                unc_slice = unc_np[0, mid_slice]
            else:
                unc_slice = unc_np[mid_slice]
        else:
            img_slice = img_np[0]
            gt_slice = gt_np[0]
            pred_slice = pred_binary[0]
            unc_slice = unc_np[0] if unc_np.ndim == 3 else unc_np

        fig, axes = plt.subplots(3, 3, figsize=(20, 18))

        # Row 1: Input, GT, Prediction
        axes[0, 0].imshow(img_slice, cmap='gray')
        axes[0, 0].set_title(f'Input Image\nEpoch {epoch}, Iter {iteration}')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt_slice, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(pred_slice, cmap='gray')
        axes[0, 2].set_title('Prediction')
        axes[0, 2].axis('off')

        # Row 2: Uncertainty map, Overlay, Histogram
        im1 = axes[1, 0].imshow(unc_slice, cmap='hot')
        axes[1, 0].set_title(f'Uncertainty Map\nMean: {unc_slice.mean():.4f}, Max: {unc_slice.max():.4f}')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        axes[1, 1].imshow(img_slice, cmap='gray', alpha=0.7)
        im2 = axes[1, 1].imshow(unc_slice, cmap='hot', alpha=0.5)
        axes[1, 1].set_title('Uncertainty Overlay')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

        axes[1, 2].hist(unc_slice.flatten(), bins=50, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Uncertainty Distribution')
        axes[1, 2].set_xlabel('Uncertainty Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)

        # Row 3: Modulation Info
        if model is not None:
            unc_strength = 2
            if hasattr(model, 'module'):
                boundary_attention = model.module.boundary_attention if hasattr(model.module, 'boundary_attention') else None
            else:
                boundary_attention = model.boundary_attention if hasattr(model, 'boundary_attention') else None

            # Get boundary attention parameters
            if boundary_attention is not None:
                alpha_values = [ba.alpha.item() for ba in boundary_attention]
                beta_values = [ba.beta.item() for ba in boundary_attention]
                alpha_str = ", ".join([f"{a:.3f}" for a in alpha_values])
                beta_str = ", ".join([f"{b:.3f}" for b in beta_values])
            else:
                alpha_str = "N/A"
                beta_str = "N/A"

            axes[2, 0].axis('off')
            modulation_text = f"""
            Modulation Parameters:
            ━━━━━━━━━━━━━━━━━━━━━━━━
            unc_strength: {unc_strength:.4f}
            
            Boundary Attention:
              alpha (residual): [{alpha_str}]
              beta (unc influence): [{beta_str}]
            
            Modulation Factor Range:
              Min: {(1.0 + unc_strength * 0.0):.4f}
              Max: {(1.0 + unc_strength * 1.0):.4f}
            
            Features amplified by up to
            {(unc_strength * 100):.1f}% in high-unc regions
            """
            axes[2, 0].text(0.1, 0.5, modulation_text,
                           fontsize=10, verticalalignment='center',
                           family='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[2, 0].set_title('Modulation Info')

            # Plot alpha and beta values
            if boundary_attention is not None:
                x = np.arange(len(alpha_values))
                width = 0.35
                axes[2, 1].bar(x - width/2, alpha_values, width, label='alpha (residual)', color='blue', alpha=0.7)
                axes[2, 1].bar(x + width/2, beta_values, width, label='beta (unc)', color='red', alpha=0.7)
                axes[2, 1].set_xlabel('Scale')
                axes[2, 1].set_ylabel('Value')
                axes[2, 1].set_title('Boundary Attention Parameters')
                axes[2, 1].set_xticks(x)
                axes[2, 1].set_xticklabels([f'Scale {i}' for i in range(len(alpha_values))])
                axes[2, 1].legend()
                axes[2, 1].grid(True, alpha=0.3)
            else:
                axes[2, 1].axis('off')

            # Modulation function visualization
            x_vals = np.linspace(0, 1, 100)
            y_vals = 1.0 + unc_strength * x_vals
            axes[2, 2].plot(x_vals, y_vals, 'b-', linewidth=2, label='Feature amplification')
            axes[2, 2].fill_between(x_vals, 1.0, y_vals, alpha=0.3)
            axes[2, 2].set_xlabel('Uncertainty Gate Value')
            axes[2, 2].set_ylabel('Feature Amplification Factor')
            axes[2, 2].set_title('Modulation Function')
            axes[2, 2].grid(True, alpha=0.3)
            axes[2, 2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No modulation')
            axes[2, 2].legend()
        else:
            for ax in axes[2, :]:
                ax.axis('off')
                ax.text(0.5, 0.5, 'Modulation Disabled',
                       ha='center', va='center', fontsize=12, alpha=0.5)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f'unc_epoch{epoch:03d}_iter{iteration:05d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved uncertainty visualization to {save_path}")

        if log_writer is not None and epoch_1000x is not None:
            log_writer.add_figure('uncertainty/visualization', fig, epoch_1000x)

        plt.close(fig)

    except Exception as e:
        print(f"Error saving uncertainty visualization: {e}")
        traceback.print_exc()