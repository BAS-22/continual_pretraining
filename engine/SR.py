import math
import sys
from typing import Iterable

import torch
import os
import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn as nn
import numpy as np
# from util.meter import PSNRMeter, SSIMMeter, LPIPSMeter
from monai.inferers import sliding_window_inference
import pdb
import pytorch_msssim
import nibabel as nib
import os
from util.feature_sim import compare_vit_encoders_with_cka_batch_SR


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    """Training function adapted for super-resolution"""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # Extract low-res and high-res images
        low_res = batch['low_res'].to(device, non_blocking=True)
        high_res = batch['high_res'].to(device, non_blocking=True)
        
        # Adjust learning rate
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        
        # Forward pass with mask ratio for training
        mask_ratio = getattr(args, 'mask_ratio', 0)
        loss, pred_img, mask = model(low_res, high_res, mask_ratio=mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validation_ori(model, data_loader_val, device, epoch, args):
    """Validation function for super-resolution"""
    model.eval()
    
    # Super-resolution loss components
    # l1_loss = nn.L1Loss()
    # l2_loss = nn.MSELoss()

    with torch.no_grad():
        loss_summary = []
        l1_summary = []
        ss_summary = []
        ff_summary = []
        
        for idx, batch in enumerate(data_loader_val):
            low_res = batch['low_res'].to(device)
            high_res = batch['high_res'].to(device)
            
            # Forward pass without masking for validation
            pred_img = model(low_res, mask_ratio=0)
            
            # Calculate losses
            # l1 = l1_loss(pred_img, high_res)
            # l2 = l2_loss(pred_img, high_res)
            # total_loss = l1 + 0.1 * l2
            total_loss, l1, ss, ff = forward_loss_sr(pred_img, high_res)

            
            loss_summary.append(total_loss.detach().cpu().numpy())
            l1_summary.append(l1.detach().cpu().numpy())
            ss_summary.append(ss.detach().cpu().numpy())
            ff_summary.append(ff.detach().cpu().numpy())
            
            print(
                "epoch: {}/{}, iter: {}/{}".format(
                    epoch, args.epochs, idx, len(data_loader_val)
                )
                + " loss: {:.4f}, L1: {:.4f}, L2: {:.4f}".format(
                    loss_summary[-1], l1_summary[-1], ss_summary[-1], ff_summary[-1]
                )
            )
        
        avg_loss = np.mean(loss_summary)
        avg_l1 = np.mean(l1_summary)
        avg_ss = np.mean(ss_summary)
        avg_ff = np.mean(ff_summary)
        
        print("Averaged stats - Total Loss: {:.4f}, L1: {:.4f}, L2: {:.4f}".format(
            avg_loss, avg_l1, avg_ss, avg_ff))
    
    return avg_loss


def validation_hybrid(model, data_loader_val, device, epoch, args):
    """Validation function for super-resolution"""
    model.eval()
    
    # Initialize meters for image quality metrics
    psnr_meter = PSNRMeter()
    ssim_meter = SSIMMeter()
    
    # Try to initialize LPIPS meter (requires lpips package)
    try:
        lpips_meter = LPIPSMeter(device=device)
        use_lpips = True
    except:
        print("LPIPS not available, skipping LPIPS evaluation")
        use_lpips = False

    with torch.no_grad():
        loss_summary = []
        l1_summary = []
        ss_summary = []
        ff_summary = []
        
        for idx, batch in enumerate(data_loader_val):
            low_res = batch['low_res'].to(device)
            high_res = batch['high_res'].to(device)
            
            # Forward pass without masking for validation
            pred_img = model(low_res, mask_ratio=0)
            
            # Calculate losses
            total_loss, l1, ss, ff = forward_loss_sr(pred_img, high_res)
            
            loss_summary.append(total_loss.detach().cpu().numpy())
            l1_summary.append(l1.detach().cpu().numpy())
            ss_summary.append(ss.detach().cpu().numpy())
            ff_summary.append(ff.detach().cpu().numpy())
            
            # Update quality metrics
            psnr_meter.update(pred_img, high_res)
            ssim_meter.update(pred_img, high_res)
            if use_lpips:
                lpips_meter.update(pred_img, high_res)
            
            print(
                "epoch: {}/{}, iter: {}/{}".format(
                    epoch, args.epochs, idx, len(data_loader_val)
                )
                + " loss: {:.4f}, L1: {:.4f}, SSIM: {:.4f}, Fourier: {:.4f}".format(
                    loss_summary[-1], l1_summary[-1], ss_summary[-1], ff_summary[-1]
                )
            )
        
        avg_loss = np.mean(loss_summary)
        avg_l1 = np.mean(l1_summary)
        avg_ss = np.mean(ss_summary)
        avg_ff = np.mean(ff_summary)
        
        # Get average quality metrics
        avg_psnr = psnr_meter.get_average()
        avg_ssim = ssim_meter.get_average()
        
        print("- Validation metrics for Super-Resolution:")
        print(f"Total Loss: {avg_loss:.4f}")
        print(f"L1 Loss: {avg_l1:.4f}")
        print(f"SSIM Loss: {avg_ss:.4f}")
        print(f"Fourier Loss: {avg_ff:.4f}")
        print(f"PSNR: {avg_psnr:.4f}")
        print(f"SSIM: {avg_ssim:.4f}")
        
        if use_lpips:
            avg_lpips = lpips_meter.get_average()
            print(f"LPIPS: {avg_lpips:.4f}")
    
    return avg_loss

def validation(model, data_loader_val, device, epoch, args):
        """Validation function for super-resolution"""
        model.eval()
        
        # Initialize meters for image quality metrics
        psnr_meter = PSNRMeter()
        ssim_meter = SSIMMeter()

        with torch.no_grad():
            loss_summary = []
            l1_summary = []
            l2_summary = []
            
            for idx, batch in enumerate(data_loader_val):
                low_res = batch['low_res'].to(device)
                high_res = batch['high_res'].to(device)
                
                # Forward pass without masking for validation
                pred_img = model(low_res, mask_ratio=0)
                
                # Calculate losses
                total_loss, l1, l2 = forward_loss_sr(pred_img, high_res)
                
                loss_summary.append(total_loss.detach().cpu().numpy())
                l1_summary.append(l1.detach().cpu().numpy())
                l2_summary.append(l2.detach().cpu().numpy())
                
                # Update quality metrics
                psnr_meter.update(pred_img, high_res)
                ssim_meter.update(pred_img, high_res)
                
                
                print(
                    "epoch: {}/{}, iter: {}/{}".format(
                        epoch, args.epochs, idx, len(data_loader_val)
                    )
                    + " loss: {:.4f}, L1: {:.4f}, L2: {:.4f}".format(
                        loss_summary[-1], l1_summary[-1], l2_summary[-1]
                    )
                )
            
            avg_loss = np.mean(loss_summary)
            avg_l1 = np.mean(l1_summary)
            avg_l2 = np.mean(l2_summary)
            
            # Get average quality metrics
            avg_psnr = psnr_meter.get_average()
            avg_ssim = ssim_meter.get_average()
            
            print("- Validation metrics for Super-Resolution:")
            print(f"Total Loss: {avg_loss:.4f}")
            print(f"L1 Loss: {avg_l1:.4f}")
            print(f"L2 Loss: {avg_l2:.4f}")
            print(f"PSNR: {avg_psnr:.4f}")
            print(f"SSIM: {avg_ssim:.4f}")
        
        return avg_loss

def test_ori(model, test_loader, args, sliding_window=False):
    """Test function for super-resolution with image quality metrics"""
    model.eval()
    
    # Load best model
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    if os.path.exists(filepath_best):
        model.load_state_dict(torch.load(filepath_best)["model"])
        print(f"Loaded best model from {filepath_best}")
    
    # Initialize meters for image quality metrics
    psnr_meter = PSNRMeter()
    ssim_meter = SSIMMeter()
    
    # Try to initialize LPIPS meter (requires lpips package)
    try:
        lpips_meter = LPIPSMeter(device=args.device)
        use_lpips = True
    except:
        print("LPIPS not available, skipping LPIPS evaluation")
        use_lpips = False
    
    log_stats = {}
    loss_summary = []
    l1_summary = []
    ss_summary = []
    ff_summary = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            low_res = batch['low_res'].to(args.device)
            high_res = batch['high_res'].to(args.device)
            
            if sliding_window:
                # For sliding window inference, you might need to adapt this
                # based on your model's input requirements
                pred_img = sliding_window_inference(
                    low_res, args.crop_spatial_size, 4, 
                    lambda x: model(x, mask_ratio=0), overlap=0.5
                )
            else:
                pred_img = model(low_res, mask_ratio=0)
            
            # Calculate losses using forward_loss_sr (same as validation)
            total_loss, l1_loss, ss_loss, ff_loss = forward_loss_sr(pred_img, high_res)
            
            loss_summary.append(total_loss.detach().cpu().numpy())
            l1_summary.append(l1_loss.detach().cpu().numpy())
            ss_summary.append(ss_loss.detach().cpu().numpy())
            ff_summary.append(ff_loss.detach().cpu().numpy())
            
            # Update quality metrics
            psnr_meter.update(pred_img, high_res)
            ssim_meter.update(pred_img, high_res)
            if use_lpips:
                lpips_meter.update(pred_img, high_res)
            
            if idx % 10 == 0:
                print(f"Processing {idx}/{len(test_loader)}")

    # Get average metrics
    avg_psnr = psnr_meter.get_average()
    avg_ssim = ssim_meter.get_average()
    avg_total_loss = np.mean(loss_summary)
    avg_l1 = np.mean(l1_summary)
    avg_ss = np.mean(ss_summary)
    avg_ff = np.mean(ff_summary)
    
    print("- Test metrics for Super-Resolution:")
    print(f"Total Loss: {avg_total_loss:.4f}")
    print(f"L1 Loss: {avg_l1:.4f}")
    print(f"SSIM Loss: {avg_ss:.4f}")
    print(f"Fourier Loss: {avg_ff:.4f}")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    
    log_stats = {
        "total_loss": avg_total_loss,
        "l1_loss": avg_l1,
        "ssim_loss": avg_ss,
        "fourier_loss": avg_ff,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
    }
    
    if use_lpips:
        avg_lpips = lpips_meter.get_average()
        print(f"LPIPS: {avg_lpips:.4f}")
        log_stats["lpips"] = avg_lpips
    
    return log_stats

def test_ori1(model, test_loader, args, sliding_window=False):
    """Test function for super-resolution with image quality metrics"""
    model.eval()
    
    # Load best model
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    if os.path.exists(filepath_best):
        model.load_state_dict(torch.load(filepath_best)["model"])
        print(f"Loaded best model from {filepath_best}")
    
    # Initialize meters for image quality metrics
    psnr_meter = PSNRMeter()
    ssim_meter = SSIMMeter()
    
    
    # Create visualization directory
    vis_dir = os.path.join(args.output_dir, "visualise")
    os.makedirs(vis_dir, exist_ok=True)
    
    log_stats = {}
    loss_summary = []
    l1_summary = []
    l2_summary = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            low_res = batch['low_res'].to(args.device)
            high_res = batch['high_res'].to(args.device)
            
            if sliding_window:
                # For sliding window inference, you might need to adapt this
                # based on your model's input requirements
                pred_img = sliding_window_inference(
                    low_res, args.crop_spatial_size, 4, 
                    lambda x: model(x, mask_ratio=0), overlap=0.5
                )
            else:
                pred_img = model(low_res, mask_ratio=0)
            
            # Save first batch as .nii.gz files
            if idx == 0:
                print(f"Saving visualization files to {vis_dir}")
                batch_size = pred_img.shape[0]
                
                for i in range(batch_size):
                    # Convert tensors to numpy arrays and remove channel dimension
                    pred_np = pred_img[i, 0].detach().cpu().numpy()  # Remove channel dim [C, D, H, W] -> [D, H, W]
                    high_res_np = high_res[i, 0].detach().cpu().numpy()
                    low_res_np = low_res[i, 0].detach().cpu().numpy()
                    
                    # Create NIfTI images with proper orientation (transpose to [H, W, D] for NIfTI)
                    pred_nii = nib.Nifti1Image(pred_np.transpose(2, 1, 0), affine=np.eye(4))
                    high_res_nii = nib.Nifti1Image(high_res_np.transpose(2, 1, 0), affine=np.eye(4))
                    low_res_nii = nib.Nifti1Image(low_res_np.transpose(2, 1, 0), affine=np.eye(4))
                    
                    # Save files
                    pred_path = os.path.join(vis_dir, f"pred_batch0_sample{i}.nii.gz")
                    high_res_path = os.path.join(vis_dir, f"ground_truth_batch0_sample{i}.nii.gz")
                    low_res_path = os.path.join(vis_dir, f"low_res_batch0_sample{i}.nii.gz")
                    
                    nib.save(pred_nii, pred_path)
                    nib.save(high_res_nii, high_res_path)
                    nib.save(low_res_nii, low_res_path)
                    
                    print(f"Saved sample {i}:")
                    print(f"  Prediction: {pred_path}")
                    print(f"  Ground truth: {high_res_path}")
                    print(f"  Low resolution: {low_res_path}")
                    print(f"  Shapes - Pred: {pred_np.shape}, GT: {high_res_np.shape}, LR: {low_res_np.shape}")
            
            # Calculate losses using forward_loss_sr (same as validation)
            total_loss, l1_loss, l2_loss = forward_loss_sr(pred_img, high_res)
            
            loss_summary.append(total_loss.detach().cpu().numpy())
            l1_summary.append(l1_loss.detach().cpu().numpy())
            l2_summary.append(l2_loss.detach().cpu().numpy())
            # Update quality metrics
            psnr_meter.update(pred_img, high_res)
            ssim_meter.update(pred_img, high_res)

    # Get average metrics
    avg_psnr = psnr_meter.get_average()
    avg_ssim = ssim_meter.get_average()
    avg_total_loss = np.mean(loss_summary)
    avg_l1 = np.mean(l1_summary)
    avg_l2 = np.mean(l2_summary)
    
    print("- Test metrics for Super-Resolution:")
    print(f"Total Loss: {avg_total_loss:.4f}")
    print(f"L1 Loss: {avg_l1:.4f}")
    print(f"L2 Loss: {avg_l2:.4f}")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    
    log_stats = {
        "total_loss": avg_total_loss,
        "l1_loss": avg_l1,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
    }
    
    return log_stats

def test(model, test_loader, args, sliding_window=False):
    """Test function for super-resolution with image quality metrics"""
    model.eval()
    
    # Load best model
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    if os.path.exists(filepath_best):
        model.load_state_dict(torch.load(filepath_best)["model"])
        print(f"Loaded best model from {filepath_best}")
    
    # Initialize meters for image quality metrics
    psnr_meter = PSNRMeter()
    ssim_meter = SSIMMeter()
    
    # Create visualization directory
    vis_dir = os.path.join(args.output_dir, "visualise")
    os.makedirs(vis_dir, exist_ok=True)
    
    log_stats = {}
    loss_summary = []
    l1_summary = []
    l2_summary = []
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            low_res = batch['low_res'].to(args.device)
            high_res = batch['high_res'].to(args.device)
            
            if sliding_window:
                # For sliding window inference, you might need to adapt this
                # based on your model's input requirements
                pred_img = sliding_window_inference(
                    low_res, args.crop_spatial_size, 4, 
                    lambda x: model(x, mask_ratio=0), overlap=0.5
                )
            else:
                pred_img = model(low_res, mask_ratio=0)
            
            # Save first batch as .nii.gz files
            if idx == 0:
                print(f"Saving visualization files to {vis_dir}")
                batch_size = pred_img.shape[0]
                
                for i in range(batch_size):
                    # Convert tensors to numpy arrays and remove channel dimension
                    pred_np = pred_img[i, 0].detach().cpu().numpy()  # Remove channel dim [C, D, H, W] -> [D, H, W]
                    high_res_np = high_res[i, 0].detach().cpu().numpy()
                    low_res_np = low_res[i, 0].detach().cpu().numpy()
                    
                    # Create NIfTI images with proper orientation (transpose to [H, W, D] for NIfTI)
                    pred_nii = nib.Nifti1Image(pred_np.transpose(2, 1, 0), affine=np.eye(4))
                    high_res_nii = nib.Nifti1Image(high_res_np.transpose(2, 1, 0), affine=np.eye(4))
                    low_res_nii = nib.Nifti1Image(low_res_np.transpose(2, 1, 0), affine=np.eye(4))
                    
                    # Save files
                    pred_path = os.path.join(vis_dir, f"pred_batch0_sample{i}.nii.gz")
                    high_res_path = os.path.join(vis_dir, f"ground_truth_batch0_sample{i}.nii.gz")
                    low_res_path = os.path.join(vis_dir, f"low_res_batch0_sample{i}.nii.gz")
                    
                    nib.save(pred_nii, pred_path)
                    nib.save(high_res_nii, high_res_path)
                    nib.save(low_res_nii, low_res_path)
                    
                    print(f"Saved sample {i}:")
                    print(f"  Prediction: {pred_path}")
                    print(f"  Ground truth: {high_res_path}")
                    print(f"  Low resolution: {low_res_path}")
                    print(f"  Shapes - Pred: {pred_np.shape}, GT: {high_res_np.shape}, LR: {low_res_np.shape}")
            
            # Calculate losses using forward_loss_sr (same as validation)
            total_loss, l1_loss, l2_loss = forward_loss_sr(pred_img, high_res)
            
            loss_summary.append(total_loss.detach().cpu().numpy())
            l1_summary.append(l1_loss.detach().cpu().numpy())
            l2_summary.append(l2_loss.detach().cpu().numpy())
            
            # Update quality metrics
            psnr_meter.update(pred_img, high_res)
            ssim_meter.update(pred_img, high_res)
            
            # Store PSNR and SSIM values for standard deviation computation
            psnr_values.append(psnr_meter.psnr_sum / psnr_meter.count)
            ssim_values.append(ssim_meter.ssim_sum / ssim_meter.count)

    # Get average metrics
    avg_psnr = psnr_meter.get_average()
    avg_ssim = ssim_meter.get_average()
    avg_total_loss = np.mean(loss_summary)
    avg_l1 = np.mean(l1_summary)
    avg_l2 = np.mean(l2_summary)
    
    # Compute standard deviations
    std_psnr = np.std(psnr_values)
    std_ssim = np.std(ssim_values)
    std_total_loss = np.std(loss_summary)
    std_l1 = np.std(l1_summary)
    std_l2 = np.std(l2_summary)
    
    print("- Test metrics for Super-Resolution:")
    print(f"Total Loss: {avg_total_loss:.4f} ± {std_total_loss:.4f}")
    print(f"L1 Loss: {avg_l1:.4f} ± {std_l1:.4f}")
    print(f"L2 Loss: {avg_l2:.4f} ± {std_l2:.4f}")
    print(f"PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    
    log_stats = {
        "total_loss": avg_total_loss,
        "total_loss_std": std_total_loss,
        "l1_loss": avg_l1,
        "l1_loss_std": std_l1,
        "l2_loss": avg_l2,
        "l2_loss_std": std_l2,
        "psnr": avg_psnr,
        "psnr_std": std_psnr,
        "ssim": avg_ssim,
        "ssim_std": std_ssim,
    }
    
    return log_stats

def test_with_uncertainty(model, test_loader, args, num_samples=10, sliding_window=False):
    """Test super-resolution with uncertainty estimation using MC Dropout"""
    model.train()  # Enable dropout for uncertainty estimation
    
    psnr_meter = PSNRMeter()
    ssim_meter = SSIMMeter()
    
    all_predictions = []
    all_ground_truth = []
    all_uncertainties = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            low_res = batch['low_res'].to(args.device)
            high_res = batch['high_res'].to(args.device)
            
            # Generate multiple predictions for the same input
            predictions = []
            for _ in range(num_samples):
                if sliding_window:
                    pred = sliding_window_inference(
                        low_res, args.crop_spatial_size, 4,
                        lambda x: model(x, mask_ratio=0), overlap=0.5
                    )
                else:
                    pred = model(low_res, mask_ratio=0)
                
                predictions.append(pred)
            
            # Stack predictions and calculate statistics
            stacked_preds = torch.stack(predictions)  # (num_samples, batch, channels, ...)
            mean_pred = torch.mean(stacked_preds, dim=0)
            uncertainty = torch.var(stacked_preds, dim=0)
            
            # Calculate metrics using mean prediction
            psnr_meter.update(mean_pred, high_res)
            ssim_meter.update(mean_pred, high_res)
            
            # Store for uncertainty analysis
            all_predictions.append(mean_pred.cpu())
            all_ground_truth.append(high_res.cpu())
            all_uncertainties.append(uncertainty.cpu())
    
    # Calculate quality metrics
    avg_psnr = psnr_meter.get_average()
    avg_ssim = ssim_meter.get_average()
    
    # Calculate uncertainty statistics
    all_uncertainties = torch.cat(all_uncertainties, dim=0)
    mean_uncertainty = torch.mean(all_uncertainties).item()
    std_uncertainty = torch.std(all_uncertainties).item()
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'mean_uncertainty': mean_uncertainty,
        'std_uncertainty': std_uncertainty,
    }


def sim_test_ori(test_loader, args, lora=False):
    """Similarity test for super-resolution models"""
    from util.feature_sim import compare_vit_encoders_with_cka_batch
    
    # Define file paths for different model stages
    filepath_pretrained = args.mae_checkpoint
    
    if lora:
        filepath_posttrained = os.path.join("output_sr_lora_799epoch/Promis/profound_vit/lora/2021", "best.pth.tar")
        filepath_finetuned = os.path.join("output_sr_lora_full_799epoch/Promis/profound_vit/lora/2021", "best.pth.tar")
    else:
        filepath_posttrained = os.path.join("output_sr_799epoch/Promis/profound_vit/finetune/2021", "best.pth.tar")
        filepath_finetuned = os.path.join("output_sr_full_799epoch/Promis/profound_vit/finetune/2021", "best.pth.tar")

    model_paths = [filepath_pretrained, filepath_posttrained, filepath_finetuned]
    cka_results = None

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            low_res = batch['low_res'].to("cuda")
            
            if idx == 0:
                print("Low-res image shape:", low_res.shape)
                cka_results = compare_vit_encoders_with_cka_batch(args, model_paths, input_batch=low_res)
            else:
                break  # Exit after first batch

    return cka_results


class PSNRMeter:
    """Peak Signal-to-Noise Ratio meter for image quality assessment"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.psnr_sum = 0.0
        self.count = 0
    
    def update(self, pred, target):
        """
        Update PSNR calculation
        pred, target: torch tensors of shape [B, C, D, H, W]
        """
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            mse = torch.mean((pred[i] - target[i]) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                max_pixel = torch.max(target[i])
                psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
            
            self.psnr_sum += psnr.item()
            self.count += 1
    
    def get_average(self):
        if self.count == 0:
            return 0.0
        return self.psnr_sum / self.count


class SSIMMeter:
    """Structural Similarity Index meter for image quality assessment"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.ssim_sum = 0.0
        self.count = 0
    
    def _ssim_3d(self, pred, target, window_size=11):
        """Calculate SSIM for 3D volumes"""
        # Simplified SSIM calculation for 3D
        # You might want to use a more sophisticated implementation
        
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        
        sigma1_sq = torch.var(pred)
        sigma2_sq = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim
    
    def update(self, pred, target):
        """
        Update SSIM calculation
        pred, target: torch tensors of shape [B, C, D, H, W]
        """
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            ssim = self._ssim_3d(pred[i], target[i])
            self.ssim_sum += ssim.item()
            self.count += 1
    
    def get_average(self):
        if self.count == 0:
            return 0.0
        return self.ssim_sum / self.count


class LPIPSMeter:
    """Learned Perceptual Image Patch Similarity meter"""
    
    def __init__(self, device='cuda'):
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.device = device
            self.reset()
        except ImportError:
            raise ImportError("LPIPS package not found. Install with: pip install lpips")
    
    def reset(self):
        self.lpips_sum = 0.0
        self.count = 0
    
    def update(self, pred, target):
        """
        Update LPIPS calculation
        pred, target: torch tensors of shape [B, C, D, H, W]
        For 3D volumes, we'll calculate LPIPS on 2D slices and average
        """
        batch_size = pred.shape[0]
        depth = pred.shape[2]
        
        for i in range(batch_size):
            lpips_scores = []
            for d in range(depth):
                # Extract 2D slices
                pred_slice = pred[i:i+1, :, d:d+1, :, :].squeeze(2)  # [1, C, H, W]
                target_slice = target[i:i+1, :, d:d+1, :, :].squeeze(2)  # [1, C, H, W]
                
                # LPIPS expects 3-channel images, so repeat if grayscale
                if pred_slice.shape[1] == 1:
                    pred_slice = pred_slice.repeat(1, 3, 1, 1)
                    target_slice = target_slice.repeat(1, 3, 1, 1)
                
                # Normalize to [-1, 1] range expected by LPIPS
                pred_slice = 2 * pred_slice - 1
                target_slice = 2 * target_slice - 1
                
                lpips_score = self.lpips_fn(pred_slice, target_slice)
                lpips_scores.append(lpips_score.item())
            
            avg_lpips = np.mean(lpips_scores)
            self.lpips_sum += avg_lpips
            self.count += 1
    
    def get_average(self):
        if self.count == 0:
            return 0.0
        return self.lpips_sum / self.count


def forward_loss_sr_hybrid(pred, target, alpha=0.7, beta=0.3, gamma=0.001):
    """
    Compute the super-resolution loss using a combination of:
    - L1 loss (voxel-wise accuracy)
    - SSIM loss (structural similarity)
    - Fourier loss (frequency consistency)
    """
    # L1 loss
    l1_loss = nn.L1Loss()(pred, target)

    # SSIM loss - compute slice-wise for 3D volumes
    ssim_loss = _compute_3d_ssim_loss(pred, target)

    # Improved Fourier loss
    fft_loss = _compute_fourier_loss(pred, target)

    # Combine losses
    total_loss = alpha * l1_loss + beta * ssim_loss + gamma * fft_loss
    return total_loss, l1_loss, ssim_loss, fft_loss

def forward_loss_sr(pred_img, high_res_target):
    """Compute super-resolution loss"""
    # pred_img = self.unpatchify_sr(pred)
    
    l1_loss = nn.L1Loss()(pred_img, high_res_target)
    l2_loss = nn.MSELoss()(pred_img, high_res_target)
    
    total_loss = l1_loss + 0.1 * l2_loss
    return total_loss, l1_loss, l2_loss

def _compute_fourier_loss(pred, target):
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

def _compute_3d_ssim_loss(pred, target):
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



def sim_test(test_loader, args, lora=True, num_batches=None, sliding_window=False):
    """
    Compute CKA similarity between different models using multiple batches of data.
    
    Args:
        test_loader: DataLoader for test data
        args: Arguments
        lora: Whether to use LoRA paths
        num_batches: Number of batches to use for averaging CKA results
    
    Returns:
        averaged CKA results
    """
    # Define file paths for pretrained, posttrained, and finetuned models
    filepath_pretrained = args.mae_checkpoint
    # filepath_pretrained = os.path.join(args.output_dir, "best.pth.tar")
    
    filepath_posttrained = (args.continual_checkpoint or args.pretrain)
    filepath_finetuned = os.path.join(args.output_dir, "best.pth.tar")

    model_paths = [filepath_pretrained, filepath_posttrained, filepath_finetuned]
    
    all_cka_results = []
    batch_count = 0
    
    print(f"Computing CKA similarity across {num_batches} batches...")

    # with torch.no_grad():
        # for idx, (img, gt, _) in enumerate(test_loader):
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            low_res = batch['low_res'].to(args.device)
            high_res = batch['high_res'].to(args.device)
            
            if batch_count >= num_batches:
                break
                
            print(f"Processing batch {batch_count+1}/{num_batches}")
            # img, gt = img.to("cuda"), gt.to("cuda")
            
            # Compute CKA for this batch
            batch_cka_results = compare_vit_encoders_with_cka_batch_SR(args, model_paths, input_batch=low_res, input_HR = high_res, sliding_window=sliding_window)
            
            if batch_cka_results is not None:
                all_cka_results.append(batch_cka_results)
                batch_count += 1
                print(f"Successfully processed batch {batch_count}")
            else:
                print(f"Skipping batch {idx} due to error in CKA computation")
    
    if not all_cka_results:
        print("No valid CKA results were computed!")
        return None
    
    # Average CKA results across all batches
    avg_cka_results = np.mean(np.stack(all_cka_results), axis=0)
    
    # Print summary statistics
    num_models = avg_cka_results.shape[0]
    num_layers = avg_cka_results.shape[2]
    
    print("\nAverage CKA similarity across layers:")
    for l in range(num_layers):
        print(f"\nLayer {l}:")
        for i in range(num_models):
            row = " ".join([f"{avg_cka_results[i, j, l]:.3f}" for j in range(num_models)])
            print(f"Model {i+1}: [{row}]")
    
    # Overall average similarity between models
    print("\nOverall average similarity between models:")
    for i in range(num_models):
        for j in range(i+1, num_models):
            avg_sim = np.mean(avg_cka_results[i, j, :])
            print(f"Models {i+1} and {j+1}: {avg_sim:.3f}")
    
    return avg_cka_results