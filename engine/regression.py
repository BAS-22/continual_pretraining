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
import torch
import os
import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
from util.feature_sim import compare_vit_encoders_with_cka_batch


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
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    loss_cal = torch.nn.MSELoss()

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    last_norm = 0.0
    for data_iter_step, (img, gt, dataidx) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        img, gt = img.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        logit = model(img)
        loss = loss_cal(logit, gt)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(
                "nan",
                torch.isnan(logit).any(),
                torch.isnan(img).any(),
                dataidx,
                last_norm,
            )
            print(
                "inf",
                torch.isinf(logit).any(),
                torch.isinf(img).any(),
                dataidx,
                last_norm,
            )
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),  1.0)
        optimizer.step()

        # last_norm = loss_scaler(loss, optimizer, parameters=model.parameters())
        # optimizer.zero_grad()
        # torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validation(model, data_loader_val, device, epoch, args):
    model.eval()
    loss_cal = torch.nn.MSELoss()
    with torch.no_grad():
        loss_summary = []
        for idx, (img, gt, _) in enumerate(data_loader_val):
            img, gt = img.to(device), gt.to(device)
            loss = loss_cal(model(img), gt)
            loss_summary.append(loss.detach().cpu().numpy())
            print(
                "epoch: {}/{}, iter: {}/{}".format(
                    epoch, args.epochs, idx, len(data_loader_val)
                )
                + " loss:"
                + str(loss_summary[-1].flatten()[0])
            )
        avg_loss = np.mean(loss_summary)
        print("Averaged stats:", str(avg_loss))
    return avg_loss

def test_ori(model, test_loader, args):
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    model.load_state_dict(torch.load(filepath_best)["model"])

    model.eval()
    log_stats = {}
    pred, gts = [], []

    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            img, gt = img.to(args.device), gt.to(args.device)
            pred.append(model(img))
            gts.append(gt)
        pred = torch.cat(pred, 0)
        gts = torch.cat(gts, 0)
        pred = pred * 500000 + 70000
        gts = gts * 500000 + 70000
        mse = torch.nn.MSELoss()(pred, gts)
        mae = torch.nn.L1Loss()(pred, gts)
    print("MSE", mse.item(), "MAE", mae.item())
    log_stats = {"MSE": mse.item(), "MAE": mae.item()}
    return log_stats

def test(model, test_loader, args):
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    model.load_state_dict(torch.load(filepath_best)["model"])

    model.eval()
    log_stats = {}
    pred, gts = [], []
    mse_values, mae_values = [], []  # To store MSE and MAE for each sample/batch

    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            img, gt = img.to(args.device), gt.to(args.device)
            pred_batch = model(img)
            pred.append(pred_batch)
            gts.append(gt)

            # Compute MSE and MAE for the current batch
            # pred_batch = pred_batch * 500000 + 70000
            # gt_batch = gt * 500000 + 70000
            pred_batch = pred_batch*1
            gt_batch = gt*1
            mse_batch = torch.nn.functional.mse_loss(pred_batch, gt_batch, reduction="none")
            mae_batch = torch.nn.functional.l1_loss(pred_batch, gt_batch, reduction="none")

            # Store the mean MSE and MAE for this batch
            mse_values.append(mse_batch.mean().item())
            mae_values.append(mae_batch.mean().item())

        # Concatenate predictions and ground truths for overall metrics
        pred = torch.cat(pred, 0)
        gts = torch.cat(gts, 0)
        # pred = pred * 500000 + 70000
        # gts = gts * 500000 + 70000

        # Compute overall MSE and MAE
        mse = torch.nn.MSELoss()(pred, gts)
        mae = torch.nn.L1Loss()(pred, gts)

        # Compute standard deviation for MSE and MAE
        mse_std = np.std(mse_values)
        mae_std = np.std(mae_values)

    print(
        "MSE", mse.item(),
        "MAE", mae.item(),
        "MSE_STD", mse_std,
        "MAE_STD", mae_std
    )
    log_stats = {
        "MSE": mse.item(),
        "MAE": mae.item(),
        "MSE_STD": mse_std,
        "MAE_STD": mae_std,
    }
    return log_stats

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
    
    filepath_continual_trained = (args.continual_checkpoint or args.pretrain)
    filepath_continual_finetuned = os.path.join(args.output_dir, "best.pth.tar")
    filepath_pretrained_finetuned = os.path.join(args.prenc_dir, "best.pth.tar") 

    model_paths = [filepath_pretrained, filepath_continual_trained, filepath_continual_finetuned, filepath_pretrained_finetuned]
    
    all_cka_results = []
    batch_count = 0
    
    print(f"Computing CKA similarity across {num_batches} batches...")

    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            if batch_count >= num_batches:
                break
                
            print(f"Processing batch {batch_count+1}/{num_batches}")
            img, gt = img.to("cuda"), gt.to("cuda")
            
            # Compute CKA for this batch
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