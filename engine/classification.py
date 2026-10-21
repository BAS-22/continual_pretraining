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
from util.metric import accuracy, ConfusionMatrix, kappa
from sklearn.metrics import (
    roc_auc_score,
    top_k_accuracy_score,
    f1_score,
    confusion_matrix,
)
from torchmetrics.classification import (
    BinarySpecificityAtSensitivity,
    BinarySensitivityAtSpecificity,
)

import pdb
from util.feature_sim import compare_vit_encoders_with_cka_batch
from sklearn.utils import resample

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

    if args.dataset == "promis":
        # pos_weight = torch.ones([args.num_classes]).to(device) * 5 
        loss_cal = torch.nn.BCEWithLogitsLoss()
    else:
        if args.num_classes > 1:
            loss_cal = torch.nn.CrossEntropyLoss()
        else:
            loss_cal = torch.nn.BCEWithLogitsLoss()
    
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
        # print("logit: ", logit.shape, "gt: ", gt.shape, "image: ", img.shape)
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
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validation(model, data_loader_val, device, epoch, args):
    model.eval()

    if args.dataset == "promis":
        loss_cal = torch.nn.BCEWithLogitsLoss()
    else:
        if args.num_classes > 1:
            loss_cal = torch.nn.CrossEntropyLoss()
        else:
            loss_cal = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        loss_summary = []
        for idx, (img, gt, _) in enumerate(data_loader_val):
            img, gt = img.to(device), gt.to(device)
            mask = model(img)
            loss = loss_cal(mask, gt)
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

def test(model, test_loader, args):
    filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    model.load_state_dict(torch.load(filepath_best)["model"])
    model.eval()
    prob, gts = [], []
    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            img, gt = img.to(args.device), gt.to(args.device)
            logit = model(img)
            prob.append(logit)
            gts.append(gt)

    if args.dataset == "risk":
        return test_risk(prob, gts)
    elif args.dataset == "screening":
        return test_screening(prob, gts)
    elif args.dataset == "promis":
        return test_promis(prob, gts)
    elif args.dataset == "promis3hist":
        return test_promis_pirads3(prob, gts)
    else:
        raise NotImplementedError(f"unknown dataset: {args.dataset}")

def test_risk_ori(prob, gts):
    log_stats = {}
    prob = torch.cat(prob, 0)
    prob = torch.softmax(prob, dim=-1).cpu().numpy()
    gts = torch.cat(gts, 0).cpu().numpy()

    score_acc = top_k_accuracy_score(gts, prob, k=1) * 100
    score_qwk = kappa(gts, np.argmax(prob, 1))
    score_auc = roc_auc_score(gts, prob, multi_class="ovr") * 100
    score_f1 = f1_score(gts, np.argmax(prob, 1), average="macro") * 100

    print("score")
    print(f"acc\t auc \t qwk \t f1")
    print(f"{score_acc:.2f} \t {score_auc:.2f} \t {score_qwk:.4f} \t {score_f1:.2f}")
    log_stats["4-class_acc"] = f"{score_acc:.2f}"
    log_stats["4-class_auc"] = f"{score_auc:.2f}"
    log_stats["4-class_qwk"] = f"{score_qwk:.4f}"
    log_stats["4-class_f1"] = f"{score_f1:.2f}"

    # 2 3 4 5 four classes 0 1 2 3

    sig_prob = np.sum(prob[:, 1:], -1)
    sig_gts = (gts > 0).astype(int)
    sig_acc = top_k_accuracy_score(sig_gts, sig_prob, k=1) * 100
    sig_auc = roc_auc_score(sig_gts, sig_prob) * 100
    sig_f1 = f1_score(sig_gts, sig_prob > 0.5) * 100

    print("Pirads >=3")
    print(f"auc \t f1 ")
    print(f"{sig_auc:.2f} \t {sig_f1:.2f}")

    log_stats["leq3_auc"]=f"{sig_auc:.2f}"
    log_stats["leq3_f1"]=f"{sig_f1:.2f}"

    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(
            torch.from_numpy(sig_prob), torch.from_numpy(sig_gts)
        )
        sig_specificity = sig_specificity * 100
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(
            torch.from_numpy(sig_prob), torch.from_numpy(sig_gts)
        )
        sig_sensitivity = sig_sensitivity* 100

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity:.2f} \t {sig_sensitivity:.2f} ")
        log_stats[f"leq3_specificity_at_{i}"]=f"{sig_specificity:.2f}"
        log_stats[f"leq3_sensitivity_at_{i}"]=f"{sig_sensitivity:.2f}"

    sig_prob = np.sum(prob[:, 2:], -1)
    sig_gts = (gts > 1).astype(int)
    sig_acc = top_k_accuracy_score(sig_gts, sig_prob, k=1) * 100
    sig_auc = roc_auc_score(sig_gts, sig_prob) * 100
    sig_f1 = f1_score(sig_gts, sig_prob > 0.5) * 100

    print("Pirads >=4")
    print(f"auc \t f1 ")
    print(f"{sig_auc:.2f} \t {sig_f1:.2f}")

    log_stats["leq4_auc"]=f"{sig_auc:.2f}"
    log_stats["leq4_f1"]=f"{sig_f1:.2f}"

    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(
            torch.from_numpy(sig_prob), torch.from_numpy(sig_gts)
        )
        sig_specificity = sig_specificity * 100
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(
            torch.from_numpy(sig_prob), torch.from_numpy(sig_gts)
        )
        sig_sensitivity = sig_sensitivity* 100

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity:.2f} \t {sig_sensitivity:.2f} ")
        log_stats[f"leq4_specificity_at_{i}"]=f"{sig_specificity:.2f}"
        log_stats[f"leq4_sensitivity_at_{i}"]=f"{sig_sensitivity:.2f}"
    return log_stats

def test_risk(prob, gts):
    log_stats = {}
    prob = torch.cat(prob, 0)
    prob = torch.softmax(prob, dim=-1).cpu().numpy()
    gts = torch.cat(gts, 0).cpu().numpy()

    # Compute metrics
    score_acc = top_k_accuracy_score(gts, prob, k=1) * 100
    score_qwk = kappa(gts, np.argmax(prob, 1))
    score_auc = roc_auc_score(gts, prob, multi_class="ovr") * 100
    score_f1 = f1_score(gts, np.argmax(prob, 1), average="macro") * 100

    print("score")
    print(f"acc\t auc \t qwk \t f1")
    print(f"{score_acc:.2f} \t {score_auc:.2f} \t {score_qwk:.4f} \t {score_f1:.2f}")
    log_stats["4-class_acc"] = f"{score_acc:.2f}"
    log_stats["4-class_auc"] = f"{score_auc:.2f}"
    log_stats["4-class_qwk"] = f"{score_qwk:.4f}"
    log_stats["4-class_f1"] = f"{score_f1:.2f}"

    # Compute AUC Standard Deviation using Bootstrapping
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    auc_scores = []

    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = rng.choice(len(gts), len(gts), replace=True)
        if len(np.unique(gts[indices])) < 2:
            # Skip if resampled data doesn't have all classes
            continue
        auc_scores.append(roc_auc_score(gts[indices], prob[indices], multi_class="ovr") * 100)

    auc_std = np.std(auc_scores)
    print(f"AUC Standard Deviation: {auc_std:.2f}")
    log_stats["4-class_auc_std"] = f"{auc_std:.2f}"

    # Binary classification for PIRADS >= 3
    sig_prob = np.sum(prob[:, 1:], -1)
    sig_gts = (gts > 0).astype(int)
    sig_acc = top_k_accuracy_score(sig_gts, sig_prob, k=1) * 100
    sig_auc = roc_auc_score(sig_gts, sig_prob) * 100
    sig_f1 = f1_score(sig_gts, sig_prob > 0.5) * 100

    print("Pirads >=3")
    print(f"auc \t f1 ")
    print(f"{sig_auc:.2f} \t {sig_f1:.2f}")

    log_stats["leq3_auc"] = f"{sig_auc:.2f}"
    log_stats["leq3_f1"] = f"{sig_f1:.2f}"

    # Compute AUC Standard Deviation for PIRADS >= 3
    auc_scores = []
    for _ in range(n_bootstraps):
        indices = rng.choice(len(sig_gts), len(sig_gts), replace=True)
        if len(np.unique(sig_gts[indices])) < 2:
            continue
        auc_scores.append(roc_auc_score(sig_gts[indices], sig_prob[indices]) * 100)

    auc_std = np.std(auc_scores)
    print(f"PIRADS >=3 AUC Standard Deviation: {auc_std:.2f}")
    log_stats["leq3_auc_std"] = f"{auc_std:.2f}"

    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(
            torch.from_numpy(sig_prob), torch.from_numpy(sig_gts)
        )
        sig_specificity = sig_specificity * 100
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(
            torch.from_numpy(sig_prob), torch.from_numpy(sig_gts)
        )
        sig_sensitivity = sig_sensitivity * 100

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity:.2f} \t {sig_sensitivity:.2f} ")
        log_stats[f"leq3_specificity_at_{i}"] = f"{sig_specificity:.2f}"
        log_stats[f"leq3_sensitivity_at_{i}"] = f"{sig_sensitivity:.2f}"

    # Binary classification for PIRADS >= 4
    sig_prob = np.sum(prob[:, 2:], -1)
    sig_gts = (gts > 1).astype(int)
    sig_acc = top_k_accuracy_score(sig_gts, sig_prob, k=1) * 100
    sig_auc = roc_auc_score(sig_gts, sig_prob) * 100
    sig_f1 = f1_score(sig_gts, sig_prob > 0.5) * 100

    print("Pirads >=4")
    print(f"auc \t f1 ")
    print(f"{sig_auc:.2f} \t {sig_f1:.2f}")

    log_stats["leq4_auc"] = f"{sig_auc:.2f}"
    log_stats["leq4_f1"] = f"{sig_f1:.2f}"

    # Compute AUC Standard Deviation for PIRADS >= 4
    auc_scores = []
    for _ in range(n_bootstraps):
        indices = rng.choice(len(sig_gts), len(sig_gts), replace=True)
        if len(np.unique(sig_gts[indices])) < 2:
            continue
        auc_scores.append(roc_auc_score(sig_gts[indices], sig_prob[indices]) * 100)

    auc_std = np.std(auc_scores)
    print(f"PIRADS >=4 AUC Standard Deviation: {auc_std:.2f}")
    log_stats["leq4_auc_std"] = f"{auc_std:.2f}"

    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(
            torch.from_numpy(sig_prob), torch.from_numpy(sig_gts)
        )
        sig_specificity = sig_specificity * 100
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(
            torch.from_numpy(sig_prob), torch.from_numpy(sig_gts)
        )
        sig_sensitivity = sig_sensitivity * 100

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity:.2f} \t {sig_sensitivity:.2f} ")
        log_stats[f"leq4_specificity_at_{i}"] = f"{sig_specificity:.2f}"
        log_stats[f"leq4_sensitivity_at_{i}"] = f"{sig_sensitivity:.2f}"

    return log_stats

def test_screening(prob, gts):
    prob = torch.cat(prob, 0)
    prob = torch.sigmoid(prob).cpu().numpy()
    gts = torch.cat(gts, 0).long().cpu().numpy()

    np.savez("result.npz", gts=gts, prob=prob)
    score_acc = top_k_accuracy_score(gts, prob, k=1) * 100
    score_auc = roc_auc_score(gts, prob) * 100
    score_f1 = f1_score(gts, np.argmax(prob, 1)) * 100

    print(f"acc\t auc \t f1")
    print(f"{score_acc:.2f} \t {score_auc:.2f} \t {score_f1:.2f}")

    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(torch.from_numpy(prob), torch.from_numpy(gts))
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(torch.from_numpy(prob), torch.from_numpy(gts))

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity* 100:.2f} \t {sig_sensitivity* 100:.2f} ")

    log_stats = None
    return log_stats

def test_promis_ori(prob, gts):
    log_stats = {}

    prob = torch.cat(prob, 0)
    prob = torch.sigmoid(prob).cpu().numpy()
    gts = torch.cat(gts, 0).cpu().numpy().astype(int)

    # print("probs:",prob[0])
    # print("gts:", gts[0])

    #zone level
    zone_prob = prob.reshape(-1)
    zone_gt = gts.reshape(-1)
    print(f"zone level performance")

    # auc = roc_auc_score(zone_prob, zone_gt) * 100
    auc = roc_auc_score(zone_gt, zone_prob) * 100
    print(f"AUC: {auc:.2f}")
    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(
            torch.from_numpy(zone_prob), torch.from_numpy(zone_gt)
        )
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(
            torch.from_numpy(zone_prob), torch.from_numpy(zone_gt)
        )

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity* 100:.2f} \t {sig_sensitivity* 100:.2f} ")




    #patient level
    patient_prob = prob.max(-1)
    patient_gt = gts.max(-1)
    
    print(f"patient level performance")

    # auc = roc_auc_score(patient_prob, patient_gt) * 100
    auc = roc_auc_score(patient_gt, patient_prob) * 100
    print(f"AUC: {auc:.2f}")
    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(
            torch.from_numpy(patient_prob), torch.from_numpy(patient_gt)
        )
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(
            torch.from_numpy(patient_prob), torch.from_numpy(patient_gt)
        )

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity* 100:.2f} \t {sig_sensitivity* 100:.2f} ")

    
    return log_stats

def test_promis(prob, gts):
    log_stats = {}

    prob = torch.cat(prob, 0)
    prob = torch.sigmoid(prob).cpu().numpy()
    gts = torch.cat(gts, 0).cpu().numpy().astype(int)

    # Zone level
    zone_prob = prob.reshape(-1)
    zone_gt = gts.reshape(-1)
    print(f"zone level performance")

    # Compute AUC
    auc = roc_auc_score(zone_gt, zone_prob) * 100
    print(f"AUC: {auc:.2f}")

    # Compute AUC Standard Deviation using Bootstrapping
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    auc_scores = []

    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = rng.choice(len(zone_gt), len(zone_gt), replace=True)
        if len(np.unique(zone_gt[indices])) < 2:
            # Skip if resampled data doesn't have both classes
            continue
        auc_scores.append(roc_auc_score(zone_gt[indices], zone_prob[indices]) * 100)

    auc_std = np.std(auc_scores)
    print(f"AUC Standard Deviation: {auc_std:.2f}")

    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(
            torch.from_numpy(zone_prob), torch.from_numpy(zone_gt)
        )
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(
            torch.from_numpy(zone_prob), torch.from_numpy(zone_gt)
        )

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity* 100:.2f} \t {sig_sensitivity* 100:.2f} ")

    # Patient level
    patient_prob = prob.max(-1)
    patient_gt = gts.max(-1)

    print(f"patient level performance")

    # Compute AUC
    auc = roc_auc_score(patient_gt, patient_prob) * 100
    print(f"AUC: {auc:.2f}")

    # Compute AUC Standard Deviation using Bootstrapping
    auc_scores = []
    for _ in range(n_bootstraps):
        indices = rng.choice(len(patient_gt), len(patient_gt), replace=True)
        if len(np.unique(patient_gt[indices])) < 2:
            continue
        auc_scores.append(roc_auc_score(patient_gt[indices], patient_prob[indices]) * 100)

    auc_std = np.std(auc_scores)
    print(f"AUC Standard Deviation: {auc_std:.2f}")

    for i in [0.8, 0.9]:
        sig_spec = BinarySpecificityAtSensitivity(min_sensitivity=i, thresholds=None)
        sig_specificity, _ = sig_spec(
            torch.from_numpy(patient_prob), torch.from_numpy(patient_gt)
        )
        sig_sens = BinarySensitivityAtSpecificity(min_specificity=i, thresholds=None)
        sig_sensitivity, _ = sig_sens(
            torch.from_numpy(patient_prob), torch.from_numpy(patient_gt)
        )

        print(f"min: {i}")
        print(f"Specificity at Sensitivity \t Sensitivity at Specificity")
        print(f"{sig_specificity* 100:.2f} \t {sig_sensitivity* 100:.2f} ")

    return log_stats

def test_promis_pirads3(prob, gts):
    # three classes: 0,1,2
    log_stats = {}
    prob = torch.cat(prob, 0)
    prob = torch.softmax(prob, dim=-1).cpu().numpy()
    gts = torch.cat(gts, 0).cpu().numpy().astype(int)

    score_acc = top_k_accuracy_score(gts, prob, k=1) * 100
    score_qwk = kappa(gts, np.argmax(prob, 1))
    score_auc = roc_auc_score(gts, prob, multi_class="ovr") * 100
    score_f1 = f1_score(gts, np.argmax(prob, 1), average="macro") * 100
    print("score")
    print(f"acc\t auc \t qwk \t f1")
    print(f"{score_acc:.2f} \t {score_auc:.2f} \t {score_qwk:.4f} \t {score_f1:.2f}")
    log_stats["3-class_acc"] = f"{score_acc:.2f}"
    log_stats["3-class_auc"] = f"{score_auc:.2f}"
    log_stats["3-class_qwk"] = f"{score_qwk:.4f}"
    log_stats["3-class_f1"] = f"{score_f1:.2f}"

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
    
    filepath_posttrained = (args.continual_checkpoint or args.pretrain)
    filepath_finetuned = os.path.join(args.output_dir, "best.pth.tar")
    # filepath_continual_finetuned = os.path.join(args.output_dir, "best.pth.tar")
    # filepath_pretrained_finetuned = os.path.join(args.prenc_dir, "best.pth.tar") 

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

