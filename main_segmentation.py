# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
r"""
Continual pretraining + segmentation entry point.

Paper: "Continual Pretraining Prostate MRI ViT Encoder for Task Alignment", ISBI 2026.

Stage 1 (continual pretraining, LoRA on PROMIS segmentation):
    python main_segmentation.py --state train --model profound_vit --train scratch --case continual \
        --dataset Promis --promis_mask gland --root /path/to/data \
        --pretrain pretrained/profound_mae_checkpoint-799.pth --output_dir outputs/continual \
        --batch_size 4 --epochs 100 --warmup_epochs 5 --lr 1e-3 --seed 2021

Stage 2 (downstream lesion / zone segmentation), --case scratch | prenc | posprenc:
    python main_segmentation.py --state train --model profound_vit --train scratch --case posprenc \
        --dataset Promis --promis_mask lesion --root /path/to/data \
        --pretrain outputs/continual/Promis/profound_vit/scratch/2021/best.pth.tar ...

CKA analysis (Table 6): --state sim_test, see scripts/cka.sh.
See scripts/ for all commands used in the paper.
"""
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
from typing import Callable, List, Optional
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_seg import (
    build_UCL_loader,
    build_Anatomy_loader,
    build_BpAnatomy_loader,
    build_Promis_loader,
    build_PromisHist_loader,
    build_PromisPirads3_loader,
)
import timm.optim.optim_factory as optim_factory
import util.misc as misc
import torch.distributed as dist
from models.build_segment import buidl_model
from engine.segment import train_one_epoch, validation, test, test_with_uncertainty, sim_test
from models.vit import vit_base_patch16_dec512d8b
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import write_log
import traceback


def get_available_gpus(min_mem_required_gb):
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        free_mem_gb = torch.cuda.mem_get_info(i)[0] / 1024**3
        print(f"GPU {i}: {free_mem_gb} GB free")
        if free_mem_gb >= min_mem_required_gb:
            available_gpus.append(str(i))
    return available_gpus
    
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def get_args_parser():
    parser = argparse.ArgumentParser("segmentation", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--root", default="data", type=str, help="data root; CSV paths in spilt/ are relative to it"
    )

    # Model parameters
    parser.add_argument("--model", help="model name")
    parser.add_argument(
        "--input_size",
        default=(64, 256, 256),
        type=tuple_type,
        help="images input size",
    )
    parser.add_argument("--crop_spatial_size", default=(64, 256, 256), type=tuple_type)

    parser.add_argument(
        "--train", choices=["fintune", "freeze", "scratch", "lora"], help="train method"
    )
    parser.add_argument(
        "--state", choices=["train", "test", "sim_test"], default="train", help="train or test"
    )
    parser.add_argument(
        "--case", default=None, type=str,
        help="encoder initialisation: scratch | prenc | continual | posprenc | test (see models/build_segment.py)"
    )

    # In the argument parser section, add:
    parser.add_argument('--sliding_window', action="store_true",
                    help='Use sliding window inference for test')

    parser.add_argument("--use_mc_dropout", action="store_true", help="Use MC Dropout during inference")
    parser.add_argument("--mc_samples", default= 30, type=int, help="Number of MC Dropout samples")
    parser.add_argument("--use_uncertainty_modulation", action="store_true", 
                    help="Use uncertainty modulation during inference")
    


        # Loss weights for uncertainty-guided training
    parser.add_argument("--boundary_loss_weight", default=0.1, type=float,
                        help="Weight for boundary-aware loss")
    parser.add_argument("--calibration_loss_weight", default=0.05, type=float,
                        help="Weight for confidence calibration loss")
    
    parser.add_argument('--use_cucl', action="store_true",
                    help='Enable Contrastive Uncertainty Calibration Loss')
    parser.add_argument('--cucl_contrastive_weight', type=float, default=0.05,
                        help='Weight for CUCL contrastive loss')
    parser.add_argument('--cucl_calibration_weight', type=float, default=0.1,
                    help='Weight for CUCL calibration loss')
    parser.add_argument('--cucl_temperature', type=float, default=0.1,
                        help='CUCL InfoNCE temperature')
    parser.add_argument('--cucl_uncertainty_threshold', type=float, default=0.5,
                        help='Threshold used to split low/high uncertainty voxels in CUCL')
    parser.add_argument('--cucl_num_samples', type=int, default=256,
                        help='Number of sampled anchors/negatives for CUCL')
    parser.add_argument('--aux_loss_warmup_epochs', type=int, default=20,
                        help='Warmup epochs for boundary/calibration/CUCL loss weights (0 disables)')
    
    parser.add_argument("--pretrain", default=None, type=str,
                        help="checkpoint to initialise from: ProFound MAE weights (--case prenc / scratch) "
                             "or the continual-pretraining checkpoint (--case posprenc / test)")
    parser.add_argument("--unfreeze_encoder", action="store_true",
                        help="--case posprenc: also train the encoder weights (used for the Anatomy 'Ours' result)")
    parser.add_argument("--promis_mask", default="lesion", choices=["lesion", "gland"],
                        help="PROMIS mask used by --dataset Promis: lesion or prostate gland")
    parser.add_argument("--mae_checkpoint", default="pretrained/profound_mae_checkpoint-799.pth", type=str,
                        help="ProFound MAE checkpoint; used as the 'pretrained' encoder in --state sim_test (CKA)")
    parser.add_argument("--continual_checkpoint", default=None, type=str,
                        help="continual-pretraining checkpoint; used as the 'continual' encoder in --state sim_test "
                             "(defaults to --pretrain)")
    parser.add_argument("--tolerance", default=5, type=int)
    parser.add_argument("--spacing", default=(1.0, 0.5, 0.5), type=tuple)
    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="weight decay (default: 1e-5)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument(
        "--output_dir",
        default="./outputseg",
        help="path where to save, empty for no saving",
    )

    # parser.add_argument(
    #     "--model_weights",
    #     default=None,
    #     type=str,
    #     help="path to trained model weights",
    # )

    parser.add_argument(
        "--log_dir", default="./outputseg", help="path where to tensorboard log"
    )
    parser.add_argument("--dataset", default="UCL", help="dataset name")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--layer_decay", type=float, default=0.6)
    parser.add_argument(
        "--layer_decay_type",
        type=str,
        choices=["single", "group"],
        default="group",
        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                        whereas the group strategy assigns the same decaying value for three consecutive layers""",
    )

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--data20", action="store_true", help="Use 20 training data")
    parser.set_defaults(data20=False)

    parser.add_argument("--data_num", default=0, type=int, help="number of train data")

    parser.add_argument(
        "--prompt", action="store_true", help="Use visual prompt tuning"
    )
    parser.set_defaults(prompt=False)

    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--min_mem", type=int, default=2, help="Minimum GPU memory required (in GB)")

    parser.add_argument('--uncertainty', action='store_true', help='Enable uncertainty estimation')
    parser.add_argument('--sim_batches', type=int, default=10, help='Number of batches for similarity test')

    return parser


def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    # Get available GPUs based on memory requirement
    # available_gpus = get_available_gpus(args.min_mem)
    # if not available_gpus:
    #     raise RuntimeError(f"No GPUs available with at least {args.min_mem} GB free memory!")

    # # Set visible devices before any CUDA operations
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available_gpus)
    # print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Now that CUDA_VISIBLE_DEVICES is set, let's check again to confirm
    torch.cuda.empty_cache()  # Optional: clear leftover memory
    device = torch.device("cuda")
    # print(f"Running on accessible GPU: {device}, accessible GPUs: {available_gpus}")

    # device = "cuda"
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if args.data20 or args.data_num > 0:
        args.log_dir = os.path.join(
            args.log_dir, args.dataset, args.model, args.train, str(args.epochs), str(args.lr), str(args.seed)
            )
        args.output_dir = os.path.join(
            args.output_dir, args.dataset, args.model, args.train, str(args.epochs), str(args.lr), str(args.seed)
        )
    else:  
        args.log_dir = os.path.join(
            args.log_dir, args.dataset, args.model, args.train, str(args.seed)
        )
        args.output_dir = os.path.join(
            args.output_dir, args.dataset, args.model, args.train, str(args.seed)
        )

    if args.dataset == "UCL":
        data_loader_train, data_loader_val, data_loader_test = build_UCL_loader(args)
        args.sliding_window = False

    elif args.dataset == "Promis":
        data_loader_train, data_loader_val, data_loader_test = build_Promis_loader(args)
        args.sliding_window = False

    elif args.dataset == "PromisHist":
        data_loader_train, data_loader_val, data_loader_test = build_PromisHist_loader(args)
        args.sliding_window = False

    elif args.dataset == "PromisPirads3":
        data_loader_train, data_loader_val, data_loader_test = build_PromisPirads3_loader(args)
        args.sliding_window = False

    elif args.dataset == "PromisSR":
        data_loader_train, data_loader_val, data_loader_test = build_Promis_loader(args)
        args.sliding_window = False

    elif args.dataset == "anatomy":
        if args.train == "fintune" or args.train == "freeze":
            if args.prompt:
                data_loader_train, data_loader_val, data_loader_test = (
                    build_Anatomy_loader(args)
                )
            else:
                data_loader_train, data_loader_val, data_loader_test = (
                    build_BpAnatomy_loader(args)
                )
        else:
            data_loader_train, data_loader_val, data_loader_test = build_Anatomy_loader(
                args
            )

        # args.sliding_window = False
        # args.sliding_window = True
    else:
        raise NotImplementedError(f"unknown schedule sampler: {args.dataset}")
    print(f"train data: {len(data_loader_train)}, val data: {len(data_loader_val)}, test data: {len(data_loader_test)}")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # define the model
    model, optimizer = buidl_model(args=args, device=device) #pretrained weights loaded

    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )

    if args.state == "train":
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()

        best_loss = np.inf
        for epoch in range(args.start_epoch, args.epochs):
            
            train_stats = train_one_epoch(
                model,
                data_loader_train,
                optimizer,
                device,
                epoch,
                None,
                log_writer=log_writer,
                args=args,
                use_mc_dropout= args.use_mc_dropout,
                mc_samples= args.mc_samples,
                use_uncertainty_modulation= args.use_uncertainty_modulation,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }

            write_log(log_writer, log_stats, args)

            if args.output_dir and (epoch % args.val_interval == 0 or epoch + 1 == args.epochs):
                epoch_loss = validation(
                    model, data_loader_val, device, epoch,
                    use_mc_dropout= args.use_mc_dropout,
                    mc_samples= args.mc_samples,
                    use_uncertainty_modulation= args.use_uncertainty_modulation, args=args)
                
                is_best = False
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    is_best = True
                misc.save_best_model(
                    args=args,
                    model=model,
                    model_without_ddp=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    is_best=is_best,
                )

                val_stats = {
                    "val_loss": str(epoch_loss),
                    "epoch": epoch,
                    "is_best": is_best,
                }
                write_log(log_writer, val_stats, args)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    elif args.state == "test":

        try:
            test_stats = test(
                model=model,
                test_loader=data_loader_test,
                args=args,
                sliding_window=args.sliding_window,
                use_mc_dropout= args.use_mc_dropout,
                mc_samples= args.mc_samples,
                use_uncertainty_modulation= args.use_uncertainty_modulation,
            )

            # If uncertainty estimation is enabled
            if args.uncertainty:
                uncertainty_stats = test_with_uncertainty(
                    model=model, 
                    test_loader=data_loader_test,
                    args=args,
                    sliding_window=args.sliding_window
                )
                test_stats.update(uncertainty_stats)


            print("Test stats:", test_stats)  # Add this line for debugging
            write_log(log_writer, test_stats, args)
        except Exception as e:
            print(f"Error during testing: {e}")
            traceback.print_exc()

    elif args.state == "sim_test":
            #similarity test
        try:
            sim_results = sim_test(
                test_loader=data_loader_val,
                args=args,
                lora=True,
                num_batches= args.sim_batches,
                sliding_window=args.sliding_window,
            )
        except Exception as e:
            print(f"Error during similarity test: {e}")
            traceback.print_exc()

        print("sim_results shape", sim_results.shape if sim_results is not None else "None")
        # Only visualize if sim_results was successfully created
        if sim_results is not None and hasattr(sim_results, 'shape') and len(sim_results.shape) >= 3:
            # Visualize results
            layer_names = ['Layer 3', 'Layer 5', 'Layer 7', 'Layer 11']
            model_names = ['pretrained', 'continual_trained', 'finetuned']
            fig_path = os.path.join(args.output_dir, "sim_figures")
            os.makedirs(fig_path, exist_ok=True)

            # Create a figure with 2x2 subplots (square arrangement)
            num_layers = sim_results.shape[2]
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()  # Flatten the 2x2 grid to make indexing easier

            for l in range(num_layers):
                ax = axes[l]
                im = ax.imshow(sim_results[:, :, l], cmap='viridis', vmin=0, vmax=1)
                ax.set_title(f'CKA Similarity - {layer_names[l]}')
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names)
                ax.set_yticks(range(len(model_names)))
                ax.set_yticklabels(model_names)
                fig.colorbar(im, ax=ax, label='CKA Similarity')

            # Add a tight layout with padding to prevent overlap
            plt.tight_layout(pad=2.0)
            plt.savefig(os.path.join(fig_path, "cka_similarity_combined.png"), dpi=300)
            plt.close()
        else:
            print("Warning: CKA similarity visualization skipped - no valid results available")
    else:
        raise NotImplementedError(f"unknown state: {args.state}")

if __name__ == "__main__":
    # Parse command line argumentss
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
