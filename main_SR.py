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
Downstream T2W super-resolution entry point (--dataset Promis, --model mae_sr_vit_base).
Encoder initialisation with --case scratch | prenc | posprenc.
See scripts/train_downstream.sh and scripts/test_downstream.sh.
"""
import argparse
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_SR import build_promis_loader_sr
import timm.optim.optim_factory as optim_factory
import util.misc as misc
import torch.distributed as dist
from models.build_SR import build_model
# from models.SuperFormer_vit import build_model
from engine.SR import train_one_epoch, validation, test, test_with_uncertainty, sim_test
from models.vit import vit_base_patch16_dec512d8b
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import write_log
import traceback
import sys

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
    parser = argparse.ArgumentParser("SR", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--sim_batches", default=10, type=int, help="number of batches for similarity test")
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
    parser.add_argument("--state", choices=["train", "test", "sim_test"], help="train or test")
    parser.add_argument("--case", default= "scratch", type=str, help="case name")
    parser.add_argument(
        "--fm_phase", choices=["pretrain", "postrain", "finetune"], help="foundation model phase"
    )
    parser.add_argument("--test", default=False, type=bool, help="test mode")
    parser.add_argument("--pretrain", default=None, type=str,
                        help="checkpoint to initialise from: ProFound MAE weights (--case prenc / scratch) "
                             "or the continual-pretraining checkpoint (--case posprenc / test)")
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
        # data_loader_train, data_loader_val, data_loader_test = build_UCL_loader(args)
        args.sliding_window = False

    elif args.dataset == "Promis":
        data_loader_train, data_loader_val, data_loader_test = build_promis_loader_sr(args)
        args.sliding_window = False

            # Check training data loader
        for batch in data_loader_train:
            print(f"TRAIN - Low-res shape: {batch['low_res'].shape}") #Low-res shape: torch.Size([4, 1, 32, 256, 256])
            print(f"TRAIN - High-res shape: {batch['high_res'].shape}") #TRAIN - High-res shape: torch.Size([4, 1, 64, 256, 256])
            print(f"TRAIN - Subject IDs: {batch['subject_id']}")
            print(f"TRAIN - Batch indices shape: {batch['idx'].shape}")
            break  # Only check first batch

        # sys.exit()

    elif args.dataset == "PromisHist":
        # data_loader_train, data_loader_val, data_loader_test = build_PromisHist_loader(args)
        args.sliding_window = False

    elif args.dataset == "PromisPirads3":
        # data_loader_train, data_loader_val, data_loader_test = build_PromisPirads3_loader(args)
        args.sliding_window = False

    # elif args.dataset == "anatomy":
    #     if args.train == "fintune" or args.train == "freeze":
    #         if args.prompt:
    #             data_loader_train, data_loader_val, data_loader_test = (
    #                 build_Anatomy_loader(args)
    #             )
    #         else:
    #             data_loader_train, data_loader_val, data_loader_test = (
    #                 build_BpAnatomy_loader(args)
    #             )
    #     else:
    #         data_loader_train, data_loader_val, data_loader_test = build_Anatomy_loader(
    #             args
    #         )

    #     args.sliding_window = True
    else:
        raise NotImplementedError(f"unknown schedule sampler: {args.dataset}")
    print(f"train data: {len(data_loader_train)}, val data: {len(data_loader_val)}, test data: {len(data_loader_test)}")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # define the model
    model, optimizer = build_model(args=args, device=device) #pretrained weights loaded

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
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }

            write_log(log_writer, log_stats, args)

            if args.output_dir and (epoch % args.val_interval == 0 or epoch + 1 == args.epochs):
                epoch_loss = validation(model, data_loader_val, device, epoch, args)
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
                test_loader=data_loader_test,
                args=args,
                lora=True,
                num_batches= args.sim_batches
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
            for l in range(sim_results.shape[2]):
                plt.figure(figsize=(8, 6))
                plt.imshow(sim_results[:, :, l], cmap='viridis', vmin=0, vmax=1)
                plt.colorbar(label='CKA Similarity')
                plt.title(f'CKA Similarity - {layer_names[l]}')
                plt.xticks(range(len(model_names)), model_names)
                plt.yticks(range(len(model_names)), model_names)
                plt.savefig(os.path.join(fig_path, f"cka_similarity_SR{layer_names[l]}.png"))
                plt.close()
        else:
            print("Warning: CKA similarity visualization skipped - no valid results available")

    else:
        raise NotImplementedError(f"unknown state: {args.state}")
    #similarity test

    # try:
    #     sim_results = sim_test(
    #         test_loader=data_loader_test,
    #         args=args,
    #         lora=True if args.train == "lora" else False,
    #     )
    # except Exception as e:
    #     print(f"Error during similarity test: {e}")
    #     traceback.print_exc()

    # print("sim_results shape", sim_results.shape if sim_results is not None else "None")
    # # Only visualize if sim_results was successfully created
    # if sim_results is not None and hasattr(sim_results, 'shape') and len(sim_results.shape) >= 3:
    #     # Visualize results
    #     layer_names = ['Layer 3', 'Layer 5', 'Layer 7', 'Layer 11']
    #     model_names = ['pretrained', 'postrained', 'finetuned']

    #     for l in range(sim_results.shape[2]):
    #         plt.figure(figsize=(8, 6))
    #         plt.imshow(sim_results[:, :, l], cmap='viridis', vmin=0, vmax=1)
    #         plt.colorbar(label='CKA Similarity')
    #         plt.title(f'CKA Similarity - {layer_names[l]}')
    #         plt.xticks(range(len(model_names)), model_names)
    #         plt.yticks(range(len(model_names)), model_names)
    #         plt.savefig(f'cka_similarity_layer_{l}.png')
    #         plt.close()
    # else:
    #     print("Warning: CKA similarity visualization skipped - no valid results available")


if __name__ == "__main__":
    # Parse command line argumentss
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
