"""Train ViLNM.

Examples
--------
# CUHK-PEDES, paper settings
python train.py --dataset_name CUHK-PEDES --root_dir data --batch_size 64

# CPU smoke test on the synthetic benchmark
python tools/make_synthetic_data.py
python train.py --dataset_name SYNTH-PEDES --name smoke --num_epoch 2 \
    --batch_size 8 --img_size 96 32 --no_pretrained True --eval_period 1
"""

import logging
import os.path as op

import torch

from datasets.build import build_dataloader
from model.build import build_model
from processor.processor import do_train
from solver.build import build_optimizer
from solver.lr_scheduler import build_lr_scheduler
from utils.iotools import load_checkpoint, save_train_configs, set_seed
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return choice


def main():
    args = get_args()
    set_seed(args.seed)
    logger = setup_logger("ViLNM", save_dir=args.output_dir, if_train=True)
    device = resolve_device(args.device)
    if args.amp and device != "cuda":
        logger.warning("--amp requires CUDA; disabling mixed precision.")
        args.amp = False
    logger.info("Device: %s", device)
    logger.info("Args: %s", vars(args))
    save_train_configs(args.output_dir, args)

    train_loader, gallery_loader, query_loader, num_classes, _ = build_dataloader(args)
    model = build_model(args, num_classes).to(device)

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    evaluator = Evaluator(gallery_loader, query_loader, args.sim_alpha,
                          args.sim_beta, device)

    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    do_train(args, model, train_loader, evaluator, optimizer, scheduler,
             device, start_epoch)


if __name__ == "__main__":
    main()
