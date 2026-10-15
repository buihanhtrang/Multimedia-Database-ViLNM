"""Train ViLNM.

Examples
--------
# CUHK-PEDES, paper settings
python train.py --dataset_name CUHK-PEDES --root_dir data --batch_size 64

# RTX 3050-class GPU: 4 epochs per session, then resume where it stopped
python train.py --dataset_name CUHK-PEDES --num_epoch 30 --batch_size 16 \
    --amp True --num_workers 6 --stop_after_epochs 4 --mlflow True
python train.py ... --resume logs/CUHK-PEDES/vilnm/last.pth

# CPU smoke test on the synthetic benchmark
python tools/make_synthetic_data.py
python train.py --dataset_name SYNTH-PEDES --name smoke --num_epoch 2 \
    --batch_size 8 --img_size 96 32 --no_pretrained True --eval_period 1
"""

import json
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
from utils.tracker import Tracker


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
    if args.resume:
        check_resume_args(args, logger)   # before configs.json is overwritten
    save_train_configs(args.output_dir, args)

    train_loader, gallery_loader, query_loader, num_classes, _ = build_dataloader(args)
    model = build_model(args, num_classes).to(device)

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    evaluator = Evaluator(gallery_loader, query_loader, args.sim_alpha,
                          args.sim_beta, device)

    resume = None
    if args.resume:
        resume = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(resume["model"])
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        if "scheduler" in resume:
            scheduler.load_state_dict(resume["scheduler"])
        if "rng" not in resume:
            logger.warning("%s holds weights only (saved with --save_optimizer False): "
                           "optimizer, NTA and RNG state start fresh", args.resume)
        where = (f"epoch {resume['epoch']} step {resume['step']}"
                 if not resume.get("epoch_complete", True)
                 else f"end of epoch {resume['epoch']}")
        logger.info("Resumed from %s (%s)", args.resume, where)

    tracker = Tracker(args, run_id=resume.get("mlflow_run_id") if resume else None)
    do_train(args, model, train_loader, evaluator, optimizer, scheduler,
             device, resume, tracker)


# Changing any of these between sessions silently changes the training run.
TRAJECTORY_ARGS = ("num_epoch", "batch_size", "lr", "lrscheduler", "warmup_epochs",
                   "seed", "sampler", "img_size", "use_nta", "use_jii", "use_ndf",
                   "use_id", "loss_on_global", "loss_on_local", "nta_rate", "amp")


def check_resume_args(args, logger):
    path = op.join(op.dirname(args.resume), "configs.json")
    if not op.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        old = json.load(f)
    now = json.loads(json.dumps(vars(args), default=str))
    for key in TRAJECTORY_ARGS:
        if key in old and old[key] != now.get(key):
            logger.warning("--%s changed since the checkpoint was written: %s -> %s",
                           key, old[key], now.get(key))


if __name__ == "__main__":
    main()
