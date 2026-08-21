"""Evaluate a trained ViLNM checkpoint.

    python test.py --config_file logs/CUHK-PEDES/vilnm/configs.json \
                   --checkpoint logs/CUHK-PEDES/vilnm/best.pth
"""

import argparse
import logging

import torch

from datasets.build import build_test_dataloader
from model.build import build_model
from utils.iotools import load_checkpoint, load_train_configs
from utils.logger import setup_logger
from utils.metrics import Evaluator


def main():
    parser = argparse.ArgumentParser("ViLNM evaluation")
    parser.add_argument("--config_file", required=True,
                        help="configs.json written by train.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sim_alpha", type=float, default=None)
    parser.add_argument("--sim_beta", type=float, default=None)
    cli = parser.parse_args()

    args = load_train_configs(cli.config_file)
    args.img_size = tuple(args.img_size)
    if cli.sim_alpha is not None:
        args.sim_alpha = cli.sim_alpha
    if cli.sim_beta is not None:
        args.sim_beta = cli.sim_beta

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if cli.device == "auto" else cli.device
    logger = setup_logger("ViLNM", save_dir=args.output_dir, if_train=False)
    logger.info("Evaluating %s on %s (device %s)", cli.checkpoint,
                args.dataset_name, device)

    gallery_loader, query_loader, num_classes = build_test_dataloader(args)
    # The checkpoint carries the CLIP weights, so skip re-reading them here.
    args.no_pretrained = True
    model = build_model(args, num_classes)
    ckpt = load_checkpoint(cli.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(device)

    Evaluator(gallery_loader, query_loader, args.sim_alpha,
              args.sim_beta, device).eval(model)


if __name__ == "__main__":
    main()
