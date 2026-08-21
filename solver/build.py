"""Optimizer construction.

The CLIP backbone is fine-tuned at `--lr`, while the randomly initialised ViLNM
modules (the two LA branches and the ID classifier) get `--lr * --lr_factor`,
matching IRRA's treatment of newly added parameters.
"""

import logging

import torch

from .lr_scheduler import build_lr_scheduler

logger = logging.getLogger("ViLNM.solver")

NEW_MODULES = ("classifier", "la_visual", "la_text")


def build_optimizer(args, model):
    params = []
    n_new = 0
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        if any(key.startswith(m) for m in NEW_MODULES):
            lr = lr * args.lr_factor
            n_new += 1
        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=args.lr,
                                      betas=(args.adam_beta1, args.adam_beta2),
                                      eps=1e-8)
    else:
        optimizer = torch.optim.Adam(params, lr=args.lr,
                                     betas=(args.adam_beta1, args.adam_beta2),
                                     eps=1e-8)

    logger.info("Built %s: %d parameter groups (%d at lr x%.1f)",
                args.optimizer, len(params), n_new, args.lr_factor)
    return optimizer
