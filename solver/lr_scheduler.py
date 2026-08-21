"""LR schedulers: linear/constant warmup followed by cosine or step decay (IRRA)."""

from bisect import bisect_right

import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupLR(LRScheduler):
    """Common warmup handling for both decay shapes."""

    def __init__(self, optimizer, num_epochs, warmup_epochs=5, warmup_factor=0.1,
                 warmup_method="linear", last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(f"unknown warmup method '{warmup_method}'")
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def _warmup_scale(self):
        if self.last_epoch >= self.warmup_epochs:
            return None
        if self.warmup_method == "constant":
            return self.warmup_factor
        alpha = self.last_epoch / max(self.warmup_epochs, 1)
        return self.warmup_factor * (1 - alpha) + alpha


class WarmupCosineLR(WarmupLR):
    def __init__(self, optimizer, num_epochs, target_lr=0.0, **kwargs):
        self.target_lr = target_lr
        super().__init__(optimizer, num_epochs, **kwargs)

    def get_lr(self):
        scale = self._warmup_scale()
        if scale is not None:
            return [base_lr * scale for base_lr in self.base_lrs]
        import math
        progress = ((self.last_epoch - self.warmup_epochs) /
                    max(self.num_epochs - self.warmup_epochs, 1))
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.target_lr + (base_lr - self.target_lr) * cos
                for base_lr in self.base_lrs]


class WarmupMultiStepLR(WarmupLR):
    def __init__(self, optimizer, num_epochs, milestones=(20, 50), gamma=0.1, **kwargs):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, num_epochs, **kwargs)

    def get_lr(self):
        scale = self._warmup_scale()
        if scale is None:
            scale = 1.0
        decay = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        return [base_lr * scale * decay for base_lr in self.base_lrs]


def build_lr_scheduler(args, optimizer):
    common = dict(warmup_epochs=args.warmup_epochs,
                  warmup_factor=args.warmup_factor,
                  warmup_method=args.warmup_method)
    if args.lrscheduler == "cosine":
        return WarmupCosineLR(optimizer, args.num_epoch, args.target_lr, **common)
    return WarmupMultiStepLR(optimizer, args.num_epoch, args.milestones,
                             args.gamma, **common)
