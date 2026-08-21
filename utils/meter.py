from collections import defaultdict


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


class MetricLogger(object):
    """Tracks a running average per named scalar."""

    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, values: dict, n: int = 1):
        for key, val in values.items():
            self.meters[key].update(float(val), n)

    def __str__(self):
        return ", ".join(f"{k} {v.avg:.4f}" for k, v in sorted(self.meters.items()))
