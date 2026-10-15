"""Training samplers.

Both samplers draw their order from a private RNG seeded by (seed, epoch), so the
order of epoch *e* is the same no matter how many times training was stopped and
resumed.  `set_epoch(e)` picks the epoch and `skip(n)` drops the first n indices,
which is how a run resumed mid-epoch continues with the batches it had not yet
seen without re-reading the images of the ones it had.

`RandomIdentitySampler` is optional, but useful for the intra-modal term of the
JII loss (Eq. 7): its negatives are restricted to *different* identities, so a
batch that contains several images per identity gives a mix of same-identity
pairs (excluded) and hard different-identity pairs (kept).
"""

import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


class _ResumableSampler(Sampler):
    def __init__(self, seed: int = 1):
        self.seed = seed
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def skip(self, n: int):
        """Drop the first n indices of the next iteration only."""
        self.start = n

    def _order(self):
        raise NotImplementedError

    def __iter__(self):
        order = self._order()[self.start:]
        self.start = 0
        return iter(order)


class EpochShuffleSampler(_ResumableSampler):
    """Plain random permutation, reproducible per epoch."""

    def __init__(self, num_samples: int, seed: int = 1):
        super().__init__(seed)
        self.num_samples = num_samples

    def _order(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return torch.randperm(self.num_samples, generator=g).tolist()

    def __len__(self):
        return self.num_samples - self.start


class RandomIdentitySampler(_ResumableSampler):
    """Sample `num_instances` samples for each of `batch_size // num_instances` ids."""

    def __init__(self, data_source, batch_size: int, num_instances: int, seed: int = 1):
        super().__init__(seed)
        if batch_size < num_instances:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= num_instances ({num_instances})")
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            self.index_dic[item[0]].append(index)
        self.pids = list(self.index_dic.keys())
        if len(self.pids) < self.num_pids_per_batch:
            raise ValueError(
                f"dataset has {len(self.pids)} identities but a batch needs "
                f"{self.num_pids_per_batch}; lower --batch_size or raise --num_instance")

        self.length = 0
        for pid in self.pids:
            num = max(len(self.index_dic[pid]), self.num_instances)
            self.length += num - num % self.num_instances

    def _order(self):
        py_rng = random.Random(self.seed * 100003 + self.epoch)
        np_rng = np.random.RandomState((self.seed * 100003 + self.epoch) % 2**32)
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np_rng.choice(idxs, size=self.num_instances, replace=True).tolist()
            py_rng.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(list(batch_idxs_dict.keys()))
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = py_rng.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                final_idxs.extend(batch_idxs_dict[pid].pop(0))
                if not batch_idxs_dict[pid]:
                    avai_pids.remove(pid)
        return final_idxs

    def __len__(self):
        return self.length - self.start
