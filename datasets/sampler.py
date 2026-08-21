"""Identity-balanced batch sampler.

Optional, but useful for the intra-modal term of the JII loss (Eq. 7): its
negatives are restricted to *different* identities, so a batch that contains
several images per identity gives a mix of same-identity pairs (excluded) and
hard different-identity pairs (kept).
"""

import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """Sample `num_instances` samples for each of `batch_size // num_instances` ids."""

    def __init__(self, data_source, batch_size: int, num_instances: int):
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

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(list(batch_idxs_dict.keys()))
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                final_idxs.extend(batch_idxs_dict[pid].pop(0))
                if not batch_idxs_dict[pid]:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length
