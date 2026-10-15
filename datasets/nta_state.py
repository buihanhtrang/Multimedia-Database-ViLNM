"""Per-sample caption state for the Noise Token Aware module (Sec. III-B).

The module needs "only one forward pass for each epoch": during epoch *e* the
model emits the elimination probabilities r' (Eq. 4) for the captions it is
currently looking at, and those probabilities are used to build the captions for
epoch *e+1* (Eq. 5).

`NTAState` keeps three things per training sample:

  original  -- the untouched word token ids, never modified;
  noise     -- the most recent r' value for each *original* token;
  current   -- the token ids actually fed to the model this epoch, together with
               `src_index`, which maps each current position back to an original
               one so `update()` can scatter r' onto the right token.

Regenerating `current` from `original` each epoch (rather than eliminating on
top of the previous epoch's result) is what keeps deletions from compounding
until the caption is empty, while still only ever costing one forward pass.
"""

import random
from typing import List, Sequence

import numpy as np

from model.nta import eliminate_tokens


class NTAState:
    def __init__(self,
                 captions: Sequence[Sequence[int]],
                 mask_token: int,
                 vocab_limit: int,
                 rate: float = 0.2,
                 p1: float = 0.6,
                 p2: float = 0.8,
                 seed: int = 1):
        self.original: List[List[int]] = [list(c) for c in captions]
        self.noise: List[np.ndarray] = [
            np.full(len(c), rate, dtype=np.float32) for c in self.original]
        self.current: List[List[int]] = [list(c) for c in self.original]
        self.src_index: List[List[int]] = [list(range(len(c))) for c in self.original]

        self.mask_token = mask_token
        self.vocab_limit = vocab_limit
        self.rate, self.p1, self.p2 = rate, p1, p2
        self.rng = random.Random(seed)
        self.last_counts = [0, 0, 0]

    def __len__(self):
        return len(self.original)

    def tokens(self, index: int) -> List[int]:
        return self.current[index]

    def update(self, index: int, noise_row: np.ndarray):
        """Scatter one row of r' (indexed by *current* word position) onto `noise`.

        `noise_row` is the r' vector for the padded caption, so word position j
        lives at `noise_row[j + 1]` ([SOS] occupies slot 0).
        """
        src = self.src_index[index]
        n = min(len(src), max(len(noise_row) - 1, 0))
        target = self.noise[index]
        for j in range(n):
            target[src[j]] = noise_row[j + 1]

    def resample(self):
        """Rebuild every training caption with Eq. (5).  Called between epochs."""
        counts = [0, 0, 0]
        for i, (tokens, noise) in enumerate(zip(self.original, self.noise)):
            new_tokens, src, c = eliminate_tokens(
                tokens, noise, self.mask_token, self.vocab_limit,
                self.p1, self.p2, self.rng)
            self.current[i], self.src_index[i] = new_tokens, src
            counts = [a + b for a, b in zip(counts, c)]
        self.last_counts = counts
        return counts

    def state_dict(self) -> dict:
        """Everything `resample()` and `tokens()` depend on, for exact resume."""
        return {"noise": self.noise, "current": self.current,
                "src_index": self.src_index, "rng": self.rng.getstate(),
                "last_counts": self.last_counts}

    def load_state_dict(self, state: dict):
        if len(state["noise"]) != len(self.original):
            raise ValueError(f"NTA state holds {len(state['noise'])} captions, "
                             f"dataset has {len(self.original)}")
        self.noise = state["noise"]
        self.current = state["current"]
        self.src_index = state["src_index"]
        self.rng.setstate(state["rng"])
        self.last_counts = state["last_counts"]

    def stats(self):
        total = sum(len(c) for c in self.original) or 1
        n_mask, n_replace, n_delete = self.last_counts
        return dict(
            mask=n_mask, replace=n_replace, delete=n_delete,
            eliminated_ratio=(n_mask + n_replace + n_delete) / total,
            mean_noise=float(np.mean([n.mean() for n in self.noise if len(n)]) or 0.0),
        )
