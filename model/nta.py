"""Noise Token Aware (NTA) module -- Sec. III-B.

Two pieces:

`NoiseTokenAware`  (parameter-free, runs inside the model's forward pass)
    Eq. (2)  Mat   = L2Norm(F^t_local)^T  L2Norm(F^v_local)
    Eq. (3)  r_x   = 1 - max_y Mat_xy
    Eq. (4)  r'_x  = r_x - mean(r) + rate

`eliminate_tokens` (runs on the CPU between epochs, on token ids)
    Eq. (5)  k ~ U(0,1);  a token is eliminated only when k < r'_x, and then
             p = k / r'_x selects the operation:
                 p  <  p1        -> random mask
                 p1 <= p < p2    -> random replace
                 p2 <= p         -> random delete

The paper notes that "we use the eliminated token sequence for model training in
the next epoch, which requires only one forward pass for each epoch".  The
noise levels are therefore cached during an epoch and the training captions for
the *next* epoch are re-derived from the original tokens (see
`datasets/nta_state.py`), which keeps a single forward pass per epoch without
letting deletions accumulate until the caption is empty.
"""

import random
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK, REPLACE, DELETE = 0, 1, 2


class NoiseTokenAware(nn.Module):
    def __init__(self, rate: float = 0.2):
        super().__init__()
        self.rate = rate

    @torch.no_grad()
    def forward(self,
                text_tokens: torch.Tensor,
                image_tokens: torch.Tensor,
                text_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens:  [B, Nt, D] word-level local features F^t_local.
            image_tokens: [B, Nv, D] patch-level local features F^v_local.
            text_mask:    [B, Nt] bool, True for real word tokens.
        Returns:
            r_prime: [B, Nt] elimination probability per text token, clamped to
                     [0, 1]; zero on padded positions.
        """
        t = F.normalize(text_tokens.float(), dim=-1)
        v = F.normalize(image_tokens.float(), dim=-1)

        mat = torch.bmm(t, v.transpose(1, 2))                  # Eq. (2) [B, Nt, Nv]
        r = 1.0 - mat.max(dim=-1).values                        # Eq. (3) [B, Nt]

        mask = text_mask.to(r.dtype)
        n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (r * mask).sum(dim=1, keepdim=True) / n_valid    # mean over valid tokens
        r_prime = r - mean + self.rate                          # Eq. (4)

        return (r_prime.clamp(0.0, 1.0) * mask)


def eliminate_tokens(tokens: Sequence[int],
                     noise: Sequence[float],
                     mask_token: int,
                     vocab_limit: int,
                     p1: float = 0.6,
                     p2: float = 0.8,
                     rng: random.Random = random):
    """Apply Eq. (5) to one caption.

    Args:
        tokens: word token ids *without* [SOS]/[EOS]/padding.
        noise:  matching r' values, same length as `tokens`.
        mask_token: id of `<|mask|>`.
        vocab_limit: exclusive upper bound for "random replace" ids (keeps the
            replacement inside the regular vocabulary, never a special token).
    Returns:
        (kept_tokens, src_index, counts) -- `src_index[j]` is the position in
        `tokens` that produced `kept_tokens[j]`, which lets the trainer scatter
        the next epoch's noise levels back onto the original caption.  `counts`
        is [n_mask, n_replace, n_delete].  At least one token is always kept so
        a caption never collapses to an empty sequence.
    """
    out: List[int] = []
    src: List[int] = []
    counts = [0, 0, 0]
    for i, (tok, r) in enumerate(zip(tokens, noise)):
        k = rng.random()
        if r <= 0.0 or k >= r:                    # not eliminated
            out.append(tok)
            src.append(i)
            continue
        p = k / r
        if p < p1:
            out.append(mask_token)
            src.append(i)
            counts[MASK] += 1
        elif p < p2:
            out.append(rng.randrange(vocab_limit))
            src.append(i)
            counts[REPLACE] += 1
        else:
            counts[DELETE] += 1                   # dropped
    if not out and len(tokens):
        # Degenerate case: keep the least-noisy original token.
        keep = min(range(len(tokens)), key=lambda i: noise[i])
        out, src = [tokens[keep]], [keep]
        counts[DELETE] -= 1
    return out, src, counts
