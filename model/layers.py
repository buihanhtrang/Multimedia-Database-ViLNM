"""Local Aggregation (LA) module -- Sec. III-A, Eq. (1).

    v_local = MaxPool( MLP(v_{1..Nv}) + FC(v_{1..Nv}) )
    t_local = MaxPool( MLP(t_{1..Nt}) + FC(t_{1..Nt}) )

The paper describes the transformation as "compress the features into a lower
dimensional space before expanding them back to a higher dimension to extract
essential information", plus "residual learning to capture finer details" and a
final max pooling.  So `MLP` is a bottleneck (d -> d/r -> d) and `FC` is the
single linear shortcut that is summed with it.
"""

import torch
import torch.nn as nn


class LocalAggregation(nn.Module):
    """Bottleneck MLP + linear shortcut, followed by max pooling over tokens."""

    def __init__(self, dim: int, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
        )
        self.fc = nn.Linear(dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            tokens: [B, N, D] local token sequence (patch-level or word-level).
            mask:   [B, N] bool, True for valid tokens.  Padding is pushed to
                    -inf before max pooling so it can never win the max.
        Returns:
            pooled: [B, D]    the local representation (v_local / t_local).
            feats:  [B, N, D] the per-token features, consumed by NTA.
        """
        feats = self.mlp(tokens) + self.fc(tokens)
        if mask is None:
            pooled = feats.max(dim=1).values
        else:
            neg_inf = torch.finfo(feats.dtype).min
            pooled = feats.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=1).values
        return pooled, feats
