"""Training objectives for ViLNM.

    L = L_NDF + L_ID + L_JII            (Eq. 9)
    L_JII = L_InterC + lambda * L_IntraC (Eq. 8)

`compute_interc`  -- Eq. (6), symmetric inter-modal InfoNCE.
`compute_intrac`  -- Eq. (7), intra-modal contrastive loss whose negatives are
                     restricted to *different* person identities.
`compute_id`      -- the identity classification loss of [26].
`compute_ndf`     -- the NDF loss of [19].  The letter only characterises it as
                     similarity-distribution matching that "prevents the model
                     from becoming overly confident by incorporating the reverse
                     KL divergence loss", so it is implemented here as the
                     forward KL of IRRA's SDM plus its reverse-KL counterpart
                     against the pid-derived label distribution.
"""

import torch
import torch.nn.functional as F


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).t()


def compute_interc(sim: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
    """Eq. (6).  `sim[i, j]` = Sim(I_i, T_j); positives are on the diagonal."""
    logits = sim / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    i2t = F.cross_entropy(logits, labels)
    t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (i2t + t2i)


def compute_intrac(sim_cross: torch.Tensor,
                   sim_v: torch.Tensor,
                   sim_t: torch.Tensor,
                   pid: torch.Tensor,
                   tau: float = 0.02) -> torch.Tensor:
    """Eq. (7).

    The positive is the matched cross-modal pair Sim(I_i, T_i); the negatives
    are the *intra-modal* similarities to samples of different identities, which
    is what pushes apart look-alike persons (inter-identity noise).

    Args:
        sim_cross: [B, B] image-text cosine similarity.
        sim_v:     [B, B] image-image cosine similarity.
        sim_t:     [B, B] text-text cosine similarity.
        pid:       [B] person identity labels.
    """
    diff_id = pid.unsqueeze(1) != pid.unsqueeze(0)          # j not in identity_i
    pos = sim_cross.diagonal() / tau                        # [B]

    def branch(sim_intra: torch.Tensor) -> torch.Tensor:
        neg = (sim_intra / tau).masked_fill(~diff_id, float("-inf"))
        # The positive is kept inside the denominator so the loss is >= 0 and
        # finite even when a row has no different-identity negatives.
        denom = torch.logsumexp(torch.cat([pos.unsqueeze(1), neg], dim=1), dim=1)
        return (denom - pos).mean()

    return 0.5 * (branch(sim_v) + branch(sim_t))


def compute_jii(v_feat: torch.Tensor,
                t_feat: torch.Tensor,
                pid: torch.Tensor,
                tau: float = 0.02,
                lambda_intra: float = 0.1):
    """Eq. (8).  Returns (total, inter, intra)."""
    v = F.normalize(v_feat, dim=-1)
    t = F.normalize(t_feat, dim=-1)
    sim_cross = v @ t.t()
    inter = compute_interc(sim_cross, tau)
    intra = compute_intrac(sim_cross, v @ v.t(), t @ t.t(), pid, tau)
    return inter + lambda_intra * intra, inter, intra


def compute_id(image_logits: torch.Tensor,
               text_logits: torch.Tensor,
               labels: torch.Tensor) -> torch.Tensor:
    """Identity classification loss [26]."""
    return 0.5 * (F.cross_entropy(image_logits, labels) +
                  F.cross_entropy(text_logits, labels))


def compute_ndf(v_feat: torch.Tensor,
                t_feat: torch.Tensor,
                pid: torch.Tensor,
                tau: float = 0.02,
                epsilon: float = 1e-8) -> torch.Tensor:
    """NDF loss [19]: bidirectional-KL similarity distribution matching.

    q is the ground-truth distribution induced by the identity labels (uniform
    over every in-batch sample sharing the query's identity); p is the softmax
    over the scaled cosine similarities.  The forward term KL(p||q) is IRRA's
    SDM; the reverse term KL(q||p) is the noise-tolerant part that stops the
    model from collapsing onto a single over-confident match.
    """
    logit_scale = 1.0 / tau
    v = F.normalize(v_feat, dim=-1)
    t = F.normalize(t_feat, dim=-1)

    i2t_logits = logit_scale * (v @ t.t())
    t2i_logits = logit_scale * (t @ v.t())

    same = (pid.unsqueeze(1) == pid.unsqueeze(0)).float()
    q = same / same.sum(dim=1, keepdim=True).clamp(min=1.0)
    log_q = torch.log(q + epsilon)

    def bidirectional_kl(logits: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        forward = (p * (log_p - log_q)).sum(dim=1)          # KL(p || q)
        reverse = (q * (log_q - log_p)).sum(dim=1)          # KL(q || p)
        return (forward + reverse).mean()

    return bidirectional_kl(i2t_logits) + bidirectional_kl(t2i_logits)
