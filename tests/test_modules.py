"""Unit tests for the ViLNM building blocks.

Run with:  python -m pytest tests/ -v      (or)  python tests/test_modules.py
"""

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.nta_state import NTAState
from model.layers import LocalAggregation
from model.nta import NoiseTokenAware, eliminate_tokens
from model import objectives
from utils.metrics import rank
from utils.simple_tokenizer import SimpleTokenizer


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def test_tokenizer_roundtrip_and_mask_token():
    tok = SimpleTokenizer()
    assert tok.sot_token == 49406 and tok.eot_token == 49407, "CLIP ids must not shift"
    assert tok.mask_token == 49408, "the <|mask|> token is appended after CLIP's specials"
    assert tok.vocab_size == 49409
    ids = tok.encode("a man in a yellow shirt")
    assert tok.decode(ids).strip() == "a man in a yellow shirt"
    assert max(ids) < tok.n_regular_tokens, "regular text must never emit special ids"


# --------------------------------------------------------------------------- #
# LA module -- Eq. (1)
# --------------------------------------------------------------------------- #
def test_la_module_shapes_and_masking():
    la = LocalAggregation(dim=32, reduction=4)
    tokens = torch.randn(3, 7, 32)
    pooled, feats = la(tokens)
    assert pooled.shape == (3, 32) and feats.shape == (3, 7, 32)

    # Padded positions must not influence the max pooling.
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[:, 4:] = False
    tokens_a = tokens.clone()
    tokens_b = tokens.clone()
    tokens_b[:, 4:] = 1e3                       # garbage in the padded region
    pooled_a, _ = la(tokens_a, mask)
    pooled_b, _ = la(tokens_b, mask)
    assert torch.allclose(pooled_a, pooled_b, atol=1e-4)


def test_la_residual_branch_is_used():
    la = LocalAggregation(dim=16, reduction=4)
    tokens = torch.randn(2, 5, 16)
    _, feats = la(tokens)
    expected = la.mlp(tokens) + la.fc(tokens)
    assert torch.allclose(feats, expected)


# --------------------------------------------------------------------------- #
# NTA -- Eqs. (2)-(5)
# --------------------------------------------------------------------------- #
def test_nta_noise_level_detects_mismatched_token():
    """A text token aligned with an image patch must score a lower r' than one
    that is orthogonal to every patch (Eqs. 2-3)."""
    nta = NoiseTokenAware(rate=0.2)
    d = 8
    v = torch.zeros(1, 4, d)
    v[0, 0, 0] = 1.0                            # one distinctive patch
    t = torch.zeros(1, 3, d)
    t[0, 0, 0] = 1.0                            # matches that patch  -> r = 0
    t[0, 1, 1] = 1.0                            # orthogonal          -> r = 1
    t[0, 2, 0] = 1.0
    mask = torch.ones(1, 3, dtype=torch.bool)

    r = nta(t, v, mask)[0]
    assert r[1] > r[0] and r[1] > r[2], f"noisy token should score highest, got {r}"
    assert (r >= 0).all() and (r <= 1).all()


def test_nta_mean_centering_and_padding():
    nta = NoiseTokenAware(rate=0.2)
    torch.manual_seed(0)
    t = torch.randn(2, 6, 16)
    v = torch.randn(2, 5, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 4:] = False

    r = nta(t, v, mask)
    assert (r[:, 4:] == 0).all(), "padded positions must carry no noise level"
    # Eq. (4) centres r on `rate` over the valid tokens (before clamping).
    valid_mean = r[:, :4].mean(dim=1)
    assert torch.allclose(valid_mean, torch.full_like(valid_mean, 0.2), atol=0.05)


def test_eliminate_tokens_operations():
    """Eq. (5): p = k/r' routes to mask / replace / delete."""
    tokens = [10, 11, 12, 13]

    # r' = 0 -> nothing is ever eliminated.
    kept, src, counts = eliminate_tokens(tokens, [0.0] * 4, 49408, 49406,
                                         rng=random.Random(0))
    assert kept == tokens and src == [0, 1, 2, 3] and counts == [0, 0, 0]

    # r' = 1 -> every token is eliminated; k is uniform so p = k, hence the
    # operation mix follows p1 = 0.6 / p2 = 0.8.
    n_mask = n_repl = n_del = 0
    rng = random.Random(7)
    for _ in range(400):
        kept, src, counts = eliminate_tokens(tokens, [1.0] * 4, 49408, 49406,
                                             p1=0.6, p2=0.8, rng=rng)
        n_mask += counts[0]
        n_repl += counts[1]
        n_del += counts[2]
        assert len(kept) == len(src)
        assert all(0 <= s < len(tokens) for s in src)
    total = n_mask + n_repl + n_del
    assert abs(n_mask / total - 0.6) < 0.06, n_mask / total
    assert abs(n_repl / total - 0.2) < 0.06, n_repl / total
    assert abs(n_del / total - 0.2) < 0.06, n_del / total


def test_eliminate_tokens_never_empties_caption():
    for seed in range(50):
        kept, src, _ = eliminate_tokens([5, 6, 7], [1.0, 1.0, 1.0], 49408, 49406,
                                        p1=0.0, p2=0.0,  # force delete
                                        rng=random.Random(seed))
        assert len(kept) >= 1 and len(src) == len(kept)


def test_eliminate_tokens_replacement_stays_in_vocab():
    kept, _, counts = eliminate_tokens([5] * 200, [1.0] * 200, 49408, 49406,
                                       p1=0.0, p2=1.0,   # force replace
                                       rng=random.Random(3))
    assert counts[1] == 200
    assert all(0 <= t < 49406 for t in kept), "replacements must avoid special ids"


# --------------------------------------------------------------------------- #
# NTA caption cache
# --------------------------------------------------------------------------- #
def test_nta_state_roundtrip():
    captions = [[1, 2, 3], [4, 5, 6, 7]]
    state = NTAState(captions, mask_token=49408, vocab_limit=49406, rate=0.2, seed=0)
    assert state.current == captions and state.src_index == [[0, 1, 2], [0, 1, 2, 3]]

    # r' rows are indexed by padded position: slot 0 is [SOS].
    state.update(0, np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert state.noise[0].tolist() == [1.0, 0.0, 0.0]

    state.resample()
    assert 1 <= len(state.current[0]) <= 3
    # Token 0 had r' = 1 so it is always eliminated; tokens 1 and 2 had r' = 0.
    assert state.current[0][-2:] == [2, 3] or len(state.current[0]) < 3

    # The original captions are never mutated, so deletions cannot compound.
    assert state.original == captions


def test_nta_state_does_not_degrade_over_epochs():
    captions = [list(range(20))]
    state = NTAState(captions, 49408, 49406, rate=0.2, seed=1)
    for _ in range(60):
        state.update(0, np.concatenate([[0.0], np.full(20, 0.3, np.float32)]))
        state.resample()
        assert len(state.current[0]) >= 10, "captions must not shrink to nothing"


def test_nta_state_scatters_through_deletions():
    """After a deletion, r' must land on the correct *original* token."""
    state = NTAState([[10, 11, 12]], 49408, 49406, rate=0.2, seed=0)
    state.current[0] = [10, 12]          # token 11 was deleted last epoch
    state.src_index[0] = [0, 2]
    state.update(0, np.array([0.0, 0.9, 0.7, 0.0], dtype=np.float32))
    # index 1 was deleted last epoch, so it keeps its previous value.
    assert np.allclose(state.noise[0], [0.9, 0.2, 0.7], atol=1e-6), state.noise[0]


# --------------------------------------------------------------------------- #
# Objectives -- Eqs. (6)-(9)
# --------------------------------------------------------------------------- #
def test_interc_is_minimised_by_aligned_features():
    torch.manual_seed(0)
    feats = torch.randn(8, 16)
    aligned = objectives.compute_interc(
        torch.nn.functional.normalize(feats, dim=-1) @
        torch.nn.functional.normalize(feats, dim=-1).t(), tau=0.02)
    shuffled = objectives.compute_interc(
        torch.nn.functional.normalize(feats, dim=-1) @
        torch.nn.functional.normalize(feats.roll(1, 0), dim=-1).t(), tau=0.02)
    assert aligned < shuffled


def test_intrac_excludes_same_identity_negatives():
    """Eq. (7): samples sharing the query's identity must not act as negatives."""
    torch.manual_seed(0)
    b, d = 6, 12
    v = torch.nn.functional.normalize(torch.randn(b, d), dim=-1)
    t = torch.nn.functional.normalize(torch.randn(b, d), dim=-1)
    sim_cross, sim_v, sim_t = v @ t.t(), v @ v.t(), t @ t.t()

    all_distinct = torch.arange(b)
    paired = torch.tensor([0, 0, 1, 1, 2, 2])
    loss_distinct = objectives.compute_intrac(sim_cross, sim_v, sim_t, all_distinct)
    loss_paired = objectives.compute_intrac(sim_cross, sim_v, sim_t, paired)
    assert loss_paired < loss_distinct, "removing negatives must lower the loss"

    # Single identity -> no negatives at all -> loss collapses to 0, not NaN.
    one_id = torch.zeros(b, dtype=torch.long)
    loss_single = objectives.compute_intrac(sim_cross, sim_v, sim_t, one_id)
    assert torch.isfinite(loss_single) and abs(loss_single.item()) < 1e-5


def test_jii_composition():
    torch.manual_seed(0)
    v, t = torch.randn(6, 16), torch.randn(6, 16)
    pid = torch.arange(6)
    total, inter, intra = objectives.compute_jii(v, t, pid, tau=0.02, lambda_intra=0.1)
    assert torch.allclose(total, inter + 0.1 * intra)


def test_ndf_is_finite_and_bidirectional():
    torch.manual_seed(0)
    v, t = torch.randn(8, 16), torch.randn(8, 16)
    pid = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = objectives.compute_ndf(v, t, pid, tau=0.02)
    assert torch.isfinite(loss) and loss > 0

    # Perfectly aligned features -> much lower NDF than misaligned ones.
    aligned = objectives.compute_ndf(v, v.clone(), torch.arange(8), tau=0.02)
    misaligned = objectives.compute_ndf(v, torch.randn(8, 16), torch.arange(8), tau=0.02)
    assert aligned < misaligned


def test_id_loss():
    logits_i = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    loss = objectives.compute_id(logits_i, logits_i, labels)
    assert loss < 1e-3


def test_losses_backprop():
    torch.manual_seed(0)
    v = torch.randn(4, 16, requires_grad=True)
    t = torch.randn(4, 16, requires_grad=True)
    pid = torch.tensor([0, 0, 1, 1])
    loss = (objectives.compute_jii(v, t, pid)[0] +
            objectives.compute_ndf(v, t, pid))
    loss.backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert t.grad is not None and torch.isfinite(t.grad).all()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_rank_perfect_retrieval():
    # 4 queries, 12 gallery items, 3 relevant each, all ranked first.
    g_pids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    q_pids = torch.tensor([0, 1, 2, 3])
    sim = (q_pids.view(-1, 1) == g_pids.view(1, -1)).float()
    cmc, mAP, mINP, _ = rank(sim, q_pids, g_pids, max_rank=10)
    assert cmc[0].item() == 100.0
    assert abs(mAP.item() - 100.0) < 1e-4
    assert abs(mINP.item() - 100.0) < 1e-4


def test_rank_minp_penalises_late_hard_match():
    g_pids = torch.tensor([0, 1, 1, 1, 0, 0, 2, 2, 2, 3, 3, 3])
    q_pids = torch.tensor([0])
    # Rank the three relevant items (pid 0) at positions 1, 2 and 12.
    sim = torch.zeros(1, 12)
    sim[0, 0], sim[0, 4] = 1.0, 0.9        # two easy hits
    sim[0, 5] = -1.0                       # the hard one, ranked last
    cmc, mAP, mINP, _ = rank(sim, q_pids, g_pids, max_rank=10)
    assert cmc[0].item() == 100.0
    assert abs(mINP.item() - 100.0 * 3 / 12) < 1e-3, mINP.item()


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
