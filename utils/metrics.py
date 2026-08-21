"""Retrieval metrics: Rank-K, mAP and mINP.

mINP follows [30]: for a query with |G| relevant gallery items whose hardest
(last) correct match sits at rank R_hard, INP = |G| / R_hard.  A high mINP means
correctly matched images are retrieved at earlier positions rather than merely
appearing somewhere in the ranking.

The final match degree at inference is `sim_alpha * S_global + sim_beta * S_local`
(Sec. III-A), with alpha = 0.25 and beta = 0.75 in the paper.
"""

import logging
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from prettytable import PrettyTable

logger = logging.getLogger("ViLNM.eval")


def rank(similarity: torch.Tensor, q_pids: torch.Tensor, g_pids: torch.Tensor,
         max_rank: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (cmc[:max_rank] in %, mAP in %, mINP in %, sorted indices)."""
    indices = torch.argsort(similarity, dim=1, descending=True)
    pred_labels = g_pids[indices]                                    # [Q, G]
    matches = pred_labels.eq(q_pids.view(-1, 1))                     # [Q, G]

    all_cmc = matches[:, :max_rank].cumsum(dim=1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(dim=0) * 100

    num_rel = matches.sum(dim=1)
    tmp_cmc = matches.cumsum(dim=1)

    inp = []
    for i in range(matches.size(0)):
        pos = matches[i].nonzero()
        if pos.numel() == 0:                # no relevant item retrieved at all
            inp.append(torch.zeros(1, dtype=torch.float32))
            continue
        last = pos[-1]
        inp.append((tmp_cmc[i][last].float() / (last.float() + 1.0)).reshape(1))
    mINP = torch.cat(inp).mean() * 100

    ranks = torch.arange(1, matches.size(1) + 1, device=matches.device).float()
    precision = tmp_cmc.float() / ranks
    AP = (precision * matches).sum(dim=1) / num_rel.clamp(min=1).float()
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator(object):
    def __init__(self, gallery_loader, query_loader, sim_alpha: float = 0.25,
                 sim_beta: float = 0.75, device: str = "cpu"):
        self.gallery_loader = gallery_loader
        self.query_loader = query_loader
        self.sim_alpha = sim_alpha
        self.sim_beta = sim_beta
        self.device = device

    @torch.no_grad()
    def _extract(self, model):
        model.eval()
        q_pids, q_global, q_local = [], [], []
        for pid, caption in self.query_loader:
            caption = caption.to(self.device)
            g, l = model.text_features(caption)
            q_pids.append(pid.view(-1))
            q_global.append(g.float().cpu())
            q_local.append(l.float().cpu())

        g_pids, g_global, g_local = [], [], []
        for pid, image in self.gallery_loader:
            image = image.to(self.device)
            g, l = model.image_features(image)
            g_pids.append(pid.view(-1))
            g_global.append(g.float().cpu())
            g_local.append(l.float().cpu())

        return (torch.cat(q_pids), torch.cat(q_global), torch.cat(q_local),
                torch.cat(g_pids), torch.cat(g_global), torch.cat(g_local))

    def eval(self, model, log: bool = True):
        start = time.time()
        q_pids, q_g, q_l, g_pids, g_g, g_l = self._extract(model)

        s_global = F.normalize(q_g, dim=-1) @ F.normalize(g_g, dim=-1).t()
        s_local = F.normalize(q_l, dim=-1) @ F.normalize(g_l, dim=-1).t()
        similarity = self.sim_alpha * s_global + self.sim_beta * s_local

        results = {}
        views = [("t2i", similarity), ("t2i_global", s_global), ("t2i_local", s_local)]
        table = PrettyTable(["task", "R@1", "R@5", "R@10", "mAP", "mINP"])
        for name, sim in views:
            cmc, mAP, mINP, _ = rank(sim, q_pids, g_pids, max_rank=10)
            results[name] = dict(R1=cmc[0].item(), R5=cmc[4].item(), R10=cmc[9].item(),
                                 mAP=mAP.item(), mINP=mINP.item())
            table.add_row([name] + [f"{results[name][k]:.3f}"
                                    for k in ("R1", "R5", "R10", "mAP", "mINP")])

        if log:
            logger.info("evaluation finished in %.1fs\n%s", time.time() - start, table)
        return results
