"""Re-score trained runs with other global/local weights (no retraining).

At test time ViLNM ranks images by S = alpha * S_global + (1 - alpha) * S_local,
alpha = 0.25 in the paper.  This takes the final checkpoint of every run that
has the LA branch, extracts the test features once, and evaluates several
alphas, so we can tell whether LA hurts because S_local gets too much weight.
Runs without LA (no_la, baseline) are listed for reference: their score is
S_global only, whatever alpha is.

    python tools/sim_weights.py
    python tools/sim_weights.py --alphas 0.25 0.5 0.75 --noise color50

Results are cached per run in <run>/sim_weights.json; the table goes to
results/sim_weights/sim_weights.md and the chart to sim_weights.png.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ablation_report import COLOR, CONFIG_LABEL, GRID, INK, INK2  # noqa: E402

LA_CONFIGS = ["full", "no_nta", "nta_random", "no_jii"]
REF_CONFIGS = ["no_la", "baseline"]
NOISES = ["clean", "color50", "sent50"]


def final_ckpt(run_dir):
    snap = os.path.join(run_dir, "snapshots", "epoch_30.pth")
    return snap if os.path.exists(snap) else os.path.join(run_dir, "best.pth")


def evaluate(run_dir, alphas, device):
    import torch
    import torch.nn.functional as F
    from datasets.build import build_test_dataloader
    from model.build import build_model
    from utils.iotools import load_checkpoint, load_train_configs
    from utils.metrics import Evaluator, rank

    args = load_train_configs(os.path.join(run_dir, "configs.json"))
    args.img_size = tuple(args.img_size)
    gallery_loader, query_loader, num_classes = build_test_dataloader(args)
    args.no_pretrained = True
    model = build_model(args, num_classes)
    ckpt = load_checkpoint(final_ckpt(run_dir), map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(device)

    q_pids, q_g, q_l, g_pids, g_g, g_l = Evaluator(
        gallery_loader, query_loader, device=device)._extract(model)
    s_global = F.normalize(q_g, dim=-1) @ F.normalize(g_g, dim=-1).t()
    s_local = F.normalize(q_l, dim=-1) @ F.normalize(g_l, dim=-1).t()

    out = {}
    for a in alphas:
        cmc, mAP, mINP, _ = rank(a * s_global + (1 - a) * s_local, q_pids, g_pids)
        out[f"{a:g}"] = dict(R1=cmc[0].item(), R5=cmc[4].item(), mAP=mAP.item(),
                             mINP=mINP.item())
    del model
    torch.cuda.empty_cache()
    return out


def load_or_eval(run_dir, alphas, device, force):
    cache = os.path.join(run_dir, "sim_weights.json")
    ckpt = final_ckpt(run_dir)
    data = {}
    if os.path.exists(cache) and not force:
        with open(cache) as f:
            data = json.load(f)
        # Weights pruned to save disk: keep the cached numbers.
        if data.get("ckpt") != ckpt and os.path.exists(ckpt):
            data = {}
    have = data.get("alphas", {})
    missing = [a for a in alphas if f"{a:g}" not in have]
    if missing:
        print(f"evaluating {os.path.basename(run_dir)} at alpha {missing} ...", flush=True)
        have.update(evaluate(run_dir, missing, device))
        data = dict(ckpt=ckpt, alphas=have)
        with open(cache, "w") as f:
            json.dump(data, f, indent=1)
    return {a: have[f"{a:g}"] for a in alphas}


def reference_r1(run_dir):
    """R@1 of a run without LA at its final evaluation (S_global only)."""
    with open(os.path.join(run_dir, "history.json")) as f:
        evals = json.load(f)["eval"]
    return evals[max(evals, key=int)]["t2i"]["R1"]


def plot(out_dir, results, alphas, noises):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(noises), figsize=(5.2 * len(noises), 4.2), squeeze=False)
    axes = axes[0]
    for ax, noise in zip(axes, noises):
        for cfg in LA_CONFIGS:
            r = results.get((cfg, noise))
            if r is None:
                continue
            ax.plot(alphas, [r[a]["R1"] for a in alphas], marker="o", lw=2,
                    color=COLOR[cfg], label=CONFIG_LABEL[cfg])
        for cfg in REF_CONFIGS:
            ref = results.get((cfg, noise))
            if ref is not None:
                ax.axhline(ref, color=COLOR[cfg], ls="--", lw=1.5,
                           label=CONFIG_LABEL[cfg] + " (chỉ global)")
        ax.axvline(0.25, color=INK2, ls=":", lw=1)
        ax.text(0.25, 1.0, " bài báo", transform=ax.get_xaxis_transform(),
                color=INK2, fontsize=8, va="top")
        ax.set_title(noise, loc="left", color=INK)
        ax.set_xlabel("trọng số global α  (0 = chỉ local, 1 = chỉ global)")
        ax.set_ylabel("R@1 (%)")
        ax.set_xticks(alphas)
        ax.grid(color=GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = os.path.join(out_dir, "sim_weights.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_md(out_dir, results, alphas, noises):
    lines = ["# Đổi tỉ lệ global / local khi chấm điểm (không train lại)", "",
             "S = α·S_global + (1−α)·S_local. Bài báo dùng α = 0,25. "
             "Checkpoint epoch cuối. Dòng *bỏ LA* / *baseline* chỉ có global nên "
             "không đổi theo α.", ""]
    for metric in ("R1", "mAP"):
        lines += [f"## {metric}", ""]
        for noise in noises:
            lines += [f"**{noise}**", "",
                      "| Cấu hình | " + " | ".join(f"α={a:g}" for a in alphas) + " |",
                      "|---|" + "---|" * len(alphas)]
            for cfg in LA_CONFIGS:
                r = results.get((cfg, noise))
                if r is None:
                    continue
                vals = [r[a][metric] for a in alphas]
                best = max(vals)
                cells = [f"**{v:.2f}**" if v == best else f"{v:.2f}" for v in vals]
                lines.append(f"| {CONFIG_LABEL[cfg]} | " + " | ".join(cells) + " |")
            if metric == "R1":
                for cfg in REF_CONFIGS:
                    ref = results.get((cfg, noise))
                    if ref is not None:
                        lines.append(f"| {CONFIG_LABEL[cfg]} | "
                                     + " | ".join(f"{ref:.2f}" for _ in alphas) + " |")
            lines.append("")
    path = os.path.join(out_dir, "sim_weights.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=os.path.join(ROOT, "logs", "VN3K"))
    ap.add_argument("--noise", nargs="+", default=NOISES)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "sim_weights"))
    cli = ap.parse_args()

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if cli.device == "auto" else cli.device
    alphas = sorted(cli.alphas)

    results = {}
    for noise in cli.noise:
        for cfg in LA_CONFIGS:
            run = os.path.join(cli.logs, f"abl_{cfg}_{noise}")
            if os.path.exists(os.path.join(run, "configs.json")):
                results[(cfg, noise)] = load_or_eval(run, alphas, device, cli.force)
        for cfg in REF_CONFIGS:
            run = os.path.join(cli.logs, f"abl_{cfg}_{noise}")
            if os.path.exists(os.path.join(run, "history.json")):
                results[(cfg, noise)] = reference_r1(run)

    os.makedirs(cli.out, exist_ok=True)
    print("->", write_md(cli.out, results, alphas, cli.noise))
    print("->", plot(cli.out, results, alphas, cli.noise))


if __name__ == "__main__":
    main()
