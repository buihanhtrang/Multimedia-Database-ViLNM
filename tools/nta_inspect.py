"""Look inside NTA: what it scores, where it looks, and whether it finds the noise.

NTA (Sec. III-B) scores every word of a caption against the image it came with:

    Mat[x, y] = cos(word x, patch y)             Eq. (2)  word-patch similarity
    r_x       = 1 - max_y Mat[x, y]              Eq. (3)  "no patch shows this word"
    r'_x      = r_x - mean(r) + rate             Eq. (4)  elimination probability

This tool takes a trained run, swaps colour words in training captions with the
VN3K colour-noise file (so we know exactly which words are wrong), and measures
how NTA scores them:

    examples.png   image + where the right / the wrong colour word looks on it
                   (row of Mat as a heat map) + r' for every word of the caption
    detection.png  r' of swapped vs untouched words, and AUC / top-1 / elimination
                   probability for each run given
    top_words.png  words NTA eliminates most / least on the run's own training
                   captions (from nta_final.pt)
    summary.md     the numbers

    python tools/nta_inspect.py                                    # full vs no_nta, clean runs
    python tools/nta_inspect.py --runs abl_full_color50 abl_no_nta_color50 --noise color50

The first run is the one drawn in examples.png and top_words.png.
"""

import argparse
import difflib
import logging
import os
import random
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ablation import NOISES                                        # noqa: E402
from ablation_report import auc, changed_token_labels              # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CLEAN_C, NOISY_C = "#2a78d6", "#eb6834"       # categorical slots 1-2 (validated)


def load_run(run_dir, device, ckpt_name="best.pth"):
    import torch
    from datasets.preprocessing import build_transforms
    from model.build import build_model
    from utils.hf_tokenizer import build_tokenizer
    from utils.iotools import load_checkpoint, load_train_configs

    cfg = load_train_configs(os.path.join(run_dir, "configs.json"))
    cfg.img_size = tuple(cfg.img_size)
    cfg.no_pretrained = True
    ckpt = load_checkpoint(os.path.join(run_dir, ckpt_name))
    model = build_model(cfg, ckpt["model"]["classifier.weight"].shape[0])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tok = build_tokenizer(cfg)
    tf = build_transforms(cfg.img_size, is_train=False, backbone=cfg.backbone)
    return model, cfg, tok, tf


def nta_scores(model, cfg, tok, tf, items, device, batch=48):
    """items: [(img_path, word_ids)] -> [(r' [n_words], Mat [n_words, Nv])].

    Mat is built from the same LA token features NTA uses in training."""
    import torch
    import torch.nn.functional as F
    from datasets.bases import pad_tokens, read_image

    out = []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        imgs = torch.stack([tf(read_image(p)) for p, _ in chunk]).to(device)
        ids = torch.stack([pad_tokens(w, tok.sot_token, tok.eot_token, cfg.text_length)
                           for _, w in chunk]).to(device)
        with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
            _, _, patches = model.encode_image(imgs)
            _, _, words, mask = model.encode_text(ids)
        r_prime = model.nta(words, patches, mask).float().cpu().numpy()
        mat = torch.bmm(F.normalize(words.float(), dim=-1),
                        F.normalize(patches.float(), dim=-1).transpose(1, 2)).cpu().numpy()
        off = model.text_offset
        for b, (_, w) in enumerate(chunk):
            n = len(w)
            out.append((r_prime[b, off:off + n], mat[b, off:off + n]))
    return out


def noisy_pairs(cfg, tok, noise, n, seed):
    """Training captions whose colour noise changed some (not all) tokens."""
    from datasets.bases import encode_caption
    from datasets.build import build_dataset
    from datasets.noise import apply_caption_noise, resolve_noise_file

    ds = build_dataset(argparse.Namespace(dataset_name="VN3K", root_dir=cfg.root_dir))
    noisy, _ = apply_caption_noise(
        ds.train, resolve_noise_file(NOISES[noise], cfg.root_dir, ds.dataset_dir),
        ds.img_dir)
    pairs = []
    for clean, dirty in zip(ds.train, noisy):
        c = encode_caption(clean[3], tok, cfg.text_length)
        d = encode_caption(dirty[3], tok, cfg.text_length)
        lab = changed_token_labels(c, d)
        if lab.any() and not lab.all():
            # First swap: its token in the noisy caption and the one it replaced.
            ops = [o for o in difflib.SequenceMatcher(None, c, d, autojunk=False).get_opcodes()
                   if o[0] == "replace"]
            swap = (ops[0][3], ops[0][1]) if ops else None
            pairs.append(dict(img=clean[2], clean=c, noisy=d, labels=lab, swap=swap))
    random.Random(seed).shuffle(pairs)
    return pairs[:n], len(pairs)


def detection_stats(scores, pairs, clean_scores):
    """AUC/top-1 plus two controls:

    auc_position  AUC of "relative position in the caption" alone -- r' grows
                  towards the end of a caption (shoes, bags come last), and so
                  do the swapped colours, so plain AUC partly measures position;
    auc_binned    mean AUC within 5 position bins, i.e. with position held fixed;
    same_slot     the swapped word vs the original word at the same slot of the
                  clean caption, same image: does the wrong colour score higher?
    """
    r = np.concatenate([s[0] for s in scores])
    lab = np.concatenate([p["labels"] for p in pairs])
    pos = np.concatenate([np.arange(len(p["noisy"])) / len(p["noisy"]) for p in pairs])
    bins = np.minimum((pos * 5).astype(int), 4)
    hits = [p["labels"][s[0] == s[0].max()].mean() for s, p in zip(scores, pairs)]
    chance = [p["labels"].mean() for p in pairs]
    slot = np.array([(s[0][p["swap"][0]], c[0][p["swap"][1]])
                     for s, c, p in zip(scores, clean_scores, pairs) if p["swap"]])
    return dict(auc=auc(r, lab), auc_position=auc(pos, lab),
                auc_binned=float(np.mean([auc(r[bins == k], lab[bins == k]) for k in range(5)])),
                top1=100 * np.mean(hits), chance=100 * np.mean(chance),
                p_noisy=100 * r[lab].mean(), p_clean=100 * r[~lab].mean(),
                slot_wrong=slot[:, 0].mean(), slot_right=slot[:, 1].mean(),
                slot_higher=100 * (slot[:, 0] > slot[:, 1]).mean(),
                r_noisy=r[lab], r_clean=r[~lab])


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pretty(tok, ids):
    return [tok.decode([t]).strip() or "·" for t in ids]


def heat_overlay(ax, img_path, row, size, title):
    import torch
    import torch.nn.functional as F
    from datasets.bases import read_image

    h, w = size
    img = np.asarray(read_image(img_path).resize((w, h)))
    grid = torch.tensor(row).reshape(1, 1, h // 16, w // 16)
    heat = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)[0, 0].numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    ax.imshow(img)
    ax.imshow(heat, cmap="magma", alpha=0.55)
    ax.set_title(title, fontsize=9, color=INK)
    ax.axis("off")


def plot_examples(path, tok, cfg, pairs, noisy_scores, clean_scores, k):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 3.4 * k), facecolor=SURFACE)
    gs = fig.add_gridspec(k, 4, width_ratios=[1, 1, 1, 6.5], wspace=0.15, hspace=0.55)
    for row in range(k):
        p, (rn, mn), (_, mc) = pairs[row], noisy_scores[row], clean_scores[row]
        j, jc = p["swap"]                                  # wrong token, the right one
        wrong, right = pretty(tok, [p["noisy"][j]])[0], pretty(tok, [p["clean"][jc]])[0]

        ax = fig.add_subplot(gs[row, 0])
        from datasets.bases import read_image
        ax.imshow(np.asarray(read_image(p["img"]).resize(cfg.img_size[::-1])))
        ax.set_title("Ảnh", fontsize=9, color=INK)
        ax.axis("off")
        heat_overlay(fig.add_subplot(gs[row, 1]), p["img"], mc[jc], cfg.img_size,
                     f"từ đúng «{right}»\nnhìn vào đâu")
        heat_overlay(fig.add_subplot(gs[row, 2]), p["img"], mn[j], cfg.img_size,
                     f"từ sai «{wrong}»\nnhìn vào đâu")

        ax = fig.add_subplot(gs[row, 3])
        style(ax)
        words = pretty(tok, p["noisy"])
        colors = [NOISY_C if l else CLEAN_C for l in p["labels"]]
        ax.bar(range(len(rn)), rn, color=colors, width=0.8)
        ax.axhline(cfg.nta_rate, color=INK2, linewidth=1, linestyle="--")
        ax.text(len(rn) - 0.5, cfg.nta_rate, f" trung bình = {cfg.nta_rate}",
                fontsize=7, color=INK2, va="bottom", ha="right")
        ax.set_xticks(range(len(rn)))
        ax.set_xticklabels(words, rotation=70, fontsize=7.5)
        for t, l in zip(ax.get_xticklabels(), p["labels"]):
            if l:
                t.set_fontweight("bold")
        ax.set_xlim(-0.6, len(rn) - 0.4)
        ax.set_ylim(0, max(0.6, float(rn.max()) * 1.15))
        ax.set_ylabel("r' (xác suất bị loại)", fontsize=8, color=INK2)
        rank = int((rn > rn[j]).sum()) + 1
        ax.set_title(f"r' từng từ — từ bị đổi (cam, in đậm) «{wrong}» r'={rn[j]:.2f}, "
                     f"xếp hạng {rank}/{len(rn)} về độ nhiễu", fontsize=9, color=INK, loc="left")
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_detection(path, stats, names):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(stats) + 1, figsize=(5 * (len(stats) + 1), 3.8),
                             facecolor=SURFACE)
    bins = np.linspace(0, 1, 41)
    for ax, st, name in zip(axes, stats, names):
        style(ax)
        for r, c, lab in ((st["r_clean"], CLEAN_C, "từ giữ nguyên"),
                          (st["r_noisy"], NOISY_C, "từ bị đổi màu")):
            ax.hist(r, bins=bins, density=True, color=c, alpha=0.55, label=lab,
                    edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(f"{name}\nAUC {st['auc']:.3f}", fontsize=9, color=INK)
        ax.set_xlabel("r' do NTA chấm", fontsize=8, color=INK2)
        ax.set_ylabel("mật độ", fontsize=8, color=INK2)
        ax.legend(fontsize=7.5, frameon=False)

    ax = axes[-1]
    style(ax)
    metrics = [("AUC×100", "auc", 100), ("top-1 %", "top1", 1),
               ("P(loại) từ sai %", "p_noisy", 1), ("P(loại) từ đúng %", "p_clean", 1)]
    width = 0.8 / len(stats)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#008300"]
    for i, (st, name) in enumerate(zip(stats, names)):
        xs = np.arange(len(metrics)) + (i - (len(stats) - 1) / 2) * width
        ys = [st[k] * m for _, k, m in metrics]
        ax.bar(xs, ys, width=width * 0.9, color=palette[i], label=name)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}", ha="center", fontsize=7, color=INK2)
    ax.axhline(50, color=GRID, linewidth=1)
    ax.plot([], [])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=7.5)
    ax.set_title(f"So sánh (top-1 ngẫu nhiên = {stats[0]['chance']:.0f}%)",
                 fontsize=9, color=INK)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def top_words(run_dir, tok, min_count=200):
    import torch
    nta = torch.load(os.path.join(run_dir, "nta_final.pt"), weights_only=False)
    acc = defaultdict(list)
    for ids, r in zip(nta["original"], nta["noise"]):
        for t, v in zip(ids, r):
            acc[t].append(float(v))
    rows = [(pretty(tok, [t])[0], float(np.mean(v)), len(v))
            for t, v in acc.items() if len(v) >= min_count]
    rows.sort(key=lambda x: -x[1])
    return rows


def plot_top_words(path, rows, n=18, rate=0.2):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor=SURFACE)
    for ax, sel, c, title in ((axes[0], rows[:n], NOISY_C, "NTA loại nhiều nhất"),
                              (axes[1], rows[-n:][::-1], CLEAN_C, "NTA giữ lại nhiều nhất")):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ys = np.arange(len(sel))[::-1]
        ax.barh(ys, [v for _, v, _ in sel], color=c, height=0.7)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{w}  ({k})" for w, _, k in sel], fontsize=8)
        ax.axvline(rate, color=INK2, linewidth=1, linestyle="--")
        ax.set_xlabel("r' trung bình (epoch cuối)", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.suptitle("Từ trong caption train, theo mức NTA cho là nhiễu (số lần xuất hiện trong ngoặc)",
                 fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["abl_full_clean", "abl_no_nta_clean"])
    ap.add_argument("--ckpt", default="best.pth",
                    help="checkpoint inside each run, e.g. snapshots/epoch_10.pth")
    ap.add_argument("--noise", default="color50", help="colour-noise level used as probe")
    ap.add_argument("--n", type=int, default=2000, help="probe captions")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "nta_inspect"))
    a = ap.parse_args()
    if not a.noise.startswith("color"):
        ap.error("--noise must be a colour level: only there do we know which words are wrong")

    import torch
    logging.getLogger("ViLNM").setLevel(logging.WARNING)
    os.chdir(ROOT)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats, first = [], None
    for name in a.runs:
        run_dir = os.path.join(ROOT, "logs", "VN3K", name)
        model, cfg, tok, tf = load_run(run_dir, device, a.ckpt)
        if first is None:
            pairs, n_total = noisy_pairs(cfg, tok, a.noise, a.n, a.seed)
            print(f"{len(pairs)} probe captions (of {n_total} with a swapped colour)")
        noisy = nta_scores(model, cfg, tok, tf, [(p["img"], p["noisy"]) for p in pairs], device)
        clean = nta_scores(model, cfg, tok, tf, [(p["img"], p["clean"]) for p in pairs], device)
        st = detection_stats(noisy, pairs, clean)
        stats.append(st)
        print(f"{name}: AUC {st['auc']:.3f} (position alone {st['auc_position']:.3f}, "
              f"position held fixed {st['auc_binned']:.3f})  top-1 {st['top1']:.1f}% (chance {st['chance']:.1f}%)"
              f"  P(eliminate) swapped {st['p_noisy']:.1f}% vs kept {st['p_clean']:.1f}%")
        if first is None:
            first = (run_dir, cfg, tok)
            ex = [i for i, p in enumerate(pairs) if p["swap"]][:a.examples]
            plot_examples(os.path.join(a.out, "examples.png"), tok, cfg,
                          [pairs[i] for i in ex], [noisy[i] for i in ex],
                          [clean[i] for i in ex], len(ex))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    plot_detection(os.path.join(a.out, "detection.png"), stats, a.runs)
    run_dir, cfg, tok = first
    rows = []
    if os.path.exists(os.path.join(run_dir, "nta_final.pt")):
        rows = top_words(run_dir, tok)
        plot_top_words(os.path.join(a.out, "top_words.png"), rows, rate=cfg.nta_rate)

    lines = [f"# NTA: phân tích ({a.noise} làm phép thử, {len(pairs)} caption)", "",
             "| Lượt | AUC | AUC chỉ theo vị trí | AUC cố định vị trí | top-1 | top-1 ngẫu nhiên "
             "| P(loại) từ bị đổi | P(loại) từ giữ nguyên | cùng chỗ: r' từ sai / từ đúng "
             "| từ sai > từ đúng |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, st in zip(a.runs, stats):
        lines.append(f"| {name} | {st['auc']:.3f} | {st['auc_position']:.3f} "
                     f"| {st['auc_binned']:.3f} | {st['top1']:.1f}% | {st['chance']:.1f}% "
                     f"| {st['p_noisy']:.1f}% | {st['p_clean']:.1f}% "
                     f"| {st['slot_wrong']:.3f} / {st['slot_right']:.3f} "
                     f"| {st['slot_higher']:.1f}% |")
    if rows:
        lines += ["", f"Từ bị loại nhiều nhất ({a.runs[0]}): " +
                  ", ".join(f"{w} ({v:.2f})" for w, v, _ in rows[:12]),
                  "", "Từ được giữ nhiều nhất: " +
                  ", ".join(f"{w} ({v:.2f})" for w, v, _ in rows[-12:][::-1])]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
