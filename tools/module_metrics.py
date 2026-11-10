"""Per-module diagnostics for the VN3K ablation runs (beyond R@1 / mAP).

Each ViLNM module has its own job, so each gets a measurement of that job:

  JII  (pull a person's images and captions together, push other people away)
       margin       test set, per caption: similarity of its best *correct* image
                    minus its best *wrong* image (S = 0.25 S_global + 0.75 S_local);
                    > 0 means the right person wins.  Reported as the mean and as
                    the share of captions with margin > 0.
       compactness  mean cosine between features of the same person minus
                    between different people, for images, captions, and
                    caption-to-image (global features).
  LA   (match single words to image regions)
       pointing     for body-part words (tóc/mũ, áo, quần/váy, giày/dép), is the
                    image row the word matches best inside the band of the image
                    where that part is?  Pedestrian crops are upright, so the bands
                    are fixed fractions of the height.  Chance = band height.
  noise memorisation  (sentence noise: a caption was moved to another person)
       on the training set with the sent50 file, for each moved caption: does
       the model retrieve the image it was wrongly paired with (memorised), or an
       image of the person it really describes (recovered)?  Runs that never saw
       sent50 give the reference level.

    python tools/module_metrics.py                 # every finished ablation run
    python tools/module_metrics.py --runs abl_full_color50 abl_no_la_color50
    python tools/module_metrics.py --limit 200 --device cpu   # quick check

Results are cached per run in <run>/module_metrics.json (keyed by checkpoint);
tables and charts go to results/module_metrics/.  The checkpoint is the final
epoch (snapshots/epoch_30.pth, else best.pth), as in the ablation report.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ablation_report import (COLOR, CONFIG_LABEL, CONFIG_ORDER, GRID, INK,  # noqa: E402
                             INK2, MARKER, SURFACE)
from nta_inspect import load_run, nta_scores                                # noqa: E402

RUNS_DIR = os.path.join(ROOT, "logs", "VN3K")
SENT_PROBE = "noise_sentence/VN3K-VI_0.5.npy"
# Body part -> words, and the rows (fraction of image height) where it is.
PARTS = {"đầu (tóc, mũ)": (("tóc", "mũ", "nón"), (0.0, 0.30)),
         "áo": (("áo",), (0.12, 0.60)),
         "quần / váy": (("quần", "váy"), (0.42, 0.88)),
         "giày / dép": (("giày", "dép"), (0.72, 1.0))}
NOISE_ORDER = ["clean", "color50", "sent50"]


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def encode(model, cfg, tok, tf, img_paths, captions, device, batch=64):
    import torch
    import torch.nn.functional as F
    from datasets.bases import pad_tokens, read_image, encode_caption

    def run(items, fn):
        g, l = [], []
        for s in range(0, len(items), batch):
            with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
                a, b = fn(items[s:s + batch])
            g.append(F.normalize(a.float(), dim=-1).cpu())
            l.append(F.normalize(b.float(), dim=-1).cpu())
        return torch.cat(g), torch.cat(l)

    img = run(img_paths, lambda c: model.image_features(
        torch.stack([tf(read_image(p)) for p in c]).to(device)))
    txt = run(captions, lambda c: model.text_features(torch.stack(
        [pad_tokens(encode_caption(t, tok, cfg.text_length), tok.sot_token,
                    tok.eot_token, cfg.text_length) for t in c]).to(device)))
    return img, txt


def similarity(cfg, img, txt):
    return (cfg.sim_alpha * txt[0] @ img[0].t() + cfg.sim_beta * txt[1] @ img[1].t())


def margins(sim, q_pids, g_pids):
    import torch
    same = q_pids.view(-1, 1) == g_pids.view(1, -1)
    pos = sim.masked_fill(~same, -1e4).max(1).values
    neg = sim.masked_fill(same, -1e4).max(1).values
    m = (pos - neg).numpy()
    return dict(margin=float(m.mean()), margin_pos_share=100.0 * float((m > 0).mean()),
                pos_sim=float(pos.mean()), hardest_neg_sim=float(neg.mean()))


def compactness(a, a_pids, b=None, b_pids=None):
    """mean cos(same person) - mean cos(different people); b=None -> within a."""
    import torch
    b, b_pids = (a, a_pids) if b is None else (b, b_pids)
    s = a @ b.t()
    same = a_pids.view(-1, 1) == b_pids.view(1, -1)
    if b is a:
        same.fill_diagonal_(False)
        other = ~same
        other.fill_diagonal_(False)
    else:
        other = ~same
    intra, inter = float(s[same].mean()), float(s[other].mean())
    return dict(intra=intra, inter=inter, gap=intra - inter)


def part_token_ids(tok):
    """word -> set of token ids it is encoded as (start of caption / mid-sentence)."""
    base = tok.encode("a")
    ids = {}
    for _, (words, _) in PARTS.items():
        for w in words:
            cands = set()
            alone = tok.encode(w)
            if len(alone) == 1:
                cands.add(alone[0])
            mid = tok.encode("a " + w)[len(base):]
            if len(mid) == 1:
                cands.add(mid[0])
            if cands:
                ids[w] = cands
    return ids


def pointing(model, cfg, tok, tf, test, device, limit):
    from datasets.bases import encode_caption

    word_ids = part_token_ids(tok)
    id_to_part = {}
    for part, (words, band) in PARTS.items():
        for w in words:
            for t in word_ids.get(w, ()):
                id_to_part[t] = part
    # Captions are stored image by image, two per image in VN3K's test split.
    assert len(test["captions"]) == 2 * len(test["img_paths"]) and all(
        q == test["image_pids"][k // 2] for k, q in enumerate(test["caption_pids"]))
    items, hits_meta = [], []
    for k, cap in enumerate(test["captions"]):
        ids = encode_caption(cap, tok, cfg.text_length)
        where = [(j, id_to_part[t]) for j, t in enumerate(ids) if t in id_to_part]
        if where:
            items.append((test["img_paths"][k // 2], ids))
            hits_meta.append(where)
        if limit and len(items) >= limit:
            break
    scores = nta_scores(model, cfg, tok, tf, items, device)
    rows = cfg.img_size[0] // 16
    hit = {p: [] for p in PARTS}
    for (_, mat), where in zip(scores, hits_meta):
        for j, part in where:
            grid = mat[j].reshape(rows, -1)
            row = int(np.unravel_index(grid.argmax(), grid.shape)[0])
            lo, hi = PARTS[part][1]
            hit[part].append(lo * rows <= row < hi * rows)
    out = {p: dict(hit=100.0 * float(np.mean(h)) if h else None, n=len(h),
                   chance=100.0 * (PARTS[p][1][1] - PARTS[p][1][0])) for p, h in hit.items()}
    allh = [x for h in hit.values() for x in h]
    out["_all"] = dict(hit=100.0 * float(np.mean(allh)) if allh else None, n=len(allh),
                       chance=float(np.mean([out[p]["chance"] for p in PARTS])))
    return out


def memorisation(model, cfg, tok, tf, device, limit):
    """Sentence-noise probe on the training set: memorised vs recovered, in %.

    Runs trained with sentence noise are probed with their own noise file (the
    wrong pairs they actually saw); every other run gets sent50 as the reference."""
    import torch
    from datasets.build import build_dataset
    from datasets.noise import resolve_noise_file

    ds = build_dataset(argparse.Namespace(dataset_name="VN3K", root_dir=cfg.root_dir))
    train = ds.train
    noise = getattr(cfg, "noise_file", "") or ""
    probe = noise if noise.startswith("noise_sentence/") else SENT_PROBE
    index = np.load(resolve_noise_file(probe, cfg.root_dir, ds.dataset_dir))
    moved = [i for i, j in enumerate(index) if train[int(j)][0] != train[i][0]]
    if limit:
        moved = moved[:limit]
    involved = None
    if limit:   # quick check: only the people the probe captions touch
        involved = {train[i][0] for i in moved} | {train[int(index[i])][0] for i in moved}
    images = sorted({t[2] for t in train if involved is None or t[0] in involved})
    pos = {p: k for k, p in enumerate(images)}
    pid_of_img = torch.tensor([0] * len(images))
    for t in train:
        if t[2] in pos:
            pid_of_img[pos[t[2]]] = t[0]
    img, txt = encode(model, cfg, tok, tf, images,
                      [train[int(index[i])][3] for i in moved], device)
    top = similarity(cfg, img, txt).argmax(1)
    wrong_img = torch.tensor([pos[train[i][2]] for i in moved])
    true_pid = torch.tensor([train[int(index[i])][0] for i in moved])
    wrong_pid = torch.tensor([train[i][0] for i in moved])
    return dict(n=len(moved), probe=probe,
                memorised_image=100.0 * float((top == wrong_img).float().mean()),
                memorised_person=100.0 * float((pid_of_img[top] == wrong_pid).float().mean()),
                recovered=100.0 * float((pid_of_img[top] == true_pid).float().mean()))


def measure(run_dir, ckpt, device, limit):
    import torch
    from datasets.build import build_dataset

    model, cfg, tok, tf = load_run(run_dir, device, ckpt)
    test = build_dataset(argparse.Namespace(dataset_name="VN3K", root_dir=cfg.root_dir)).test
    n_img = len(test["img_paths"]) if not limit else min(limit, len(test["img_paths"]))
    paths, i_pids = test["img_paths"][:n_img], torch.tensor(test["image_pids"][:n_img])
    caps, q_pids = test["captions"][:2 * n_img], torch.tensor(test["caption_pids"][:2 * n_img])
    img, txt = encode(model, cfg, tok, tf, paths, caps, device)

    out = dict(ckpt=ckpt, limit=limit)
    out["margin"] = margins(similarity(cfg, img, txt), q_pids, i_pids)
    out["margin_global"] = margins(txt[0] @ img[0].t(), q_pids, i_pids)
    out["margin_local"] = margins(txt[1] @ img[1].t(), q_pids, i_pids)
    out["compact_image"] = compactness(img[0], i_pids)
    out["compact_text"] = compactness(txt[0], q_pids)
    out["compact_cross"] = compactness(txt[0], q_pids, img[0], i_pids)
    out["pointing"] = pointing(model, cfg, tok, tf, test, device, limit)
    out["memorisation"] = memorisation(model, cfg, tok, tf, device, limit)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax, results, noises, configs, value, ylabel, title, ref=None):
    width = 0.8 / max(len(configs), 1)
    for i, c in enumerate(configs):
        xs, ys = [], []
        for k, n in enumerate(noises):
            v = results.get((c, n))
            v = None if v is None else value(v)
            if v is not None:
                xs.append(k + (i - (len(configs) - 1) / 2) * width)
                ys.append(v)
        ax.bar(xs, ys, width=width * 0.9, color=COLOR[c], label=CONFIG_LABEL[c],
               edgecolor=SURFACE, linewidth=0.5)
    if ref is not None:
        ax.axhline(ref[0], color=INK2, linewidth=1, linestyle="--")
        ax.text(0.01, ref[0], ref[1], fontsize=7, color=INK2, ha="left", va="bottom",
                transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels(noises, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def plot(out_dir, results, noises, configs):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE)
    for ax in axes.flat:
        style(ax)
    grouped_bars(axes[0, 0], results, noises, configs, lambda r: r["margin"]["margin"],
                 "điểm giống ảnh đúng − ảnh sai giống nhất",
                 "JII · Khoảng cách đúng/sai (càng cao càng ít nhầm)")
    grouped_bars(axes[0, 1], results, noises, configs, lambda r: r["compact_cross"]["gap"],
                 "cùng người − khác người (cosine)",
                 "JII · Câu–ảnh cùng một người gom lại gần nhau")
    chance = next(iter(results.values()))["pointing"]["_all"]["chance"]
    grouped_bars(axes[1, 0], results, noises, configs,
                 lambda r: r["pointing"]["_all"]["hit"],
                 "% từ chỉ bộ phận nhìn đúng vùng ảnh",
                 "LA · Từ 'tóc/áo/quần/giày' có nhìn đúng chỗ không",
                 ref=(chance, f"ngẫu nhiên ≈ {chance:.0f}%"))
    grouped_bars(axes[1, 1], results, noises, configs,
                 lambda r: r["memorisation"]["memorised_person"],
                 "% câu bị tráo vẫn kéo về người bị gán sai",
                 "Học thuộc nhiễu (phép thử nhiễu câu trên tập train; thấp = tốt)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(configs), fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "module_metrics.png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)

    # pointing per body part
    parts = list(PARTS)
    fig, axes = plt.subplots(1, len(noises), figsize=(5 * len(noises), 3.8),
                             facecolor=SURFACE, squeeze=False)
    for ax, n in zip(axes[0], noises):
        style(ax)
        width = 0.8 / len(configs)
        for i, c in enumerate(configs):
            r = results.get((c, n))
            if r is None:
                continue
            xs = np.arange(len(parts)) + (i - (len(configs) - 1) / 2) * width
            ax.bar(xs, [r["pointing"][p]["hit"] or 0 for p in parts], width=width * 0.9,
                   color=COLOR[c], label=CONFIG_LABEL[c])
        for k, p in enumerate(parts):
            ch = PARTS[p][1][1] - PARTS[p][1][0]
            ax.plot([k - 0.45, k + 0.45], [100 * ch] * 2, color=INK2, linestyle="--",
                    linewidth=1)
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels(parts, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(f"{n} · % nhìn đúng vùng (gạch = ngẫu nhiên)", fontsize=9,
                     color=INK, loc="left")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "la_pointing_parts.png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)


def write_md(out_dir, results, noises, configs):
    def table(title, fn, digits=3):
        lines = [f"**{title}**", "", "| Cấu hình | " + " | ".join(noises) + " |",
                 "|---|" + "---|" * len(noises)]
        for c in configs:
            cells = []
            for n in noises:
                r = results.get((c, n))
                v = None if r is None else fn(r)
                cells.append("—" if v is None else f"{v:.{digits}f}")
            lines.append(f"| {CONFIG_LABEL[c]} | " + " | ".join(cells) + " |")
        return lines + [""]

    L = ["# Metric theo từng module (epoch cuối)", "",
         "## JII: tách người đúng khỏi người sai", ""]
    L += table("Khoảng cách: điểm giống ảnh đúng − ảnh sai giống nhất (S kết hợp)",
               lambda r: r["margin"]["margin"])
    L += table("% caption mà ảnh đúng thắng ảnh sai giống nhất",
               lambda r: r["margin"]["margin_pos_share"], 1)
    L += table("Gom cụm câu–ảnh: cosine cùng người − khác người",
               lambda r: r["compact_cross"]["gap"])
    L += table("Gom cụm ảnh–ảnh", lambda r: r["compact_image"]["gap"])
    L += table("Gom cụm câu–câu", lambda r: r["compact_text"]["gap"])
    L += ["## LA: từ có nhìn đúng vùng ảnh không", ""]
    chance = next(iter(results.values()))["pointing"]["_all"]["chance"]
    L += table(f"% từ chỉ bộ phận nhìn đúng vùng (ngẫu nhiên ≈ {chance:.0f}%)",
               lambda r: r["pointing"]["_all"]["hit"], 1)
    L += table("Khoảng cách đúng/sai chỉ dùng nhánh local",
               lambda r: r["margin_local"]["margin"])
    L += table("Khoảng cách đúng/sai chỉ dùng nhánh global",
               lambda r: r["margin_global"]["margin"])
    L += ["## Học thuộc nhiễu (phép thử nhiễu câu trên tập train)", "",
          "Câu bị chuyển sang người khác: model kéo nó về người bị gán sai (học thuộc) "
          "hay về đúng người nó mô tả (phục hồi)? Lượt train với nhiễu câu được thử bằng chính file nhiễu của nó; các lượt khác thử bằng sent50 làm mức tham chiếu.", ""]
    L += table("% học thuộc (về người bị gán sai)",
               lambda r: r["memorisation"]["memorised_person"], 1)
    L += table("% phục hồi (về đúng người)", lambda r: r["memorisation"]["recovered"], 1)
    with open(os.path.join(out_dir, "module_metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="subset size, for a quick check")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "module_metrics"))
    a = ap.parse_args()

    import torch
    logging.getLogger("ViLNM").setLevel(logging.WARNING)
    os.chdir(ROOT)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    names = a.runs or sorted(d for d in os.listdir(RUNS_DIR)
                             if os.path.exists(os.path.join(RUNS_DIR, d, "ablation.json")))

    results = {}
    for name in names:
        run_dir = os.path.join(RUNS_DIR, name)
        meta = json.load(open(os.path.join(run_dir, "ablation.json"), encoding="utf-8"))
        status = json.load(open(os.path.join(run_dir, "status.json"), encoding="utf-8")) \
            if os.path.exists(os.path.join(run_dir, "status.json")) else {}
        if status.get("state") != "finished" or meta.get("seed", 1) != 1:
            continue
        ckpt = ("snapshots/epoch_30.pth"
                if os.path.exists(os.path.join(run_dir, "snapshots", "epoch_30.pth"))
                else "best.pth")
        cache = os.path.join(run_dir, "module_metrics.json")
        r = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else None
        if not os.path.exists(os.path.join(run_dir, ckpt)):
            # Weights pruned to save disk: the cached numbers are all that is left.
            if r is None or r.get("limit") != a.limit:
                print(f"skipping {name}: no weights and no cached result")
                continue
        elif a.force or r is None or r.get("ckpt") != ckpt or r.get("limit") != a.limit:
            print(f"measuring {name} ({ckpt}) ...", flush=True)
            r = measure(run_dir, ckpt, device, a.limit)
            if not a.limit:
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump(r, f, indent=1)
        results[(meta["config"], meta["noise"])] = r
        print(f"{name}: margin {r['margin']['margin']:.3f}  "
              f"pointing {r['pointing']['_all']['hit']:.1f}%  "
              f"memorised {r['memorisation']['memorised_person']:.1f}%")
    if not results:
        print("no finished runs")
        return
    configs = [c for c in CONFIG_ORDER if any(k[0] == c for k in results)]
    noises = [n for n in NOISE_ORDER if any(k[1] == n for k in results)] + \
        sorted({k[1] for k in results} - set(NOISE_ORDER))
    os.makedirs(a.out, exist_ok=True)
    plot(a.out, results, noises, configs)
    write_md(a.out, results, noises, configs)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
