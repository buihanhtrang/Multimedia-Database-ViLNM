"""Tables and charts for the VN3K ablation grid (`tools/ablation.py`).

Reads every `logs/VN3K/abl_*` run (finished or not) and writes to `--out`:

  results.csv          one row per run: final-epoch and best-on-test metrics
  report.md            R@1 / mAP / mINP matrices (config x noise), the drop
                       when each module is removed, NTA noise-detection scores
  r1_vs_noise.png      final R@1 against the *measured* noise rate, per noise type
  delta_r1.png         R@1(full) - R@1(without module), per noise level
  curves.png           R@1 over training epochs, one panel per noise level

Main numbers are from the **final epoch**: VN3K has no validation split, so
picking the best epoch on the test set would flatter every configuration by a
different amount.  Best-on-test is reported alongside, in brackets.

NTA detection (colour noise only, where we know exactly which words were
changed): the final per-token noise levels r' that NTA assigned to each training
caption are compared with the token-level diff between the clean and the noisy
caption.  AUC = probability that a changed token gets a higher r' than an
unchanged token of the same caption set (0.5 = chance); top-1 = how often the
single highest-r' token of a changed caption is a changed token, against the
chance rate of picking one at random.

    python tools/ablation_report.py
"""

import argparse
import csv
import difflib
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONFIG_ORDER = ["full", "no_nta", "nta_random", "no_la", "no_jii", "baseline"]
CONFIG_LABEL = {"full": "Đầy đủ (NTA+LA+JII)", "no_nta": "Bỏ NTA",
                "nta_random": "NTA ngẫu nhiên (đối chứng)", "no_la": "Bỏ LA",
                "no_jii": "Bỏ JII", "baseline": "Baseline (bỏ cả 3)"}
# Fixed categorical slots (validated palette, light surface); colour follows the
# configuration, never its rank.  Markers are the secondary encoding.
COLOR = {"full": "#2a78d6", "no_nta": "#eb6834", "no_la": "#008300",
         "no_jii": "#eda100", "baseline": "#e87ba4", "nta_random": "#1baf7a"}
MARKER = {"full": "o", "no_nta": "s", "no_la": "^", "no_jii": "D", "baseline": "v",
          "nta_random": "P"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def noise_kind(noise):
    return "clean" if noise == "clean" else "sentence" if noise.startswith("sent") else "color"


def noise_sort_key(noise):
    order = {"clean": 0, "sentence": 1, "color": 2}[noise_kind(noise)]
    return order, int("".join(ch for ch in noise if ch.isdigit()) or 0)


# --------------------------------------------------------------------------- #
# collecting runs
# --------------------------------------------------------------------------- #
def collect(runs_dir):
    runs = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "abl_*"))):
        meta = read_json(os.path.join(d, "ablation.json"))
        cfg = read_json(os.path.join(d, "configs.json"))
        if not meta or not cfg:
            continue
        status = read_json(os.path.join(d, "status.json")) or {}
        hist = read_json(os.path.join(d, "history.json")) or {"train": {}, "eval": {}}
        evals = {int(k): v for k, v in hist.get("eval", {}).items()}
        last_ep = max(evals) if evals else None
        num_epoch = cfg["num_epoch"]
        best = status.get("best") or {}
        runs.append(dict(
            dir=d, name=os.path.basename(d), cfg=cfg, **meta,
            state=status.get("state", "new"), epoch=status.get("epoch", 0),
            num_epoch=num_epoch,
            final_epoch=last_ep, complete=last_ep == num_epoch,
            final=evals[last_ep]["t2i"] if evals else None,
            final_global=evals[last_ep]["t2i_global"] if evals else None,
            final_local=evals[last_ep]["t2i_local"] if evals else None,
            best=best if best.get("R1", -1) >= 0 else None,
            curve=sorted((e, v["t2i"]["R1"]) for e, v in evals.items()),
            train={int(k): v for k, v in hist.get("train", {}).items()}))
    return runs


def measured_noise_rates(noises, root_dir):
    """Share of training captions whose text is actually wrong, per noise level (%)."""
    import logging
    from datasets.build import build_dataset
    from datasets.noise import apply_caption_noise, resolve_noise_file
    from tools.ablation import NOISES
    logging.getLogger("ViLNM").setLevel(logging.WARNING)

    ds = build_dataset(argparse.Namespace(dataset_name="VN3K", root_dir=root_dir))
    rates = {}
    for noise in noises:
        if noise == "clean":
            rates[noise] = 0.0
            continue
        path = resolve_noise_file(NOISES[noise], root_dir, ds.dataset_dir)
        _, st = apply_caption_noise(ds.train, path, ds.img_dir)
        bad = st["wrong_identity"] if st["kind"] == "sentence" else st["actually_changed"]
        rates[noise] = 100.0 * bad / st["total"]
    return rates


# --------------------------------------------------------------------------- #
# NTA noise detection
# --------------------------------------------------------------------------- #
def auc(scores, labels):
    """Mann-Whitney AUC with average ranks for ties."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def changed_token_labels(clean_ids, noisy_ids):
    """1 for every token of the noisy caption that is not in the clean one."""
    labels = np.zeros(len(noisy_ids), dtype=bool)
    sm = difflib.SequenceMatcher(None, clean_ids, noisy_ids, autojunk=False)
    for tag, _, _, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            labels[j1:j2] = True
    return labels


def nta_detection(run, root_dir):
    import logging
    import torch
    from datasets.bases import encode_caption
    from datasets.build import build_dataset
    from datasets.noise import apply_caption_noise, resolve_noise_file
    from utils.hf_tokenizer import build_tokenizer
    logging.getLogger("ViLNM").setLevel(logging.WARNING)

    nta = torch.load(os.path.join(run["dir"], "nta_final.pt"), weights_only=False)
    cfg = argparse.Namespace(**run["cfg"])
    tok = build_tokenizer(cfg)
    ds = build_dataset(argparse.Namespace(dataset_name="VN3K", root_dir=root_dir))
    noisy, _ = apply_caption_noise(
        ds.train, resolve_noise_file(run["noise_file"], root_dir, ds.dataset_dir),
        ds.img_dir)

    scores, labels, hits, chance, mismatched = [], [], [], [], 0
    for i, (clean, dirty) in enumerate(zip(ds.train, noisy)):
        noisy_ids = encode_caption(dirty[3], tok, cfg.text_length)
        if noisy_ids != list(nta["original"][i]):
            mismatched += 1
            continue
        lab = changed_token_labels(encode_caption(clean[3], tok, cfg.text_length),
                                   noisy_ids)
        if not lab.any() or lab.all():
            continue
        r = np.asarray(nta["noise"][i], float)
        scores.append(r)
        labels.append(lab)
        # Ties (e.g. a caption NTA never updated keeps r' = rate everywhere)
        # share the hit instead of crediting whichever token comes first.
        hits.append(lab[r == r.max()].mean())
        chance.append(lab.mean())
    if not scores:
        return None
    return dict(captions=len(scores), mismatched=mismatched,
                auc=auc(np.concatenate(scores), np.concatenate(labels)),
                top1=100.0 * float(np.mean(hits)), top1_chance=100.0 * float(np.mean(chance)))


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def matrix(runs, noises, configs, cell):
    head = "| Cấu hình | " + " | ".join(noises) + " |"
    lines = [head, "|---|" + "---|" * len(noises)]
    by = {(r["config"], r["noise"]): r for r in runs}
    for c in configs:
        lines.append(f"| {CONFIG_LABEL[c]} | " +
                     " | ".join(cell(by.get((c, n))) for n in noises) + " |")
    return "\n".join(lines)


def write_markdown(path, runs, noises, configs, rates, detection):
    def metric_cell(key):
        def cell(r):
            if r is None or r["final"] is None:
                return "—"
            s = fmt(r["final"][key])
            if key == "R1" and r["best"]:
                s += f" ({fmt(r['best']['R1'])})"
            return s + ("" if r["complete"] else f" *ep{r['final_epoch']}*")
        return cell

    def delta_cell(by, n, key):
        def cell(r):
            full = by.get(("full", n))
            if r is None or full is None or not r["final"] or not full["final"]:
                return "—"
            return f"{full['final'][key] - r['final'][key]:+.2f}"
        return cell

    by = {(r["config"], r["noise"]): r for r in runs}
    out = ["# Ablation ViLNM trên VN3K", ""]
    first = runs[0]["cfg"]
    out += [f"Backbone `{first.get('backbone')}` ({first.get('hf_model')}), ảnh "
            f"{first['img_size'][0]}×{first['img_size'][1]}, batch {first['batch_size']}, "
            f"{first['num_epoch']} epoch, seed {first['seed']}. Tập test luôn sạch.", ""]
    out += ["Mức nhiễu (tỉ lệ caption train thực sự sai, đo từ dữ liệu):", ""]
    out += ["| " + " | ".join(noises) + " |", "|" + "---|" * len(noises),
            "| " + " | ".join(f"{rates[n]:.1f}%" for n in noises) + " |", ""]

    out += ["## R@1 (epoch cuối; trong ngoặc: best-on-test)", "",
            matrix(runs, noises, configs, metric_cell("R1")), "",
            "*epN* = lượt chưa xong, số lấy ở lần đánh giá gần nhất (epoch N).", ""]
    for key, title in (("mAP", "mAP"), ("mINP", "mINP")):
        out += [f"## {title} (epoch cuối)", "",
                matrix(runs, noises, configs, metric_cell(key)), ""]

    out += ["## Mức tụt khi bỏ module (Đầy đủ − cấu hình, điểm %)", "",
            "Dương = bỏ module làm kết quả kém đi, tức module có ích.", ""]
    for key in ("R1", "mAP"):
        head = "| Cấu hình | " + " | ".join(noises) + " |"
        lines = [f"**{key}**", "", head, "|---|" + "---|" * len(noises)]
        for c in configs:
            if c == "full":
                continue
            lines.append(f"| {CONFIG_LABEL[c]} | " + " | ".join(
                delta_cell(by, n, key)(by.get((c, n))) for n in noises) + " |")
        out += lines + [""]

    def robust_cell(n):
        def cell(r):
            clean = by.get((r["config"], "clean")) if r else None
            if not r or not r["final"] or not clean or not clean["final"]:
                return "—"
            return f"{100 * r['final']['R1'] / clean['final']['R1']:.0f}%"
        return cell
    noisy = [n for n in noises if n != "clean"]
    if noisy and "clean" in noises:
        out += ["## Độ bền với nhiễu (R@1 khi nhiễu ÷ R@1 khi sạch, cùng cấu hình)", "",
                "100% = nhiễu không làm hại gì; càng thấp càng bị nhiễu kéo xuống.", "",
                "| Cấu hình | " + " | ".join(noisy) + " |", "|---|" + "---|" * len(noisy)]
        for c in configs:
            out.append(f"| {CONFIG_LABEL[c]} | " + " | ".join(
                robust_cell(n)(by.get((c, n))) for n in noisy) + " |")
        out.append("")

    def overfit_cell(r):
        if not r or not r["final"] or not r["best"] or not r["complete"]:
            return "—"
        drop = r["best"]["R1"] - r["final"]["R1"]
        return "0" if drop < 0.005 else f"−{drop:.2f} (đỉnh ep{r['best']['epoch']})"
    out += ["## Dấu hiệu học thuộc (R@1 tụt từ đỉnh đến epoch cuối)", "",
            "0 = vẫn còn tăng đến cuối; tụt nhiều = model bắt đầu học thuộc caption sai.", "",
            matrix(runs, noises, configs, overfit_cell), ""]

    def la_cell(r):
        if not r or not r["final"]:
            return "—"
        return (f"{fmt(r['final_global']['R1'])} / {fmt(r['final_local']['R1'])} / "
                f"**{fmt(r['final']['R1'])}**")
    out += ["## LA: global / local / kết hợp (R@1 epoch cuối)", "",
            "Kết hợp cao hơn global bao nhiêu = phần LA đóng góp khi chấm điểm. "
            "Bỏ LA thì chỉ còn global.", "",
            matrix(runs, noises, configs, la_cell), ""]

    out += ["## Toàn cục và cục bộ (R@1 epoch cuối)", "",
            "S = 0,25·S_global + 0,75·S_local. Cột local cho thấy phần LA đóng góp.", "",
            "| Lượt | t2i | t2i_global | t2i_local |", "|---|---|---|---|"]
    for r in runs:
        if r["final"]:
            out.append(f"| {r['name']} | {fmt(r['final']['R1'])} | "
                       f"{fmt(r['final_global']['R1'])} | {fmt(r['final_local']['R1'])} |")
    out.append("")

    if detection:
        out += ["## NTA có bắt đúng từ bị nhiễu không? (nhiễu màu)", "",
                "AUC: 0,5 = đoán bừa, 1,0 = luôn chấm từ bị đổi nhiễu hơn từ đúng. "
                "Top-1: token NTA cho là nhiễu nhất có đúng là từ bị đổi không.", "",
                "| Lượt | caption có từ bị đổi | AUC | top-1 | top-1 ngẫu nhiên |",
                "|---|---|---|---|---|"]
        for name, d in detection.items():
            out.append(f"| {name} | {d['captions']} | {d['auc']:.3f} | "
                       f"{d['top1']:.1f}% | {d['top1_chance']:.1f}% |")
        out.append("")

    out += ["## Trạng thái các lượt", "", "| Lượt | Trạng thái | Epoch |", "|---|---|---|"]
    for r in runs:
        out.append(f"| {r['name']} | {r['state']} | {r['epoch']}/{r['num_epoch']} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def write_csv(path, runs, rates):
    keys = ("R1", "R5", "R10", "mAP", "mINP")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "config", "noise", "noise_kind", "noise_rate_pct", "state",
                    "final_epoch", *[f"final_{k}" for k in keys], "final_global_R1",
                    "final_local_R1", "best_R1", "best_epoch"])
        for r in runs:
            fin = r["final"] or {}
            w.writerow([r["name"], r["config"], r["noise"], noise_kind(r["noise"]),
                        f"{rates.get(r['noise'], float('nan')):.2f}", r["state"],
                        r["final_epoch"], *[fin.get(k) for k in keys],
                        (r["final_global"] or {}).get("R1"),
                        (r["final_local"] or {}).get("R1"),
                        (r["best"] or {}).get("R1"), (r["best"] or {}).get("epoch")])


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_all(out_dir, runs, noises, configs, rates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    by = {(r["config"], r["noise"]): r for r in runs}
    written = []

    # --- R@1 vs measured noise rate, one panel per noise type ---------------
    kinds = [k for k in ("sentence", "color") if any(noise_kind(n) == k for n in noises)]
    if kinds:
        fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4),
                                 squeeze=False, facecolor=SURFACE)
        for ax, kind in zip(axes[0], kinds):
            levels = [n for n in noises if noise_kind(n) in ("clean", kind)]
            for c in configs:
                pts = [(rates[n], by[(c, n)]["final"]["R1"]) for n in levels
                       if (c, n) in by and by[(c, n)]["final"]]
                if pts:
                    xs, ys = zip(*sorted(pts))
                    ax.plot(xs, ys, color=COLOR[c], marker=MARKER[c], markersize=7,
                            linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5,
                            label=CONFIG_LABEL[c])
            name = "Nhiễu câu (tráo caption)" if kind == "sentence" else "Nhiễu màu (đổi từ chỉ màu)"
            style(ax, name, "caption train bị sai (%)", "R@1 epoch cuối (%)")
        axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        written.append(os.path.join(out_dir, "r1_vs_noise.png"))
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    # --- delta bars ---------------------------------------------------------
    removed = [c for c in configs if c != "full"]
    have = [n for n in noises if ("full", n) in by and by[("full", n)]["final"]]
    if have and removed:
        fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(have) + 2), 4), facecolor=SURFACE)
        width = 0.8 / len(removed)
        for k, c in enumerate(removed):
            xs, ys = [], []
            for i, n in enumerate(have):
                r = by.get((c, n))
                if r and r["final"]:
                    xs.append(i - 0.4 + width * (k + 0.5))
                    ys.append(by[("full", n)]["final"]["R1"] - r["final"]["R1"])
            ax.bar(xs, ys, width=width * 0.9, color=COLOR[c], label=CONFIG_LABEL[c],
                   edgecolor=SURFACE, linewidth=1)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(range(len(have)))
        ax.set_xticklabels([f"{n}\n({rates[n]:.0f}%)" for n in have])
        style(ax, "R@1 tụt bao nhiêu khi bỏ module (Đầy đủ − cấu hình)",
              "mức nhiễu (tỉ lệ caption sai thật)", "Δ R@1 (điểm %)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        written.append(os.path.join(out_dir, "delta_r1.png"))
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    # --- training curves ------------------------------------------------------
    with_curves = [n for n in noises if any(by.get((c, n), {}).get("curve") for c in configs)]
    if with_curves:
        cols = min(3, len(with_curves))
        rows = (len(with_curves) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows),
                                 squeeze=False, sharey=True, facecolor=SURFACE)
        for idx, n in enumerate(with_curves):
            ax = axes[idx // cols][idx % cols]
            for c in configs:
                r = by.get((c, n))
                if r and r["curve"]:
                    xs, ys = zip(*r["curve"])
                    ax.plot(xs, ys, color=COLOR[c], marker=MARKER[c], markersize=5,
                            linewidth=2, label=CONFIG_LABEL[c])
            style(ax, f"{n} ({rates[n]:.0f}% sai)", "epoch", "R@1 (%)")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for idx in range(len(with_curves), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)
        axes[0][0].legend(frameon=False, fontsize=7, labelcolor=INK)
        fig.tight_layout()
        written.append(os.path.join(out_dir, "curves.png"))
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    # --- loss components ------------------------------------------------------
    losses = [("loss", "loss tổng"), ("jii_loss", "L_JII"), ("ndf_loss", "L_NDF"),
              ("id_loss", "L_ID")]
    with_train = [n for n in noises if any(by.get((c, n), {}).get("train") for c in configs)]
    if with_train:
        fig, axes = plt.subplots(len(with_train), len(losses),
                                 figsize=(3.6 * len(losses), 2.8 * len(with_train)),
                                 squeeze=False, facecolor=SURFACE)
        for i, n in enumerate(with_train):
            for j, (key, title) in enumerate(losses):
                ax = axes[i][j]
                for c in configs:
                    tr = (by.get((c, n)) or {}).get("train") or {}
                    pts = sorted((e, v[key]) for e, v in tr.items() if key in v)
                    if pts:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, color=COLOR[c], linewidth=1.8, label=CONFIG_LABEL[c])
                style(ax, f"{n} · {title}", "epoch" if i == len(with_train) - 1 else "",
                      "loss train" if j == 0 else "")
                ax.set_yscale("log")
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        seen = {}
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(l, h)
        fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=len(seen),
                   frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        written.append(os.path.join(out_dir, "losses.png"))
        fig.savefig(written[-1], dpi=130)
        plt.close(fig)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default=os.path.join(ROOT, "logs", "VN3K"))
    ap.add_argument("--root_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ablation_VN3K"))
    ap.add_argument("--no_nta_detection", action="store_true",
                    help="skip the NTA token-level analysis (slowest part)")
    a = ap.parse_args()

    runs = collect(a.runs_dir)
    if not runs:
        print(f"no ablation runs under {a.runs_dir}; start them with tools/ablation.py")
        return 1
    noises = sorted({r["noise"] for r in runs}, key=noise_sort_key)
    configs = [c for c in CONFIG_ORDER if any(r["config"] == c for r in runs)]
    runs.sort(key=lambda r: (noise_sort_key(r["noise"]), CONFIG_ORDER.index(r["config"])))
    rates = measured_noise_rates(noises, a.root_dir)

    detection = {}
    if not a.no_nta_detection:
        for r in runs:
            if (noise_kind(r["noise"]) == "color" and r["state"] == "finished"
                    and os.path.exists(os.path.join(r["dir"], "nta_final.pt"))):
                d = nta_detection(r, a.root_dir)
                if d:
                    detection[r["name"]] = d

    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "results.csv"), runs, rates)
    write_markdown(os.path.join(a.out, "report.md"), runs, noises, configs, rates, detection)
    charts = plot_all(a.out, runs, noises, configs, rates)
    print(f"{len(runs)} runs -> {a.out}")
    for p in ["results.csv", "report.md"] + [os.path.basename(c) for c in charts]:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
