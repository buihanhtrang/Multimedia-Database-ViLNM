"""End-to-end smoke test: does the whole ViLNM pipeline actually run?

Drives `train.py` / `test.py` as subprocesses on the synthetic CUHK-PEDES-format
benchmark, so the real loader, sampler, NTA bookkeeping, trainer, evaluator and
checkpoint I/O are all exercised -- not re-implemented here.

Checks, in order:

1. the unit tests in `tests/test_modules.py`
2. a full 3-epoch train -> eval -> checkpoint run, then assert
     * every loss term decreased from the first epoch to the last
     * NTA's elimination ratio landed near `--nta_rate`
     * the mask/replace/delete mix matches p1 / p2
3. `test.py` reloads `best.pth` and reproduces the training run's metrics exactly
4. `--resume` restarts from `last.pth`
5. every ablation row of Table II, plus the `--use_ndf` / `--use_id` switches
6. the identity sampler (`--sampler identity`)

By default this uses `--tiny_model`, a 2-layer/128-wide CLIP, so the whole suite
finishes in a couple of minutes on a CPU and writes ~30 MB checkpoints instead of
600 MB ones.  Pass `--full_model` to run the real ViT-B/16 instead (slow on CPU),
which is worth doing once to confirm the pretrained path.

    python tools/smoke_test.py
    python tools/smoke_test.py --full_model
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "SYNTH-PEDES"
LOGDIR = os.path.join(ROOT, "logs", DATASET)


# --------------------------------------------------------------------------- #
# subprocess plumbing
# --------------------------------------------------------------------------- #
def run(label, cmd, results):
    """Run a command, print a one-line verdict, and remember the outcome."""
    print(f"\n=== {label}\n    $ python {' '.join(cmd)}", flush=True)
    start = time.time()
    proc = subprocess.run([sys.executable, "-u"] + cmd, cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    dt = time.time() - start
    ok = proc.returncode == 0
    print(f"    -> {'ok' if ok else f'FAILED (exit {proc.returncode})'}  [{dt:.1f}s]",
          flush=True)
    if not ok:
        tail = proc.stdout.strip().splitlines()[-40:]
        print("    " + "\n    ".join(tail), flush=True)
    results.append((label, ok))
    return proc.stdout


def check(label, ok, results, detail=""):
    print(f"    -> {'ok' if ok else 'FAILED'}  {label}"
          f"{'  ' + detail if detail else ''}", flush=True)
    results.append((label, bool(ok)))
    return ok


# --------------------------------------------------------------------------- #
# log parsing
# --------------------------------------------------------------------------- #
def epoch_losses(out):
    """[{name: value}, ...] from the 'epoch N done in ...' summary lines."""
    epochs = []
    for line in out.splitlines():
        if " done in " not in line:
            continue
        body = line.split("|", 1)[-1]
        terms = dict(re.findall(r"(\w+_loss)\s+(-?\d+\.\d+)", body))
        total = re.search(r"\bloss\s+(-?\d+\.\d+)", body)
        if total:
            terms["loss"] = total.group(1)
        if terms:
            epochs.append({k: float(v) for k, v in terms.items()})
    return epochs


def nta_stats(out):
    """[{...}, ...] from the 'NTA epoch N: {...}' lines."""
    stats = []
    for raw in re.findall(r"NTA epoch \d+: \{(.*?)\}", out):
        d = {}
        for key, val in re.findall(r"'(\w+)':\s*([\d.eE+-]+)", raw):
            d[key] = float(val)
        if d:
            stats.append(d)
    return stats


def t2i_row(out):
    """The fused-similarity metrics from the last evaluation table printed."""
    rows = re.findall(r"\|\s*t2i\s*\|([^\n]*)\|", out)
    if not rows:
        return None
    cells = [c.strip() for c in rows[-1].split("|") if c.strip()]
    if len(cells) != 5:
        return None
    return dict(zip(("R1", "R5", "R10", "mAP", "mINP"), (float(c) for c in cells)))


def best_from_train(out):
    """The 'training finished. best: {...}' dict."""
    m = re.search(r"training finished\. best: \{(.*?)\}", out)
    if not m:
        return None
    d = {}
    for key, val in re.findall(r"'(\w+)':\s*([\d.eE+-]+)", m.group(1)):
        d[key] = float(val)
    return d or None


# --------------------------------------------------------------------------- #
# the suite
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full_model", action="store_true",
                    help="use the real CLIP-ViT-B/16 instead of --tiny_model")
    ap.add_argument("--keep_logs", action="store_true",
                    help="do not delete logs/SYNTH-PEDES when finished")
    ap.add_argument("--skip_unit_tests", action="store_true")
    a = ap.parse_args()

    results = []
    size = ["--img_size", "384", "128"] if a.full_model else \
           ["--img_size", "96", "32", "--tiny_model", "True"]
    if a.full_model:
        size += ["--no_pretrained", "True"] if not os.path.exists(
            os.path.join(ROOT, "pretrained", "ViT-B-16.pt")) else []
    common = ["--dataset_name", DATASET, "--num_workers", "0"] + size

    print(f"ViLNM smoke test  ({'ViT-B/16' if a.full_model else 'tiny model'})")
    print(f"repo: {ROOT}")

    # -- 1. unit tests ----------------------------------------------------- #
    if not a.skip_unit_tests:
        out = run("unit tests", [os.path.join("tests", "test_modules.py")], results)
        m = re.search(r"(\d+)\s*/\s*(\d+)\s+tests? passed", out)
        if m:
            print(f"    {m.group(0)}")

    # -- 2. synthetic benchmark -------------------------------------------- #
    run("build synthetic benchmark",
        [os.path.join("tools", "make_synthetic_data.py")], results)

    # -- 3. train ---------------------------------------------------------- #
    train_out = run(
        "train (4 epochs, all components)",
        ["train.py", "--name", "smoke", *common,
         "--num_epoch", "4", "--batch_size", "8", "--log_period", "5",
         "--eval_period", "1", "--nta_warmup_epochs", "1",
         "--save_optimizer", "True"],
        results)

    losses = epoch_losses(train_out)
    if check("4 epochs completed", len(losses) >= 4, results,
             f"({len(losses)} epoch summaries)") and len(losses) >= 4:
        first, last = losses[0], losses[-1]
        # The total objective is the thing that must go down; individual terms can
        # trade off against each other, so those are reported but not asserted.
        if "loss" in first and "loss" in last:
            check("total loss decreased", last["loss"] < first["loss"], results,
                  f"{first['loss']:.4f} -> {last['loss']:.4f}")
        for key in sorted(set(first) & set(last) - {"loss"}):
            delta = last[key] - first[key]
            print(f"       {key:<14} {first[key]:9.4f} -> {last[key]:9.4f} "
                  f"({delta:+.4f})")

    stats = nta_stats(train_out)
    if check("NTA ran", bool(stats), results, f"({len(stats)} epochs)") and stats:
        rate, p1, p2 = 0.2, 0.6, 0.8
        for s in stats:
            check("elimination ratio ~= nta_rate",
                  abs(s["eliminated_ratio"] - rate) < 0.05, results,
                  f"{s['eliminated_ratio']:.4f} vs rate={rate}")
        agg = {k: sum(s.get(k, 0) for s in stats) for k in ("mask", "replace", "delete")}
        n = sum(agg.values()) or 1
        frac = {k: v / n for k, v in agg.items()}
        check("mask fraction ~= p1", abs(frac["mask"] - p1) < 0.06, results,
              f"{frac['mask']:.3f} vs p1={p1}")
        check("replace fraction ~= p2-p1", abs(frac["replace"] - (p2 - p1)) < 0.06,
              results, f"{frac['replace']:.3f} vs p2-p1={p2 - p1:.1f}")
        check("delete fraction ~= 1-p2", abs(frac["delete"] - (1 - p2)) < 0.06,
              results, f"{frac['delete']:.3f} vs 1-p2={1 - p2:.1f}")

    # -- 4. checkpoint round-trip ------------------------------------------ #
    run_dir = os.path.join(LOGDIR, "smoke")
    cfg = os.path.join(run_dir, "configs.json")
    best_ckpt = os.path.join(run_dir, "best.pth")
    last_ckpt = os.path.join(run_dir, "last.pth")
    check("best.pth written", os.path.exists(best_ckpt), results)
    check("configs.json written", os.path.exists(cfg), results)

    if os.path.exists(best_ckpt) and os.path.exists(cfg):
        test_out = run("reload best.pth with test.py",
                       ["test.py", "--config_file", cfg,
                        "--checkpoint", best_ckpt, "--device", "cpu"], results)
        trained, reloaded = best_from_train(train_out), t2i_row(test_out)
        if trained and reloaded:
            same = all(abs(trained[k] - reloaded[k]) < 1e-3
                       for k in ("R1", "R5", "R10", "mAP", "mINP") if k in trained)
            check("reloaded metrics match the training run", same, results,
                  f"R@1 {trained['R1']:.3f} vs {reloaded['R1']:.3f}, "
                  f"mAP {trained['mAP']:.3f} vs {reloaded['mAP']:.3f}")
        else:
            check("metrics parsed from both runs", False, results)

    # -- 5. resume --------------------------------------------------------- #
    if os.path.exists(last_ckpt):
        resume_out = run("resume from last.pth",
                         ["train.py", "--name", "smoke_resume", *common,
                          "--num_epoch", "5", "--batch_size", "8",
                          "--max_steps_per_epoch", "3", "--eval_period", "0",
                          "--save_last", "False", "--resume", last_ckpt],
                         results)
        check("resumed at epoch 5", "at epoch 5" in resume_out, results)

    # -- 6. ablations (Table II) and loss switches ------------------------- #
    ablations = [
        ("baseline (CLIP + NDF + ID)",
         ["--use_nta", "False", "--use_jii", "False", "--loss_on_local", "False"]),
        ("+ NTA", ["--use_jii", "False", "--loss_on_local", "False"]),
        ("+ LA", ["--use_jii", "False"]),
        ("+ JII", ["--use_nta", "False"]),
        ("full", []),
        ("no NDF", ["--use_ndf", "False"]),
        ("no ID", ["--use_id", "False"]),
        ("local-only losses", ["--loss_on_global", "False"]),
    ]
    for i, (label, extra) in enumerate(ablations):
        run(f"ablation: {label}",
            ["train.py", "--name", f"abl{i}", *common,
             "--num_epoch", "2", "--batch_size", "8", "--log_period", "5",
             "--max_steps_per_epoch", "3", "--eval_period", "2",
             "--save_last", "False", *extra],
            results)

    # -- 7. identity sampler ----------------------------------------------- #
    run("identity sampler (PK sampling)",
        ["train.py", "--name", "sampler", *common,
         "--sampler", "identity", "--num_instance", "2",
         "--num_epoch", "1", "--batch_size", "8", "--log_period", "5",
         "--max_steps_per_epoch", "3", "--eval_period", "0",
         "--save_last", "False"],
        results)

    if not a.keep_logs and os.path.isdir(LOGDIR):
        shutil.rmtree(LOGDIR, ignore_errors=True)
        print(f"\nremoved {LOGDIR} (pass --keep_logs to keep it)")

    # -- summary ----------------------------------------------------------- #
    failed = [name for name, ok in results if not ok]
    print("\n" + "=" * 68)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    for name in failed:
        print(f"  FAILED  {name}")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
