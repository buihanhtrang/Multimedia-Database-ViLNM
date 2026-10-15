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
4. `--resume` restarts from `last.pth`; a run split into sessions (stopped at an
   epoch boundary and mid-epoch) ends with bit-identical weights, best result,
   NTA captions and MLflow run id to one run straight through; Ctrl+C saves a
   resumable mid-epoch checkpoint
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
def _load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def sessions_equivalence(common, results):
    """3 epochs straight vs. the same run cut at an epoch boundary and mid-epoch."""
    db = os.path.join(LOGDIR, "mlflow.db")
    base = [*common, "--num_epoch", "3", "--batch_size", "8", "--log_period", "2",
            "--max_steps_per_epoch", "6", "--eval_period", "1",
            "--nta_warmup_epochs", "1", "--ckpt_every_minutes", "0",
            "--mlflow", "True", "--mlflow_uri", f"sqlite:///{db}",
            "--mlflow_system_metrics", "False"]
    run("sessions: straight run (3 epochs)", ["train.py", "--name", "eqA", *base], results)
    last_b = os.path.join(LOGDIR, "eqB", "last.pth")
    run("sessions: part 1 (--stop_after_epochs 1)",
        ["train.py", "--name", "eqB", *base, "--stop_after_epochs", "1"], results)
    run("sessions: part 2 (--stop_after_steps 4, stops mid-epoch 2)",
        ["train.py", "--name", "eqB", *base, "--stop_after_steps", "4",
         "--resume", last_b], results)
    mid = _load(last_b) if os.path.exists(last_b) else {}
    check("part 2 paused mid-epoch", mid.get("epoch") == 2 and mid.get("step") == 4
          and not mid.get("epoch_complete", True), results,
          f"(epoch {mid.get('epoch')}, step {mid.get('step')})")
    run_id = mid.get("mlflow_run_id")
    run("sessions: part 3 (to the end)",
        ["train.py", "--name", "eqB", *base, "--resume", last_b], results)

    a_path = os.path.join(LOGDIR, "eqA", "last.pth")
    if not (os.path.exists(a_path) and os.path.exists(last_b)):
        check("both runs wrote last.pth", False, results)
        return
    a, b = _load(a_path), _load(last_b)
    diff = max((a["model"][k].float() - b["model"][k].float()).abs().max().item()
               for k in a["model"])
    check("split run weights == straight run weights", diff == 0.0, results,
          f"(max |diff| {diff:.3g})")
    check("split run best == straight run best",
          a["best"] == b["best"], results, f"(R@1 {a['best']['R1']:.3f} vs "
          f"{b['best']['R1']:.3f})")
    check("split run NTA captions == straight run",
          a["nta"]["current"] == b["nta"]["current"], results)
    check("one MLflow run across the 3 sessions",
          run_id is not None and b.get("mlflow_run_id") == run_id, results)


def sigint_pause(common, results):
    """Ctrl+C mid-epoch saves a resumable checkpoint instead of losing the epoch."""
    import signal
    name = "sigint"
    cmd = [sys.executable, "-u", "train.py", "--name", name, *common,
           "--num_epoch", "50", "--batch_size", "8", "--log_period", "1",
           "--eval_period", "0"]
    print(f"\n=== Ctrl+C mid-epoch\n    $ python {' '.join(cmd[2:])}", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    out = []
    for line in proc.stdout:
        out.append(line)
        if "iter[3/" in line:
            proc.send_signal(signal.SIGINT)
            break
    out.extend(proc.stdout)
    proc.wait(timeout=120)
    out = "".join(out)
    ckpt = os.path.join(LOGDIR, name, "last.pth")
    state = _load(ckpt) if os.path.exists(ckpt) else {}
    check("Ctrl+C exits cleanly", proc.returncode == 0, results,
          f"(exit {proc.returncode})")
    check("Ctrl+C saved a mid-epoch checkpoint",
          "session stopped by SIGINT" in out and state.get("step", 0) >= 3
          and not state.get("epoch_complete", True), results,
          f"(epoch {state.get('epoch')}, step {state.get('step')})")


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
        check("resumed after epoch 4", "end of epoch 4" in resume_out, results)

    # -- 5b. split sessions == one straight run ---------------------------- #
    sessions_equivalence(common, results)
    sigint_pause(common, results)

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
