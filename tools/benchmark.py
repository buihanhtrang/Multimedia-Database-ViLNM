"""How big a batch fits, and how long will training take?

Runs a few real training steps (real data loader, real model, AMP, the full
ViLNM objective) for each candidate batch size, each in its own subprocess so an
out-of-memory error cannot poison the next measurement, then prints peak VRAM,
step time split into data-wait vs. compute, and the projected time per epoch and
per run.

    python tools/benchmark.py                                   # bs 8 16 24 32
    python tools/benchmark.py --batch_sizes 16 24 --num_workers 8 --eval
    python tools/benchmark.py --dataset_name RSTPReid --root_dir data

Extra arguments after `--` go to train.py's parser unchanged, e.g.
`-- --img_size 384 128 --amp False`.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def measure(cfg, train_argv):
    """Child process: time `cfg.steps` training steps at one batch size."""
    import torch

    from datasets.build import build_dataloader
    from model.build import build_model
    from processor.processor import _total_loss
    from solver.build import build_optimizer
    from utils.iotools import set_seed
    from utils.metrics import Evaluator
    from utils.options import get_args

    args = get_args(train_argv)
    set_seed(args.seed)
    device = "cuda"
    t0 = time.time()
    train_loader, gallery_loader, query_loader, num_classes, train_set = build_dataloader(args)
    model = build_model(args, num_classes).to(device)
    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    setup_s = time.time() - t0
    train_loader.dataset.use_nta = args.use_nta
    torch.cuda.reset_peak_memory_stats()

    out = {"batch_size": args.batch_size, "num_workers": args.num_workers,
           "amp": args.amp, "train_pairs": len(train_set),
           "steps_per_epoch": len(train_loader), "setup_s": setup_s}
    data_s = compute_s = 0.0
    batches = iter(train_loader)
    model.train()
    try:
        for i in range(cfg.warmup + cfg.steps):
            t_data = time.time()
            batch = next(batches)
            batch.pop("index")
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            torch.cuda.synchronize()
            t_compute = time.time()
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss = _total_loss(model(batch))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if i >= cfg.warmup:
                data_s += t_compute - t_data
                compute_s += time.time() - t_compute
    except torch.OutOfMemoryError:
        out["oom"] = True
        return out

    out.update(
        oom=False,
        peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
        total_gb=torch.cuda.get_device_properties(0).total_memory / 2**30,
        data_s_per_step=data_s / cfg.steps,
        compute_s_per_step=compute_s / cfg.steps,
        loss=float(loss.item()))
    out["s_per_step"] = out["data_s_per_step"] + out["compute_s_per_step"]

    if cfg.eval:
        del optimizer, scaler, batches
        torch.cuda.empty_cache()
        t0 = time.time()
        Evaluator(gallery_loader, query_loader, args.sim_alpha, args.sim_beta,
                  device).eval(model, log=False)
        out["eval_s"] = time.time() - t0
    return out


def fmt_h(seconds):
    h, m = divmod(int(round(seconds / 60.0)), 60)
    return f"{h}h{m:02d}" if h else f"{m}m"


def main():
    if "--_child" in sys.argv:
        i = sys.argv.index("--_child")
        cfg = argparse.Namespace(**json.loads(sys.argv[i + 1]))
        res = measure(cfg, sys.argv[i + 2:])
        print("BENCH_RESULT " + json.dumps(res), flush=True)
        return

    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 24, 32])
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--dataset_name", default="CUHK-PEDES")
    ap.add_argument("--root_dir", default="data")
    ap.add_argument("--warmup", type=int, default=10, help="untimed steps")
    ap.add_argument("--steps", type=int, default=40, help="timed steps")
    ap.add_argument("--eval", action="store_true",
                    help="also time one full evaluation pass (at the largest batch that fit)")
    ap.add_argument("--epochs", type=int, nargs="+", default=[30, 60],
                    help="run lengths to project")
    ap.add_argument("--eval_period", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "benchmark.json"))
    a = ap.parse_args(argv)

    results = []
    fits = []
    for bs in a.batch_sizes:
        want_eval = a.eval and bs == max(a.batch_sizes)
        cfg = dict(warmup=a.warmup, steps=a.steps, eval=want_eval)
        train_argv = ["--dataset_name", a.dataset_name, "--root_dir", a.root_dir,
                      "--batch_size", str(bs), "--num_workers", str(a.num_workers),
                      "--amp", "True", "--mlflow", "False", *extra]
        print(f"batch {bs:>3}: measuring ...", end=" ", flush=True)
        start = time.time()
        proc = subprocess.run(
            [sys.executable, "-u", os.path.abspath(__file__), "--_child", json.dumps(cfg),
             *train_argv], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace")
        line = next((l for l in proc.stdout.splitlines() if l.startswith("BENCH_RESULT ")),
                    None)
        if line is None:
            oom = "OutOfMemory" in proc.stdout or "out of memory" in proc.stdout
            print("OOM" if oom else f"FAILED (exit {proc.returncode})")
            if not oom:
                print("\n".join(proc.stdout.strip().splitlines()[-25:]))
            results.append({"batch_size": bs, "oom": oom, "failed": not oom})
            continue
        r = json.loads(line[len("BENCH_RESULT "):])
        results.append(r)
        if r["oom"]:
            print("OOM")
            continue
        fits.append(r)
        print(f"ok in {time.time() - start:.0f}s")

    # Retry the eval timing at the largest batch that fit, if the largest did not.
    if a.eval and fits and "eval_s" not in fits[-1]:
        bs = fits[-1]["batch_size"]
        print(f"batch {bs:>3}: timing evaluation ...", flush=True)
        cfg = dict(warmup=1, steps=1, eval=True)
        proc = subprocess.run(
            [sys.executable, "-u", os.path.abspath(__file__), "--_child", json.dumps(cfg),
             "--dataset_name", a.dataset_name, "--root_dir", a.root_dir,
             "--batch_size", str(bs), "--num_workers", str(a.num_workers),
             "--amp", "True", "--mlflow", "False", *extra],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace")
        line = next((l for l in proc.stdout.splitlines() if l.startswith("BENCH_RESULT ")),
                    None)
        if line:
            fits[-1]["eval_s"] = json.loads(line[len("BENCH_RESULT "):]).get("eval_s")

    eval_s = next((r["eval_s"] for r in fits if r.get("eval_s")), None)

    print()
    hdr = (f"{'batch':>5} {'VRAM peak':>10} {'s/step':>7} {'data%':>6} "
           f"{'img/s':>6} {'epoch':>7} " + " ".join(f"{e:>3}ep" .rjust(8) for e in a.epochs))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.get("oom") or r.get("failed"):
            print(f"{r['batch_size']:>5} {'OOM' if r.get('oom') else 'failed':>10}")
            continue
        epoch_s = r["steps_per_epoch"] * r["s_per_step"]
        r["epoch_s"] = epoch_s
        cells = []
        for e in a.epochs:
            total = e * epoch_s + (e // a.eval_period) * (eval_s or 0.0)
            r[f"run_{e}ep_s"] = total
            cells.append(fmt_h(total).rjust(8))
        print(f"{r['batch_size']:>5} {r['peak_reserved_gb']:>6.2f}/{r['total_gb']:.0f}G "
              f"{r['s_per_step']:>7.3f} {100 * r['data_s_per_step'] / r['s_per_step']:>5.0f}% "
              f"{r['batch_size'] / r['s_per_step']:>6.1f} {fmt_h(epoch_s):>7} "
              + " ".join(cells))
    if eval_s:
        print(f"\none evaluation pass: {eval_s:.0f}s (included above every "
              f"{a.eval_period} epochs)")
    if fits:
        busy = max(fits, key=lambda r: r["data_s_per_step"] / r["s_per_step"])
        if busy["data_s_per_step"] / busy["s_per_step"] > 0.15:
            print(f"data loading takes up to "
                  f"{100 * busy['data_s_per_step'] / busy['s_per_step']:.0f}% of a step: "
                  f"try a larger --num_workers (now {a.num_workers})")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "args": vars(a),
                   "extra": extra, "results": results, "eval_s": eval_s}, f, indent=2)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
