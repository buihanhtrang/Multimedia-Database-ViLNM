"""Run the ViLNM ablation grid on VN3K: module configurations x noise levels.

Each cell of the grid is an ordinary `train.py` run in
`logs/VN3K/abl_<config>_<noise>/`, so everything the trainer offers (resume,
MLflow, status.json, history.json) applies.  Run this script whenever the
machine is free; it skips finished cells, resumes a paused one, and starts the
rest in order (noise level by noise level, so a complete comparison at one noise
level is available as early as possible).

    python tools/ablation.py --status                  # what is done, what is left
    python tools/ablation.py --dry_run                 # print the commands only
    python tools/ablation.py                           # batch 1 (NTA): {full, no_nta,
                                                       #   nta_random} x {clean, color30, color50}
    python tools/ablation.py --hours 8                 # stop (and save) after 8 hours
    python tools/ablation.py --configs full no_la no_jii baseline --noise clean color50
    python tools/ablation.py --configs full no_nta --noise color20
    python tools/ablation.py --configs full no_nta nta_random --noise sent20 color20 \
        --keep_nta --rest_minutes 10        # NTA study, ~1-2 GB per run

Stopping: Ctrl+C saves the current run mid-epoch and exits; or, from another
terminal, `touch logs/VN3K/STOP_ABLATION` to do the same (the file is removed
when the next invocation starts).  Tables and charts:
`python tools/ablation_report.py`, at any time, finished or not.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "VN3K"
RUNS_DIR = os.path.join(ROOT, "logs", DATASET)
STOP_FILE = os.path.join(RUNS_DIR, "STOP_ABLATION")

# Module switches.  Without LA, S_local comes from an untrained LA head, so the
# similarity must also drop it (sim_alpha 1, sim_beta 0) -- otherwise 75% of the
# retrieval score would be noise.
NO_LA = ["--loss_on_local", "False", "--sim_alpha", "1", "--sim_beta", "0"]
CONFIGS = {
    "full": [],
    "no_nta": ["--use_nta", "False"],
    # Same 20% token elimination as NTA, chosen at random: separates "NTA finds
    # the noisy words" from "dropping words regularises".
    "nta_random": ["--nta_random", "True"],
    "no_la": NO_LA,
    "no_jii": ["--use_jii", "False"],
    "baseline": ["--use_nta", "False", "--use_jii", "False", *NO_LA],
}

NOISES = {"clean": ""}
NOISES.update({f"sent{k}0": f"noise_sentence/VN3K-VI_0.{k}.npy" for k in range(1, 9)})
NOISES.update({f"color{k}0": f"noisy_color/vn3k_attribute_noise_{k}0pct.json"
               for k in range(1, 8)})


def run_name(config, noise, seed):
    return f"abl_{config}_{noise}" + ("" if seed == 1 else f"_s{seed}")


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run_state(run_dir):
    """('finished' | 'paused' | 'running' | 'new', status dict or None)."""
    status = read_json(os.path.join(run_dir, "status.json"))
    if status is None:
        return ("paused" if os.path.exists(os.path.join(run_dir, "last.pth"))
                else "new"), None
    return status.get("state", "new"), status


def train_command(a, config, noise, run_dir, hours_left):
    cmd = [sys.executable, "-u", "train.py",
           "--dataset_name", DATASET, "--backbone", "siglip2",
           "--name", os.path.basename(run_dir),
           "--img_size", *map(str, a.img_size), "--batch_size", str(a.batch_size),
           "--num_epoch", str(a.num_epoch), "--eval_period", str(a.eval_period),
           "--num_workers", str(a.num_workers), "--seed", str(a.seed), "--amp", "True",
           "--mlflow", "True", "--mlflow_experiment", "VN3K-ablation",
           *CONFIGS[config]]
    if NOISES[noise]:
        cmd += ["--noise_file", NOISES[noise]]
    if hours_left is not None:
        cmd += ["--max_hours", f"{max(hours_left, 0.01):.3f}"]
    if os.path.exists(os.path.join(run_dir, "last.pth")):
        cmd += ["--resume", os.path.join(run_dir, "last.pth")]
    return cmd + a.extra


def print_status(cells, a):
    print(f"{'run':<26} {'state':<9} {'epoch':>7} {'final R@1':>9} {'best R@1':>9}")
    left = 0
    for config, noise in cells:
        run_dir = os.path.join(RUNS_DIR, run_name(config, noise, a.seed))
        state, status = run_state(run_dir)
        hist = read_json(os.path.join(run_dir, "history.json")) or {}
        final = hist.get("eval", {}).get(str(a.num_epoch), {}).get("t2i", {}).get("R1")
        best = (status or {}).get("best", {}).get("R1")
        epoch = f"{(status or {}).get('epoch', 0)}/{a.num_epoch}"
        final = "-" if final is None else f"{final:.2f}"
        best = "-" if best in (None, -1.0) else f"{best:.2f}"
        print(f"{run_name(config, noise, a.seed):<26} {state:<9} {epoch:>7} "
              f"{final:>9} {best:>9}")
        if state != "finished":
            done = (status or {}).get("epoch", 0) if state != "new" else 0
            left += (a.num_epoch - done) / a.num_epoch
    print(f"\n{len(cells)} runs, about {left:.1f} runs of work left "
          f"(~{left * a.hours_per_run:.0f} h at {a.hours_per_run} h per run)")


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["full", "no_nta", "nta_random"],
                    choices=list(CONFIGS))
    ap.add_argument("--noise", nargs="+", default=["clean", "color30", "color50"],
                    choices=list(NOISES))
    ap.add_argument("--img_size", type=int, nargs=2, default=[384, 128])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_epoch", type=int, default=30)
    ap.add_argument("--eval_period", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hours", type=float, default=0.0,
                    help="time budget for this invocation (0 = until the grid is done)")
    ap.add_argument("--keep_last", action="store_true",
                    help="keep last.pth (3 GB) of finished runs; by default it is "
                         "deleted once the run is finished -- best.pth, history.json "
                         "and nta_final.pt stay")
    ap.add_argument("--keep_all", action="store_true",
                    help="keep everything for later analysis: last.pth (continue "
                         "training later) plus snapshots/ (weights at every "
                         "evaluation, NTA state every epoch; ~7 GB per run)")
    ap.add_argument("--keep_nta", action="store_true",
                    help="lean history for the NTA study: the NTA state every epoch "
                         "(snapshots/nta_epoch_XX.pt) and the last epoch's weights "
                         "(snapshots/epoch_XX.pth); best.pth only when the best "
                         "evaluation is not the last one; last.pth deleted (~1-2 GB per run)")
    ap.add_argument("--rest_minutes", type=float, default=0.0,
                    help="pause between two runs, to let the GPU cool down")
    ap.add_argument("--status", action="store_true", help="print the grid state and exit")
    ap.add_argument("--dry_run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--hours_per_run", type=float, default=1.9,
                    help="for the --status estimate (benchmark: 384x128, batch 32)")
    a = ap.parse_args(argv)
    a.extra = extra
    if a.keep_all:
        a.keep_last = True
        a.extra = ["--save_snapshots", "True"] + a.extra
    elif a.keep_nta:
        a.extra = ["--save_snapshots", "True", "--snapshot_weights", "final"] + a.extra

    cells = [(c, n) for n in a.noise for c in a.configs]
    if a.status:
        print_status(cells, a)
        return 0

    if os.path.exists(STOP_FILE) and not a.dry_run:
        os.remove(STOP_FILE)
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)        # conda's hook breaks the venv's cuDNN
    # Less fragmentation: the 6 GB card is shared with the desktop.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    inhibit = []
    if shutil.which("systemd-inhibit"):
        inhibit = ["systemd-inhibit", "--what=sleep:idle", "--who=ViLNM",
                   "--why=ViLNM ablation"]
    start = time.time()
    trained = False

    for config, noise in cells:
        name = run_name(config, noise, a.seed)
        run_dir = os.path.join(RUNS_DIR, name)
        state, _ = run_state(run_dir)
        if state == "finished":
            print(f"== {name}: finished, skipping")
            continue
        hours_left = None
        if a.hours:
            hours_left = a.hours - (time.time() - start) / 3600
            if hours_left <= 0:
                print(f"== time budget of {a.hours} h used up")
                break
        if trained and a.rest_minutes > 0 and not a.dry_run:
            print(f"== resting {a.rest_minutes:g} min before {name}", flush=True)
            rest_end = time.time() + a.rest_minutes * 60
            while time.time() < rest_end:
                if os.path.exists(STOP_FILE):
                    print("== stop requested during the rest")
                    return 0
                time.sleep(5)
            if a.hours:
                hours_left = a.hours - (time.time() - start) / 3600
                if hours_left <= 0:
                    print(f"== time budget of {a.hours} h used up")
                    break
        cmd = train_command(a, config, noise, run_dir, hours_left)
        print(f"== {name}: {'resuming' if '--resume' in cmd else 'starting'}")
        print("   $ python " + " ".join(cmd[2:]), flush=True)
        if a.dry_run:
            continue

        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "ablation.json"), "w", encoding="utf-8") as f:
            json.dump({"config": config, "noise": noise, "noise_file": NOISES[noise],
                       "seed": a.seed}, f, indent=2)
        proc = subprocess.Popen(inhibit + cmd, cwd=ROOT, env=env)
        try:
            while proc.poll() is None:
                if os.path.exists(STOP_FILE):
                    # The trainer polls its own STOP file; it saves and exits.
                    open(os.path.join(run_dir, "STOP"), "a").close()
                time.sleep(5)
        except KeyboardInterrupt:
            # The child got the same SIGINT and is saving; wait for it.
            proc.wait()
            print(f"\n== interrupted; {name} saved, rerun this script to continue")
            return 130

        state, status = run_state(run_dir)
        if proc.returncode != 0:
            print(f"== {name} FAILED (exit {proc.returncode}); see {run_dir}/train_log.txt")
            return proc.returncode
        if state != "finished":
            print(f"== {name} paused ({(status or {}).get('reason', '?')}); "
                  f"rerun this script to continue")
            return 0
        if not a.keep_last and os.path.exists(os.path.join(run_dir, "last.pth")):
            os.remove(os.path.join(run_dir, "last.pth"))
        best_pth = os.path.join(run_dir, "best.pth")
        if (a.keep_nta and status["best"].get("epoch") == a.num_epoch
                and os.path.exists(best_pth)):
            os.remove(best_pth)             # same weights as snapshots/epoch_<last>.pth
        trained = True
        print(f"== {name} finished: best R@1 {status['best']['R1']:.2f}")

    print()
    print_status(cells, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
