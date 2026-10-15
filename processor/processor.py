"""Training / evaluation loop.

The NTA bookkeeping is the only non-standard part: during each epoch the model
returns r' (Eq. 4) for the captions it saw, those values are scattered onto the
original tokens, and at the end of the epoch `NTAState.resample()` applies
Eq. (5) once to build the captions for the next epoch.  That is what "requires
only one forward pass for each epoch" means in Sec. III-B.

Training can be split over several sessions.  `last.pth` holds everything the
loop depends on -- weights, optimizer, scheduler, AMP scaler, NTA captions and
noise levels, every RNG, the best result so far and the position inside the
epoch -- so a run stopped and resumed any number of times follows the same
trajectory as one run straight through (bit-exact on CPU with
`--num_workers 0`; with workers, a resume *mid-epoch* re-seeds the augmentation
RNG of the remaining batches, which changes nothing statistically).
"""

import json
import logging
import os
import os.path as op
import signal
import time

import torch

from utils.iotools import get_rng_state, save_checkpoint, set_rng_state
from utils.meter import MetricLogger
from utils.metrics import Evaluator

logger = logging.getLogger("ViLNM.train")

LOSS_KEYS = ("jii_loss", "ndf_loss", "id_loss")


def _total_loss(ret):
    return sum(ret[k] for k in LOSS_KEYS if k in ret)


class StopController:
    """Decides when this session should save and exit.

    Triggers: SIGINT/SIGTERM (first one only; a second SIGINT aborts at once),
    a file named STOP in the output directory, `--max_hours`,
    `--stop_after_steps` and `--stop_after_epochs`.
    """

    def __init__(self, args):
        self.args = args
        self.start = time.time()
        self.steps = 0
        self.epochs = 0
        self.signalled = None
        self.stop_file = op.join(args.output_dir, "STOP")
        if op.exists(self.stop_file):
            os.remove(self.stop_file)          # stale request from an earlier session
        self._previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.signal(sig, self._handle)

    def _handle(self, signum, frame):
        if self.signalled:                     # second signal: give up immediately
            self.restore()
            raise KeyboardInterrupt
        self.signalled = signal.Signals(signum).name
        logger.warning("%s received: saving after the current step "
                       "(press Ctrl+C again to abort without saving)", self.signalled)

    def restore(self):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)

    def hours(self):
        return (time.time() - self.start) / 3600.0

    def step_reason(self):
        if self.signalled:
            return self.signalled
        if op.exists(self.stop_file):
            os.remove(self.stop_file)
            return "STOP file"
        if self.args.max_hours and self.hours() >= self.args.max_hours:
            return f"--max_hours {self.args.max_hours}"
        if self.args.stop_after_steps and self.steps >= self.args.stop_after_steps:
            return f"--stop_after_steps {self.args.stop_after_steps}"
        return None

    def epoch_reason(self):
        if self.args.stop_after_epochs and self.epochs >= self.args.stop_after_epochs:
            return f"--stop_after_epochs {self.args.stop_after_epochs}"
        return self.step_reason()


def _write_status(args, **status):
    status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = op.join(args.output_dir, "status.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, op.join(args.output_dir, "status.json"))


def do_train(args, model, train_loader, evaluator: Evaluator, optimizer,
             scheduler, device, resume: dict = None, tracker=None):
    """Train until `--num_epoch` or until a stop trigger fires.

    `resume` is the content of a `last.pth`; returns (best, finished).
    """
    from utils.tracker import Tracker
    resume = resume or {}
    tracker = tracker or Tracker(args, resume.get("mlflow_run_id"))
    meters = MetricLogger()
    dataset = train_loader.dataset
    nta_state = getattr(dataset, "nta_state", None)
    sampler = train_loader.sampler
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")
    best = resume.get("best", {"R1": -1.0, "epoch": -1})
    global_step = resume.get("global_step", 0)
    stopper = StopController(args)

    if "scaler" in resume:
        scaler.load_state_dict(resume["scaler"])
    if nta_state is not None and "nta" in resume:
        nta_state.load_state_dict(resume["nta"])
    rng_state = resume.get("rng")

    if not resume:
        start_epoch, start_step, pending_eval = 1, 0, False
    elif resume.get("epoch_complete", True):
        start_epoch, start_step = resume["epoch"] + 1, 0
        pending_eval = not resume.get("eval_done", True)
    else:
        start_epoch, start_step, pending_eval = resume["epoch"], resume["step"], False

    def checkpoint(epoch, step, epoch_complete, eval_done=True):
        payload = {"epoch": epoch, "model": model.state_dict()}
        # A mid-epoch checkpoint is useless without the position and RNG state.
        if args.save_optimizer or not epoch_complete:
            payload.update(
                step=step, epoch_complete=epoch_complete, eval_done=eval_done,
                global_step=global_step, best=best,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), rng=get_rng_state(),
                mlflow_run_id=tracker.run_id)
            if nta_state is not None:
                payload["nta"] = nta_state.state_dict()
        save_checkpoint(payload, op.join(args.output_dir, "last.pth"))

    def evaluate(epoch):
        nonlocal best
        results = evaluator.eval(model)
        tracker.log_metrics(results, step=epoch, prefix="eval")
        r1 = results["t2i"]["R1"]
        if r1 > best["R1"]:
            best = {"R1": r1, "epoch": epoch, **results["t2i"]}
            save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                             "results": results},
                            op.join(args.output_dir, "best.pth"))
            tracker.set_tags({"best_R1": f"{r1:.3f}", "best_epoch": epoch})
            logger.info("new best R@1 = %.3f at epoch %d", r1, epoch)

    def pause(epoch, step, epoch_complete, reason, eval_done=True):
        checkpoint(epoch, step, epoch_complete, eval_done)
        where = f"end of epoch {epoch}" if epoch_complete else f"epoch {epoch} step {step}"
        logger.info("session stopped by %s at %s after %.2f h; resume with "
                    "--resume %s", reason, where, stopper.hours(),
                    op.join(args.output_dir, "last.pth"))
        _write_status(args, state="paused", epoch=epoch, step=step,
                      epoch_complete=epoch_complete, num_epoch=args.num_epoch,
                      global_step=global_step, best=best, reason=reason)
        tracker.set_tags({"state": "paused", "paused_at": where})
        stopper.restore()
        tracker.close("KILLED")
        return best, False

    if pending_eval:
        # The previous session stopped after finishing epoch N but before (or
        # during) its evaluation; run it now, from the RNG state it would have had.
        if rng_state is not None:
            set_rng_state(rng_state)
            rng_state = None
        evaluate(start_epoch - 1)
        checkpoint(start_epoch - 1, 0, True)

    tracker.set_tags({"state": "running"})
    last_ckpt_time = time.time()

    for epoch in range(start_epoch, args.num_epoch + 1):
        model.train()
        meters.reset()
        epoch_start = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Freeze NTA elimination for the first epochs: r' is meaningless before
        # the encoders have adapted, and Eq. (4) would drop informative tokens.
        dataset.use_nta = args.use_nta and epoch > args.nta_warmup_epochs

        steps_per_epoch = len(train_loader)
        skip = start_step if epoch == start_epoch else 0
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
            if skip:
                sampler.skip(skip * args.batch_size)
                logger.info("resuming epoch %d at step %d/%d", epoch, skip, steps_per_epoch)
        # At an epoch boundary the RNG goes back before the iterator draws its
        # worker seeds; mid-epoch it goes back after, where the original run was.
        if rng_state is not None and not skip:
            set_rng_state(rng_state)
            rng_state = None
        batches = iter(train_loader)
        if rng_state is not None:
            set_rng_state(rng_state)
            rng_state = None

        step = skip
        while True:
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            try:
                batch = next(batches)
            except StopIteration:
                break
            except RuntimeError:
                # Ctrl+C reaches the DataLoader workers too and kills them.
                if stopper.signalled:
                    return pause(epoch, step, False, stopper.signalled)
                raise
            step += 1

            indices = batch.pop("index")
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=args.amp and device == "cuda"):
                ret = model(batch)
                loss = _total_loss(ret)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            stopper.steps += 1

            # --- cache this batch's noise levels for the next epoch ---
            if nta_state is not None and "nta_noise" in ret:
                noise = ret["nta_noise"].detach().float().cpu().numpy()
                for row, idx in enumerate(indices.tolist()):
                    nta_state.update(idx, noise[row])

            logs = {"loss": loss.detach().item()}
            logs.update({k: v.detach().item() if torch.is_tensor(v) else float(v)
                         for k, v in ret.items() if k.endswith("_loss")})
            meters.update(logs, n=1)

            if step % args.log_period == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info("epoch[%d] iter[%d/%d] %s, lr %.2e",
                            epoch, step, steps_per_epoch, meters, lr)
                tracker.log_metrics({**{k: m.avg for k, m in meters.meters.items()},
                                     "lr": lr}, step=global_step, prefix="train")

            reason = stopper.step_reason()
            if reason:
                return pause(epoch, step, False, reason)
            if (args.ckpt_every_minutes
                    and time.time() - last_ckpt_time >= args.ckpt_every_minutes * 60):
                checkpoint(epoch, step, False)
                last_ckpt_time = time.time()
                logger.info("mid-epoch checkpoint at epoch %d step %d", epoch, step)

        scheduler.step()

        # --- Eq. (5): rebuild the captions used by the next epoch ---
        nta_stats = None
        if nta_state is not None and args.use_nta and epoch >= args.nta_warmup_epochs:
            nta_state.resample()
            nta_stats = nta_state.stats()
            logger.info("NTA epoch %d: %s", epoch, nta_stats)

        epoch_time = time.time() - epoch_start
        logger.info("epoch %d done in %.1fs | %s", epoch, epoch_time, meters)
        epoch_metrics = {"time_min": epoch_time / 60.0,
                         **{k: m.avg for k, m in meters.meters.items()}}
        if device == "cuda":
            epoch_metrics["gpu_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if nta_stats:
            epoch_metrics.update({f"nta_{k}": v for k, v in nta_stats.items()})
        tracker.log_metrics(epoch_metrics, step=epoch, prefix="epoch")

        do_eval = bool(args.eval_period) and (epoch % args.eval_period == 0
                                              or epoch == args.num_epoch)
        if do_eval:
            # Save first, so a stop during evaluation does not redo the epoch.
            if args.save_last:
                checkpoint(epoch, steps_per_epoch, True, eval_done=False)
            try:
                evaluate(epoch)
            except (RuntimeError, KeyboardInterrupt):
                if stopper.signalled:
                    return pause(epoch, steps_per_epoch, True, stopper.signalled,
                                 eval_done=False)
                raise
        if args.save_last:
            checkpoint(epoch, steps_per_epoch, True)
        last_ckpt_time = time.time()
        stopper.epochs += 1
        _write_status(args, state="running", epoch=epoch, step=steps_per_epoch,
                      epoch_complete=True, num_epoch=args.num_epoch,
                      global_step=global_step, best=best,
                      last_epoch_min=round(epoch_time / 60.0, 2))

        if epoch < args.num_epoch:
            reason = stopper.epoch_reason()
            if reason:
                return pause(epoch, steps_per_epoch, True, reason)

    logger.info("training finished. best: %s", best)
    _write_status(args, state="finished", epoch=args.num_epoch, num_epoch=args.num_epoch,
                  global_step=global_step, best=best)
    tracker.set_tags({"state": "finished"})
    tracker.log_artifact(op.join(args.output_dir, "configs.json"))
    tracker.log_artifact(op.join(args.output_dir, "train_log.txt"))
    stopper.restore()
    tracker.close("FINISHED")
    return best, True


def do_inference(model, evaluator: Evaluator):
    return evaluator.eval(model)
