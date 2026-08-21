"""Training / evaluation loop.

The NTA bookkeeping is the only non-standard part: during each epoch the model
returns r' (Eq. 4) for the captions it saw, those values are scattered onto the
original tokens, and at the end of the epoch `NTAState.resample()` applies
Eq. (5) once to build the captions for the next epoch.  That is what "requires
only one forward pass for each epoch" means in Sec. III-B.
"""

import logging
import os.path as op
import time

import torch

from utils.iotools import save_checkpoint
from utils.meter import MetricLogger
from utils.metrics import Evaluator

logger = logging.getLogger("ViLNM.train")

LOSS_KEYS = ("jii_loss", "ndf_loss", "id_loss")


def _total_loss(ret):
    return sum(ret[k] for k in LOSS_KEYS if k in ret)


def do_train(args, model, train_loader, evaluator: Evaluator, optimizer,
             scheduler, device, start_epoch: int = 1):
    meters = MetricLogger()
    nta_state = getattr(train_loader.dataset, "nta_state", None)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")
    best = {"R1": -1.0, "epoch": -1}

    for epoch in range(start_epoch, args.num_epoch + 1):
        model.train()
        meters.reset()
        epoch_start = time.time()

        # Freeze NTA elimination for the first epochs: r' is meaningless before
        # the encoders have adapted, and Eq. (4) would drop informative tokens.
        train_loader.dataset.use_nta = args.use_nta and epoch > args.nta_warmup_epochs

        for step, batch in enumerate(train_loader, start=1):
            if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
                break

            indices = batch.pop("index")
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=args.amp and device == "cuda"):
                ret = model(batch)
                loss = _total_loss(ret)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
                logger.info("epoch[%d] iter[%d/%d] %s, lr %.2e",
                            epoch, step, len(train_loader), meters,
                            scheduler.get_last_lr()[0])

        scheduler.step()

        # --- Eq. (5): rebuild the captions used by the next epoch ---
        if nta_state is not None and args.use_nta and epoch >= args.nta_warmup_epochs:
            nta_state.resample()
            logger.info("NTA epoch %d: %s", epoch, nta_state.stats())

        logger.info("epoch %d done in %.1fs | %s", epoch,
                    time.time() - epoch_start, meters)

        if args.eval_period and epoch % args.eval_period == 0:
            results = evaluator.eval(model)
            r1 = results["t2i"]["R1"]
            if r1 > best["R1"]:
                best = {"R1": r1, "epoch": epoch, **results["t2i"]}
                save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                                 "results": results},
                                op.join(args.output_dir, "best.pth"))
                logger.info("new best R@1 = %.3f at epoch %d", r1, epoch)

        if args.save_last:
            payload = {"epoch": epoch, "model": model.state_dict()}
            if args.save_optimizer:
                payload["optimizer"] = optimizer.state_dict()
                payload["scheduler"] = scheduler.state_dict()
            save_checkpoint(payload, op.join(args.output_dir, "last.pth"))

    logger.info("training finished. best: %s", best)
    return best


def do_inference(model, evaluator: Evaluator):
    return evaluator.eval(model)
