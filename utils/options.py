"""Command-line configuration for ViLNM.

Defaults reproduce the paper's setup (Sec. IV-A):

    backbone        CLIP-ViT-B/16, same version as IRRA [9]
    image size      384 x 128, patch/stride 16
    sim_alpha       0.25          weight of S_global at inference
    sim_beta        0.75          weight of S_local at inference
    temperature     0.02          tau in Eqs. (6)-(7)
    nta_rate        0.2           average mask probability "rate" in Eq. (4)
    nta_p1 / p2     0.6 / 0.8     operation thresholds in Eq. (5)
    lambda_intra    0.1           lambda in Eq. (8)

Optimisation follows IRRA: Adam, lr 1e-5, 5 warmup epochs, cosine decay,
60 epochs, batch size 64, lr x5 for the randomly initialised modules.
"""

import argparse
import os.path as op


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def get_args(argv=None):
    p = argparse.ArgumentParser("ViLNM: Visual-Language Noise Modeling for TPR")

    # ---------------- general ----------------
    p.add_argument("--name", default="vilnm", help="run name")
    p.add_argument("--output_dir", default="logs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--training", type=str2bool, default=True)
    p.add_argument("--resume", default="", help="checkpoint to resume from")
    p.add_argument("--amp", type=str2bool, default=False,
                   help="mixed precision (CUDA only)")

    # ---------------- data ----------------
    p.add_argument("--dataset_name", default="CUHK-PEDES",
                   choices=["CUHK-PEDES", "ICFG-PEDES", "RSTPReid", "SYNTH-PEDES"])
    p.add_argument("--root_dir", default="data")
    p.add_argument("--val_dataset", default="test", choices=["test", "val"])
    p.add_argument("--img_size", type=int, nargs=2, default=(384, 128),
                   metavar=("H", "W"))
    p.add_argument("--stride_size", type=int, default=16)
    p.add_argument("--text_length", type=int, default=77)
    p.add_argument("--vocab_size", type=int, default=49409,
                   help="49408 CLIP tokens + 1 <|mask|> token for NTA")
    p.add_argument("--img_aug", type=str2bool, default=True)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--sampler", default="random", choices=["random", "identity"])
    p.add_argument("--num_instance", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--test_batch_size", type=int, default=128)

    # ---------------- model ----------------
    p.add_argument("--pretrain_choice", default="pretrained/ViT-B-16.pt",
                   help="path to the OpenAI CLIP checkpoint")
    p.add_argument("--no_pretrained", type=str2bool, default=False,
                   help="skip loading CLIP weights (random init; smoke tests only)")
    p.add_argument("--tiny_model", type=str2bool, default=False,
                   help="build a 2-layer/128-wide CLIP instead of ViT-B/16 "
                        "(implies --no_pretrained; smoke tests only)")
    p.add_argument("--la_reduction", type=int, default=4,
                   help="bottleneck ratio of the LA module MLP")
    p.add_argument("--la_dropout", type=float, default=0.0)

    # ---------------- ViLNM hyperparameters ----------------
    p.add_argument("--sim_alpha", type=float, default=0.25, help="weight of S_global")
    p.add_argument("--sim_beta", type=float, default=0.75, help="weight of S_local")
    p.add_argument("--temperature", type=float, default=0.02, help="tau")
    p.add_argument("--lambda_intra", type=float, default=0.1, help="lambda in Eq. (8)")
    p.add_argument("--nta_rate", type=float, default=0.2, help="rate in Eq. (4)")
    p.add_argument("--nta_p1", type=float, default=0.6, help="mask/replace split")
    p.add_argument("--nta_p2", type=float, default=0.8, help="replace/delete split")
    p.add_argument("--nta_warmup_epochs", type=int, default=1,
                   help="epochs trained on clean captions before NTA starts eliminating")

    # ---------------- loss switches (for the ablation in Table II) ----------------
    p.add_argument("--use_nta", type=str2bool, default=True)
    p.add_argument("--use_jii", type=str2bool, default=True)
    p.add_argument("--use_ndf", type=str2bool, default=True)
    p.add_argument("--use_id", type=str2bool, default=True)
    p.add_argument("--loss_on_global", type=str2bool, default=True)
    p.add_argument("--loss_on_local", type=str2bool, default=True)

    # ---------------- solver ----------------
    p.add_argument("--optimizer", default="Adam", choices=["Adam", "AdamW", "SGD"])
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_factor", type=float, default=5.0,
                   help="lr multiplier for randomly initialised modules")
    p.add_argument("--bias_lr_factor", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=4e-5)
    p.add_argument("--weight_decay_bias", type=float, default=0.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num_epoch", type=int, default=60)
    p.add_argument("--lrscheduler", default="cosine", choices=["cosine", "step"])
    p.add_argument("--milestones", type=int, nargs="+", default=(20, 50))
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--target_lr", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--warmup_factor", type=float, default=0.1)
    p.add_argument("--warmup_method", default="linear", choices=["linear", "constant"])

    # ---------------- logging ----------------
    p.add_argument("--log_period", type=int, default=100)
    p.add_argument("--eval_period", type=int, default=1)
    p.add_argument("--max_steps_per_epoch", type=int, default=0,
                   help="cap the number of iterations per epoch (0 = no cap); "
                        "used by the smoke tests")
    p.add_argument("--save_last", type=str2bool, default=True,
                   help="write last.pth after every epoch")
    p.add_argument("--save_optimizer", type=str2bool, default=False,
                   help="include optimizer/scheduler state in last.pth so training "
                        "can be resumed exactly; triples the file size")

    args = p.parse_args(argv)
    args.img_size = tuple(args.img_size)
    args.output_dir = op.join(args.output_dir, args.dataset_name, args.name)
    args.eot_token = 49407          # CLIP's <|endoftext|>
    if args.tiny_model:
        args.no_pretrained = True
    return args
