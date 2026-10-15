import json
import os
import os.path as op
import random

import numpy as np
import torch


def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_if_missing(directory: str):
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_train_configs(path: str, args):
    mkdir_if_missing(path)
    with open(op.join(path, "configs.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, default=str)


def load_train_configs(path: str):
    import argparse
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    args.training = False
    return args


def save_checkpoint(state: dict, path: str):
    """Write atomically: a crash mid-save leaves the previous file intact."""
    mkdir_if_missing(op.dirname(path))
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def get_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def load_checkpoint(path: str, map_location="cpu"):
    if not op.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
