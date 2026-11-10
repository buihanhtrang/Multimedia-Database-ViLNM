"""Inject noisy training captions from a precomputed noise file.

Only the training split is touched; evaluation always uses the clean captions.
Two formats are supported, matching the files shipped with VN3K:

`.npy` -- noisy correspondence (RDE-style).  An int array with one entry per
    training caption, in annotation order (image by image, caption by
    caption).  Caption k is replaced by the text of caption `x[k]`, which
    usually belongs to another identity; the image, identity label and image
    id stay those of sample k.

`.json` -- attribute noise.  A list of per-caption records
    {"file_path", "caption_index", "split", "caption", "is_real"}; the text of
    each training caption is replaced by the record's `caption`.  The text is
    used as is, whatever `is_real` says: in the VN3K files many captions marked
    `is_real: true` were also altered, so the flag is reported but not trusted.
"""

import json
import logging
import os.path as op
import re
from collections import defaultdict
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("ViLNM.dataset")


def _norm(text: str) -> str:
    # The attribute-noise files re-join tokens with spaces before punctuation
    # ("áo , quần"); undo that so unchanged captions tokenize like the originals.
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def resolve_noise_file(path: str, root_dir: str, dataset_dir: str) -> str:
    for cand in (path, op.join(root_dir, dataset_dir, path)):
        if op.exists(cand):
            return cand
    raise FileNotFoundError(f"noise file not found: {path} "
                            f"(also tried under {op.join(root_dir, dataset_dir)})")


def apply_caption_noise(train: List[Tuple], path: str, img_dir: str):
    """Returns (noisy_train, stats)."""
    if path.endswith(".npy"):
        index = np.load(path, allow_pickle=False)
        if index.shape != (len(train),):
            raise ValueError(f"{path}: {index.shape[0]} entries for "
                             f"{len(train)} training captions")
        noisy = [(pid, image_id, img, train[int(j)][3])
                 for (pid, image_id, img, _), j in zip(train, index)]
        swapped = int((index != np.arange(len(train))).sum())
        wrong_id = sum(train[int(j)][0] != pid
                       for (pid, *_), j in zip(train, index))
        stats = dict(kind="sentence", swapped=swapped, wrong_identity=wrong_id)

    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        table = {(op.normpath(op.join(img_dir, r["file_path"])), int(r["caption_index"])): r
                 for r in records if r.get("split", "train") == "train"}
        seen = defaultdict(int)
        noisy, changed, flagged, missing = [], 0, 0, 0
        for pid, image_id, img, caption in train:
            k = (op.normpath(img), seen[image_id])
            seen[image_id] += 1
            rec = table.get(k)
            if rec is None:
                missing += 1
                noisy.append((pid, image_id, img, caption))
                continue
            text = _norm(rec["caption"])
            changed += text.lower() != _norm(caption).lower()
            flagged += not rec.get("is_real", True)
            noisy.append((pid, image_id, img, text))
        if missing:
            raise ValueError(f"{path}: no record for {missing} training captions")
        stats = dict(kind="attribute", flagged_noisy=flagged, actually_changed=changed)
    else:
        raise ValueError(f"unsupported noise file '{path}' (expected .npy or .json)")

    stats["total"] = len(train)
    pct = {k: f"{100 * v / len(train):.1f}%" for k, v in stats.items()
           if isinstance(v, int) and k != "total"}
    logger.info("caption noise from %s: %s %s", path, stats, pct)
    return noisy, stats
