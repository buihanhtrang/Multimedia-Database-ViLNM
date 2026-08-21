"""CUHK-PEDES [1] and its annotation format.

`reid_raw.json` is a list of records:
    {"file_path": "...", "captions": ["...", "..."], "split": "train", "id": 1}
"""

import json
import logging
import os.path as op
from typing import Dict, List, Tuple

from .bases import BaseDataset


class CUHKPEDES(BaseDataset):
    """11,003 identities / 40,206 images / 80,412 captions."""

    dataset_dir = "CUHK-PEDES"
    anno_name = "reid_raw.json"
    img_subdir = "imgs"
    path_key = "file_path"

    def __init__(self, root: str = "", verbose: bool = True):
        super().__init__()
        self.logger = logging.getLogger("ViLNM.dataset")
        self.root = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.root, self.img_subdir)
        self.anno_path = op.join(self.root, self.anno_name)
        self._check_before_run()

        annos = self._split_anno(self.anno_path)
        self.train, self.train_id_container = self._process_anno(annos["train"], training=True)
        self.test, self.test_id_container = self._process_anno(annos["test"])
        self.val, self.val_id_container = self._process_anno(annos["val"])

        if verbose:
            self.logger.info("=> %s loaded", self.__class__.__name__)
            self.show_dataset_info()

    def _check_before_run(self):
        for path in (self.root, self.img_dir, self.anno_path):
            if not op.exists(path):
                raise RuntimeError(
                    f"'{path}' is not available. Download the dataset and lay it out as\n"
                    f"  {self.root}/{self.img_subdir}/...\n  {self.anno_path}")

    def _split_anno(self, anno_path: str) -> Dict[str, List[dict]]:
        splits = {"train": [], "test": [], "val": []}
        with open(anno_path, "r", encoding="utf-8") as f:
            annos = json.load(f)
        for anno in annos:
            splits.setdefault(anno["split"], []).append(anno)
        return splits

    def _process_anno(self, annos: List[dict], training: bool = False):
        pid_container = set()
        if training:
            # Remap raw identity labels onto a contiguous 0..N-1 range so the ID
            # loss classifier always sees in-range targets (also makes training
            # on a subset of identities safe).
            raw_ids = sorted({int(a["id"]) for a in annos})
            pid2label = {raw: idx for idx, raw in enumerate(raw_ids)}

            dataset = []
            image_id = 0
            for anno in annos:
                pid = pid2label[int(anno["id"])]
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno[self.path_key])
                for caption in anno["captions"]:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            return dataset, pid_container

        image_pids, img_paths, caption_pids, captions = [], [], [], []
        for anno in annos:
            pid = int(anno["id"])
            pid_container.add(pid)
            image_pids.append(pid)
            img_paths.append(op.join(self.img_dir, anno[self.path_key]))
            for caption in anno["captions"]:
                caption_pids.append(pid)
                captions.append(caption)
        return dict(image_pids=image_pids, img_paths=img_paths,
                    caption_pids=caption_pids, captions=captions), pid_container
