"""ICFG-PEDES [29].

`ICFG-PEDES.json` uses the same record layout as CUHK-PEDES but has only
train/test splits and exactly one caption per image.
"""

from .cuhkpedes import CUHKPEDES


class ICFGPEDES(CUHKPEDES):
    """4,102 identities / 54,522 images / 54,522 captions."""

    dataset_dir = "ICFG-PEDES"
    anno_name = "ICFG-PEDES.json"
    img_subdir = "imgs"
    path_key = "file_path"

    def _split_anno(self, anno_path):
        splits = super()._split_anno(anno_path)
        splits.setdefault("val", [])
        return splits
