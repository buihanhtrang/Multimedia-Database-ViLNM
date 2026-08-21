"""RSTPReid [11].

`data_captions.json` records look like
    {"img_path": "...", "id": 1, "captions": ["...", "..."], "split": "train"}
so only the image-path key differs from CUHK-PEDES.
"""

from .cuhkpedes import CUHKPEDES


class RSTPReid(CUHKPEDES):
    """4,101 identities / 20,505 images / 41,010 captions."""

    dataset_dir = "RSTPReid"
    anno_name = "data_captions.json"
    img_subdir = "imgs"
    path_key = "img_path"
