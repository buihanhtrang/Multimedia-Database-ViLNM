"""VN3K: Vietnamese text-based person retrieval.

`vn3k.json` uses the CUHK-PEDES record format with Vietnamese captions:
    {"": 0, "id": 1, "file_path": "TEST_DATA/Person1.2/output_2.jpg",
     "split": "train", "captions": ["...", "..."], "processed_tokens": [...]}

2,000 train identities / 4,302 images / 8,604 captions and 1,000 test
identities / 2,000 images; there is no val split.  Noisy training captions
live next to it (see `datasets/noise.py`):

    data/VN3K/noise_sentence/VN3K-VI_0.2.npy                  whole-caption swaps
    data/VN3K/noisy_color/vn3k_attribute_noise_20pct.json     colour-word swaps
"""

from .cuhkpedes import CUHKPEDES


class VN3K(CUHKPEDES):
    dataset_dir = "VN3K"
    anno_name = "vn3k.json"
    img_subdir = "images"
    path_key = "file_path"
