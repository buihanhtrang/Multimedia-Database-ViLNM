"""A tiny synthetic benchmark used to verify the pipeline end to end.

It reuses the CUHK-PEDES annotation format verbatim, so running on it exercises
exactly the same loader, sampler, trainer and evaluator code paths as the real
datasets.  Generate it with `python tools/make_synthetic_data.py`.
"""

from .cuhkpedes import CUHKPEDES


class SyntheticPEDES(CUHKPEDES):
    dataset_dir = "SYNTH-PEDES"
    anno_name = "reid_raw.json"
    img_subdir = "imgs"
    path_key = "file_path"
