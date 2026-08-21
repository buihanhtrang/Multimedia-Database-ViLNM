"""Dataset base classes and the tokenizer glue."""

import logging
import os
from typing import Dict, List, Sequence

import torch
from PIL import Image
from prettytable import PrettyTable
from torch.utils.data import Dataset

from utils.simple_tokenizer import SimpleTokenizer
from .nta_state import NTAState


def read_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise IOError(f"{path} does not exist")
    while True:
        try:
            return Image.open(path).convert("RGB")
        except IOError:
            logging.getLogger("ViLNM.dataset").warning(
                "IOError while reading %s, retrying", path)


def encode_caption(caption: str, tokenizer: SimpleTokenizer,
                   text_length: int = 77) -> List[int]:
    """BPE-encode a caption to word ids, truncated to fit [SOS] ... [EOS]."""
    return tokenizer.encode(caption)[:text_length - 2]


def pad_tokens(word_ids: Sequence[int], sot: int, eot: int,
               text_length: int = 77) -> torch.Tensor:
    """Wrap word ids with [SOS]/[EOS] and zero-pad to `text_length`."""
    tokens = [sot] + list(word_ids)[:text_length - 2] + [eot]
    out = torch.zeros(text_length, dtype=torch.long)
    out[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return out


def tokenize(caption: str, tokenizer: SimpleTokenizer,
             text_length: int = 77) -> torch.Tensor:
    return pad_tokens(encode_caption(caption, tokenizer, text_length),
                      tokenizer.sot_token, tokenizer.eot_token, text_length)


class BaseDataset(object):
    """Holds the train/test/val splits produced by the concrete dataset classes."""

    logger = logging.getLogger("ViLNM.dataset")

    def _split_stats(self, split) -> tuple:
        pids, img_paths, captions = [], [], []
        for pid, img_id, path, caption in split:
            pids.append(pid)
            img_paths.append(path)
            captions.append(caption)
        return len(set(pids)), len(set(img_paths)), len(captions)

    def show_dataset_info(self):
        table = PrettyTable(["subset", "ids", "images", "captions"])
        for name in ("train", "test", "val"):
            split = getattr(self, name, None)
            if split is None:
                continue
            if isinstance(split, dict):
                table.add_row([name, len(set(split["image_pids"])),
                               len(split["img_paths"]), len(split["captions"])])
            else:
                table.add_row([name, *self._split_stats(split)])
        self.logger.info("%s dataset statistics:\n%s", self.__class__.__name__, table)


class ImageTextDataset(Dataset):
    """Training set: one item per (image, caption) pair.

    Yields `index` so the trainer can write NTA noise levels back into
    `self.nta_state`.
    """

    def __init__(self, dataset, transform=None, text_length: int = 77,
                 tokenizer: SimpleTokenizer = None, use_nta: bool = True,
                 nta_rate: float = 0.2, nta_p1: float = 0.6, nta_p2: float = 0.8,
                 seed: int = 1):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.use_nta = use_nta

        self.pids = [d[0] for d in dataset]
        self.image_ids = [d[1] for d in dataset]
        self.img_paths = [d[2] for d in dataset]
        self.word_ids = [encode_caption(d[3], self.tokenizer, text_length)
                         for d in dataset]

        self.nta_state = NTAState(
            self.word_ids, self.tokenizer.mask_token,
            self.tokenizer.n_regular_tokens, nta_rate, nta_p1, nta_p2, seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        words = self.nta_state.tokens(index) if self.use_nta else self.word_ids[index]
        img = read_image(self.img_paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return {
            "index": index,
            "pids": self.pids[index],
            "image_ids": self.image_ids[index],
            "images": img,
            "caption_ids": pad_tokens(words, self.tokenizer.sot_token,
                                      self.tokenizer.eot_token, self.text_length),
        }


class ImageDataset(Dataset):
    """Gallery side of evaluation."""

    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        img = read_image(self.img_paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return self.image_pids[index], img


class TextDataset(Dataset):
    """Query side of evaluation."""

    def __init__(self, caption_pids, captions, text_length: int = 77,
                 tokenizer: SimpleTokenizer = None):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.tokenizer = tokenizer or SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        return (self.caption_pids[index],
                tokenize(self.captions[index], self.tokenizer, self.text_length))
