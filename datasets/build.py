"""Dataloader construction."""

import logging

import torch
from torch.utils.data import DataLoader

from utils.simple_tokenizer import SimpleTokenizer
from .bases import ImageDataset, ImageTextDataset, TextDataset
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .sampler import EpochShuffleSampler, RandomIdentitySampler
from .synthetic import SyntheticPEDES
from .preprocessing import build_transforms

logger = logging.getLogger("ViLNM.dataset")

__factory = {
    "CUHK-PEDES": CUHKPEDES,
    "ICFG-PEDES": ICFGPEDES,
    "RSTPReid": RSTPReid,
    "SYNTH-PEDES": SyntheticPEDES,
}


def build_dataset(args):
    if args.dataset_name not in __factory:
        raise KeyError(f"unknown dataset '{args.dataset_name}', "
                       f"choose from {sorted(__factory)}")
    return __factory[args.dataset_name](root=args.root_dir)


def build_dataloader(args):
    """Returns (train_loader, gallery_loader, query_loader, num_classes, train_set)."""
    dataset = build_dataset(args)
    num_classes = len(dataset.train_id_container)
    tokenizer = SimpleTokenizer()

    train_transforms = build_transforms(args.img_size, is_train=True, aug=args.img_aug)
    test_transforms = build_transforms(args.img_size, is_train=False)

    train_set = ImageTextDataset(
        dataset.train, train_transforms, text_length=args.text_length,
        tokenizer=tokenizer, use_nta=args.use_nta, nta_rate=args.nta_rate,
        nta_p1=args.nta_p1, nta_p2=args.nta_p2, seed=args.seed)

    # No persistent_workers: each epoch's workers must be re-forked so they see
    # the captions NTAState.resample() rebuilt in the main process.
    common = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    if args.sampler == "identity":
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, drop_last=True,
            sampler=RandomIdentitySampler(dataset.train, args.batch_size,
                                          args.num_instance, seed=args.seed),
            **common)
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, drop_last=True,
            sampler=EpochShuffleSampler(len(train_set), seed=args.seed), **common)

    split = dataset.val if args.val_dataset == "val" and dataset.val else dataset.test
    gallery = ImageDataset(split["image_pids"], split["img_paths"], test_transforms)
    query = TextDataset(split["caption_pids"], split["captions"],
                        text_length=args.text_length, tokenizer=tokenizer)
    gallery_loader = DataLoader(gallery, batch_size=args.test_batch_size, **common)
    query_loader = DataLoader(query, batch_size=args.test_batch_size, **common)

    logger.info("train: %d pairs (%d ids) | gallery: %d images | query: %d captions",
                len(train_set), num_classes, len(gallery), len(query))
    return train_loader, gallery_loader, query_loader, num_classes, train_set


def build_test_dataloader(args):
    """Gallery/query loaders for the test split only."""
    dataset = build_dataset(args)
    tokenizer = SimpleTokenizer()
    test_transforms = build_transforms(args.img_size, is_train=False)
    split = dataset.test
    gallery = ImageDataset(split["image_pids"], split["img_paths"], test_transforms)
    query = TextDataset(split["caption_pids"], split["captions"],
                        text_length=args.text_length, tokenizer=tokenizer)
    common = dict(num_workers=args.num_workers, pin_memory=False)
    return (DataLoader(gallery, batch_size=args.test_batch_size, **common),
            DataLoader(query, batch_size=args.test_batch_size, **common),
            len(dataset.train_id_container))
