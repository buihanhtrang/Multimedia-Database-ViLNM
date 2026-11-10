"""Hugging Face glue for the SigLIP 2 backbone (`--backbone siglip2`).

Everything Hugging Face downloads lives under `pretrained/hf` (HF_HOME) inside
the repo rather than `~/.cache`, so the project, its venv and its weights can be
moved, containerised or deleted as one directory.  Weights are read from that
cache first and only fetched from the Hub when missing.

`SiglipTokenizer` exposes the attributes the dataset code reads from
`SimpleTokenizer`, with SigLIP's caption layout instead of CLIP's:

    CLIP     <sot> w_1 ... w_n <eot> 0 ... 0           (77 positions)
    SigLIP         w_1 ... w_n <eos> <pad> ... <pad>   (64 positions)

There is no start token (`sot_token is None`), text is lower-cased as in SigLIP 2
training, and captions are always padded to the full length because the text
tower pools the *last* position.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "pretrained", "hf"))
# Fast tokenizers warn (and may deadlock) when used after a DataLoader fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# SigLIP 2 configs carry CLIP's bos/eos ids, which transformers warns about on
# every load; they are unused (ViLNM's tokenizer supplies the real ids).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def hf_from_pretrained(cls, ref: str, **kwargs):
    """`cls.from_pretrained(ref)`, preferring the local cache over the network."""
    try:
        return cls.from_pretrained(ref, local_files_only=True, **kwargs)
    except OSError:
        return cls.from_pretrained(ref, **kwargs)


class SiglipTokenizer(object):
    def __init__(self, ref: str = "google/siglip2-base-patch16-256"):
        from transformers import AutoTokenizer

        self._tok = hf_from_pretrained(AutoTokenizer, ref)
        self.sot_token = None
        self.eot_token = self._tok.eos_token_id          # 1
        self.pad_token = self._tok.pad_token_id          # 0
        self.mask_token = self._tok.mask_token_id        # 4, used by NTA "random mask"
        self.n_regular_tokens = len(self._tok)
        # 256k multilingual pieces: NTA's "random replace" must draw from the
        # words the training captions actually use, not from the whole table.
        self.replace_from_corpus = True
        self._special = set(self._tok.all_special_ids)

    @property
    def vocab_size(self):
        return len(self._tok)

    def encode(self, text: str):
        ids = self._tok(text.lower(), add_special_tokens=False)["input_ids"]
        # A literal "<eos>" in a caption would otherwise end it early.
        return [t for t in ids if t not in self._special]

    def decode(self, tokens):
        return self._tok.decode(tokens)


def build_tokenizer(args):
    if getattr(args, "backbone", "clip") == "siglip2":
        return SiglipTokenizer(args.hf_model)
    from utils.simple_tokenizer import SimpleTokenizer
    return SimpleTokenizer()
