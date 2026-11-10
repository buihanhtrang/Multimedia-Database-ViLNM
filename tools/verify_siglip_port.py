"""Verify that `--backbone siglip2` is a faithful wrapper of SigLIP 2.

Three checks:

1. **Exact match.** At the native 256x256 (no position resizing), the ViLNM
   global vectors must equal Hugging Face's `get_image_features` /
   `get_text_features` up to float noise.
2. **Same tokens.** Every caption of the chosen split, run through the ViLNM
   dataset code, must give exactly the ids SigLIP's own processor gives
   (lower-cased, padded to 64).
3. **Zero-shot through the real pipeline.** Untrained ViLNM (global similarity
   only, `t2i_global`) on the test split at each `--sizes` H W.  On VN3K
   the standalone screening
   (`tools/zeroshot_encoders.py`, model A) gave R@1 18.68; small differences
   come from bicubic vs. bilinear resizing.

    unset LD_LIBRARY_PATH        # conda hook breaks cuDNN, see docs/TRAINING.md
    python tools/verify_siglip_port.py                 # VN3K
    python tools/verify_siglip_port.py --skip_zeroshot
    python tools/verify_siglip_port.py --sizes 256 256 384 128 320 160
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from datasets.bases import tokenize                                # noqa: E402
from datasets.build import build_dataset, build_test_dataloader    # noqa: E402
from model.build import build_model                                # noqa: E402
from utils.hf_tokenizer import SiglipTokenizer, hf_from_pretrained # noqa: E402
from utils.metrics import Evaluator                                # noqa: E402
from utils.options import get_args                                 # noqa: E402


def exact_match(ref, device):
    from transformers import AutoModel
    hf = hf_from_pretrained(AutoModel, ref, dtype=torch.float32).to(device).eval()
    # Trainable fp32 token table: the exact port; the frozen fp16 table used for
    # training is compared separately below.
    args = get_args(["--backbone", "siglip2", "--hf_model", ref, "--img_size", "256", "256",
                     "--freeze_token_embedding", "False"])
    ours = build_model(args, num_classes=2).to(device).eval()
    args.freeze_token_embedding = True
    frozen = build_model(args, num_classes=2).to(device).eval()
    tok = SiglipTokenizer(ref)

    torch.manual_seed(0)
    px = torch.randn(4, 3, 256, 256, device=device)
    caps = ["người phụ nữ mặc váy đỏ, đi giày cao gót màu đen",
            "a man in a white shirt and blue jeans carrying a black backpack",
            "Cô gái tóc dài, áo khoác xám", "x"]
    ids = torch.stack([tokenize(c, tok, 64) for c in caps]).to(device)
    with torch.no_grad():
        img_ref = hf.get_image_features(pixel_values=px).pooler_output
        txt_ref = hf.get_text_features(input_ids=ids).pooler_output
        img, _ = ours.image_features(px)
        txt, _ = ours.text_features(ids)
        txt16, _ = frozen.text_features(ids)
    di = (img - img_ref).abs().max().item()
    dt = (txt - txt_ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(txt16, txt_ref).min().item()
    ok = di < 1e-3 and dt < 1e-3 and cos > 0.999
    print(f"[1] exact match vs HF     image max|diff| {di:.2e}  text {dt:.2e}  "
          f"fp16 token table: min cos {cos:.6f}  {'OK' if ok else 'FAIL'}")
    return ok


def same_tokens(ref, dataset_name, root_dir):
    from transformers import AutoTokenizer
    hf = hf_from_pretrained(AutoTokenizer, ref)
    tok = SiglipTokenizer(ref)
    ds = build_dataset(argparse.Namespace(dataset_name=dataset_name, root_dir=root_dir))
    caps = ds.test["captions"] + [c for _, _, _, c in ds.train]
    bad = 0
    for c in caps:
        want = hf(c.lower(), padding="max_length", max_length=64,
                  truncation=True)["input_ids"]
        # HF truncation drops <eos> on over-long captions; ViLNM always keeps it.
        if len(tok.encode(c)) >= 64:
            continue
        bad += tokenize(c, tok, 64).tolist() != want
    ok = bad == 0
    print(f"[2] tokens vs processor   {len(caps)} captions, {bad} differ  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def zeroshot(ref, dataset_name, root_dir, device, sizes):
    rows = []
    for size in zip(sizes[::2], sizes[1::2]):
        size = (str(size[0]), str(size[1]))
        args = get_args(["--backbone", "siglip2", "--hf_model", ref,
                         "--dataset_name", dataset_name, "--root_dir", root_dir,
                         "--img_size", *size, "--num_workers", "8",
                         "--test_batch_size", "64"])
        gallery, query, n = build_test_dataloader(args)
        model = build_model(args, n).to(device).eval()
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            res = Evaluator(gallery, query, 1.0, 0.0, device).eval(model, log=False)
        r = res["t2i_global"]
        rows.append(r)
        print(f"[3] zero-shot {'x'.join(size):>7}   R@1 {r['R1']:6.2f}  R@5 {r['R5']:6.2f}  "
              f"R@10 {r['R10']:6.2f}  mAP {r['mAP']:6.2f}")
        del model
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_model", default="google/siglip2-base-patch16-256")
    ap.add_argument("--dataset_name", default="VN3K")
    ap.add_argument("--root_dir", default="data")
    ap.add_argument("--skip_zeroshot", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 256, 384, 128],
                    help="H W pairs to evaluate zero-shot at")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.getLogger("ViLNM").setLevel(logging.WARNING)

    ok = exact_match(a.hf_model, device)
    ok &= same_tokens(a.hf_model, a.dataset_name, a.root_dir)
    if not a.skip_zeroshot:
        zeroshot(a.hf_model, a.dataset_name, a.root_dir, device, a.sizes)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
