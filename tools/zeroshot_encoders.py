"""Zero-shot text-to-image retrieval on the VN3K test split, per candidate encoder.

Picks the image-text backbone to put under ViLNM for Vietnamese captions: each
model encodes the 2,000 test images and 4,000 Vietnamese test captions with its
own preprocessing and native resolution, no training, and is scored with the
repo's own `utils.metrics.rank` so the numbers are comparable with later ViLNM
runs.

    python tools/zeroshot_encoders.py                    # clip A B C
    python tools/zeroshot_encoders.py --models A --batch_size 32
    python tools/zeroshot_encoders.py --dataset CUHK-PEDES --models clip A

Each model runs in its own subprocess (an OOM or a broken download cannot take
the others down).  Weights are cached under pretrained/hf/ (HF_HOME), never in
~/.cache.  Needs the optional dependency group:  uv sync --group encoders
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS = {
    "clip": dict(kind="openai", ref="pretrained/ViT-B-16.pt",
                 note="current ViLNM backbone (English CLIP), baseline"),
    "A": dict(kind="hf", ref="google/siglip2-base-patch16-256",
              note="SigLIP 2 base/16, multilingual (Gemma tokenizer)"),
    "B": dict(kind="hf", ref="google/siglip-base-patch16-256-multilingual",
              note="SigLIP base/16 multilingual"),
    "C": dict(kind="open_clip", ref="xlm-roberta-base-ViT-B-32", tag="laion5b_s13b_b90k",
              note="OpenCLIP ViT-B/32 + XLM-RoBERTa base"),
    "A384": dict(kind="hf", ref="google/siglip2-base-patch16-384",
                 note="SigLIP 2 base/16 at 384x384"),
    "Anaflex": dict(kind="hf", ref="google/siglip2-base-patch16-naflex",
                    note="SigLIP 2 base/16 NaFlex: keeps the person crop's aspect ratio"),
    "L": dict(kind="hf", ref="google/siglip2-large-patch16-256",
              note="SigLIP 2 large/16 (zero-shot reference; heavy to fine-tune on 6 GB)"),
    "SO": dict(kind="hf", ref="google/siglip2-so400m-patch16-256",
               note="SigLIP 2 so400m/16 (zero-shot reference; too heavy to fine-tune on 6 GB)"),
    "SOnaflex": dict(kind="hf", ref="google/siglip2-so400m-patch16-naflex",
                     note="SigLIP 2 so400m/16 NaFlex (zero-shot reference)"),
}
# Vietnamese-specific dual encoders (DINOv3 ViT-B/16 + Vietnamese-SBERT/PhoBERT).
# Their remote code needs transformers<5: run this script with .venv-viclip/bin/python.
# Variants: image centre-crop (the model's own processor) vs. squash to a square,
# and raw text vs. pyvi word segmentation (PhoBERT is normally fed segmented text).
for _key, _ref in (("VI", "minhnguyent546/ViCLIP-OT"), ("VIS", "minhnguyent546/ViSigLIP-OT")):
    for _crop in (True, False):
        for _seg in (False, True):
            MODELS[_key + ("" if _crop else "_sq") + ("_seg" if _seg else "")] = dict(
                kind="viclip", ref=_ref, crop=_crop, seg=_seg,
                note=f"{_ref.split('/')[1]}, image {'centre-crop' if _crop else 'squash'}, "
                     f"text {'pyvi-segmented' if _seg else 'raw'}")


# --------------------------------------------------------------------------- #
# child: load one model, encode, score
# --------------------------------------------------------------------------- #
def _features(out):
    """transformers 5 may return an output object instead of a tensor."""
    import torch
    if torch.is_tensor(out):
        return out
    return out.pooler_output


def load(spec, device):
    """Returns (encode_images(list[PIL]) -> Tensor, encode_texts(list[str]) -> Tensor,
    count_tokens(list[str]) -> list[int], max_tokens, n_params)."""
    import torch

    if spec["kind"] == "viclip":
        import torchvision.transforms as T
        from transformers import AutoModel
        model = AutoModel.from_pretrained(spec["ref"], trust_remote_code=True).to(device).eval()
        tok = model._tokenizer
        max_tokens = 256                     # PhoBERT: 258 positions incl. <s> </s>
        if spec["crop"]:
            proc = model.processor.image_processor
            to_pixels = lambda ims: proc(images=ims, return_tensors="pt")["pixel_values"]
        else:
            tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                            T.ToTensor(), T.Normalize((0.485, 0.456, 0.406),
                                                      (0.229, 0.224, 0.225))])
            to_pixels = lambda ims: torch.stack([tf(im) for im in ims])
        if spec["seg"]:
            from pyvi import ViTokenizer
            prep = lambda texts: [ViTokenizer.tokenize(t) for t in texts]
        else:
            prep = lambda texts: list(texts)
        amp = torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda")

        def enc_img(images):
            with amp:
                return model.get_image_features(pixel_values=to_pixels(images).to(device),
                                                normalize=False)

        def enc_txt(texts):
            t = tok(prep(texts), padding=True, truncation=True, max_length=max_tokens,
                    return_tensors="pt").to(device)
            with amp:
                return model.get_text_features(input_ids=t, normalize=False)

        def count(texts):
            return [len(ids) for ids in tok(prep(texts))["input_ids"]]
        n_params = sum(p.numel() for p in model.parameters())
        return enc_img, enc_txt, count, max_tokens, n_params

    if spec["kind"] == "hf":
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(spec["ref"], dtype=torch.float16).to(device).eval()
        proc = AutoProcessor.from_pretrained(spec["ref"])
        tok = proc.tokenizer
        max_tokens = 64        # SigLIP text towers are trained on 64-token sequences

        def enc_img(images):
            # NaFlex processors also return pixel_attention_mask / spatial_shapes.
            x = proc(images=images, return_tensors="pt")
            x = {k: v.to(device, torch.float16) if v.is_floating_point() else v.to(device)
                 for k, v in x.items() if k != "input_ids"}
            return _features(model.get_image_features(**x))

        def enc_txt(texts):
            # SigLIP (2) was trained on lowercased text padded to max_length.
            t = tok([s.lower() for s in texts], padding="max_length", truncation=True,
                    max_length=max_tokens, return_tensors="pt")
            return _features(model.get_text_features(input_ids=t["input_ids"].to(device)))

        def count(texts):
            return [len(ids) for ids in tok([s.lower() for s in texts],
                                            add_special_tokens=True)["input_ids"]]
        n_params = sum(p.numel() for p in model.parameters())
        return enc_img, enc_txt, count, max_tokens, n_params

    import open_clip
    if spec["kind"] == "openai":
        from open_clip.openai import load_openai_model
        model = load_openai_model(os.path.join(ROOT, spec["ref"]), device=device)
        preprocess = open_clip.image_transform(model.visual.image_size, is_train=False)
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec["ref"], pretrained=spec["tag"], device=device)
        tokenizer = open_clip.get_tokenizer(spec["ref"])
    # fp32 weights + fp16 autocast: open_clip's LayerNorms reject a fully .half() model.
    model = model.float().eval()
    max_tokens = tokenizer.context_length
    amp = torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda")

    def enc_img(images):
        x = torch.stack([preprocess(im) for im in images]).to(device)
        with amp:
            return model.encode_image(x)

    def enc_txt(texts):
        with amp:
            return model.encode_text(tokenizer(texts).to(device))

    if hasattr(tokenizer, "tokenizer"):              # HF tokenizer wrapper (XLM-R)
        def count(texts):
            return [len(ids) for ids in tokenizer.tokenizer(texts)["input_ids"]]
    else:                                            # CLIP BPE: + <sot> <eot>
        def count(texts):
            return [len(tokenizer.encode(t)) + 2 for t in texts]
    n_params = sum(p.numel() for p in model.parameters())
    return enc_img, enc_txt, count, max_tokens, n_params


def evaluate(name, batch_size, dataset):
    import statistics

    import torch
    import torch.nn.functional as F
    from PIL import Image

    from datasets.build import build_dataset
    from utils.metrics import rank
    from utils.options import get_args

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = MODELS[name]
    test = build_dataset(get_args(["--dataset_name", dataset])).test

    t0 = time.time()
    enc_img, enc_txt, count, max_tokens, n_params = load(spec, device)
    load_s = time.time() - t0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.inference_mode():
        img_feats = []
        paths = test["img_paths"]
        for i in range(0, len(paths), batch_size):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            img_feats.append(F.normalize(enc_img(ims).float(), dim=-1))
        txt_feats = []
        caps = test["captions"]
        for i in range(0, len(caps), batch_size):
            txt_feats.append(F.normalize(enc_txt(caps[i:i + batch_size]).float(), dim=-1))
        sim = torch.cat(txt_feats) @ torch.cat(img_feats).t()
    encode_s = time.time() - t0

    q_pids = torch.tensor(test["caption_pids"], device=sim.device)
    g_pids = torch.tensor(test["image_pids"], device=sim.device)
    cmc, mAP, mINP, _ = rank(sim, q_pids, g_pids, max_rank=10)
    lengths = count(caps)
    return dict(
        model=name, dataset=dataset, ref=spec["ref"], note=spec["note"],
        R1=cmc[0].item(), R5=cmc[4].item(), R10=cmc[9].item(),
        mAP=mAP.item(), mINP=mINP.item(),
        tokens_median=statistics.median(lengths),
        tokens_p95=sorted(lengths)[int(0.95 * len(lengths))],
        max_tokens=max_tokens,
        truncated=sum(n > max_tokens for n in lengths) / len(lengths),
        params_m=n_params / 1e6,
        vram_gb=(torch.cuda.max_memory_allocated() / 2**30) if device == "cuda" else None,
        load_s=load_s, encode_s=encode_s, device=device)


# --------------------------------------------------------------------------- #
# parent: run every model in its own process, print the comparison
# --------------------------------------------------------------------------- #
def main():
    if "--_child" in sys.argv:
        i = sys.argv.index("--_child")
        res = evaluate(sys.argv[i + 1], int(sys.argv[i + 2]), sys.argv[i + 3])
        print("ZS_RESULT " + json.dumps(res), flush=True)
        return

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["clip", "A", "B", "C"],
                    choices=list(MODELS))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dataset", default="VN3K", choices=["VN3K", "CUHK-PEDES"],
                    help="CUHK-PEDES (English) calibrates zero-shot against fine-tuned numbers")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.out = a.out or os.path.join(ROOT, "logs", f"zeroshot_{a.dataset}.json")

    env = dict(os.environ)
    env.setdefault("HF_HOME", os.path.join(ROOT, "pretrained", "hf"))
    env.pop("LD_LIBRARY_PATH", None)         # conda's CUDA libs break the venv's cuDNN
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    results = []
    for name in a.models:
        print(f"[{name}] {MODELS[name]['ref']} ...", flush=True)
        start = time.time()
        proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--_child",
                               name, str(a.batch_size), a.dataset], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace")
        line = next((l for l in proc.stdout.splitlines() if l.startswith("ZS_RESULT ")), None)
        if line is None:
            print(f"    FAILED (exit {proc.returncode}) after {time.time() - start:.0f}s")
            print("\n".join("    " + l for l in proc.stdout.strip().splitlines()[-25:]))
            results.append(dict(model=name, ref=MODELS[name]["ref"], failed=True))
            continue
        r = json.loads(line[len("ZS_RESULT "):])
        results.append(r)
        print(f"    R@1 {r['R1']:.2f}  ({time.time() - start:.0f}s incl. download/load)")

    print(f"\n{'model':<6}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'mAP':>7}{'mINP':>7}"
          f"{'tok med':>9}{'cut@max':>12}{'params':>9}{'VRAM':>7}  ref")
    for r in results:
        if r.get("failed"):
            print(f"{r['model']:<6}{'failed':>7}  {r['ref']}")
            continue
        print(f"{r['model']:<6}{r['R1']:>7.2f}{r['R5']:>7.2f}{r['R10']:>7.2f}{r['mAP']:>7.2f}"
              f"{r['mINP']:>7.2f}{r['tokens_median']:>9.0f}"
              f"{100 * r['truncated']:>7.1f}%@{r['max_tokens']:<3}{r['params_m']:>8.0f}M"
              f"{(r['vram_gb'] or 0):>6.1f}G  {r['ref']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "split": f"{a.dataset} test",
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
