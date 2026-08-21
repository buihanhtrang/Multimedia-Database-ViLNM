"""Download the OpenAI CLIP ViT-B/16 checkpoint the paper initialises from.

Sec. III-A: "we initialize our ViLNM using the full CLIP visual and textual
encoders", and Sec. IV-A specifies the same CLIP-ViT-B/16 version used by
IRRA [9].  That is OpenAI's released `ViT-B-16.pt` (~335 MB TorchScript archive).

    python tools/download_clip.py                 # -> pretrained/ViT-B-16.pt
    python tools/download_clip.py --model ViT-B/32

The SHA-256 is embedded in OpenAI's URLs, so the download is verified against it;
an existing file with the right hash is left alone.
"""

import argparse
import hashlib
import os
import sys
import urllib.request

# OpenAI CLIP's model registry (github.com/openai/CLIP, clip/clip.py).
MODELS = {
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target):
    expected = url.split("/")[-2]          # OpenAI puts the digest in the path

    if os.path.exists(target):
        if sha256(target) == expected:
            print(f"{target} already present and verified.")
            return target
        print(f"{target} exists but the checksum differs; re-downloading.")

    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    tmp = target + ".part"
    print(f"downloading {url}\n         -> {target}")

    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                pct = 100 * done / total
                print(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MiB "
                      f"({pct:5.1f}%)", end="", flush=True)
    print()

    actual = sha256(tmp)
    if actual != expected:
        os.remove(tmp)
        raise RuntimeError(f"checksum mismatch\n  expected {expected}\n  got      {actual}")
    os.replace(tmp, target)
    print(f"verified sha256 {actual[:16]}...")
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="ViT-B/16", choices=sorted(MODELS))
    p.add_argument("--output_dir", default="pretrained")
    a = p.parse_args()

    dest = os.path.join(a.output_dir, MODELS[a.model].split("/")[-1])
    try:
        download(MODELS[a.model], dest)
    except Exception as exc:                                   # noqa: BLE001
        sys.exit(f"download failed: {exc}\n"
                 f"Fetch it manually and pass the path via --pretrain_choice.")
    print(f"\nUse it with:  python train.py --pretrain_choice {dest}")
