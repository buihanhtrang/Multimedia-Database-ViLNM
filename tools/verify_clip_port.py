"""Verify that `model/clip_model.py` is a faithful port of OpenAI CLIP.

Two checks:

1. **Exact match.** Built at 224x224 (so no positional-embedding resizing is
   involved), our encoders must reproduce OpenAI's released TorchScript model
   bit-for-bit up to float noise.  The archive is traced on CUDA, so its graph is
   patched to CPU/fp32 the same way `clip.load()` does.
2. **Zero-shot sanity.** With the weights loaded, CLIP must still pick the right
   caption for a solid-colour image -- this catches a checkpoint that loads with
   matching shapes but scrambled semantics.

    python tools/verify_clip_port.py
"""

import logging
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from datasets.preprocessing import build_transforms                # noqa: E402
from model.clip_model import build_CLIP_from_openai_pretrained     # noqa: E402
from utils.simple_tokenizer import SimpleTokenizer                 # noqa: E402

CKPT = os.path.join("pretrained", "ViT-B-16.pt")


def _const_value(node):
    """Read a prim::Constant's `value` attribute.

    torch >= 2.x dropped the `node["value"]` subscript that OpenAI's loader used,
    so the attribute kind has to be queried and the matching typed accessor
    called ('s' for the Device string, 'i' for the dtype int).
    """
    if "value" not in node.attributeNames():
        return None
    kind = node.kindOf("value")
    getter = getattr(node, kind, None)
    if getter is None:
        return None
    try:
        return getter("value")
    except Exception:                                        # noqa: BLE001
        return None


def _graphs(module):
    try:
        graphs = [module.graph] if hasattr(module, "graph") else []
    except RuntimeError:
        graphs = []
    if hasattr(module, "forward1"):
        graphs.append(module.forward1.graph)
    return graphs


def load_reference(path):
    """Load OpenAI's TorchScript archive and rewrite it for CPU/fp32."""
    model = torch.jit.load(path, map_location="cpu").eval()

    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device("cpu")), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant")
                   if "Device" in repr(n)][-1]

    def patch_device(module):
        for graph in _graphs(module):
            for node in graph.findAllNodes("prim::Constant"):
                value = _const_value(node)
                if isinstance(value, str) and value.startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # fp16 -> fp32
    float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
    float_node = list(float_holder.graph.findNode("aten::to").inputs())[1].node()

    def patch_float(module):
        for graph in _graphs(module):
            for node in graph.findAllNodes("aten::to"):
                inputs = list(node.inputs())
                for i in (1, 2):
                    if _const_value(inputs[i].node()) == 5:      # torch.half
                        inputs[i].node().copyAttributes(float_node)

    model.apply(patch_float)
    patch_float(model.encode_image)
    patch_float(model.encode_text)
    model.float()
    return model


def encode(text, tok, length=77):
    ids = [tok.sot_token] + tok.encode(text)[:length - 2] + [tok.eot_token]
    out = torch.zeros(length, dtype=torch.long)
    out[:len(ids)] = torch.tensor(ids)
    return out


def check_exact_match():
    print("\n[1] exact match against OpenAI's released model (224x224)")
    ours, _ = build_CLIP_from_openai_pretrained(CKPT, (224, 224), 16)
    ours.eval()
    ref = load_reference(CKPT)
    tok = SimpleTokenizer()

    torch.manual_seed(0)
    img = torch.randn(2, 3, 224, 224)
    txt = torch.stack([encode("a photo of a red car", tok),
                       encode("a photo of a blue dog", tok)])

    with torch.no_grad():
        ours_v = ours.encode_image(img)[:, 0, :]
        ref_v = ref.encode_image(img)
        eot = (txt == tok.eot_token).float().argmax(dim=-1)
        ours_t = ours.encode_text(txt)[torch.arange(txt.size(0)), eot]
        ref_t = ref.encode_text(txt)

    dv = (ours_v - ref_v).abs().max().item()
    dt = (ours_t - ref_t).abs().max().item()
    cv = torch.nn.functional.cosine_similarity(ours_v, ref_v).mean().item()
    ct = torch.nn.functional.cosine_similarity(ours_t, ref_t).mean().item()
    print(f"    image  max|diff| = {dv:.3e}   cosine = {cv:.6f}")
    print(f"    text   max|diff| = {dt:.3e}   cosine = {ct:.6f}")
    ok = cv > 0.9999 and ct > 0.9999
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def check_zero_shot():
    print("\n[2] zero-shot sanity at the paper's 384x128 resolution")
    ours, _ = build_CLIP_from_openai_pretrained(CKPT, (384, 128), 16)
    ours.eval()
    tok = SimpleTokenizer()
    transform = build_transforms((384, 128), is_train=False)

    colors = {"red": (220, 30, 30), "blue": (30, 40, 220), "green": (30, 170, 60)}
    prompts = [f"a {name} photo" for name in colors]
    txt = torch.stack([encode(p, tok) for p in prompts])
    images = torch.stack([transform(Image.new("RGB", (128, 384), rgb))
                          for rgb in colors.values()])

    with torch.no_grad():
        v = ours.encode_image(images)[:, 0, :]
        eot = (txt == tok.eot_token).float().argmax(dim=-1)
        t = ours.encode_text(txt)[torch.arange(txt.size(0)), eot]
        sim = (torch.nn.functional.normalize(v, dim=-1) @
               torch.nn.functional.normalize(t, dim=-1).t())

    print("            " + "".join(f"{p:>16}" for p in prompts))
    correct = 0
    for i, name in enumerate(colors):
        row = "".join(f"{sim[i, j].item():>16.4f}" for j in range(len(prompts)))
        hit = sim[i].argmax().item() == i
        correct += hit
        print(f"    {name:<8}{row}   {'ok' if hit else 'MISMATCH'}")
    ok = correct == len(colors)
    print(f"    {correct}/{len(colors)} images matched their colour prompt -- "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not os.path.exists(CKPT):
        sys.exit(f"missing {CKPT}; run tools/download_clip.py first")
    results = [check_exact_match(), check_zero_shot()]
    print("\n" + ("all CLIP checks passed" if all(results) else "SOME CLIP CHECKS FAILED"))
    sys.exit(0 if all(results) else 1)
