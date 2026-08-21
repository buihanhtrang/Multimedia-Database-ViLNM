"""Generate a tiny synthetic benchmark in the CUHK-PEDES annotation format.

The point is verification, not realism: it produces enough identities, images,
captions and *deliberate noisy correspondences* to drive every code path in
ViLNM (loader, identity sampler, NTA caption cache, JII loss, evaluator) on a
CPU in seconds.

    python tools/make_synthetic_data.py --out data/SYNTH-PEDES

Layout produced:
    data/SYNTH-PEDES/reid_raw.json
    data/SYNTH-PEDES/imgs/id_0000/0.png ...
"""

import argparse
import json
import os
import os.path as op
import random

from PIL import Image, ImageDraw

COLORS = {
    "red": (200, 40, 40), "blue": (40, 60, 200), "green": (40, 160, 60),
    "yellow": (220, 200, 40), "black": (30, 30, 30), "white": (235, 235, 235),
    "grey": (128, 128, 128), "pink": (235, 130, 180),
}
TOPS = ["t-shirt", "shirt", "jacket", "hoodie", "sweater"]
BOTTOMS = ["jeans", "shorts", "trousers", "skirt"]
BAGS = ["a backpack", "a handbag", "a shoulder bag", "no bag"]
HAIR = ["short", "long", "curly", "straight"]


def draw_person(size, top_color, bottom_color, seed):
    """A crude 'person': head + torso + legs, with per-identity jitter."""
    rng = random.Random(seed)
    w, h = size
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)

    cx = w // 2 + rng.randint(-2, 2)
    head_r = max(w // 6, 3)
    head_y = h // 8
    d.ellipse([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r],
              fill=(210, 180, 150))

    torso_top = head_y + head_r
    torso_bot = int(h * 0.55)
    d.rectangle([cx - w // 3, torso_top, cx + w // 3, torso_bot], fill=top_color)

    d.rectangle([cx - w // 4, torso_bot, cx - 1, int(h * 0.95)], fill=bottom_color)
    d.rectangle([cx + 1, torso_bot, cx + w // 4, int(h * 0.95)], fill=bottom_color)

    for _ in range(6):                      # texture so images are not identical
        x, y = rng.randrange(w), rng.randrange(h)
        d.point((x, y), fill=(rng.randrange(256),) * 3)
    return img


def caption_for(top_color, bottom_color, rng, noisy=False):
    top = rng.choice(TOPS)
    bottom = rng.choice(BOTTOMS)
    bag = rng.choice(BAGS)
    hair = rng.choice(HAIR)
    if noisy:
        # Inject a wrong colour word -- exactly the failure mode NTA targets
        # (Fig. 1(b) / Fig. 3: the mislabelled word "yellow").
        wrong = rng.choice([c for c in COLORS if c != top_color])
        return (f"the person has {hair} hair and wears a {wrong} {top} "
                f"with {bottom_color} {bottom} and carries {bag} "
                f"while walking across the street")
    return (f"the person has {hair} hair and wears a {top_color} {top} "
            f"with {bottom_color} {bottom} and carries {bag} "
            f"while standing on the pavement")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=op.join("data", "SYNTH-PEDES"))
    p.add_argument("--train_ids", type=int, default=24)
    p.add_argument("--test_ids", type=int, default=12)
    p.add_argument("--val_ids", type=int, default=6)
    p.add_argument("--images_per_id", type=int, default=4)
    p.add_argument("--captions_per_image", type=int, default=2)
    p.add_argument("--noise_ratio", type=float, default=0.3,
                   help="fraction of captions given a deliberately wrong colour")
    p.add_argument("--img_size", type=int, nargs=2, default=(128, 64),
                   metavar=("H", "W"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    img_dir = op.join(args.out, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    pid = 1
    color_names = list(COLORS)
    splits = [("train", args.train_ids), ("test", args.test_ids), ("val", args.val_ids)]
    n_noisy = 0

    for split, n_ids in splits:
        for _ in range(n_ids):
            top_name = rng.choice(color_names)
            bottom_name = rng.choice(color_names)
            id_dir = op.join(img_dir, f"id_{pid:04d}")
            os.makedirs(id_dir, exist_ok=True)

            for k in range(args.images_per_id):
                img = draw_person((args.img_size[1], args.img_size[0]),
                                  COLORS[top_name], COLORS[bottom_name],
                                  seed=pid * 100 + k)
                rel = op.join(f"id_{pid:04d}", f"{k}.png").replace("\\", "/")
                img.save(op.join(img_dir, rel))

                captions = []
                for _ in range(args.captions_per_image):
                    noisy = split == "train" and rng.random() < args.noise_ratio
                    n_noisy += int(noisy)
                    captions.append(caption_for(top_name, bottom_name, rng, noisy))
                records.append({"file_path": rel, "id": pid,
                                "captions": captions, "split": split})
            pid += 1

    anno_path = op.join(args.out, "reid_raw.json")
    with open(anno_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)

    n_caps = sum(len(r["captions"]) for r in records)
    print(f"wrote {anno_path}")
    print(f"  identities : {pid - 1}")
    print(f"  images     : {len(records)}")
    print(f"  captions   : {n_caps} ({n_noisy} with an injected wrong colour word)")
    print(f"  image size : {args.img_size[0]}x{args.img_size[1]}")


if __name__ == "__main__":
    main()
