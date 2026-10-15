# ViLNM — Visual-Language Noise Modeling for Text-to-Image Person Retrieval

Unofficial PyTorch implementation of

> G. Xu, Y. Feng, Y. Chen, G. Duan, M. Zhou.
> **ViLNM: Visual-Language Noise Modeling for Text-to-Image Person Retrieval.**
> *IEEE Signal Processing Letters*, vol. 32, pp. 1386–1390, 2025.
> DOI [10.1109/LSP.2025.3553424](https://doi.org/10.1109/LSP.2025.3553424)

Built from the letter alone (`../ViLNM_Visual-Language_Noise_Modeling_for_Text-to-Image_Person_Retrieval.pdf`).
The authors' own release is announced at <https://github.com/xgl0626/ViLNM.git>; this
code was **not** derived from it.

---

## What is implemented

| Paper component | Where | Notes |
|---|---|---|
| CLIP-ViT-B/16 backbone, global `[CLS]`/`[EOS]` features (Sec. III-A) | `model/clip_model.py` | positional embedding resized from 14×14 to 24×8 for 384×128 crops |
| **LA** — Local Aggregation, Eq. (1) | `model/layers.py` | bottleneck MLP + linear shortcut, then max pooling |
| **NTA** — Noise Token Aware, Eqs. (2)–(5) | `model/nta.py`, `datasets/nta_state.py` | token-wise similarity → noise level → mask / replace / delete |
| **JII** — Joint Inter & Intra-Modal Contrastive Loss, Eqs. (6)–(8) | `model/objectives.py` | intra-modal negatives restricted to different identities |
| ID loss [26] and NDF loss [19] | `model/objectives.py` | combined per Eq. (9) |
| Inference fusion `α·S_global + β·S_local` | `utils/metrics.py` | α = 0.25, β = 0.75 |
| Rank-K / mAP / mINP [30] | `utils/metrics.py` | reported for the fused, global-only and local-only similarities |
| Augmentations of Sec. IV-A | `datasets/preprocessing.py` | colour jitter, h-flip, pad+crop, erasing, rotation, grayscale |

Default hyperparameters are the paper's: α=0.25, β=0.75, τ=0.02, `rate`=0.2,
p1=0.6, p2=0.8, λ=0.1. Optimisation follows IRRA [9]: Adam, lr 1e-5 (×5 for the
randomly initialised LA branches and ID classifier), 5 warm-up epochs, cosine
decay, 60 epochs, batch size 64.

---

## Install

```bash
pip install -r requirements.txt

# CLIP ViT-B/16 weights (350 MB) -> pretrained/ViT-B-16.pt
python tools/download_clip.py
```

## Quick verification (CPU, no dataset needed)

```bash
python tests/test_modules.py              # 19 unit tests over NTA / LA / losses / metrics
python tools/smoke_test.py                # 28-check end-to-end suite (~2 min on CPU)
python tools/verify_clip_port.py          # CLIP port == OpenAI's released model
```

`tools/smoke_test.py` builds the synthetic benchmark itself and then drives
`train.py` / `test.py` as subprocesses, so the real loader, sampler, NTA
bookkeeping, trainer, evaluator and checkpoint I/O are all exercised. It checks
that the total loss decreases, that NTA's elimination ratio lands near
`--nta_rate` with the mask/replace/delete mix implied by p1/p2, that `test.py`
reloading `best.pth` reproduces the training run's metrics exactly, that
`--resume` works, and that every ablation row of Table II plus `--sampler
identity` runs. It defaults to `--tiny_model` (a 2-layer/128-wide CLIP) to keep
the suite quick; `--full_model` runs the real ViT-B/16 instead.

`tools/make_synthetic_data.py` writes `data/SYNTH-PEDES/` in the exact CUHK-PEDES
annotation format, so the smoke test drives the same code paths used for the real
benchmarks. 30 % of its training captions get a deliberately wrong colour word,
which is the noisy-correspondence pattern of Fig. 1(b) / Fig. 3.

## Train on the real benchmarks

> Training on a small local GPU, in resumable sessions with MLflow tracking:
> see [docs/TRAINING.md](docs/TRAINING.md) (Vietnamese) and `tools/train_session.sh`.

Lay the datasets out under `--root_dir`:

```
data/
├── CUHK-PEDES/   imgs/…                reid_raw.json
├── ICFG-PEDES/   imgs/…                ICFG-PEDES.json
└── RSTPReid/     imgs/…                data_captions.json
```

```bash
python train.py --dataset_name CUHK-PEDES --root_dir data --batch_size 64
python train.py --dataset_name ICFG-PEDES --root_dir data --batch_size 64
python train.py --dataset_name RSTPReid   --root_dir data --batch_size 64
```

Evaluate a checkpoint:

```bash
python test.py --config_file logs/CUHK-PEDES/vilnm/configs.json \
               --checkpoint  logs/CUHK-PEDES/vilnm/best.pth
```

`last.pth` holds weights only; add `--save_optimizer True` if you want to be able
to `--resume` with the optimizer and scheduler state restored exactly.

### Ablation study (Table II)

```bash
python train.py --name baseline    --use_nta False --use_jii False --loss_on_local False
python train.py --name plus_nta    --use_jii False --loss_on_local False
python train.py --name plus_la     --use_jii False
python train.py --name plus_jii    --use_nta False
python train.py --name full                                    # all components
```

`--loss_on_local False` disables the LA branch's contribution to the objective,
which is how the "no fine-grained alignment" rows of Table II are produced.

---

## How NTA is wired into training

Sec. III-B specifies that NTA "requires only one forward pass for each epoch", so
the module is *not* a second encoder pass:

1. During epoch *e* the model returns `r'` (Eq. 4) for every caption it sees.
2. `NTAState.update()` scatters those values onto the **original** token
   positions, using the index map recorded when the epoch's captions were built.
3. At the end of epoch *e*, `NTAState.resample()` applies Eq. (5) once to
   regenerate the captions used by epoch *e+1*.

Captions are always regenerated **from the untouched originals**, never from the
previous epoch's eliminated version. Eliminating on top of an already-eliminated
caption would compound deletions and empty the caption within a few dozen
epochs; re-deriving from the original keeps the single-forward-pass cost while
letting a token that was dropped once come back later. `--nta_warmup_epochs`
(default 1) trains on clean captions first, because Eq. (4) notes that "in the
initial training stage, the values of elements in r may be high".

A healthy run logs an elimination ratio close to `--nta_rate` with the
mask/replace/delete split implied by p1 and p2 — mean-centring in Eq. (4) is what
pins the average elimination probability to `rate`:

```
NTA epoch 2: {'mask': 530, 'replace': 172, 'delete': 178,
              'eliminated_ratio': 0.2019, 'mean_noise': 0.2001}
```

---

## Choices the paper leaves open

The letter is five pages, so several details had to be settled here. All of them
are localised and configurable.

1. **`<|mask|>` token.** CLIP has no `[MASK]`, so one row (id 49408) is appended
   to the token embedding and initialised with the mean of the pretrained rows.
   CLIP's own ids are untouched. Because 49408 > `<|endoftext|>` (49407), the
   usual `argmax` trick for locating `[EOS]` would break, so the model finds the
   EOS position by explicit comparison (`model/build.py:text_word_mask`).
2. **Features fed to NTA.** Eqs. (2)–(3) need text and image tokens in one
   comparable space. The LA-transformed token sequences are used, which live in
   the shared 512-d space that CLIP's `visual.proj` / `text_projection` map into.
3. **Which similarity the losses see.** Eqs. (6)–(7) write `Sim(I,T)` without
   saying whether it is global, local or both. Here every objective is applied to
   the global *and* local views and averaged, since the LA module needs a
   training signal. Toggle with `--loss_on_global` / `--loss_on_local`.
4. **Intra-modal denominator.** Eq. (7) as printed omits the positive term from
   the denominator, which would make the loss unbounded below. The standard
   contrastive form (positive included) is used, so the loss is ≥ 0 and stays
   finite when a batch row has no different-identity negatives.
5. **NDF loss.** Reference [19] is described in one sentence — distribution
   matching that "prevents the model from becoming overly confident by
   incorporating the reverse KL divergence loss". It is implemented as IRRA's SDM
   forward KL(p‖q) plus the reverse KL(q‖p), against the identity-derived label
   distribution q. This is the least-assumption reading; swap
   `objectives.compute_ndf` if you have the original formulation.
6. **LA bottleneck ratio.** Unspecified; `--la_reduction` defaults to 4.

---

## Verified behaviour

Everything below was run on this machine (Windows, **CPU-only** torch 2.11):

- **`tools/verify_clip_port.py` — 2/2.** Built at 224×224, this port reproduces
  OpenAI's released TorchScript model to float noise (`max|diff|` 4.8e-06 image /
  1.4e-06 text, cosine 1.000000 both). At the paper's 384×128 with the positional
  embedding resized 14×14 → 24×8, zero-shot colour prompts still resolve 3/3, so
  the resize preserves semantics. `tools/download_clip.py` confirms the local
  `ViT-B-16.pt` matches OpenAI's published SHA-256.
- **`tests/test_modules.py` — 19/19.** Covers the noise-level ordering of
  Eqs. (2)–(3), the mean-centring of Eq. (4), the operation mix of Eq. (5), the
  index bookkeeping across deletions, the identity masking of Eq. (7), and mAP /
  mINP against hand-computed values.
- **`tools/smoke_test.py` — 28/28**, in ~2 min. Four epochs with every component
  on: total loss 43.09 → 42.00, `test.py` reloading `best.pth` reproduced the
  training metrics to the third decimal (R@1 8.333, mAP 19.522), `--resume`
  restarted at the right epoch, and all eight ablation configurations plus
  `--sampler identity` ran. NTA's measured statistics match its configuration —
  elimination ratio 0.191/0.206/0.210/0.204 against `rate`=0.2, and a
  mask/replace/delete split of 0.605/0.197/0.198 against p1=0.6, p2=0.8.
- **Pretrained CLIP-ViT-B/16 at 384×128** loads with `missing=0, unexpected=0`
  (12 vision + 12 text layers, 150.4M trainable parameters) and trains and
  evaluates at that resolution.

One number from that last run is worth reading carefully. After only 9 gradient
steps the global branch reaches R@1 85.4 on the synthetic set — CLIP already knows
colours zero-shot — while the local branch sits at 12.5, i.e. chance, because the
LA module is randomly initialised and has barely moved. The fused score is
therefore *below* global-only (76.0), since β=0.75 puts most of the weight on the
untrained branch. That is the expected shape of an untrained LA head, and the
evaluator reports all three columns precisely so this is visible rather than
hidden inside the fusion.

Accuracy numbers from these runs are **not** comparable to Table I: they come
from a 42-identity synthetic set of drawn figures, with a handful of iterations.
Reproducing the paper's results needs the real benchmarks and a GPU (no CUDA
device is present here, so that was not attempted).

## Layout

```
train.py  test.py
model/       clip_model.py  layers.py (LA)  nta.py (NTA)  objectives.py  build.py
datasets/    bases.py  nta_state.py  cuhkpedes.py  icfgpedes.py  rstpreid.py
             synthetic.py  sampler.py  preprocessing.py  build.py
solver/      build.py  lr_scheduler.py
processor/   processor.py
utils/       options.py  metrics.py  simple_tokenizer.py  logger.py  meter.py  iotools.py
tools/       download_clip.py  make_synthetic_data.py  smoke_test.py
             verify_clip_port.py
tests/       test_modules.py
```

## Attribution

`utils/simple_tokenizer.py` and `model/clip_model.py` are adapted from
[OpenAI CLIP](https://github.com/openai/CLIP) (MIT). The training recipe,
dataset layout and evaluation protocol follow
[IRRA](https://github.com/anosorae/IRRA), as the paper specifies.
