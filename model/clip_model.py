"""CLIP backbone for ViLNM.

Adapted from the official OpenAI CLIP implementation, with the two changes the
paper needs (Sec. III-A, "we initialize our ViLNM using the full CLIP visual
and textual encoders"):

1.  Non-square input resolution.  Person crops are 384x128, so the ViT
    positional embedding is bilinearly resized from the pretrained 14x14 grid
    to the 24x8 grid produced by a 16x16 patch stride.
2.  Both encoders return the *full token sequence* (already mapped into the
    shared 512-d embedding space by `visual.proj` / `text_projection`), because
    ViLNM needs the patch-level features {v_n} and the word-level features
    {t_n} for the LA and NTA modules -- not just the pooled [CLS]/[EOS]
    vectors.
3.  The token embedding matrix is padded by one row for the `<|mask|>` token
    used by NTA's random-mask operation.
"""

import logging
import math
from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger("ViLNM.model")


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], patch_size: int,
                 stride_size: int, width: int, layers: int, heads: int,
                 output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        # (H, W) grid of patches -- 24 x 8 for a 384x128 crop with stride 16.
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        num_patches = self.num_x * self.num_y

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size,
                               stride=stride_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)                                    # [B, width, gh, gw]
        x = x.reshape(x.shape[0], x.shape[1], -1)            # [B, width, gh*gw]
        x = x.permute(0, 2, 1)                               # [B, gh*gw, width]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                       # [B, 1+N, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj                                # [B, 1+N, output_dim]
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: Union[int, Tuple[int, int]],
                 vision_layers: int,
                 vision_width: int,
                 vision_patch_size: int,
                 stride_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            stride_size=stride_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """Causal mask, as in CLIP's text encoder."""
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """Return the full visual token sequence, [B, 1 + Nv, embed_dim]."""
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """Return the full textual token sequence, [B, context_length, embed_dim]."""
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        return x

    def forward(self, image, text):
        return self.encode_image(image), self.encode_text(text)


def resize_pos_embed(posemb, height, width):
    """Bilinearly resize a square ViT positional embedding to a height x width grid."""
    posemb = posemb.unsqueeze(0)
    cls_tok, grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(math.sqrt(len(grid)))
    logger.info("Resized visual position embedding from %dx%d to %dx%d",
                gs_old, gs_old, height, width)
    grid = grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(height, width), mode="bilinear", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    return torch.cat([cls_tok, grid], dim=1).squeeze(0)


def state_dict_from_checkpoint(model_path: str):
    """Load OpenAI's TorchScript archive (or a plain state dict) into a dict."""
    try:
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        return jit_model.state_dict()
    except RuntimeError:
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
        return obj.get("state_dict", obj) if isinstance(obj, dict) else obj.state_dict()


TINY_CFG = dict(vision_width=128, vision_layers=2, vision_patch_size=16, grid_size=14,
                embed_dim=64, context_length=77, transformer_width=128,
                transformer_heads=2, transformer_layers=2)


def build_CLIP_from_openai_pretrained(model_path: str,
                                      image_size: Tuple[int, int],
                                      stride_size: int,
                                      vocab_size: int = 49409,
                                      pretrained: bool = True,
                                      tiny: bool = False):
    """Instantiate a CLIP model shaped like the checkpoint and load the weights.

    Returns (model, cfg_dict).  When `pretrained` is False (or the checkpoint is
    missing) a randomly initialised CLIP is returned instead; `tiny=True`
    additionally shrinks the geometry to a few million parameters, which is what
    the CPU smoke tests use.
    """
    state_dict = None
    if pretrained:
        try:
            state_dict = state_dict_from_checkpoint(model_path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("Could not read CLIP checkpoint %s (%s); "
                           "falling back to random initialisation.", model_path, exc)

    if state_dict is not None:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        # visual.transformer.resblocks.<i>.attn.in_proj_weight -> segment 3
        vision_layers = len({k.split(".")[3] for k in state_dict
                             if k.startswith("visual.transformer.resblocks")})
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        # transformer.resblocks.<i>.attn.in_proj_weight -> segment 2
        transformer_layers = len({k.split(".")[2] for k in state_dict
                                  if k.startswith("transformer.resblocks")})
        ckpt_vocab = state_dict["token_embedding.weight"].shape[0]
    else:
        cfg_src = TINY_CFG if tiny else dict(
            vision_width=768, vision_layers=12, vision_patch_size=16, grid_size=14,
            embed_dim=512, context_length=77, transformer_width=512,
            transformer_heads=8, transformer_layers=12)
        vision_width = cfg_src["vision_width"]
        vision_layers = cfg_src["vision_layers"]
        vision_patch_size = cfg_src["vision_patch_size"]
        embed_dim = cfg_src["embed_dim"]
        context_length = cfg_src["context_length"]
        transformer_width = cfg_src["transformer_width"]
        transformer_heads = cfg_src["transformer_heads"]
        transformer_layers = cfg_src["transformer_layers"]
        ckpt_vocab = 49408

    model = CLIP(embed_dim, image_size, vision_layers, vision_width,
                 vision_patch_size, stride_size, context_length,
                 max(vocab_size, ckpt_vocab), transformer_width,
                 transformer_heads, transformer_layers)

    cfg = dict(embed_dim=embed_dim, image_size=image_size, stride_size=stride_size,
               vision_width=vision_width, vision_layers=vision_layers,
               vision_patch_size=vision_patch_size, context_length=context_length,
               vocab_size=max(vocab_size, ckpt_vocab),
               transformer_width=transformer_width, transformer_heads=transformer_heads,
               transformer_layers=transformer_layers, pretrained=state_dict is not None)

    if state_dict is None:
        return model, cfg

    state_dict = dict(state_dict)
    for key in ("input_resolution", "context_length", "vocab_size"):
        state_dict.pop(key, None)

    # 1. Resize the visual positional embedding to the person-crop aspect ratio.
    if state_dict["visual.positional_embedding"].shape != model.visual.positional_embedding.shape:
        state_dict["visual.positional_embedding"] = resize_pos_embed(
            state_dict["visual.positional_embedding"], model.visual.num_y, model.visual.num_x)

    # 2. Pad the token embedding for the extra <|mask|> token used by NTA.
    tok = state_dict["token_embedding.weight"]
    if tok.shape[0] < model.vocab_size:
        pad = tok.mean(dim=0, keepdim=True).repeat(model.vocab_size - tok.shape[0], 1)
        state_dict["token_embedding.weight"] = torch.cat([tok, pad], dim=0)
        logger.info("Padded token embedding %d -> %d for the <|mask|> token.",
                    tok.shape[0], model.vocab_size)

    state_dict = {k: v.float() for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded CLIP weights from %s (missing=%d, unexpected=%d)",
                model_path, len(missing), len(unexpected))
    # A non-empty diff means the model geometry was inferred wrongly and part of
    # the checkpoint would be silently discarded, so refuse rather than train a
    # half-initialised backbone.
    if missing or unexpected:
        raise RuntimeError(
            f"CLIP checkpoint does not match the constructed model.\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}")
    logger.info("CLIP-ViT-B/%d: %d vision layers, %d text layers, embed_dim %d",
                vision_patch_size, vision_layers, transformer_layers, embed_dim)
    return model, cfg
