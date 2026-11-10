"""SigLIP 2 as a ViLNM backbone (`--backbone siglip2`).

ViLNM needs one global vector plus a token sequence from each tower: the global
vectors feed L_JII / L_NDF / L_ID and S_global, the token sequences feed LA and
NTA.  CLIP gives both from one sequence ([CLS]/[EOS] + the rest).  SigLIP 2 has
no [CLS] token, so:

  image  global = the MAP attention-pooling head over the patches
                  (`pooler_output`, the vector SigLIP aligned with text);
         tokens = the post-LayerNorm patch sequence, [B, Nv, D]
  text   global = `head` at the last position (SigLIP pools the last position
                  of a max-length padded caption; `pooler_output`);
         tokens = `head` applied at every position, so word tokens share the
                  global vector's projected space, as CLIP's do.

The 16x16 position grid is resized once at build time to the training
resolution (bicubic, as HF's `interpolate_pos_encoding`), so the resized table
is what gets fine-tuned instead of being re-interpolated every forward pass.
"""

import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hf_tokenizer import hf_from_pretrained

logger = logging.getLogger("ViLNM.model")

# 2 layers, 64 wide: CPU smoke tests only.  The vocabulary must stay 256k so the
# real Gemma tokenizer's ids fit.
TINY_CFG = dict(
    text_config=dict(vocab_size=256000, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=2, num_attention_heads=2,
                     max_position_embeddings=64, projection_size=64),
    vision_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=2, image_size=32, patch_size=16),
)


def resize_position_embedding(embeddings: nn.Module, height: int, width: int):
    """Resize a SigLIP vision embedding's square position grid to height x width pixels."""
    patch = embeddings.patch_size
    if height % patch or width % patch:
        raise ValueError(f"image size {height}x{width} is not a multiple of the "
                         f"patch size {patch}")
    gh, gw = height // patch, width // patch
    old = embeddings.position_embedding.weight.data
    n = int(math.sqrt(old.shape[0]))
    if (gh, gw) != (n, n):
        grid = old.float().reshape(1, n, n, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(gh, gw), mode="bicubic", align_corners=False)
        new = nn.Embedding(gh * gw, old.shape[1])
        new.weight.data.copy_(grid.permute(0, 2, 3, 1).reshape(gh * gw, -1))
        embeddings.position_embedding = new
        logger.info("Resized SigLIP position embedding from %dx%d to %dx%d", n, n, gh, gw)
    embeddings.num_patches = embeddings.num_positions = gh * gw
    embeddings.position_ids = torch.arange(gh * gw).expand((1, -1))


class SiglipBackbone(nn.Module):
    def __init__(self, ref: str, image_size: Tuple[int, int], pretrained: bool = True,
                 tiny: bool = False, freeze_token_embedding: bool = True,
                 grad_ckpt: bool = False):
        super().__init__()
        from transformers import AutoConfig, AutoModel, SiglipConfig

        if tiny:
            model = AutoModel.from_config(SiglipConfig(**TINY_CFG))
        elif pretrained:
            model = hf_from_pretrained(AutoModel, ref, dtype=torch.float32)
        else:
            # Weights come from a ViLNM checkpoint afterwards (test.py).
            model = AutoModel.from_config(hf_from_pretrained(AutoConfig, ref))
        if grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        # logit_scale / logit_bias belong to SigLIP's own loss and are dropped.
        self.visual = model.vision_model
        self.text = model.text_model
        resize_position_embedding(self.visual.embeddings, *image_size)
        self.context_length = model.config.text_config.max_position_embeddings

        self.embed_dim = self.text.head.out_features
        if model.config.vision_config.hidden_size != self.embed_dim:
            raise ValueError("ViLNM's LA/NTA need image patches and projected words "
                             "of the same width; this SigLIP variant differs")
        if freeze_token_embedding:
            # 256k x 768 = 197M parameters: frozen, it saves ~2.4 GB of Adam
            # state and gradients, and the Vietnamese pieces are already trained.
            # Stored in fp16 (another ~0.4 GB); the lookup result is promoted to
            # fp32 when the position embedding is added.
            table = self.text.embeddings.token_embedding
            table.weight.requires_grad_(False)
            table.half()

    def encode_image(self, image):
        """Returns (global [B, D], patch tokens [B, Nv, D])."""
        out = self.visual(pixel_values=image)
        return out.pooler_output, out.last_hidden_state

    def encode_text(self, text):
        """Returns (global [B, D], projected tokens [B, L, D])."""
        if text.size(1) != self.context_length:
            raise ValueError(f"SigLIP pools the last position, so captions must be "
                             f"padded to {self.context_length} tokens, got {text.size(1)}")
        out = self.text(input_ids=text)
        return out.pooler_output, self.text.head(out.last_hidden_state)
