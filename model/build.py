"""The ViLNM model (Fig. 2)."""

import logging

import torch
import torch.nn as nn

from .clip_model import build_CLIP_from_openai_pretrained
from .layers import LocalAggregation
from .nta import NoiseTokenAware
from . import objectives

logger = logging.getLogger("ViLNM.model")


class ViLNM(nn.Module):
    """Visual-Language Noise Modeling for text-to-image person retrieval.

    Global features come from CLIP's [CLS]/[EOS] tokens; local features come
    from the LA module (Eq. 1) applied to the patch-/word-level token
    sequences.  The same token sequences feed the NTA module (Eqs. 2-5), whose
    output is the per-token elimination probability used to rebuild the training
    captions for the next epoch.
    """

    def __init__(self, args, num_classes: int):
        super().__init__()
        self.args = args
        self.num_classes = num_classes

        self.base_model, self.clip_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, (args.img_size[0], args.img_size[1]),
            args.stride_size, vocab_size=args.vocab_size,
            pretrained=not args.no_pretrained,
            tiny=getattr(args, "tiny_model", False))
        self.embed_dim = self.clip_cfg["embed_dim"]
        self.eot_token = args.eot_token

        # --- Local Aggregation (Eq. 1) -------------------------------------
        self.la_visual = LocalAggregation(self.embed_dim, args.la_reduction, args.la_dropout)
        self.la_text = LocalAggregation(self.embed_dim, args.la_reduction, args.la_dropout)

        # --- Noise Token Aware (Eqs. 2-4) ---------------------------------
        self.nta = NoiseTokenAware(rate=args.nta_rate)

        # --- ID loss classifier [26] --------------------------------------
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)

    # ------------------------------------------------------------------ #
    # Encoders
    # ------------------------------------------------------------------ #
    def encode_image(self, image):
        """Returns (v_global, v_local, patch_tokens)."""
        x = self.base_model.encode_image(image)          # [B, 1+Nv, D]
        v_global = x[:, 0, :]                            # [CLS] -> Eq. Sec III-A
        v_local, patch_tokens = self.la_visual(x[:, 1:, :])
        return v_global, v_local, patch_tokens

    def text_word_mask(self, caption_ids):
        """True for real word tokens, i.e. everything strictly between SOS and EOS."""
        is_eot = caption_ids == self.eot_token
        # First EOS position per row; captions always carry exactly one.
        eot_pos = is_eot.float().argmax(dim=-1)                       # [B]
        idx = torch.arange(caption_ids.size(1), device=caption_ids.device)
        return (idx.unsqueeze(0) > 0) & (idx.unsqueeze(0) < eot_pos.unsqueeze(1))

    def encode_text(self, caption_ids):
        """Returns (t_global, t_local, word_tokens, word_mask)."""
        x = self.base_model.encode_text(caption_ids)     # [B, L, D]
        is_eot = caption_ids == self.eot_token
        eot_pos = is_eot.float().argmax(dim=-1)
        t_global = x[torch.arange(x.size(0), device=x.device), eot_pos]

        mask = self.text_word_mask(caption_ids)
        t_local, word_tokens = self.la_text(x, mask)
        return t_global, t_local, word_tokens, mask

    # ------------------------------------------------------------------ #
    # Inference helpers (used by utils/metrics.py)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def image_features(self, image):
        v_global, v_local, _ = self.encode_image(image)
        return v_global, v_local

    @torch.no_grad()
    def text_features(self, caption_ids):
        t_global, t_local, _, _ = self.encode_text(caption_ids)
        return t_global, t_local

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        images = batch["images"]
        caption_ids = batch["caption_ids"]
        pids = batch["pids"]

        v_global, v_local, patch_tokens = self.encode_image(images)
        t_global, t_local, word_tokens, word_mask = self.encode_text(caption_ids)

        args = self.args
        ret = {}

        views = []
        if args.loss_on_global:
            views.append((v_global, t_global))
        if args.loss_on_local:
            views.append((v_local, t_local))
        assert views, "at least one of --loss_on_global / --loss_on_local must be set"
        w = 1.0 / len(views)

        # --- L_JII (Eqs. 6-8) --------------------------------------------
        if args.use_jii:
            jii = inter = intra = 0.0
            for vf, tf in views:
                j, i_, a_ = objectives.compute_jii(vf, tf, pids, args.temperature,
                                                   args.lambda_intra)
                jii, inter, intra = jii + w * j, inter + w * i_, intra + w * a_
            ret["jii_loss"] = jii
            ret["interc_loss"] = inter.detach()
            ret["intrac_loss"] = intra.detach()

        # --- L_NDF [19] ---------------------------------------------------
        if args.use_ndf:
            ndf = 0.0
            for vf, tf in views:
                ndf = ndf + w * objectives.compute_ndf(vf, tf, pids, args.temperature)
            ret["ndf_loss"] = ndf

        # --- L_ID [26] ----------------------------------------------------
        if args.use_id:
            ret["id_loss"] = objectives.compute_id(self.classifier(v_global),
                                                   self.classifier(t_global),
                                                   pids)

        # --- NTA noise levels for the next epoch (Eqs. 2-4) ---------------
        if args.use_nta:
            ret["nta_noise"] = self.nta(word_tokens, patch_tokens, word_mask)
            ret["nta_mask"] = word_mask

        return ret


def build_model(args, num_classes: int = 11003) -> ViLNM:
    model = ViLNM(args, num_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Built ViLNM: embed_dim=%d, num_classes=%d, trainable params=%.1fM",
                model.embed_dim, num_classes, n_params / 1e6)
    return model
