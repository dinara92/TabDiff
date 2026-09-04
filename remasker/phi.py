"""Frozen ReMasker encoder as a second, learned phi -- stacked on top of
drifting_true/decoder_phi.py's decoded (table) representation, per the plan
(velvet-cuddling-finch.md): "decode, then featurize," ReMasker as the
featurizer, not a replacement for the decode step.

Mask ratio is forced to 0 here (no masking at feature-extraction time --
MAE masking is a pretraining-only objective, matching how upstream's own
get_activations runs its encoder with train=False and no masking,
mae_model.py:287-319). Single-scale readout: the final encoder block's
output only, all column tokens flattened into one vector -- no [CLS]
pooling, no multi-stream combination of intermediate layers.
"""

import torch
import torch.nn as nn

from remasker.model import ReMaskerMAE, split_flat_features


class RemaskerPhi(nn.Module):
    def __init__(self, ckpt_path, device="cpu"):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device)

        self.d_numerical = ckpt['d_numerical']
        self.categories = list(ckpt['categories'])
        self.model = ReMaskerMAE(
            self.d_numerical, self.categories, embed_dim=ckpt['embed_dim'],
            encoder_layers=ckpt['encoder_layers'], decoder_layers=ckpt['decoder_layers'],
            n_head=ckpt['n_head'], d_ffn_factor=ckpt['d_ffn_factor'],
        )
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.embed_dim = ckpt['embed_dim']
        self.num_tokens = self.model.num_tokens
        self.out_dim = self.num_tokens * self.embed_dim
        # Per-dimension, not per-token: term 2's whitening (A.6-style
        # normalization, see the plan) operates on this uniform-width
        # embedding output at full per-dimension granularity, unlike
        # decoder_phi's col_widths (which reflects genuinely heterogeneous
        # numeric-vs-categorical column widths in table space).
        self.col_widths = [1] * self.out_dim

        self.to(device)

    def forward(self, decoded_flat):
        """decoded_flat: [..., d_numerical + sum(categories)] (decoder_phi's
        output, real or generated) -> [..., out_dim]."""
        lead_shape = decoded_flat.shape[:-1]
        flat = decoded_flat.reshape(-1, self.d_numerical + sum(self.categories))
        x_num, x_cat_blocks = split_flat_features(flat, self.d_numerical, self.categories)
        encoded = self.model.encode(x_num, x_cat_blocks, mask=None)  # [-1, num_tokens, embed_dim], full visibility
        feat = encoded.reshape(-1, self.out_dim)
        return feat.view(*lead_shape, self.out_dim)
