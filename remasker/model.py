"""Tabular masked-autoencoder (ReMasker-style) for use as a learned phi.

"ReMasker: Imputing Tabular Data with Masked Autoencoding" (ICLR 2024) --
mask part of a tabular row's columns, train a transformer to reconstruct
them from the rest. Pretrained here on real Adult rows; the frozen encoder
is later used as a second, learned feature space for drift_loss, stacked on
top of drifting_true/decoder_phi.py's decoded (table) representation --
see the plan in velvet-cuddling-finch.md for why it belongs there rather
than on the raw VAE latent.

Reuses tabsyn.vae.model.Transformer (the same prenormalized multi-head-
attention + FFN stack TabSyn's own VAE encoder/decoder already use) for
both this model's encoder and decoder, rather than reimplementing attention
mechanics -- Transformer.forward operates on [B, n_tokens, d_token] and
returns the same shape, exactly what's needed here.

Hyperparameters (embedding width 64, encoder 8 layers, decoder 6-8 layers,
mask ratio 0.3-0.5) are drifting/PROGRESS_REPORT.md Section 7.3's citation
of the ReMasker paper. n_head, d_ffn_factor, and decoder embedding width are
NOT independently confirmed against the paper's own hyperparameter table
(not available locally) -- chosen as reasonable, clearly-labeled defaults
consistent with this repo's own similarly-scaled transformer (TabSyn's VAE,
d_token=4/n_head=1); flagged here rather than asserted as verified.
"""

import torch
import torch.nn as nn

# Vendored from the TabSyn port: TabDiff's Transformer/MultiheadAttention are
# byte-identical to tabsyn/vae/model.py's, so checkpoints trained there load
# here unchanged (verified by comparing both import paths on the adult ckpt).
from tabdiff.modules.transformer import Transformer


class ColumnTokenizer(nn.Module):
    """Per-column linear projection into embed_dim, numeric and categorical alike.

    Numeric columns (width 1) get a shared weight/bias parameter per column,
    mirroring tabsyn.vae.model.Tokenizer's own numeric handling. Categorical
    columns get one nn.Linear(categories[i], embed_dim) each, applied to
    that column's full probability vector -- real-row pretraining feeds it
    a one-hot vector, drift-time feeds it decoder_phi's softmax output;
    one-hot is just the degenerate case of the same linear map, so no
    separate code path is needed for "hard" vs. "soft" categoricals.
    """

    def __init__(self, d_numerical, categories, embed_dim):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = list(categories)

        self.num_weight = nn.Parameter(torch.empty(d_numerical, embed_dim))
        self.num_bias = nn.Parameter(torch.zeros(d_numerical, embed_dim))
        nn.init.kaiming_uniform_(self.num_weight, a=5 ** 0.5)

        self.cat_proj = nn.ModuleList([nn.Linear(c, embed_dim) for c in self.categories])

    def forward(self, x_num, x_cat_blocks):
        """
        Args:
            x_num: [B, d_numerical]
            x_cat_blocks: list of [B, categories[i]] tensors (one-hot or softmax)
        Returns:
            tokens: [B, d_numerical + len(categories), embed_dim]
        """
        num_tokens = self.num_weight[None] * x_num[:, :, None] + self.num_bias[None]  # [B, d_numerical, E]
        cat_tokens = [proj(block) for proj, block in zip(self.cat_proj, x_cat_blocks)]  # each [B, E]
        cat_tokens = torch.stack(cat_tokens, dim=1)  # [B, len(categories), E]
        return torch.cat([num_tokens, cat_tokens], dim=1)


class ReconstructionHeads(nn.Module):
    """Per-column heads reading the decoder's output back to column space.

    Numeric: linear regression head (1 output). Categorical: linear
    classification head (categories[i] logits). Mirrors
    tabsyn.vae.model.Reconstructor's per-column structure.
    """

    def __init__(self, d_numerical, categories, embed_dim):
        super().__init__()
        self.d_numerical = d_numerical
        self.num_head = nn.Linear(embed_dim, 1)
        self.cat_heads = nn.ModuleList([nn.Linear(embed_dim, c) for c in categories])

    def forward(self, decoded_tokens):
        """decoded_tokens: [B, num_tokens, embed_dim] -> (x_num_pred [B, d_numerical], list of [B, categories[i]] logits)"""
        num_out = self.num_head(decoded_tokens[:, :self.d_numerical]).squeeze(-1)  # [B, d_numerical]
        cat_out = [head(decoded_tokens[:, self.d_numerical + i]) for i, head in enumerate(self.cat_heads)]
        return num_out, cat_out


def split_flat_features(flat, d_numerical, categories):
    """Inverse of concatenation: [*, d_numerical + sum(categories)] ->
    (x_num [*, d_numerical], x_cat_blocks list of [*, categories[i]]).
    Matches drifting_true/decoder_phi.py's DecoderPhi.forward concatenation
    order (numeric columns first, then categoricals) -- needed because
    RemaskerPhi's tokenizer wants per-column blocks, but decoder_phi hands
    off one flat vector (drifting_true/main.py's decoded_gen)."""
    x_num = flat[..., :d_numerical]
    blocks = []
    offset = d_numerical
    for c in categories:
        blocks.append(flat[..., offset:offset + c])
        offset += c
    return x_num, blocks


def sample_mask(batch_size, num_tokens, mask_ratio_min, mask_ratio_max, device):
    """One mask ratio per row, sampled uniformly in [min, max]; per-token
    Bernoulli draws against that ratio. Returns bool mask [B, num_tokens],
    True = masked. Mirrors upstream's make_patch_mask (models/mae_model.py)."""
    ratio = torch.empty(batch_size, 1, device=device).uniform_(mask_ratio_min, mask_ratio_max)
    noise = torch.rand(batch_size, num_tokens, device=device)
    return noise < ratio


class ReMaskerMAE(nn.Module):
    def __init__(
        self, d_numerical, categories, embed_dim=64,
        encoder_layers=8, decoder_layers=6, n_head=4, d_ffn_factor=4,
    ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = list(categories)
        self.num_tokens = d_numerical + len(categories)
        self.embed_dim = embed_dim

        self.tokenizer = ColumnTokenizer(d_numerical, categories, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = Transformer(encoder_layers, embed_dim, n_head, embed_dim, d_ffn_factor)
        self.decoder = Transformer(decoder_layers, embed_dim, n_head, embed_dim, d_ffn_factor)
        self.heads = ReconstructionHeads(d_numerical, categories, embed_dim)

    def encode(self, x_num, x_cat_blocks, mask=None):
        """mask: optional bool [B, num_tokens], True = replace with mask_token before the encoder."""
        tokens = self.tokenizer(x_num, x_cat_blocks)
        if mask is not None:
            tokens = torch.where(mask[:, :, None], self.mask_token, tokens)
        tokens = tokens + self.pos_embed
        return self.encoder(tokens)

    def forward(self, x_num, x_cat_blocks, mask_ratio_min=0.3, mask_ratio_max=0.5):
        """Pretraining forward: mask, encode, decode, reconstruct. Returns
        (x_num_pred, x_cat_pred_logits, mask) -- mask needed by the caller to
        compute the masked-positions-only reconstruction loss."""
        B, device = x_num.shape[0], x_num.device
        mask = sample_mask(B, self.num_tokens, mask_ratio_min, mask_ratio_max, device)
        encoded = self.encode(x_num, x_cat_blocks, mask)
        decoded = self.decoder(encoded + self.pos_embed)
        x_num_pred, x_cat_pred = self.heads(decoded)
        return x_num_pred, x_cat_pred, mask
