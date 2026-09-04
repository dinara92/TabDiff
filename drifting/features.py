"""Feature extractors (phi) that define the metric space the drift kernel acts in.

Contract: ``__call__(x_num, x_cat_onehot) -> Dict[str, Tensor]``, each value
shaped ``[N, F, D]``. ``F`` is a token axis that the trainer folds into the
kernel *group* axis, exactly as the reference does for image patch tokens
(``train.py``: ``rearrange(f, 'b x f d -> (b f) x d')``). So ``F == 1`` means
one kernel over the whole row; ``F == n_columns`` means one kernel per column,
each in that column's own embedding space.

Both real and generated rows must be pushed through the same phi, and must
share support -- feed hard (straight-through) one-hots for categoricals, never
soft simplex vectors, or the kernel measures "hard vs soft" instead of
measuring the distribution gap.
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def forward(self, x_num: torch.Tensor, x_cat_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class WhitenedIdentity(FeatureExtractor):
    """Raw space, whitened so every column contributes equally to distance.

    This is the control that the learned extractors have to beat, and it is the
    principled version of the ad-hoc scaling problem in mixed-type data: a
    12-way one-hot block otherwise dominates a standardised numerical column in
    a plain euclidean metric purely because it has more dimensions.

    Equalisation target: for two independent rows, every column contributes an
    expected squared distance of 2.

      * standardised numerical column -> E[(a-b)^2] = 2 * var = 2
      * one-hot column with class probs p -> E||a-b||^2 = 2 * (1 - sum p_i^2)

    so each one-hot block is scaled by ``1 / sqrt(1 - sum p_i^2)``.
    """

    def __init__(self, categories: Sequence[int]):
        super().__init__()
        self.categories = list(categories)
        self.register_buffer("num_mean", torch.zeros(0))
        self.register_buffer("num_std", torch.ones(0))
        self.register_buffer("cat_scale", torch.ones(0))
        self.fitted = False

    @torch.no_grad()
    def fit(self, x_num: torch.Tensor, x_cat_onehot: torch.Tensor, eps: float = 1e-3):
        self.num_mean = x_num.mean(0)
        self.num_std = x_num.std(0).clamp(min=eps)

        scales: List[torch.Tensor] = []
        start = 0
        for k in self.categories:
            block = x_cat_onehot[:, start : start + k]
            p = block.mean(0)
            spread = (1.0 - (p ** 2).sum()).clamp(min=eps)
            # A near-constant column gets a bounded, not exploding, weight.
            scales.append(spread.rsqrt().expand(k))
            start += k
        self.cat_scale = (
            torch.cat(scales) if scales else x_cat_onehot.new_ones(0)
        )
        self.fitted = True
        return self

    def forward(self, x_num, x_cat_onehot):
        if not self.fitted:
            raise RuntimeError("WhitenedIdentity.fit() must be called on train data first")
        parts = []
        if x_num.shape[1] > 0:
            parts.append((x_num - self.num_mean) / self.num_std)
        if x_cat_onehot.shape[1] > 0:
            parts.append(x_cat_onehot * self.cat_scale)
        row = torch.cat(parts, dim=-1)
        return {"raw": row.unsqueeze(1)}  # [N, 1, D] -> one kernel over the row


class ReMaskerFeatures(FeatureExtractor):
    """Frozen tabular MAE used as phi, with one kernel per column.

    This is the direct analogue of the image MAE in the Drifting release: same
    self-supervised masked-reconstruction objective, and its per-column tokens
    play the role of patch tokens. Because each column is only ever compared
    against itself, the mixed-type distance problem never has to be solved.

    ``encoder`` must map ``[N, n_columns, *]`` value embeddings to a stack of
    per-layer activations ``[N, n_columns (+1 CLS), d]`` and be differentiable
    w.r.t. its inputs -- gradients reach the generator through this module even
    though its parameters are frozen.

    ``include_row_token`` keeps the pooled/CLS token as an extra group. Do not
    drop it: per-column kernels alone cannot see cross-column dependence, which
    is exactly what the pairwise-correlation fidelity metrics measure.
    """

    def __init__(self, encoder: nn.Module, layers: Sequence[int], include_row_token: bool = True):
        super().__init__()
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.layers = list(layers)
        self.include_row_token = include_row_token

    def forward(self, x_num, x_cat_onehot):
        acts = self.encoder(x_num, x_cat_onehot, return_hidden=True)
        out = {}
        for i in self.layers:
            h = acts[i]  # [N, n_tokens, d]
            out[f"layer{i}"] = h if self.include_row_token else h[:, 1:]
        return out


class TabDiffEncoderFeatures(FeatureExtractor):
    """Fallback phi: the Tokenizer+Transformer encoder of a trained TabDiff.

    Free, since TabDiff is trained anyway as the multi-step baseline, and it is
    the tabular analogue of using diffusion features as a perceptual space.
    """

    def __init__(self, tokenizer: nn.Module, encoder: nn.Module, drop_cls: bool = False):
        super().__init__()
        self.tokenizer = tokenizer.eval()
        self.encoder = encoder.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.drop_cls = drop_cls

    def forward(self, x_num, x_cat_onehot):
        e = self.tokenizer(x_num, x_cat_onehot)
        h = self.encoder(e[:, 1:, :] if self.drop_cls else e)
        return {"enc": h}
