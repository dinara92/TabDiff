"""One-step mixed-type generator built on TabDiff's backbone.

TabDiff's diffusion process is removed entirely: no timestep conditioning, no
noise schedules, no [MASK] class. What survives is the column-token transformer
(`tabdiff.modules.transformer`), which already carries the per-column inductive
bias this port wants, and the preprocessing, so numbers stay comparable with
the multi-step baseline.

Sampling is 1 NFE: z ~ N(0, I) -> tokens -> transformer -> (x_num, cat logits).
"""

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabdiff.modules.transformer import Reconstructor, Transformer


class DriftGenerator(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        categories: Sequence[int],
        num_layers: int = 4,
        d_token: int = 128,
        n_head: int = 4,
        factor: int = 4,
        d_z: int = 128,
        tau: float = 0.5,
    ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = list(categories)
        self.n_tokens = d_numerical + len(self.categories)
        self.d_z = d_z
        self.tau = tau

        self.z_proj = nn.Linear(d_z, self.n_tokens * d_token)
        # Per-column embedding: without it the tokens are exchangeable and the
        # transformer cannot tell which column it is writing.
        self.column_embed = nn.Parameter(torch.empty(self.n_tokens, d_token))
        nn.init.normal_(self.column_embed, std=1.0 / math.sqrt(d_token))

        self.backbone = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, self.categories, d_token)
        self.d_token = d_token

    def sample_z(self, n: int, device) -> torch.Tensor:
        return torch.randn(n, self.d_z, device=device)

    def forward(self, n: int = None, z: torch.Tensor = None, hard: bool = True):
        """Returns ``(x_num, x_cat_onehot, cat_logits)``.

        ``hard=True`` gives straight-through one-hots: phi then sees the same
        support for real and generated rows, while gradients still reach the
        logits. Soft outputs would let the kernel separate real from generated
        on encoding style alone. Mirrors the gumbel_softmax(hard=True) already
        used in TabDiff's q_xt.
        """
        if z is None:
            assert n is not None
            z = self.sample_z(n, self.column_embed.device)
        h = self.z_proj(z).view(z.shape[0], self.n_tokens, self.d_token)
        h = h + self.column_embed[None]
        h = self.backbone(h)

        x_num, cat_logits = self.detokenizer(h)

        onehots: List[torch.Tensor] = []
        for logits in cat_logits:
            onehots.append(F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1))
        x_cat_onehot = (
            torch.cat(onehots, dim=-1) if onehots else x_num.new_zeros((z.shape[0], 0))
        )
        return x_num, x_cat_onehot, cat_logits

    @torch.no_grad()
    def sample(self, n: int, device, batch_size: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generation for evaluation: returns ``(x_num, x_cat_idx)``."""
        self.eval()
        nums, cats = [], []
        done = 0
        while done < n:
            b = min(batch_size, n - done)
            x_num, _, cat_logits = self.forward(b, hard=True)
            nums.append(x_num.cpu())
            if cat_logits:
                cats.append(torch.stack([l.argmax(-1) for l in cat_logits], dim=-1).cpu())
            done += b
        x_num = torch.cat(nums)[:n]
        x_cat = torch.cat(cats)[:n] if cats else torch.zeros(n, 0, dtype=torch.long)
        return x_num, x_cat
