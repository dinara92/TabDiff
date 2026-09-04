"""Group-wise ring buffer of real rows, ported from the Drifting release.

The reference (``/playpen-shared/dinara/drifting/memory_bank.py``) keys the
buffer by ImageNet class. Here the key is a *group id*: the binned target
column when training grouped, or a single bucket when training unconditionally.
"""

from typing import Optional, Tuple

import numpy as np
import torch


class ArrayMemoryBank:
    """Per-group ring buffers holding real rows for the attraction term."""

    def __init__(self, num_groups: int = 1, max_size: int = 1024, dtype=np.float32):
        self.num_groups = int(num_groups)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.bank: Optional[np.ndarray] = None
        self.row_shape: Optional[Tuple[int, ...]] = None
        self.ptr = np.zeros(self.num_groups, dtype=np.int64)
        self.count = np.zeros(self.num_groups, dtype=np.int64)

    def _init_bank(self, row_shape: Tuple[int, ...]) -> None:
        self.row_shape = tuple(row_shape)
        self.bank = np.zeros((self.num_groups, self.max_size, *self.row_shape), dtype=self.dtype)

    def add(self, rows, groups) -> None:
        """Insert rows ``(N, *row_shape)`` under integer group ids ``(N,)``."""
        rows = np.asarray(rows, dtype=self.dtype)
        groups = np.asarray(groups).astype(np.int64).reshape(-1)
        if self.bank is None:
            self._init_bank(rows.shape[1:])
        for i in range(groups.shape[0]):
            g = int(groups[i])
            idx = self.ptr[g]
            self.bank[g, idx] = rows[i]
            self.ptr[g] = (idx + 1) % self.max_size
            if self.count[g] < self.max_size:
                self.count[g] += 1

    def sample(self, groups, n_samples: int, device=None) -> torch.Tensor:
        """Draw ``n_samples`` stored rows per requested group.

        Returns ``[len(groups), n_samples, *row_shape]``.
        """
        if self.bank is None:
            raise RuntimeError("MemoryBank is empty; call add() before sample().")
        groups = np.asarray(groups).astype(np.int64).reshape(-1)
        idx = np.empty((groups.shape[0], n_samples), dtype=np.int64)
        for i, g in enumerate(groups):
            valid = int(self.count[g])
            if valid <= 0:
                raise RuntimeError(f"group {g} has no stored rows yet")
            idx[i] = np.random.choice(valid, n_samples, replace=(valid < n_samples))
        out = self.bank[groups[:, None], idx]
        return torch.as_tensor(out, device=device)

    @property
    def is_ready(self) -> bool:
        return self.bank is not None and bool((self.count > 0).all())


def bin_target(y: np.ndarray, num_groups: int, is_regression: bool) -> np.ndarray:
    """Map a target column to group ids.

    Classification targets are used directly. Regression targets are cut at
    empirical quantiles, which is what gives the kernel a partition to work
    within without inventing a conditioning signal.
    """
    y = np.asarray(y).reshape(-1)
    if not is_regression:
        return y.astype(np.int64)
    if num_groups <= 1:
        return np.zeros_like(y, dtype=np.int64)
    edges = np.quantile(y, np.linspace(0, 1, num_groups + 1)[1:-1])
    return np.digitize(y, edges).astype(np.int64)
