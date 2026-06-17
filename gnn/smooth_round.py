# https://arxiv.org/abs/2504.19026

import torch
from torch import Tensor
from torch.nn import Module

class SmoothRound(Module):
    __constants__ = ["k", "_k_half", "base_range"]

    k: int

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self._k_half = self.k * .5
        self.base_range = torch.arange(-2, 3)

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        floored = input.floor()
        frac = input - floored

        k_frac_plus_n = (frac[:, None] + self.base_range) * self.k
        floored_minus_n = floored[:, None] - self.base_range
        zwerg = torch.sigmoid(k_frac_plus_n + self._k_half) - torch.sigmoid(k_frac_plus_n - self._k_half)
        return (floored_minus_n * zwerg).sum(dim=-1)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        k_str = f"k={self.k}"
        return k_str
