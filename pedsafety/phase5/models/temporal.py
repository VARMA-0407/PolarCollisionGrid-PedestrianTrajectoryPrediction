"""Causal dilated temporal convolutional encoder over each agent's history sequence.

Input  x        [M, T, C]  per-agent sequences of spatial embeddings (M = batch x agent slots, T = history samples, oldest first)
       present  [M, T]     bool, True where the agent has an observed node at that sample
Output y        [M, T, D]  context-aware per-step embeddings (exactly 0 where ``present`` is False)

Block l (dilation d_l, kernel k, one causal convolution per block):
    u_t   = x_t * present_t                                  (absent samples contribute exactly 0; nothing is copied or filled)
    c_t   = sum_{i=0}^{k-1} W_i u_{t - i d_l} + b            (left-only zero padding of (k-1) d_l samples: no future access)
    y_t   = LayerNorm( x_t + Dropout(GELU(c_t)) ) * present_t
LayerNorm is applied per timestep over channels only (no statistics across time or agents), so causality and
agent-independence are preserved. The presence mask is applied before every convolution and to every block output.
The first block additionally receives the presence bit as an extra input channel so that "absent" is distinguishable
from a genuinely zero embedding.

Receptive field (samples) = 1 + (k - 1) * sum_l d_l. With k = 3 and dilations (1, 2): 1 + 2 * 3 = 7 >= H = 6, so the
output at the last history sample can depend on all six history samples (and the seventh position is left padding).
Output at sample t depends only on inputs at samples <= t (verified by prefix-invariance tests).
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import WindowContractError


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.kernel, self.dilation = kernel, dilation
        self.left_pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, dim, kernel_size=kernel, dilation=dilation, padding=0)       # padding applied manually, left only
        self.skip = nn.Identity() if in_ch == dim else nn.Linear(in_ch, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        m = present.unsqueeze(-1)
        xm = torch.where(m, x, torch.zeros_like(x))                    # where (not multiply): a NaN at an absent step must not survive
        u = xm.transpose(1, 2)                                        # [M, C, T]
        u = F.pad(u, (self.left_pad, 0))                              # left-only padding
        c = self.conv(u).transpose(1, 2)                              # [M, T, D]
        y = self.norm(self.skip(xm) + self.drop(F.gelu(c)))
        return torch.where(m, y, torch.zeros_like(y))


class CausalDilatedTCN(nn.Module):
    def __init__(self, in_dim: int, dim: int, kernel: int, dilations: Sequence[int], dropout: float = 0.0):
        super().__init__()
        self.kernel, self.dilations = kernel, tuple(dilations)
        self.blocks = nn.ModuleList()
        ch = in_dim + 1                                               # + presence channel
        for d in self.dilations:
            self.blocks.append(CausalConvBlock(ch, dim, kernel, d, dropout))
            ch = dim

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or present.shape != x.shape[:2]:
            raise WindowContractError("TEMPORAL_SHAPE", "x must be [M, T, C] and present [M, T]; got %s and %s" % (tuple(x.shape), tuple(present.shape)))
        if present.dtype != torch.bool:
            raise WindowContractError("TEMPORAL_SHAPE", "present must be a boolean mask")
        h = torch.cat([torch.where(present.unsqueeze(-1), x, torch.zeros_like(x)), present.unsqueeze(-1).to(x.dtype)], dim=-1)
        for blk in self.blocks:
            h = blk(h, present)
        return h


__all__ = ["CausalDilatedTCN", "CausalConvBlock"]
