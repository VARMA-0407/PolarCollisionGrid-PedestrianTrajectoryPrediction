"""Relation-aware heterogeneous GATv2 spatial encoder (plain PyTorch, sparse COO edges, no PyTorch Geometric).

Graph layout (flattened over batch and history frames, so each frame is an independent graph):
    x          [N, D]   node states; ``node_valid`` [N] bool marks real nodes (padding / absent agents are invalid)
    node_type  [N] long local type index into ``config.node_types``
    edges      {relation: (src [E], dst [E], edge_feat [E, Fe])}  edge j -> i has source ``src`` = j (neighbour) and
               destination ``dst`` = i (target of the message), as in Task 3 (``edge_index[0]`` source, ``[1]`` destination)

One layer, for destination node i of node type t(i), per approved relation r = (s -> d) with d = t(i)
(heads k = 1..K, per-head width c = D / K; ``W`` matrices are relation-specific, ``self`` matrices are type-specific):

    z_ij^r,k  = LeakyReLU( Ws^r,k h_j + Wd^r,k h_i + We^r,k ê_ij )                       (GATv2: nonlinearity BEFORE the attention vector)
    logit_ij^r,k = a^r,k . z_ij^r,k
    alpha_ij^r,k = softmax over { j : (j -> i) valid edge of relation r }  of logit_ij^r,k  (normalised per destination and per relation)
    m_ij^r,k  = alpha_ij^r,k * ( Ws^r,k h_j + Wm^r,k ê_ij )
    agg_i^r   = Wo^r  concat_k  sum_j m_ij^r,k                                           (zero if i has no valid incoming edge of r)
    h_i'      = LayerNorm( h_i + GELU( Wself^{t(i)} h_i + sum_r agg_i^r ) )

``ê_ij`` is the edge input: the scaled Task 3 edge attributes with masked entries set to exactly 0, concatenated with their
validity masks (and, optionally, the six sensitivity flags). The edge attributes therefore enter BOTH the attention logit
(through We) and the message (through Wm). The self/root term Wself h_i keeps nodes with no incoming valid edge fully
representable. Attention is normalised only over the valid incoming neighbours of the same relation; relation messages are
summed (so relation identity is preserved and relations never compete in one softmax). Edges are dropped (and counted as
invalid) if either endpoint is not a valid node. Node outputs for invalid nodes are exactly zero and invalid nodes never send
messages. The layer is permutation-equivariant in node order and invariant to edge order (up to floating-point summation order).
Image-space attributes are used as given; nothing is interpreted as a physical quantity.
"""

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import WindowContractError
from .config import EncoderConfig

EdgeTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def segment_softmax(logit: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of ``logit`` [E, K] over edges sharing the same ``dst`` (per head); empty groups produce no values."""
    if logit.shape[0] == 0:
        return logit
    idx = dst[:, None].expand_as(logit)
    with torch.no_grad():                                           # the shift is only for stability (softmax is shift-invariant)
        mx = torch.zeros(num_nodes, logit.shape[1], dtype=logit.dtype, device=logit.device).scatter_reduce(0, idx, logit.detach(), "amax", include_self=False)
    ex = torch.exp(logit - mx[dst])
    den = torch.zeros(num_nodes, logit.shape[1], dtype=logit.dtype, device=logit.device).index_add(0, dst, ex)
    return ex / den[dst]


class RelationGATv2(nn.Module):
    """Attention and message parameters of ONE relation in ONE layer."""

    def __init__(self, dim: int, heads: int, edge_dim: int, slope: float, attn_dropout: float):
        super().__init__()
        self.heads, self.c, self.slope = heads, dim // heads, slope
        self.w_src = nn.Linear(dim, dim, bias=False)
        self.w_dst = nn.Linear(dim, dim, bias=True)
        self.w_edge_att = nn.Linear(edge_dim, dim, bias=False)
        self.w_edge_msg = nn.Linear(edge_dim, dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.c))
        nn.init.xavier_uniform_(self.att)
        self.out = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        if src.shape[0] == 0:
            return h.new_zeros(n, h.shape[1])
        xs = self.w_src(h).view(n, self.heads, self.c)
        xd = self.w_dst(h).view(n, self.heads, self.c)
        ea = self.w_edge_att(e).view(-1, self.heads, self.c)
        em = self.w_edge_msg(e).view(-1, self.heads, self.c)
        z = F.leaky_relu(xs[src] + xd[dst] + ea, self.slope)
        alpha = segment_softmax((z * self.att).sum(-1), dst, n)               # [E, K]
        alpha = self.attn_dropout(alpha)
        msg = (xs[src] + em) * alpha.unsqueeze(-1)
        agg = torch.zeros(n, self.heads, self.c, dtype=h.dtype, device=h.device).index_add(0, dst, msg)
        return self.out(agg.reshape(n, -1))


class HeteroGATv2Layer(nn.Module):
    def __init__(self, cfg: EncoderConfig, edge_dim: int):
        super().__init__()
        d = cfg.hidden_dim
        self.relations = cfg.relations
        self.rel = nn.ModuleDict({r: RelationGATv2(d, cfg.num_heads, edge_dim, cfg.leaky_relu_slope, cfg.attention_dropout) for r in cfg.relations})
        self.self_proj = nn.ModuleList([nn.Linear(d, d) for _ in cfg.node_types])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor, node_type: torch.Tensor, node_valid: torch.Tensor, edges: Mapping[str, EdgeTriplet]) -> torch.Tensor:
        root = torch.zeros_like(h)
        for t, proj in enumerate(self.self_proj):
            sel = node_type == t
            if sel.any():
                root = root.index_copy(0, sel.nonzero(as_tuple=True)[0], proj(h[sel]))
        agg = torch.zeros_like(h)
        for r in self.relations:
            if r in edges:
                src, dst, e = edges[r]
                agg = agg + self.rel[r](h, src, dst, e)
        out = self.norm(h + self.drop(F.gelu(root + agg)))
        return out * node_valid.unsqueeze(-1).to(out.dtype)


class HeteroGATv2Encoder(nn.Module):
    """Stack of ``num_spatial_layers`` heterogeneous GATv2 layers over per-frame graphs."""

    def __init__(self, cfg: EncoderConfig, node_in_dim: int, edge_in_dim: int):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.ModuleList([nn.Linear(node_in_dim, cfg.hidden_dim) for _ in cfg.node_types])     # type-specific input projection
        self.in_norm = nn.LayerNorm(cfg.hidden_dim)
        self.layers = nn.ModuleList([HeteroGATv2Layer(cfg, edge_in_dim) for _ in range(cfg.num_spatial_layers)])

    def forward(self, x: torch.Tensor, node_type: torch.Tensor, node_valid: torch.Tensor, edges: Mapping[str, EdgeTriplet]) -> torch.Tensor:
        if x.ndim != 2 or node_type.shape != (x.shape[0],) or node_valid.shape != (x.shape[0],):
            raise WindowContractError("SPATIAL_SHAPE", "x must be [N, F]; node_type and node_valid must be [N]")
        unknown = [r for r in edges if r not in self.cfg.relations]
        if unknown:
            raise WindowContractError("UNSUPPORTED_RELATION", "relations %s are not enabled in this encoder (enabled: %s)" % (unknown, list(self.cfg.relations)))
        if node_valid.any() and (node_type[node_valid] < 0).any():
            raise WindowContractError("UNSUPPORTED_NODE_TYPE", "a valid node has a type outside the encoder's node types")
        h = torch.zeros(x.shape[0], self.cfg.hidden_dim, dtype=x.dtype, device=x.device)
        for t, proj in enumerate(self.input_proj):
            sel = (node_type == t) & node_valid
            if sel.any():
                h = h.index_copy(0, sel.nonzero(as_tuple=True)[0], proj(x[sel]))
        h = self.in_norm(h) * node_valid.unsqueeze(-1).to(h.dtype)
        clean: Dict[str, EdgeTriplet] = {}
        for r, (src, dst, e) in edges.items():
            if src.shape != dst.shape or e.shape[0] != src.shape[0] or src.ndim != 1:
                raise WindowContractError("SPATIAL_SHAPE", "edge arrays of relation %s are inconsistent" % r)
            if src.numel() and (min(int(src.min()), int(dst.min())) < 0 or max(int(src.max()), int(dst.max())) >= x.shape[0]):
                raise WindowContractError("EDGE_ENDPOINT", "relation %s has an endpoint outside [0, N)" % r)
            keep = node_valid[src] & node_valid[dst]                      # edges touching an absent/padded node are dropped
            clean[r] = (src[keep], dst[keep], e[keep])
        for layer in self.layers:
            h = layer(h, node_type, node_valid, clean)
        return h


__all__ = ["HeteroGATv2Encoder", "HeteroGATv2Layer", "RelationGATv2", "segment_softmax"]
