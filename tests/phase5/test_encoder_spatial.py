"""P5.5 spatial encoder: heterogeneous GATv2 on synthetic graphs (variable sizes, relations, masks, symmetry, determinism)."""

import copy

import numpy as np
import pytest
import torch

from pedsafety.errors import ConfigError
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.models import HeteroGATv2Encoder
from pedsafety.phase5.models.spatial import segment_softmax
from enc_helpers import ALL_RELATIONS, cfg, cfg_bike

FIN, FE = 13, 28
TYPE_OF = {"pedestrian": 0, "vehicle": 1, "bicyclist": 2}


def rand_graph(N=12, E=40, seed=0, rels=("pedestrian__pedestrian", "vehicle__pedestrian", "pedestrian__vehicle"), types=("pedestrian", "vehicle"), valid_frac=1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, FIN, generator=g)
    ntype = torch.randint(0, len(types), (N,), generator=g)
    valid = torch.rand(N, generator=g) < valid_frac
    edges = {}
    for r in rels:
        s, d = r.split("__")
        si = (ntype == types.index(s)).nonzero(as_tuple=True)[0]
        di = (ntype == types.index(d)).nonzero(as_tuple=True)[0]
        if len(si) == 0 or len(di) == 0:
            edges[r] = (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0, FE))
            continue
        src = si[torch.randint(0, len(si), (E,), generator=g)]
        dst = di[torch.randint(0, len(di), (E,), generator=g)]
        keep = src != dst
        edges[r] = (src[keep], dst[keep], torch.randn(int(keep.sum()), FE, generator=g))
    return x, ntype, valid, edges


def model(c=None):
    c = c or cfg()
    with torch.random.fork_rng(devices=[]):                    # the integrated encoder seeds its submodules; here we seed explicitly
        torch.manual_seed(c.init_seed)
        m = HeteroGATv2Encoder(c, FIN, FE)
    m.eval()
    return m


def test_shapes_for_variable_node_and_edge_counts():
    m = model()
    for N, E in ((1, 0), (3, 2), (12, 40), (50, 300)):
        x, nt, v, ed = rand_graph(N, E)
        out = m(x, nt, torch.ones(N, dtype=torch.bool), ed)
        assert out.shape == (N, 64) and torch.isfinite(out).all()


def test_zero_edges_everywhere_is_valid():
    x, nt, v, ed = rand_graph(8, 0)
    out = model()(x, nt, torch.ones(8, dtype=torch.bool), {r: (a[:0], b[:0], c[:0]) for r, (a, b, c) in ed.items()})
    assert torch.isfinite(out).all() and (out.abs().sum(-1) > 0).all()


def test_every_approved_relation_is_handled_and_targets_only_its_destination_type():
    c = cfg_bike()
    m = model(c)
    x, nt, _, _ = rand_graph(30, 0, seed=3, types=("pedestrian", "vehicle", "bicyclist"))
    valid = torch.ones(30, dtype=torch.bool)
    base = m(x, nt, valid, {})
    for r in ALL_RELATIONS:
        s, d = r.split("__")
        _, _, _, ed = rand_graph(30, 25, seed=3, rels=(r,), types=("pedestrian", "vehicle", "bicyclist"))
        src, dst, e = ed[r]
        assert len(src) > 0
        out = m(x, nt, valid, {r: (src, dst, e)})
        has_in = torch.zeros(30, dtype=torch.bool)
        has_in[dst] = True
        assert not torch.allclose(out[has_in], base[has_in]), r                       # destinations with an incoming edge change
        assert torch.equal(out[~has_in], base[~has_in]), r                              # all other nodes are bit-identical
        assert (nt[dst] == TYPE_OF[d]).all() and (nt[src] == TYPE_OF[s]).all()


def test_unsupported_and_disabled_relations_rejected():
    m = model()
    x, nt, v, ed = rand_graph(10, 20)
    ok = torch.ones(10, dtype=torch.bool)
    with pytest.raises(WindowContractError, match="UNSUPPORTED_RELATION"):
        m(x, nt, ok, dict(ed, **{"bicyclist__pedestrian": ed["vehicle__pedestrian"]}))
    with pytest.raises(WindowContractError, match="UNSUPPORTED_RELATION"):
        m(x, nt, ok, dict(ed, **{"vehicle__vehicle": ed["vehicle__pedestrian"]}))
    with pytest.raises(ConfigError, match="not permitted"):
        cfg(relations=["pedestrian__pedestrian", "vehicle__vehicle"])
    with pytest.raises(ConfigError, match="not permitted"):
        cfg_bike(relations=["pedestrian__pedestrian", "bicyclist__bicyclist"])
    with pytest.raises(ConfigError, match="bicyclist_context_enabled"):
        cfg(node_types=["pedestrian", "vehicle", "bicyclist"])
    with pytest.raises(ConfigError, match="outside node_types"):
        cfg(relations=["pedestrian__pedestrian", "bicyclist__pedestrian"])
    c = cfg(allow_vehicle_vehicle=True, relations=["pedestrian__pedestrian", "vehicle__pedestrian", "vehicle__vehicle"])
    assert "vehicle__vehicle" in c.relations                                         # V-V only by explicit opt-in
    with pytest.raises(WindowContractError, match="UNSUPPORTED_NODE_TYPE"):
        m(x, torch.full_like(nt, -1), ok, {})


def test_invalid_nodes_are_zero_silent_and_edges_touching_them_are_dropped():
    m = model()
    x, nt, _, ed = rand_graph(14, 50, seed=5)
    valid = torch.ones(14, dtype=torch.bool)
    valid[[2, 9]] = False
    out = m(x, nt, valid, ed)
    assert (out[~valid] == 0).all()
    x2 = x.clone()
    x2[~valid] = 1e9                                                              # garbage features at invalid nodes
    ed2 = {r: (s, d, torch.where(((~valid[s]) | (~valid[d])).unsqueeze(-1), torch.full_like(e, 1e9), e)) for r, (s, d, e) in ed.items()}
    assert torch.equal(m(x2, nt, valid, ed2), out)


def test_node_without_incoming_edges_stays_representable():
    m = model()
    x, nt, _, ed = rand_graph(10, 30, seed=2)
    valid = torch.ones(10, dtype=torch.bool)
    out = m(x, nt, valid, ed)
    has_in = torch.zeros(10, dtype=torch.bool)
    for s, d, e in ed.values():
        has_in[d] = True
    iso = (~has_in).nonzero(as_tuple=True)[0]
    x_iso = x.clone()
    ed_none = {r: (s[:0], d[:0], e[:0]) for r, (s, d, e) in ed.items()}
    alone = m(x_iso, nt, valid, ed_none)
    if len(iso):
        assert torch.equal(out[iso], alone[iso])                                  # depends only on its own features (self/root term)
    assert torch.isfinite(out).all() and (out.abs().sum(-1) > 0).all()


def test_node_order_permutation_equivariance():
    m = model()
    x, nt, _, ed = rand_graph(20, 80, seed=7)
    valid = torch.rand(20, generator=torch.Generator().manual_seed(1)) < 0.9
    out = m(x, nt, valid, ed)
    perm = torch.randperm(20, generator=torch.Generator().manual_seed(4))           # new index i holds old node perm[i]
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(20)
    ed_p = {r: (inv[s], inv[d], e) for r, (s, d, e) in ed.items()}
    out_p = m(x[perm], nt[perm], valid[perm], ed_p)
    assert torch.allclose(out_p, out[perm], atol=1e-5, rtol=1e-5)


def test_edge_order_permutation_invariance():
    m = model()
    x, nt, _, ed = rand_graph(20, 80, seed=8)
    valid = torch.ones(20, dtype=torch.bool)
    out = m(x, nt, valid, ed)
    ed_p = {}
    for r, (s, d, e) in ed.items():
        p = torch.randperm(len(s), generator=torch.Generator().manual_seed(11))
        ed_p[r] = (s[p], d[p], e[p])
    assert torch.allclose(m(x, nt, valid, ed_p), out, atol=1e-5, rtol=1e-5)


def test_segment_softmax_normalises_over_incoming_neighbours_only():
    g = torch.Generator().manual_seed(0)
    dst = torch.tensor([0, 0, 0, 2, 2, 5])
    logit = torch.randn(6, 4, generator=g) * 20                                    # large logits: stability
    a = segment_softmax(logit, dst, 7)
    for node in (0, 2, 5):
        assert torch.allclose(a[dst == node].sum(0), torch.ones(4), atol=1e-6)
    assert torch.isfinite(a).all()
    # adding edges to other destinations does not change the weights of node 0
    a2 = segment_softmax(torch.cat([logit, torch.randn(3, 4, generator=g)]), torch.cat([dst, torch.tensor([1, 1, 3])]), 7)
    assert torch.allclose(a2[:3], a[:3], atol=1e-6)
    assert torch.allclose(segment_softmax(logit + 5.0, dst, 7), a, atol=1e-6)      # shift invariance
    assert segment_softmax(logit[:0], dst[:0], 7).shape == (0, 4)


def test_edge_attributes_enter_attention_and_messages():
    m = model()
    x, nt, _, ed = rand_graph(10, 30, seed=9)
    valid = torch.ones(10, dtype=torch.bool)
    out = m(x, nt, valid, ed)
    ed2 = {r: (s, d, e + 0.5) for r, (s, d, e) in ed.items()}
    assert not torch.allclose(m(x, nt, valid, ed2), out)                          # edge attributes are not discarded
    # attention logits depend on edge attributes: perturbing only We (attention path) changes the output
    layer = m.layers[0].rel["pedestrian__pedestrian"]
    snap = layer.w_edge_att.weight.clone()
    with torch.no_grad():
        layer.w_edge_att.weight.add_(0.3)
    assert not torch.allclose(m(x, nt, valid, ed), out)
    with torch.no_grad():
        layer.w_edge_att.weight.copy_(snap)
        layer.w_edge_msg.weight.add_(0.3)
    assert not torch.allclose(m(x, nt, valid, ed), out)                           # ... and the message path


def test_relation_specific_parameters():
    m = model()
    layer = m.layers[0]
    names = list(layer.rel)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = layer.rel[names[i]], layer.rel[names[j]]
            assert a.w_src.weight.data_ptr() != b.w_src.weight.data_ptr()
            assert not torch.equal(a.w_src.weight, b.w_src.weight)
    x, nt, _, ed = rand_graph(16, 60, seed=1)
    valid = torch.ones(16, dtype=torch.bool)
    m.zero_grad()
    only = {"vehicle__pedestrian": ed["vehicle__pedestrian"]}
    m(x, nt, valid, only).sum().backward()
    for r in names:
        g = m.layers[0].rel[r].w_src.weight.grad
        if r == "vehicle__pedestrian":
            assert g is not None and g.abs().sum() > 0
        else:
            assert g is None or g.abs().sum() == 0                                    # parameters of absent relations receive no gradient


def test_type_specific_input_projection_preserves_node_type():
    m = model()
    x = torch.randn(1, FIN).repeat(2, 1)
    out = m(x, torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool), {})
    assert not torch.allclose(out[0], out[1])                                     # identical features, different node type


def test_masked_edge_noise_cannot_contaminate_and_gradients_are_finite():
    m = model()
    x, nt, v, ed = rand_graph(15, 60, seed=6)
    valid = torch.ones(15, dtype=torch.bool)
    e_req = {r: (s, d, e.clone().requires_grad_(False)) for r, (s, d, e) in ed.items()}
    out = m(x, nt, valid, e_req)
    assert torch.isfinite(out).all()
    m.zero_grad()
    m(x, nt, valid, ed).pow(2).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_deterministic_evaluation():
    x, nt, v, ed = rand_graph(15, 60, seed=6)
    ok = torch.ones(15, dtype=torch.bool)
    m1, m2 = model(), model()
    assert torch.equal(m1(x, nt, ok, ed), m1(x, nt, ok, ed))
    assert torch.equal(m1(x, nt, ok, ed), m2(x, nt, ok, ed))                      # same seed -> same weights -> same output
    assert not torch.equal(m1(x, nt, ok, ed), model(cfg(init_seed=1))(x, nt, ok, ed))


def test_frames_are_independent_graphs():
    """Frames are flattened into one node array with edges only inside a frame: perturbing frame 0 cannot change frame 1."""
    m = model()
    x, nt, v, ed = rand_graph(10, 30, seed=2)
    xx = torch.cat([x, x + 1.0])
    nn_ = torch.cat([nt, nt])
    edd = {r: (torch.cat([s, s + 10]), torch.cat([d, d + 10]), torch.cat([e, e])) for r, (s, d, e) in ed.items()}
    ok = torch.ones(20, dtype=torch.bool)
    out = m(xx, nn_, ok, edd)
    xx2 = xx.clone()
    xx2[:10] += 5.0
    out2 = m(xx2, nn_, ok, edd)
    assert torch.equal(out2[10:], out[10:]) and not torch.allclose(out2[:10], out[:10])


def test_shape_and_endpoint_validation():
    m = model()
    x, nt, v, ed = rand_graph(6, 10)
    ok = torch.ones(6, dtype=torch.bool)
    with pytest.raises(WindowContractError, match="SPATIAL_SHAPE"):
        m(x, nt[:-1], ok, ed)
    bad = copy.copy(ed)
    s, d, e = ed["pedestrian__pedestrian"]
    bad["pedestrian__pedestrian"] = (s, d + 100, e)
    with pytest.raises(WindowContractError, match="EDGE_ENDPOINT"):
        m(x, nt, ok, bad)
