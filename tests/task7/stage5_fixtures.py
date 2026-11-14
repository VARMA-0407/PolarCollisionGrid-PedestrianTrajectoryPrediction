"""Synthetic Stage 5 corpus: several scenes x sequences x tracks of analytic image tracks
pushed through the real Stage 2 code. All numbers are invented."""

import copy
import functools

from pedsafety.agent_state import build_agent_states
from pedsafety.dataset_builder import build_split_dataset
from pedsafety.splitting import GroupingPolicy, SplitConfig
from pedsafety.trajectory import TrajectoryConfig
from pedsafety.windows import WindowConfig
from stage4_fixtures import make_track

GROUP = GroupingPolicy("scene", "synthetic scenes are independent by construction")
SPLIT = SplitConfig(seed=7)
WIN = WindowConfig(obs_len=4, pred_len=3)


def corpus_observations(n_scenes=6, seqs_per_scene=2, tracks_per_seq=3, n_frames=20, dataset="SYN", scene_offset=0, drop_by_track=None):
    obs = []
    for s in range(scene_offset, scene_offset + n_scenes):
        for q in range(seqs_per_scene):
            for t in range(tracks_per_seq):
                drop = (drop_by_track or {}).get(t, ())
                obs += make_track("t%d" % t, frames=range(n_frames), x0=200.0 + 40.0 * s + 10.0 * q + 5.0 * t, y0=300.0 + 20.0 * t,
                                  vx=10.0 + 3.0 * t + s, vy=-5.0 + q, ax=0.5 * s, ay=0.3 * t, seq="s%d_%d" % (s, q), scene="sc%d" % s,
                                  dataset=dataset, drop=drop)
    return obs


@functools.lru_cache(maxsize=16)
def _cached(key):
    res = build_agent_states(corpus_observations(**dict(key)))
    assert res.report.ok, res.report.summary()
    return res.records, res.provenance


def corpus(**kw):
    """(records, provenance) deep-copied so tests may mutate them freely."""
    recs, prov = _cached(tuple(sorted(kw.items(), key=lambda kv: kv[0]))) if all(isinstance(v, (int, str)) for v in kw.values()) else (None, None)
    if recs is None:
        res = build_agent_states(corpus_observations(**kw))
        assert res.report.ok
        recs, prov = res.records, res.provenance
    return copy.deepcopy(recs), list(prov)


def dataset(records=None, provenance=None, projections=None, grouping=GROUP, split=SPLIT, windows=WIN, trajectory=None, view="causal", **kw):
    if records is None:
        records, provenance = corpus(**kw)
    return build_split_dataset(records, provenance, projections, grouping=grouping, split=split, windows=windows,
                               trajectory=trajectory or TrajectoryConfig(), view=view)
