"""Stage 6 visualisation (matplotlib, Agg backend, imported lazily).

Rules the figures follow:
* coordinate identities are NEVER drawn on one axis (one panel per identity);
* axis labels come from the data's own unit status: image pixels say "pixel", planar and
  projective values say "units unverified / not metric"; only a manifest-eligible identity
  is labelled metric, and even then "software manifest checks only";
* image-space y grows downward, so image panels invert the y axis;
* a figure that cannot be drawn returns ``(False, reason)`` instead of an empty picture.

``axis_spec`` and ``identity_key`` are pure functions so the labelling rules are testable
without rendering.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from .trajectory import CoordinateMode

MAX_PANELS = 6
FIGURES: Dict[str, Any] = {}


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def identity_key(t) -> Tuple[Any, ...]:
    return (t.coordinate_mode, t.unit_status, t.position_unit, t.frame_name, t.calibration_id, t.manifest_sha256)


def axis_spec(t) -> Dict[str, Any]:
    """Labels/banner for one trajectory's coordinate identity."""
    mode, unit = t.coordinate_mode, t.position_unit
    invert = mode == CoordinateMode.IMAGE_PIXEL.value
    if t.physical:
        banner = "metric by software manifest checks only (not independently verified)"
    elif mode == CoordinateMode.IMAGE_PIXEL.value:
        banner = "image pixels - not metric"
    else:
        banner = "units UNVERIFIED - not metric (%s)" % mode
    return {"xlabel": "x [%s]" % unit, "ylabel": "y [%s]" % unit, "invert_y": invert, "banner": banner,
            "speed_label": "speed [%s]" % t.velocity_unit, "physical": bool(t.physical), "mode": mode}


def _groups(trajs) -> List[Tuple[Tuple[Any, ...], list]]:
    out: Dict[Tuple[Any, ...], list] = {}
    for t in trajs:
        out.setdefault(identity_key(t), []).append(t)
    return sorted(out.items(), key=lambda kv: str(kv[0]))


def _colours(trajs):
    classes = sorted({t.agent_class for t in trajs})
    cmap = {c: "C%d" % i for i, c in enumerate(classes)}
    return cmap


def _save(fig, path) -> None:
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    _plt().close(fig)


def _panels(n):
    plt = _plt()
    n = min(n, MAX_PANELS)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    return fig, axes[0]


def plot_trajectories(path, *, trajectories, max_tracks=60, **_):
    groups = [(k, v) for k, v in _groups(trajectories) if len(v)]
    if not groups:
        return False, "no trajectories to draw (check the coordinate mode and input)"
    fig, axes = _panels(len(groups))
    for ax, (_, trajs) in zip(axes, groups):
        spec = axis_spec(trajs[0])
        cmap = _colours(trajs)
        for t in sorted(trajs, key=lambda t: t.track_id)[:max_tracks]:
            x, y = t["x"], t["y"]
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            ax.plot(x[ok], y[ok], lw=0.8, color=cmap[t.agent_class], alpha=0.8)
            last = np.flatnonzero(ok)[-1]
            ax.plot(x[last], y[last], "o", ms=2.5, color=cmap[t.agent_class])
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["ylabel"])
        if spec["invert_y"]:
            ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("%s\n%d track(s), first %d drawn" % (spec["banner"], len(trajs), min(len(trajs), max_tracks)), fontsize=8)
        ax.legend(handles=[_line(c, col) for c, col in cmap.items()], fontsize=7, loc="best")
    _save(fig, path)
    return True, ""


def _line(label, colour):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=colour, lw=1.5, label=label)


def plot_kinematics(path, *, trajectories, max_tracks=12, **_):
    groups = _groups(trajectories)
    if not groups:
        return False, "no trajectories to draw"
    fig, axes = _panels(len(groups))
    drawn = 0
    for ax, (_, trajs) in zip(axes, groups):
        spec = axis_spec(trajs[0])
        for t in sorted(trajs, key=lambda t: t.track_id)[:max_tracks]:
            sp = t["speed"]
            ok = np.isfinite(sp)
            if ok.any():
                ax.plot(t["timestamp"][ok], sp[ok], lw=0.8)
                drawn += 1
        ax.set_xlabel("time [s]")
        ax.set_ylabel(spec["speed_label"])
        ax.set_title(spec["banner"], fontsize=8)
    if not drawn:
        _plt().close(fig)
        return False, "no finite speed values (velocity is unavailable in this coordinate mode or tracks are too short)"
    _save(fig, path)
    return True, ""


def plot_projection(path, *, ingest, config, **_):
    ing = ingest
    if ing.projections is None or config.calibration is None:
        return False, "no calibration supplied, so there is no projection to show"
    man = config.calibration.manifest
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    status = {}
    for pr in ing.projections:
        if pr.image_xy_used is not None:
            status.setdefault(pr.status.value, []).append(pr.image_xy_used)
    if not status:
        plt.close(fig)
        return False, "no record had an image position to show"
    for i, (s, pts) in enumerate(sorted(status.items())):
        a = np.array(pts)
        ax.scatter(a[:, 0], a[:, 1], s=3, label="%s (%d)" % (s, len(pts)), color="C%d" % i)
    r = man.raster
    ax.plot([0, r.width, r.width, 0, 0], [0, 0, r.height, r.height, 0], "k-", lw=0.8, label="calibration raster")
    if man.valid_region:
        reg = np.array(list(man.valid_region) + [man.valid_region[0]])
        ax.plot(reg[:, 0], reg[:, 1], "r--", lw=1, label="valid region")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    elig = ing.transform.preflight.eligibility.manifest_eligible
    ax.set_title("projection status (image pixels). manifest_eligible=%s (software checks only; not physical verification)" % elig, fontsize=7)
    ax.legend(fontsize=7)
    _save(fig, path)
    return True, ""


def plot_split(path, *, dataset, **_):
    if dataset is None:
        return False, "no split dataset"
    rep = dataset.report
    plt = _plt()
    parts = ("train", "validation", "test")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, title in zip(axes, ("n_groups", "n_observations", "n_windows"), ("groups", "observations", "windows")):
        vals = [rep["partitions"][p][key] for p in parts]
        ax.bar(parts, vals, color=["C0", "C1", "C2"])
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.suptitle("split by %s (seed %s, plan %s...) - %s" % (rep["grouping_unit"], rep["seed"], rep["split_plan_sha256"][:12],
                                                              "independence is a stated claim, not verified"), fontsize=8)
    _save(fig, path)
    return True, ""


def plot_windows(path, *, dataset, n_examples=6, **_):
    if dataset is None:
        return False, "no split dataset"
    wins = [w for p in ("train", "validation", "test") for w in dataset.partitions[p].windows][:0]
    for p in ("train", "validation", "test"):
        ws = dataset.partitions[p].windows
        wins += ws[:: max(1, len(ws) // 2)][:2] if ws else []
    wins = wins[:n_examples]
    if not wins:
        return False, "no windows were generated"
    plt = _plt()
    cols = min(3, len(wins))
    rows = (len(wins) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, w in zip(axes.ravel(), wins):
        ax.axis("on")
        ok = w.history_valid
        ax.plot(w.history["xs"][ok], w.history["ys"][ok], "o-", ms=3, label="history")
        ax.plot(w.target_xy[:, 0], w.target_xy[:, 1], "s--", ms=3, label="target (raw)")
        if w.coordinate_mode == "image_pixel":
            ax.invert_yaxis()
        ax.set_xlabel("x [%s]" % w.position_unit)
        ax.set_ylabel("y [%s]" % w.position_unit)
        tag = "metric (manifest checks only)" if w.physical else "not metric"
        ax.set_title("%s | %s\n%s | %s" % (w.partition, w.agent_class, w.coordinate_mode, tag), fontsize=7)
        ax.legend(fontsize=6)
    fig.subplots_adjust(hspace=0.75, wspace=0.35)
    _save(fig, path)
    return True, ""


FIGURES.update({"trajectories": plot_trajectories, "kinematics": plot_kinematics, "projection": plot_projection,
                "split": plot_split, "windows": plot_windows})
