"""Rebuild paper tables and monochrome figures for the SNAP email study.

All panels are derived from saved episodes, model predictions, and bitstring
measurement decisions. The figures deliberately do not reuse LANL assets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import torch

from .email_qrouter import CONDITIONS, load_episodes
from .email_qrouter_shots import LOOKS, _set
from .qrag_moe import EntangledQuantumRouter


INK = "#202020"
DARK = "#454545"
MID = "#757575"
LIGHT = "#a8a8a8"
PALE = "#d5d5d5"
PAPER = "#f5f5f3"
MODEL_COLORS = {
    "temporal": LIGHT, "graph": INK, "fixed_mix": MID,
    "quantum": DARK, "classical": "#909090",
}
MODEL_MARKERS = {
    "temporal": "o", "graph": "s", "fixed_mix": "D",
    "quantum": "^", "classical": "v",
}
MODEL_STYLES = {
    "temporal": ":", "graph": "-", "fixed_mix": "--",
    "quantum": "-.", "classical": (0, (5, 2)),
}
CONDITION_LABELS = ("Complete", "Query half", "History half", "Joint half")
SEED_DIRS = ("shots_model0_repeat0", "shots_model1_repeat0", "shots_model2_repeat0")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.edgecolor": MID, "axes.linewidth": 0.75,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.color": INK, "ytick.color": INK,
        "grid.color": PALE, "grid.linewidth": 0.5,
        "savefig.facecolor": "white", "pdf.fonttype": 42,
    })


def _axes() -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), dpi=220)
    figure.subplots_adjust(left=.075, right=.975, bottom=.095, top=.88,
                           wspace=.25, hspace=.39)
    return figure, axes


def _panel(axis: plt.Axes, letter: str, title: str, grid: str = "y") -> None:
    axis.set_title(f"({letter})  {title}", loc="left", color=INK, weight="bold", pad=11)
    if grid:
        axis.grid(axis=grid, alpha=.65, zorder=0)
    axis.set_axisbelow(True)


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=.16)
    figure.savefig(output / f"{stem}.png", bbox_inches="tight", pad_inches=.16)
    plt.close(figure)


def _rank_reciprocal(distribution: np.ndarray, truth: np.ndarray) -> np.ndarray:
    actual = distribution[np.arange(len(truth)), truth]
    rank = 1 + (distribution > actual[:, None]).sum(axis=1)
    return 1.0 / rank


def _summarize_models(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    model_names = ("temporal", "graph", "fixed_mix", "quantum", "classical")
    for condition in CONDITIONS:
        for model in model_names:
            if model in ("quantum", "classical"):
                selected = metrics[(metrics.condition == condition) &
                                   metrics.model.str.startswith(f"{model}_seed")]
            else:
                selected = metrics[(metrics.condition == condition) &
                                   (metrics.model == model)]
            row = {"condition": condition, "model": model,
                   "model_seeds": len(selected), "test_episodes": int(selected.episodes.iloc[0])}
            for column in ("mrr", "hit10", "nll", "mean_gate"):
                row[f"{column}_mean"] = float(selected[column].mean())
                row[f"{column}_sd"] = float(selected[column].std(ddof=0))
            records.append(row)
    return pd.DataFrame(records)


def _resource_summary(decisions: list[pd.DataFrame]) -> pd.DataFrame:
    records = []
    for seed, frame in enumerate(decisions):
        for condition in (*CONDITIONS, "all"):
            subset = frame if condition == "all" else frame[frame.condition == condition]
            records.append({
                "model_seed": seed, "condition": condition, "decisions": len(subset),
                "mean_shots": float(subset.shots.mean()),
                "shot_sd": float(subset.shots.std(ddof=0)),
                "savings_vs_fixed512_pct": float(100 * (1 - subset.shots.mean() / 512)),
                "observable_bound_mean_shots": float(subset.observable_bound_shots.mean()),
                "scalar_shots_saved_vs_observable": float(
                    (subset.observable_bound_shots - subset.shots).mean()),
                "certified_pct": float(100 * subset.certified.mean()),
                "observable_certified_pct": float(100 * subset.observable_bound_certified.mean()),
                "fixed512_exact_match_pct": float(100 * subset.fixed512_matches_exact.mean()),
                "certified_mismatches": int(((subset.certified == 1) &
                                             (subset.top10_matches_exact == 0)).sum()),
                "zero_shot_decisions": int(subset.zero_shot.sum()),
            })
    return pd.DataFrame(records)


def _overlap_bin(values: np.ndarray) -> pd.Categorical:
    return pd.cut(values, bins=(-.01, .001, .20001, .40001, 1.01),
                  labels=("0", "0.1-0.2", "0.3-0.4", "0.5-1.0"),
                  include_lowest=True)


def _joined_decisions(frame: pd.DataFrame, episodes) -> pd.DataFrame:
    result = frame.copy()
    indices = result.condition.map({condition: index for index, condition in enumerate(CONDITIONS)})
    episode_ids = result.episode.to_numpy(dtype=np.int32)
    condition_ids = indices.to_numpy(dtype=np.int32)
    result["expert_top10_overlap"] = episodes.features[episode_ids, condition_ids, 8]
    result["candidate_count"] = episodes.candidate_count[episode_ids]
    result["query_activity_log"] = episodes.features[episode_ids, condition_ids, 2]
    result["history_activity_log"] = episodes.features[episode_ids, condition_ids, 3]
    result["overlap_bin"] = _overlap_bin(result.expert_top10_overlap.to_numpy())
    return result


def _overlap_summary(decisions: list[pd.DataFrame]) -> pd.DataFrame:
    records = []
    for condition in CONDITIONS:
        for band in ("0", "0.1-0.2", "0.3-0.4", "0.5-1.0"):
            seed_groups = [frame[(frame.condition == condition) &
                                 (frame.overlap_bin == band)] for frame in decisions]
            subset = pd.concat(seed_groups, ignore_index=True)
            admitted = subset[subset.certified == 1]
            records.append({
                "condition": condition, "expert_top10_overlap": band,
                "paired_decisions": len(seed_groups[0]), "model_seeds": len(decisions),
                "median_candidates": float(subset.candidate_count.median()),
                "mean_recent_source_activity_log": float(subset.query_activity_log.mean()),
                "mean_history_source_activity_log": float(subset.history_activity_log.mean()),
                "mean_shots": float(subset.shots.mean()),
                "shot_sd_across_seeds": float(np.std(
                    [group.shots.mean() for group in seed_groups], ddof=0)),
                "shots_saved_vs_separate_observables": float(
                    (subset.observable_bound_shots - subset.shots).mean()),
                "certified_pct": float(100 * subset.certified.mean()),
                "observable_certified_pct": float(100 * subset.observable_bound_certified.mean()),
                "admitted_hit10_pct": float(100 * admitted.top10_contains_target.mean()),
                "exact_hit10_all_pct": float(100 * subset.exact_top10_contains_target.mean()),
                "certified_mismatches": int((admitted.top10_matches_exact == 0).sum()),
            })
    return pd.DataFrame(records)


def _dataset_table(frame: pd.DataFrame, episodes, protocol: dict,
                   decisions: list[pd.DataFrame]) -> pd.DataFrame:
    unique_pairs = len(frame[["src", "dst"]].drop_duplicates())
    active_days = int(frame.day.nunique())
    test = decisions[0]
    candidates = episodes.candidate_count[test.episode.to_numpy(dtype=np.int32)]
    rows = [
        ("Network", "Observed directed events", len(frame), "Timestamped emails"),
        ("Network", "Distinct members", episodes.audit["nodes"], "Observed source or recipient"),
        ("Network", "Directed source-recipient pairs", unique_pairs, "Distinct ordered pairs"),
        ("Network", "Active days", active_days, "Days containing an email"),
        ("Network", "Event span in days", int(frame.day.max() + 1), "First to last recorded day"),
        ("Episodes", "Training episodes", protocol["split_episodes"]["train"], "Chronological"),
        ("Episodes", "Validation episodes", protocol["split_episodes"]["validation"], "Chronological"),
        ("Episodes", "Test episodes", protocol["split_episodes"]["test"], "88 test days"),
        ("Episodes", "Paired test decisions", len(test), "Four conditions per episode"),
        ("Candidates", "Median test candidate count", int(np.median(candidates)), "Previously observed members"),
        ("Candidates", "Test candidate count minimum", int(np.min(candidates)), "Sender excluded"),
        ("Candidates", "Test candidate count maximum", int(np.max(candidates)), "Sender excluded"),
        ("History", "Recent query window", 7, "Days before label"),
        ("History", "Historical graph window", 83, "Days 8 through 90 before label"),
        ("Training", "Quantum qubits", 4, "Two circuit layers"),
        ("Training", "Quantum trainable values", 25, "Rotation and readout parameters"),
        ("Training", "Classical trainable values", 23, "Two hidden tanh units"),
        ("Training", "Independent model seeds", protocol["seeds"], "Same chronological test set"),
        ("Measurement", "Maximum shots per decision", LOOKS[-1], "Six prespecified looks"),
        ("Measurement", "Nominal decision confidence", 0.95, "Ideal independent bitstrings"),
    ]
    result = pd.DataFrame(rows, columns=("block", "measure", "value", "definition"))
    result["value"] = result.value.map(
        lambda value: str(int(value)) if float(value).is_integer() else f"{value:g}")
    return result


def _figure1(output: Path, frame: pd.DataFrame, episodes,
             decisions: list[pd.DataFrame], training: Path) -> None:
    figure, axes = _axes()
    axis = axes[0, 0]
    _panel(axis, "a", "Observed communication and chronological split")
    counts = frame.groupby("day").size()
    active_index = np.arange(len(counts))
    rolling = counts.rolling(14, center=True, min_periods=1).mean()
    axis.plot(active_index, counts.to_numpy(), color=PALE, lw=.55, alpha=.8,
              rasterized=True, label="Daily events")
    axis.plot(active_index, rolling.to_numpy(), color=INK, lw=1.35,
              label="14-day mean")
    split_points = np.searchsorted(counts.index.to_numpy(), (90, 354, 444))
    for start, end, shade, label in ((split_points[0], split_points[1], "#ededed", "Training"),
                                     (split_points[1], split_points[2], "#dddddd", "Validation"),
                                     (split_points[2], len(counts), "#f7f7f7", "Test")):
        axis.axvspan(start, end, color=shade, alpha=.55, zorder=-1)
        axis.text((start + end) / 2, .98, label, ha="center", va="top",
                  transform=axis.get_xaxis_transform(), fontsize=7.5, color=DARK)
    axis.set(xlabel="Active-day index (calendar gaps omitted)",
             ylabel="Observed events per active day", xlim=(0, len(counts) - 1))
    axis.legend(loc="lower right", frameon=False, fontsize=7.5)

    axis = axes[0, 1]
    _panel(axis, "b", "Complementarity of fixed experts", grid="")
    test_ids = np.unique(decisions[0].episode.to_numpy(dtype=np.int32))
    temporal = episodes.temporal[test_ids].reshape(-1, episodes.temporal.shape[-1])
    graph = episodes.graph[test_ids].reshape(-1, episodes.graph.shape[-1])
    truth = np.repeat(episodes.truth[test_ids], len(CONDITIONS))
    rr_t = _rank_reciprocal(temporal, truth)
    rr_g = _rank_reciprocal(graph, truth)
    hexes = axis.hexbin(rr_t, rr_g, xscale="log", yscale="log", gridsize=31,
                        mincnt=1, cmap="Greys", norm=LogNorm(), linewidths=0,
                        extent=(-3.1, 0, -3.1, 0), rasterized=True)
    axis.plot([.001, 1], [.001, 1], color=INK, lw=.8, linestyle="--")
    axis.set(xlabel="Temporal reciprocal rank", ylabel="Graph reciprocal rank",
             xlim=(.001, 1.03), ylim=(.001, 1.03))
    cb = figure.colorbar(hexes, ax=axis, fraction=.045, pad=.02)
    cb.set_label("Paired decisions", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    axis.text(.04, .96, f"Temporal wins  {(rr_t > rr_g).mean():.1%}\n"
                        f"Graph wins  {(rr_g > rr_t).mean():.1%}",
              transform=axis.transAxes, va="top", fontsize=7.5,
              bbox=dict(facecolor="white", edgecolor=PALE, pad=3))

    axis = axes[1, 0]
    _panel(axis, "c", "Recent activity and realized graph advantage", grid="")
    activity = episodes.features[test_ids, :, 2].ravel()
    true_temporal = temporal[np.arange(len(truth)), truth]
    true_graph = graph[np.arange(len(truth)), truth]
    advantage = np.log(np.maximum(true_graph, 1e-12)) - np.log(
        np.maximum(true_temporal, 1e-12))
    hexes = axis.hexbin(activity, np.clip(advantage, -8, 8), gridsize=(38, 26),
                        mincnt=1, cmap="Greys", norm=LogNorm(), linewidths=0,
                        extent=(0, 5.6, -8, 8), rasterized=True)
    axis.axhline(0, color=INK, linestyle="--", lw=.9)
    axis.set(xlabel="Log recent source activity", ylabel="Graph log-loss advantage",
             xlim=(0, 5.6), ylim=(-8, 8))
    cb = figure.colorbar(hexes, ax=axis, fraction=.045, pad=.02)
    cb.set_label("Paired decisions", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    axis = axes[1, 1]
    _panel(axis, "d", "Observed top-ten decision interval", grid="y")
    candidate = decisions[0][(decisions[0].certificate == "confidence") &
                             (decisions[0].shots.between(32, 128)) &
                             (decisions[0].expert_top10_overlap.between(.1, .5))]
    if candidate.empty:
        candidate = decisions[0][decisions[0].certificate == "confidence"]
    example = candidate.iloc[len(candidate) // 2]
    e = int(example.episode)
    c = CONDITIONS.index(example.condition)
    temporal_e = episodes.temporal[e, c]
    graph_e = episodes.graph[e, c]
    gate = float(example.measured_gate)
    reference = _set(temporal_e, graph_e, gate)
    grid = np.linspace(0, 1, 401)
    overlap = [len(_set(temporal_e, graph_e, float(a)) & reference) / 10 for a in grid]
    model = EntangledQuantumRouter(feature_count=episodes.features.shape[-1],
                                   n_qubits=4, depth=2)
    model.load_state_dict(torch.load(training / "quantum_seed0.pt", map_location="cpu",
                                     weights_only=True))
    signs = model.observable_signs.detach().numpy().T
    weights = model.readout.weight.detach().numpy().ravel()
    outcomes = signs @ weights
    radius = (outcomes.max() - outcomes.min()) * np.sqrt(
        np.log(2 * len(LOOKS) / .05) / (2 * int(example.shots)))
    logit = np.log(np.clip(gate, 1e-8, 1 - 1e-8) /
                   (1 - np.clip(gate, 1e-8, 1 - 1e-8)))
    left = 1 / (1 + np.exp(-np.clip(logit - radius, -30, 30)))
    right = 1 / (1 + np.exp(-np.clip(logit + radius, -30, 30)))
    axis.axvspan(left, right, color=PALE, alpha=.5, label="95% gate interval")
    axis.step(grid, overlap, where="mid", color=INK, lw=1.2,
              label="Top-ten overlap with measured set")
    axis.axvline(gate, color=DARK, lw=1, linestyle="--", label="Measured gate")
    axis.set(xlabel=r"Graph authority $\alpha$", ylabel="Top-ten set overlap",
             xlim=(0, 1), ylim=(-.03, 1.04))
    axis.legend(loc="lower left", fontsize=7, frameon=False)
    axis.text(.98, .13, f"Day {int(example.day)}  |  {int(example.shots)} shots",
              transform=axis.transAxes, ha="right", fontsize=7.5,
              bbox=dict(facecolor="white", edgecolor=PALE, pad=2))
    _save(figure, output, "figure1_temporal_quantum_method")


def _architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(15.8, 4.7), dpi=220)
    figure.subplots_adjust(left=.015, right=.985, top=.97, bottom=.04)
    axis.set(xlim=(0, 16.4), ylim=(0, 5))
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, title: str,
            detail: str, fill: str = "#f3f3f3") -> None:
        axis.add_patch(FancyBboxPatch((x, y), width, height,
                       boxstyle="round,pad=0.035", facecolor=fill,
                       edgecolor=DARK, linewidth=.9))
        axis.text(x + width / 2, y + height * .80, title,
                  ha="center", va="center", fontsize=14, weight="bold")
        axis.text(x + width / 2, y + height * .29, detail,
                  ha="center", va="center", fontsize=12, linespacing=1.2)

    def arrow(start: tuple[float, float], end: tuple[float, float],
              style: str = "-") -> None:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                       mutation_scale=11, linewidth=.9,
                       linestyle=style, color=DARK))

    box(.20, 2.55, 1.70, 1.20, "Email events",
        "Directed events\nLabel withheld")
    box(2.30, 3.36, 1.83, .92, "Recent window",
        "Previous 7 days\nFull or half kept")
    box(2.30, 2.05, 1.83, .92, "History window",
        "Days 8 through 90\nFull or half kept")
    box(4.48, 3.36, 1.93, .92, "Temporal expert",
        "Direct contacts\nRecent popularity")
    box(4.48, 2.05, 1.93, .92, "Graph expert",
        "Source-conditioned\nTwo-hop diffusion")
    box(6.76, 2.62, 1.95, 1.24, "Router context",
        "Retention, activity\nEntropy, margins\nTop-ten overlap")
    box(9.06, 2.62, 1.85, 1.24, "Quantum gate",
        "Four qubits\nTwo circuit layers\nZ-basis readout", "#e6e6e6")
    box(11.26, 2.62, 1.85, 1.24, "Joint readout",
        "Bounded scalar\nGate interval at\nprespecified looks", "#e6e6e6")
    box(13.46, 2.62, 2.64, 1.24, "Decision certificate",
        "Mix cached experts\nCompare top-ten sets\nat interval endpoints", "#dedede")

    for start, end in (((1.90, 3.15), (2.30, 3.82)),
                       ((1.90, 3.15), (2.30, 2.51)),
                       ((4.13, 3.82), (4.48, 3.82)),
                       ((4.13, 2.51), (4.48, 2.51)),
                       ((6.41, 3.82), (6.76, 3.43)),
                       ((6.41, 2.51), (6.76, 3.05)),
                       ((8.71, 3.24), (9.06, 3.24)),
                       ((10.91, 3.24), (11.26, 3.24)),
                       ((13.11, 3.24), (13.46, 3.24))):
        arrow(start, end)

    box(.60, .63, 7.88, .98, "Chronological model development",
        "Chronological training and validation. Mixture NLL trains the gate.\n"
        "Matched classical and fixed-mixture policies are controls.",
        "#fafafa")
    box(9.05, .63, 7.05, .98, "Adaptive inference",
        "Shot looks 16, 32, 64, 128, 256, 512.\n"
        "Return a certified top-ten set or an unresolved decision.",
        "#fafafa")
    arrow((7.65, 1.61), (9.45, 2.62), "--")
    arrow((12.50, 1.61), (12.28, 2.62), "--")
    axis.plot([6.41, 6.59, 6.59, 14.78], [3.82, 3.82, 4.48, 4.48],
              linestyle="--", linewidth=.8, color=DARK)
    axis.plot([6.41, 6.59, 6.59], [2.51, 2.51, 4.48],
              linestyle="--", linewidth=.8, color=DARK)
    arrow((14.78, 4.48), (14.78, 3.90), "--")
    axis.text(10.65, 4.68, "Cached recent and graph distributions",
              ha="center", fontsize=13, color=DARK)
    axis.text(3.75, 4.68, "All windows precede the labelled day",
              ha="center", fontsize=13, color=DARK)
    _save(figure, output, "architecture_email_quantum_router")


def _figure2(output: Path, metrics: pd.DataFrame, episodes,
             predictions: dict[str, np.ndarray], split: np.ndarray) -> None:
    figure, axes = _axes()
    del metrics
    episode_ids = np.flatnonzero(split == "test")
    nodes = episodes.temporal.shape[-1]
    temporal = episodes.temporal.reshape(-1, nodes)
    graph = episodes.graph.reshape(-1, nodes)
    target = np.repeat(episodes.truth, len(CONDITIONS))
    indices = (episode_ids[:, None] * len(CONDITIONS) +
               np.arange(len(CONDITIONS))[None, :]).ravel()
    truth = target[indices]
    true_t = temporal[indices, truth]
    true_g = graph[indices, truth]
    curves = {}
    losses = {}
    for model in ("temporal", "graph", "fixed_mix", "quantum", "classical"):
        names = ([f"{model}_seed{seed}" for seed in range(3)] if model in
                 ("quantum", "classical") else [model])
        model_ranks = []
        model_losses = []
        for name in names:
            gate = predictions[name][indices]
            fused = (1 - gate[:, None]) * temporal[indices] + gate[:, None] * graph[indices]
            model_ranks.append(1 / _rank_reciprocal(fused, truth))
            probability = (1 - gate) * true_t + gate * true_g
            model_losses.append(-np.log(np.maximum(probability, 1e-12)))
        curves[model] = np.asarray(model_ranks)
        losses[model] = np.asarray(model_losses)

    axis = axes[0, 0]
    _panel(axis, "a", "Ground-truth rank spectrum")
    rank_grid = np.unique(np.rint(np.geomspace(1, nodes, 180)).astype(int))
    for model in curves:
        cdf = np.asarray([[(rank <= cutoff).mean() for cutoff in rank_grid]
                          for rank in curves[model]])
        axis.plot(rank_grid, cdf.mean(axis=0), color=MODEL_COLORS[model],
                  linestyle=MODEL_STYLES[model], lw=1.45,
                  label=model.replace("_", " "))
        if len(cdf) > 1:
            axis.fill_between(rank_grid, cdf.min(axis=0), cdf.max(axis=0),
                              color=MODEL_COLORS[model], alpha=.13, linewidth=0)
    axis.set(xscale="log", xlabel="Ground-truth candidate rank",
             ylabel="Fraction retrieved by rank", xlim=(1, nodes), ylim=(0, 1.01))

    axis = axes[0, 1]
    _panel(axis, "b", "Log-loss tail beyond threshold")
    loss_grid = np.linspace(0, 12, 151)
    for model in losses:
        survival = np.asarray([[(loss > threshold).mean() for threshold in loss_grid]
                               for loss in losses[model]])
        axis.plot(loss_grid, survival.mean(axis=0), color=MODEL_COLORS[model],
                  linestyle=MODEL_STYLES[model], lw=1.45)
        if len(survival) > 1:
            axis.fill_between(loss_grid, survival.min(axis=0), survival.max(axis=0),
                              color=MODEL_COLORS[model], alpha=.13, linewidth=0)
    axis.set(xlabel="Held-out decision log loss", ylabel="Fraction exceeding threshold",
             xlim=(0, 12), ylim=(0, 1.01))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .975),
                  ncol=5, frameon=False, columnspacing=1.6)

    axis = axes[1, 0]
    _panel(axis, "c", "Routing under retrospective expert advantage")
    advantage = np.log(np.maximum(true_g, 1e-12)) - np.log(np.maximum(true_t, 1e-12))
    edges = np.quantile(advantage, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    bin_id = np.clip(np.searchsorted(edges[1:-1], advantage), 0, len(edges) - 2)
    centers = np.asarray([np.median(advantage[bin_id == index])
                          for index in range(len(edges) - 1)])
    for model, color, linestyle, marker in (("quantum", DARK, "-", "o"),
                                             ("classical", MID, "--", "s")):
        values = np.asarray([[predictions[f"{model}_seed{seed}"][indices][bin_id == index].mean()
                              for index in range(len(edges) - 1)] for seed in range(3)])
        axis.fill_between(centers, values.min(axis=0), values.max(axis=0),
                          color=color, alpha=.16, linewidth=0)
        axis.plot(centers, values.mean(axis=0), color=color, linestyle=linestyle,
                  marker=marker, ms=3.7, lw=1.4, label=f"{model.capitalize()} gate")
    axis.axvline(0, color=INK, linestyle=":", lw=1)
    axis.set(xlabel="Retrospective graph log-loss advantage",
             ylabel="Mean graph routing weight", ylim=(0, 1))
    axis.legend(loc="upper left", frameon=False, fontsize=7.4)

    axis = axes[1, 1]
    _panel(axis, "d", "Paired quantum minus classical loss", grid="y")
    difference = (losses["quantum"] - losses["classical"]).mean(axis=0).reshape(
        len(episode_ids), len(CONDITIONS))
    bodies = axis.violinplot([difference[:, c] for c in range(4)],
                             positions=np.arange(4), widths=.62,
                             showmedians=True, showextrema=False)
    for body in bodies["bodies"]:
        body.set_facecolor(PALE)
        body.set_edgecolor(DARK)
        body.set_alpha(.85)
    bodies["cmedians"].set_color(INK)
    bodies["cmedians"].set_linewidth(1.5)
    rng = np.random.default_rng(831)
    for condition_id in range(4):
        sample = rng.choice(len(episode_ids), min(140, len(episode_ids)), replace=False)
        axis.scatter(condition_id + rng.normal(0, .08, len(sample)),
                     difference[sample, condition_id], s=5, color=DARK,
                     alpha=.25, linewidths=0, rasterized=True, zorder=2)
    axis.axhline(0, color=INK, lw=.9, linestyle="--")
    axis.set(xticks=range(4), xticklabels=CONDITION_LABELS,
             ylabel="Quantum minus classical NLL")
    axis.tick_params(axis="x", rotation=13)
    _save(figure, output, "figure2_predictive_audit")


def _figure3(output: Path, decisions: list[pd.DataFrame], training: Path,
             feature_count: int) -> None:
    figure, axes = _axes()
    budgets = np.asarray(LOOKS)
    scalar_curve, observable_curve, fixed_curve = [], [], []
    for frame in decisions:
        scalar_curve.append([((frame.certified == 1) & (frame.shots <= budget)).mean()
                             for budget in budgets])
        observable_curve.append([((frame.observable_bound_certified == 1) &
                                  (frame.observable_bound_shots <= budget)).mean()
                                 for budget in budgets])
        fixed_curve.append([frame[f"fixed{budget}_matches_exact"].mean()
                            for budget in LOOKS])
    scalar_curve = np.asarray(scalar_curve)
    observable_curve = np.asarray(observable_curve)
    fixed_curve = np.asarray(fixed_curve)

    axis = axes[0, 0]
    _panel(axis, "a", "Cumulative certified coverage")
    for array, color, linestyle, label in (
        (scalar_curve, INK, "-", "Joint scalar bound"),
        (observable_curve, MID, "--", "Separate-observable bound"),
    ):
        average = array.mean(axis=0)
        axis.fill_between(budgets, array.min(axis=0), array.max(axis=0),
                          color=color, alpha=.12, linewidth=0)
        axis.step(budgets, average, where="post", color=color,
                  linestyle=linestyle, lw=1.55, label=label)
        axis.scatter(budgets, average, color=color, s=18, zorder=3)
    axis.set(xlabel="Maximum measured shots", ylabel="Certified fraction",
             xlim=(12, 560), ylim=(0, .86))
    axis.set_xscale("log")
    axis.set_xticks(LOOKS, labels=[str(value) for value in LOOKS])
    axis.legend(loc="upper left", frameon=False, fontsize=7.3)

    axis = axes[0, 1]
    _panel(axis, "b", "Fixed-shot fidelity to ideal circuit")
    average = fixed_curve.mean(axis=0)
    axis.fill_between(LOOKS, fixed_curve.min(axis=0), fixed_curve.max(axis=0),
                      color=LIGHT, alpha=.35, linewidth=0)
    axis.plot(LOOKS, average, color=INK, lw=1.55, marker="o", ms=4.5)
    axis.set(xlabel="Fixed quantum shots", ylabel="Exact top-ten set match",
             xlim=(8, 520), ylim=(.88, 1.005))
    axis.set_xscale("log")
    axis.set_xticks(LOOKS, labels=[str(value) for value in LOOKS])

    axis = axes[1, 0]
    _panel(axis, "c", "Where measurement stops", grid="y")
    merged = pd.concat(decisions, ignore_index=True)
    stack_labels = ("0", "16", "32", "64", "128", "256", "512 certified", "512 unresolved")
    shades = ("#ececec", "#dedede", "#c9c9c9", "#adadad",
              "#8c8c8c", "#6d6d6d", "#4e4e4e", "#222222")
    bottom = np.zeros(4)
    for index, label in enumerate(stack_labels):
        fractions = []
        for condition in CONDITIONS:
            subset = merged[merged.condition == condition]
            if label == "512 unresolved":
                mask = (subset.shots == 512) & (subset.certified == 0)
            elif label == "512 certified":
                mask = (subset.shots == 512) & (subset.certified == 1)
            else:
                mask = subset.shots == int(label)
            fractions.append(float(mask.mean()))
        fractions = np.asarray(fractions)
        axis.bar(range(4), fractions, bottom=bottom, width=.64,
                 color=shades[index], edgecolor="white", linewidth=.45,
                 hatch="////" if label == "512 unresolved" else None,
                 label=label, zorder=3)
        bottom += fractions
    axis.set(xticks=range(4), xticklabels=CONDITION_LABELS,
             ylabel="Fraction of decisions", ylim=(0, 1.03))
    axis.tick_params(axis="x", rotation=13)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .985),
                  ncol=4, frameon=False, fontsize=7.4, columnspacing=1.4)

    axis = axes[1, 1]
    _panel(axis, "d", "Joint-readout concentration radius")
    scalar_radii, observable_radii = [], []
    for seed in range(3):
        model = EntangledQuantumRouter(feature_count=feature_count, n_qubits=4, depth=2)
        model.load_state_dict(torch.load(training / f"quantum_seed{seed}.pt",
                                         map_location="cpu", weights_only=True))
        signs = model.observable_signs.detach().numpy().T
        weights = model.readout.weight.detach().numpy().ravel()
        scalar_range = np.ptp(signs @ weights)
        scalar_radii.append(scalar_range * np.sqrt(
            np.log(2 * len(LOOKS) / .05) / (2 * np.asarray(LOOKS))))
        observable_radii.append(np.abs(weights).sum() * np.minimum(
            2, np.sqrt(2 * np.log(2 * signs.shape[1] * len(LOOKS) / .05) /
                       np.asarray(LOOKS))))
    for array, color, linestyle in (
        (np.asarray(scalar_radii), INK, "-"),
        (np.asarray(observable_radii), MID, "--"),
    ):
        axis.fill_between(LOOKS, array.min(axis=0), array.max(axis=0),
                          color=color, alpha=.13, linewidth=0)
        axis.plot(LOOKS, array.mean(axis=0), color=color, linestyle=linestyle,
                  marker="o", ms=3.4, lw=1.4)
    axis.set(xlabel="Joint bitstring shots", ylabel="Readout-logit confidence radius",
             xscale="log", yscale="log")
    axis.set_xticks(LOOKS, labels=[str(value) for value in LOOKS])
    axis.text(.97, .92, "Solid  joint scalar\nDashed  separate observables",
              transform=axis.transAxes, ha="right", va="top", fontsize=7.4,
              bbox=dict(facecolor="white", edgecolor=PALE, pad=2))
    _save(figure, output, "figure3_measurement_frontier")


def _figure4(output: Path, decisions: list[pd.DataFrame],
             repeat: pd.DataFrame) -> None:
    figure, axes = _axes()
    figure.subplots_adjust(top=.84)
    day_sets = []
    for seed, frame in enumerate(decisions):
        daily = frame.assign(gain=frame.observable_bound_shots - frame.shots).groupby(
            "day", sort=True).agg(gain=("gain", "mean"),
                                  shots=("shots", "mean"),
                                  certified=("certified", "mean"))
        daily["model_seed"] = seed
        day_sets.append(daily.reset_index())
    day_data = pd.concat(day_sets, ignore_index=True)

    axis = axes[0, 0]
    _panel(axis, "a", "Daily paired shot savings")
    per_day = day_data.pivot(index="day", columns="model_seed", values="gain")
    day_index = np.arange(len(per_day))
    for seed, color, linestyle in ((0, PALE, ":"), (1, DARK, "--"), (2, MID, "-.")):
        axis.plot(day_index, per_day[seed], color=color, lw=.72,
                  linestyle=linestyle, alpha=.72, label=f"Model seed {seed}")
    axis.plot(day_index, per_day.mean(axis=1).rolling(9, center=True,
              min_periods=1).mean(), color=INK, lw=1.85, label="Nine-day mean")
    axis.axhline(0, color=INK, lw=.8)
    axis.set(xlabel="Held-out active-day index (calendar gaps omitted)",
             ylabel="Scalar-bound shots saved\nversus separate observables")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, 1.30), ncol=4,
                frameon=False, fontsize=7.2, columnspacing=.8)

    axis = axes[0, 1]
    _panel(axis, "b", "Paired savings by telemetry regime", grid="y")
    values = []
    for condition in CONDITIONS:
        merged = []
        for frame in decisions:
            subset = frame[frame.condition == condition]
            gain = (subset.observable_bound_shots - subset.shots).groupby(subset.day).mean()
            merged.extend(gain.to_list())
        values.append(np.asarray(merged))
    parts = axis.violinplot(values, positions=range(4), widths=.62,
                             showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(PALE)
        body.set_edgecolor(DARK)
        body.set_alpha(.85)
    parts["cmedians"].set_color(INK)
    parts["cmedians"].set_linewidth(1.35)
    axis.axhline(0, color=INK, linestyle="--", lw=.8)
    axis.set(xticks=range(4), xticklabels=CONDITION_LABELS,
             ylabel="Daily mean shots saved")
    axis.tick_params(axis="x", rotation=13)

    axis = axes[1, 0]
    _panel(axis, "c", "Measurement allocation by expert agreement", grid="")
    merged = pd.concat(decisions, ignore_index=True)
    overlaps = np.rint(merged.expert_top10_overlap.to_numpy() * 10).astype(int)
    levels = (0, *LOOKS)
    matrix = np.zeros((len(levels), 11), dtype=np.int32)
    for level, overlap in zip(merged.shots.to_numpy(), overlaps):
        matrix[levels.index(int(level)), overlap] += 1
    display = np.ma.masked_equal(matrix, 0)
    heatmap = axis.imshow(display, origin="lower", aspect="auto", cmap="Greys",
                          norm=LogNorm(vmin=1, vmax=matrix.max()), interpolation="none")
    axis.set(xticks=range(0, 11, 2),
             xticklabels=[f"{value / 10:.1f}" for value in range(0, 11, 2)],
             yticks=range(len(levels)), yticklabels=[str(value) for value in levels],
             xlabel="Fixed-expert top-ten set overlap", ylabel="Shots at stop")
    colorbar = figure.colorbar(heatmap, ax=axis, fraction=.045, pad=.025)
    colorbar.set_label("Decisions", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    axis = axes[1, 1]
    _panel(axis, "d", "Independent measurement repeat", grid="")
    first = decisions[0].groupby("day").shots.mean()
    second = repeat.groupby("day").shots.mean()
    shared_days = first.index.intersection(second.index)
    x = first.loc[shared_days].to_numpy()
    y = second.loc[shared_days].to_numpy()
    axis.scatter(x, y, s=19, color=DARK, alpha=.66, edgecolor="white",
                 linewidth=.35, rasterized=True)
    bounds = (min(x.min(), y.min()) - 10, max(x.max(), y.max()) + 10)
    axis.plot(bounds, bounds, color=INK, lw=.9, linestyle="--")
    axis.set(xlabel="Seed 0, measurement stream A\nmean shots per day",
             ylabel="Seed 0, measurement stream B\nmean shots per day",
             xlim=bounds, ylim=bounds)
    axis.set_aspect("equal", adjustable="box")
    _save(figure, output, "figure4_robustness_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/email-Eu-core-temporal.txt.gz"))
    parser.add_argument("--episodes", type=Path, default=Path("outputs/email_qrouter_v1/episodes.npz"))
    parser.add_argument("--training", type=Path, default=Path("outputs/email_qrouter_v1/train"))
    parser.add_argument("--decisions", type=Path, default=Path("outputs/email_qrouter_v2"))
    parser.add_argument("--output", type=Path, default=Path("email_quantum_paper/assets"))
    args = parser.parse_args()
    _style()
    episodes = load_episodes(args.episodes)
    with np.load(args.training / "predictions.npz", allow_pickle=False) as fit:
        predictions = {str(name): fit["weights"][index] for index, name in
                       enumerate(fit["weight_names"])}
        split = fit["split"]
    metrics = pd.read_csv(args.training / "test_metrics.csv")
    protocol = json.loads((args.training / "protocol.json").read_text())
    frame = pd.read_csv(args.data, sep=r"\s+", header=None, comment="#",
                        names=("src", "dst", "timestamp"))
    frame["day"] = frame.timestamp // 86400
    decisions = [_joined_decisions(pd.read_csv(args.decisions / folder / "decisions.csv"),
                                   episodes) for folder in SEED_DIRS]
    repeat = pd.read_csv(args.decisions / "shots_model0_repeat1" / "decisions.csv")
    expected = set(zip(decisions[0].episode, decisions[0].condition))
    if any(set(zip(item.episode, item.condition)) != expected for item in decisions[1:]):
        raise ValueError("model seeds must share the same held-out episode-condition keys")
    args.output.mkdir(parents=True, exist_ok=True)
    _dataset_table(frame, episodes, protocol, decisions).to_csv(
        args.output / "table1_data_protocol.csv", index=False)
    _summarize_models(metrics).to_csv(args.output / "table2_predictive_audit.csv", index=False)
    _resource_summary(decisions).to_csv(args.output / "table3_shots_certification.csv", index=False)
    _overlap_summary(decisions).to_csv(args.output / "table4_disagreement_strata.csv", index=False)
    _figure1(args.output, frame, episodes, decisions, args.training)
    _architecture(args.output)
    _figure2(args.output, metrics, episodes, predictions, split)
    _figure3(args.output, decisions, args.training, episodes.features.shape[-1])
    _figure4(args.output, decisions, repeat)
    manifest = {
        "source": episodes.audit["dataset"],
        "test_episodes": protocol["split_episodes"]["test"],
        "model_seeds": len(decisions), "measurement_repeat": "model seed 0 only",
        "table_files": [f"table{index}_{name}.csv" for index, name in (
            (1, "data_protocol"), (2, "predictive_audit"),
            (3, "shots_certification"), (4, "disagreement_strata"))],
        "figure_stems": ["figure1_temporal_quantum_method", "figure2_predictive_audit",
                         "figure3_measurement_frontier", "figure4_robustness_diagnostics"],
        "architecture_stem": "architecture_email_quantum_router",
        "cautions": [
            "All shot outcomes are ideal-circuit simulations, not quantum hardware runs.",
            "Four masks per sender-day and three model seeds are paired, not independent samples.",
            "Certified hit rate is selected-case performance, not improvement over all decisions.",
            "Quantum predictive superiority over the classical router is not established.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tables": 4, "figures": 4,
                      "panels_per_figure": 4}, indent=2))


if __name__ == "__main__":
    main()
