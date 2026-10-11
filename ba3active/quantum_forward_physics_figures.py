"""Generate dense multi-panel figures from the observed LANL quantum-routing audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


COLORS = {
    "complete": "#154c79",
    "random50": "#d05a45",
    "hub50": "#bc8a20",
    "auth_only": "#188977",
    "dns_only": "#7d5a99",
}
NAMES = {
    "complete": "Complete",
    "random50": "Random half",
    "hub50": "Hub half",
    "auth_only": "Authentication only",
    "dns_only": "Name resolution only",
}
ORDER = list(NAMES)
QMODEL = "quantum_moe_selected"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "axes.labelsize": 8.6,
        "axes.titlesize": 9.1,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3,
        "figure.dpi": 170,
        "savefig.dpi": 340,
    })


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#d7d7d7", linewidth=.45, alpha=.75)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def qrows(results: pd.DataFrame) -> pd.DataFrame:
    return results[results.model.eq(QMODEL)].copy()


def join_experts(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold", "seed", "episode", "timestamp", "condition"]
    keep = keys + ["rank"]
    temporal = results[results.model.eq("temporal")][keep].rename(columns={"rank": "temporal_rank"})
    graph = results[results.model.eq("graph_only")][keep].rename(columns={"rank": "graph_rank"})
    quantum = qrows(results)[keys + ["alpha", "temporal_entropy", "graph_entropy", "top10_overlap"]]
    joined = quantum.merge(temporal, on=keys).merge(graph, on=keys)
    joined["regret"] = np.log1p(joined.temporal_rank) - np.log1p(joined.graph_rank)
    return joined


def figure_protocol(results: pd.DataFrame, output: Path) -> None:
    q = qrows(results)
    fig = plt.figure(figsize=(7.35, 7.35), constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=(1.02, .83, 1.22))

    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 15); ax.set_ylim(-.2, 4.6); ax.axis("off")
    ax.text(.05, 4.25, "a  Chronological evidence construction", weight="bold", fontsize=10)
    for y in (3.45, 2.45, 1.45, .45):
        ax.plot([.4, 14.6], [y, y], color="#aeb4b7", lw=.8, zorder=0)
    for x in (1.1, 2.0, 2.9, 5.5, 6.4, 7.3, 10.5, 11.4, 12.3, 13.2):
        for y in (3.45, 2.45, .45):
            ax.scatter(x, y, s=20, color="#b7b9ba", zorder=2)
    ax.add_patch(plt.Rectangle((2.0, 1.75), 4.45, 1.45, facecolor="#5ba69b", edgecolor="#075c59", lw=1.3, alpha=.72))
    ax.add_patch(plt.Rectangle((6.45, .75), 4.95, 1.45, facecolor="#eda862", edgecolor="#aa5d1e", lw=1.3, alpha=.72))
    for x, y in ((2,2.45),(2.55,3.45),(2.55,2.45),(3.5,2.45),(4.4,2.45),(5.5,3.45),(5.5,2.45),(6.05,2.45),
                 (6.45,1.45),(7.0,1.45),(7.55,1.45),(8.15,1.45),(9.05,1.45),(9.6,1.45),(10.5,1.45),(10.9,1.45)):
        ax.scatter(x, y, s=29, color="#24272a", zorder=3)
    ax.annotate("retained edge history", (4.2, 3.12), ha="center", va="bottom", color="#075c59", fontsize=8)
    ax.annotate("next destination label", (8.9, .68), ha="center", va="top", color="#974b13", fontsize=8)
    ax.annotate("earlier windows", (.75, 4.0), fontsize=7.5, color="#53575a")
    ax.annotate("observation window", (6.85, 4.0), fontsize=7.5, color="#53575a")
    ax.annotate("target bucket", (11.1, 4.0), fontsize=7.5, color="#53575a")
    ax.arrow(1.1, -.02, 12.2, 0, width=.012, head_width=.13, head_length=.28, length_includes_head=True, color="#3f474c")
    ax.text(7.3, -.16, "time", ha="center", va="top", fontsize=8)

    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 15); ax.set_ylim(0, 4.1); ax.axis("off")
    ax.text(.05, 3.85, "b  Episode level quantum authority rather than candidate level classification", weight="bold", fontsize=10)
    labels = ["missingness", "source history", "temporal uncertainty", "graph uncertainty", "top ten overlap"]
    for i, label in enumerate(labels):
        y = 3.2 - i * .48
        ax.add_patch(plt.Rectangle((.55, y - .21), 2.55, .34, facecolor="#f6f6f6", edgecolor="#8d969c", lw=.7))
        ax.text(.68, y - .04, label, va="center", fontsize=7.4)
        for j in range(6):
            ax.add_patch(plt.Rectangle((1.92 + .16*j, y - .16), .1, .2, facecolor="#138c86" if (i+j)%3 else "#e5985f", edgecolor="none"))
    ax.annotate("context vector", (5.0, 2.0), (3.25, 2.0), arrowprops=dict(arrowstyle="->", lw=1.0, color="#59636a"), va="center", fontsize=8)
    for wire in range(4):
        y = 3.15 - .72*wire
        ax.plot([5.1, 10.5], [y, y], color="#2f3336", lw=.8)
        for x, color, text in ((5.55,"#75b8b2","R"),(6.55,"#f0b076","Y"),(7.65,"#75b8b2","R"),(8.65,"#f0b076","Y")):
            ax.add_patch(plt.Rectangle((x, y-.17), .42, .34, facecolor=color, edgecolor="#363b3f", lw=.55))
            ax.text(x+.21, y, text, ha="center", va="center", fontsize=6.8, weight="bold")
        ax.scatter(9.7, y, marker="o", s=18, color="#303438")
    ax.text(7.9, .14, "two reuploading layers and local parity measurements", ha="center", fontsize=7.2)
    ax.annotate("measured context", (11.62, 2.0), (10.55, 2.0), arrowprops=dict(arrowstyle="->", lw=1.0, color="#59636a"), va="center", fontsize=8)
    ax.add_patch(plt.Rectangle((11.75, .78), 2.55, 2.35, facecolor="#f5f7f8", edgecolor="#8d969c", lw=.8))
    ax.text(13.02, 2.75, "bounded authority", ha="center", fontsize=8, weight="bold")
    xx = np.linspace(12.05, 13.98, 100)
    ax.plot(xx, 1.55 + .72/(1 + np.exp(-4*(xx-13.0))), color="#bd3f47", lw=1.55)
    ax.text(13.02, .98, r"$\alpha_e$ controls probability mixing", ha="center", fontsize=7.2)

    ax = fig.add_subplot(gs[2, 0])
    panel(ax, "c", "Authority map over observed episode context")
    hb = ax.hexbin(q.temporal_entropy, q.top10_overlap, C=q.alpha, reduce_C_function=np.mean,
                   gridsize=19, mincnt=2, cmap="RdYlBu_r", linewidths=.2, edgecolors="white")
    ax.set_xlabel("Temporal posterior entropy")
    ax.set_ylabel("Top ten expert overlap")
    cb = fig.colorbar(hb, ax=ax, pad=.02, fraction=.05)
    cb.set_label("Mean quantum authority", fontsize=7.6)
    cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[2, 1])
    panel(ax, "d", "Authority distributions across observation regimes")
    values = [q[q.condition.eq(c)].alpha.to_numpy() for c in ORDER]
    vp = ax.violinplot(values, positions=np.arange(1, 6), widths=.72, showmeans=True, showextrema=False)
    for body, condition in zip(vp["bodies"], ORDER):
        body.set_facecolor(COLORS[condition]); body.set_alpha(.55); body.set_edgecolor("#42484d")
    vp["cmeans"].set_color("#202326"); vp["cmeans"].set_linewidth(1.2)
    rng = np.random.default_rng(13)
    for i, (condition, value) in enumerate(zip(ORDER, values), 1):
        take = rng.choice(value, min(len(value), 90), replace=False)
        ax.scatter(i + rng.normal(0, .08, len(take)), take, s=4, alpha=.18, color=COLORS[condition], edgecolor="none")
    ax.set_xticks(range(1, 6), ["Full", "Rand", "Hub", "Auth", "DNS"])
    ax.set_ylabel("Quantum authority")
    ax.set_ylim(0, 1.04)
    save(fig, output, "figure1_routing_protocol")


def figure_landscape(results: pd.DataFrame, output: Path) -> None:
    joined = join_experts(results)
    q = qrows(results)
    fig = plt.figure(figsize=(7.35, 6.6), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=(1.26, .94))

    ax = fig.add_subplot(gs[0, :], projection="3d")
    ax.set_title("a  Quantum authority response surface from observed episode descriptors", loc="left", fontweight="bold", pad=5)
    x = q.temporal_entropy.to_numpy(); y = q.graph_entropy.to_numpy(); z = q.alpha.to_numpy()
    xb = np.linspace(np.quantile(x, .02), np.quantile(x, .98), 12)
    yb = np.linspace(np.quantile(y, .02), np.quantile(y, .98), 12)
    points = []
    for i in range(len(xb)-1):
        for j in range(len(yb)-1):
            select = (x >= xb[i]) & (x < xb[i+1]) & (y >= yb[j]) & (y < yb[j+1])
            if select.sum() >= 6:
                points.append(((xb[i]+xb[i+1])/2, (yb[j]+yb[j+1])/2, z[select].mean()))
    pts = np.asarray(points)
    surface = ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], cmap="RdBu", linewidth=.18, edgecolor="#eeeeee", alpha=.92)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, color="#30363a", alpha=.35)
    ax.set_xlabel("Temporal entropy", labelpad=5); ax.set_ylabel("Graph entropy", labelpad=5); ax.set_zlabel("Authority", labelpad=5)
    ax.set_zlim(0, 1); ax.view_init(elev=27, azim=-125)
    cb = fig.colorbar(surface, ax=ax, shrink=.56, pad=.08); cb.set_label("Mean authority", fontsize=7.5); cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "b", "Cross fitted regret response")
    joined["bin"] = pd.qcut(joined.regret, q=14, duplicates="drop")
    work = joined.groupby("bin", observed=True).agg(regret=("regret", "mean"), alpha=("alpha", "mean"), count=("alpha", "size"), std=("alpha", "std")).reset_index(drop=True)
    se = work["std"].fillna(0).to_numpy() / np.sqrt(work["count"].to_numpy())
    ax.fill_between(work.regret, work.alpha-se, work.alpha+se, color="#65a7a0", alpha=.25, linewidth=0)
    ax.plot(work.regret, work.alpha, color="#087f78", marker="o", ms=3.5, lw=1.6, label="Observed quantum router")
    ax.axvline(0, color="#7a7d80", lw=.8, ls="--")
    ax.set_xlabel("Cross fitted log rank regret")
    ax.set_ylabel("Mean authority")
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "c", "Condition level authority follows graph utility")
    merged = results[results.model.isin((QMODEL, "temporal", "graph_only"))].pivot_table(
        index=["fold", "seed", "episode", "condition"], columns="model", values="mrr", aggfunc="mean").reset_index()
    rows = []
    for condition in ORDER:
        gain = (merged[merged.condition.eq(condition)].graph_only - merged[merged.condition.eq(condition)].temporal)
        authority = q[q.condition.eq(condition)].alpha
        rows.append((condition, gain.mean(), gain.std()/np.sqrt(len(gain)), authority.mean(), authority.std()/np.sqrt(len(authority))))
    offsets = {"complete": (5, 5), "random50": (5, 5), "hub50": (5, 5), "auth_only": (5, -10), "dns_only": (5, 5)}
    for condition, gain, xerr, authority, yerr in rows:
        ax.errorbar(gain, authority, xerr=xerr, yerr=yerr, fmt="o", ms=6, capsize=2.5,
                    color=COLORS[condition], markeredgecolor="black", markeredgewidth=.35)
        ax.annotate(NAMES[condition], (gain, authority), xytext=offsets[condition], textcoords="offset points", fontsize=7.1)
    ax.axvline(0, color="#74787c", lw=.75, ls="--")
    ax.set_xlabel("Graph MRR minus temporal MRR")
    ax.set_ylabel("Mean quantum authority")
    ax.set_ylim(.45, 1.01)
    save(fig, output, "figure2_authority_landscape")


def figure_stability(results: pd.DataFrame, resources: pd.DataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(7.35, 6.6), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=(1.08, .92))
    shots = [32, 64, 128, 256, 512, 1024]

    ax = fig.add_subplot(gs[0, :])
    panel(ax, "a", "Finite shot deviation from exact quantum inference")
    for condition in ORDER:
        means = []
        exact = results[(results.condition.eq(condition)) & (results.model.eq(QMODEL))]
        keys = ["fold", "seed", "episode", "condition"]
        exact = exact.set_index(keys).mrr
        for shot in shots:
            values = results[(results.condition.eq(condition)) & (results.model.eq(f"quantum_shot_{shot}"))].set_index(keys).mrr
            means.append(np.abs(values - exact.reindex(values.index)).mean() * 1e6)
        ax.plot(shots, means, marker="o", ms=3, lw=1.4, color=COLORS[condition], label=NAMES[condition])
    ax.set_xscale("log", base=2); ax.set_xticks(shots, [str(v) for v in shots])
    ax.set_xlabel("Simulated measurement shots")
    ax.set_ylabel(r"Mean absolute MRR drift $\times 10^{-6}$")
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.01), columnspacing=1.0)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "b", "Noisy circuit sensitivity")
    specs = [(0., 0.), (.005, .010), (.010, .020), (.020, .030), (.050, .050)]
    labels = ["ideal", ".005", ".010", ".020", ".050"]
    base = results[results.model.eq(QMODEL)].groupby(["fold", "seed", "episode", "condition"]).mrr.mean()
    for condition in ORDER:
        drift = []
        for p, r in specs:
            name = f"quantum_noise_p{p:.3f}_r{r:.3f}"
            noisy = results[(results.model.eq(name)) & (results.condition.eq(condition))].groupby(["fold", "seed", "episode", "condition"]).mrr.mean()
            paired = noisy - base.reindex(noisy.index)
            drift.append(paired.mean())
        ax.plot(range(len(specs)), drift, marker="o", ms=3, lw=1.35, color=COLORS[condition])
    ax.axhline(0, color="#777b7e", lw=.75, ls="--")
    ax.set_xticks(range(len(specs)), labels)
    ax.set_xlabel("Depolarizing probability")
    ax.set_ylabel("MRR shift from ideal")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "c", "Validation accuracy and computational cost")
    work = resources.groupby("model", as_index=False).agg(params=("parameters", "mean"), seconds=("training_seconds", "mean"), loss=("validation_loss", "mean"))
    pretty = {"classical_moe":"Classical", "quantum_moe_q2d2":"Q two by two", "quantum_moe_q4d1":"Q four by one",
              "quantum_moe_q4d2":"Q four by two", "quantum_moe_q4d3":"Q four by three", "quantum_moe_local_only":"Local only",
              "quantum_reservoir_r2":"Reservoir two", "quantum_reservoir_r4":"Reservoir four", "quantum_moe_selected":"Selected"}
    work = work[work.model.isin(pretty)]
    for row in work.itertuples():
        is_selected = row.model == QMODEL
        color = "#087f78" if is_selected else ("#c94c45" if row.model == "classical_moe" else "#9aa5a9")
        ax.scatter(row.seconds, row.loss, s=55 if is_selected else 35, color=color, edgecolor="#303437", linewidth=.45, zorder=3)
        if row.model in {QMODEL, "classical_moe", "quantum_reservoir_r4", "quantum_moe_q2d2"}:
            ax.annotate(pretty[row.model], (row.seconds, row.loss), xytext=(4, 3), textcoords="offset points", fontsize=6.9)
    ax.set_xlabel("Training seconds per fold and seed")
    ax.set_ylabel("Validation loss")
    ax.set_xscale("log")
    save(fig, output, "figure3_stability_audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    style()
    results = pd.read_csv(args.results)
    resources = pd.read_csv(args.resources)
    figure_protocol(results, args.output)
    figure_landscape(results, args.output)
    figure_stability(results, resources, args.output)


if __name__ == "__main__":
    main()
