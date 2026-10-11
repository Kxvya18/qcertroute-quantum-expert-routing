"""Create dense publication figures for the quantum routing limits study."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "temporal": "#1f3b5b",
    "graph_only": "#b69a5b",
    "classical_moe": "#557a95",
    "quantum_moe_selected": "#0f766e",
    "quantum_no_graph_context": "#b85c38",
    "quantum_no_temporal_context": "#7c4d79",
    "quantum_no_cross_expert_context": "#8c8f88",
    "quantum_moe_local_only": "#356c9b",
}
LABELS = {
    "temporal": "Temporal",
    "graph_only": "Graph diffusion",
    "classical_moe": "Classical router",
    "quantum_moe_selected": "Quantum router",
    "quantum_no_graph_context": "No graph context",
    "quantum_no_temporal_context": "No temporal context",
    "quantum_no_cross_expert_context": "No cross context",
    "quantum_moe_local_only": "Local observables",
}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 220,
        "savefig.dpi": 360,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, count: int = 1000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), float(values.mean()), float(values.mean())
    draws = rng.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png")
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def figure_rank_spectra(results: pd.DataFrame, output: Path) -> None:
    models = ["temporal", "graph_only", "classical_moe", "quantum_moe_selected"]
    second = (
        ("history_random50", "Random half graph history loss")
        if "history_random50" in set(results.condition)
        else ("dns_only", "Name resolution only")
    )
    conditions = [("random50", "Random half query loss"), second]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.1), constrained_layout=True)
    bins = np.unique(np.round(np.geomspace(1, 400, 27)).astype(int))
    for col, (condition, title) in enumerate(conditions):
        ax = axes[0, col]
        work = results[results.condition.eq(condition)]
        for index, model in enumerate(models):
            ranks = work.loc[work.model.eq(model), "rank"].to_numpy(float)
            hist, edges = np.histogram(ranks, bins=bins)
            centers = np.sqrt(edges[:-1] * edges[1:])
            mass = hist / max(hist.sum(), 1)
            err = np.sqrt(np.maximum(hist, 1)) / max(hist.sum(), 1)
            baseline = len(models) - 1 - index
            height = 2.30 * mass
            visible = mass > 0
            ax.bar(centers[visible], height[visible], width=np.diff(edges)[visible], bottom=baseline,
                   align="center", color=COLORS[model], alpha=.72, edgecolor="white", linewidth=.18)
            ax.errorbar(centers[visible], baseline + height[visible], yerr=2.30 * err[visible],
                        fmt="none", ecolor="black", elinewidth=.46, capsize=1.0, alpha=.72)
        ax.set_xscale("log")
        ax.set_ylim(-.18, len(models) + .34)
        ax.set_yticks(range(len(models)), [LABELS[m] for m in models[::-1]])
        ax.set_xlabel("True destination rank")
        ax.set_ylabel("Separate model spectra" if col == 0 else "")
        ax.set_title(chr(97 + col) + "  " + title, loc="left")
        ax.grid(alpha=.22, which="both", axis="x")
        ax.text(.02, .03, "Bar height is probability mass\nError bars are Poisson uncertainty", transform=ax.transAxes,
                va="bottom", fontsize=7.1, bbox=dict(facecolor="white", edgecolor="0.45", alpha=.92, pad=2.3))

    for col, (condition, title) in enumerate(conditions):
        ax = axes[1, col]
        work = results[results.condition.eq(condition)]
        for model in models:
            ranks = np.sort(work.loc[work.model.eq(model), "rank"].to_numpy(float))
            survival = 1.0 - np.arange(len(ranks)) / len(ranks)
            ax.step(ranks, survival, where="post", color=COLORS[model], lw=1.7, label=LABELS[model])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.1)
        ax.set_xlabel("True destination rank")
        ax.set_ylabel("Survival probability" if col == 0 else "")
        survival_title = "c  Query loss rank survival" if col == 0 else (
            "d  History loss rank survival" if condition == "history_random50" else "d  DNS only rank survival"
        )
        ax.set_title(survival_title, loc="left")
        ax.grid(alpha=.22, which="both")
        if col == 1:
            ax.legend(frameon=True, edgecolor="0.3", loc="lower left")
    save(fig, output, "figure1_rank_spectra")


def figure_router_landscape(results: pd.DataFrame, output: Path) -> None:
    keys = ["fold", "seed", "episode", "timestamp", "condition"]
    temporal = results[results.model.eq("temporal")][keys + ["rank"]].rename(columns={"rank": "rank_t"})
    graph = results[results.model.eq("graph_only")][keys + ["rank"]].rename(columns={"rank": "rank_g"})
    quantum = results[results.model.eq("quantum_moe_selected")][keys + ["alpha"]]
    classical = results[results.model.eq("classical_moe")][keys + ["alpha"]].rename(columns={"alpha": "alpha_c"})
    joined = quantum.merge(classical, on=keys).merge(temporal, on=keys).merge(graph, on=keys)
    joined["graph_better"] = (joined.rank_g < joined.rank_t).astype(float)
    joined["regret"] = np.log1p(joined.rank_t) - np.log1p(joined.rank_g)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.1), constrained_layout=True)
    ax = axes[0, 0]
    hb = ax.hexbin(joined.regret, joined.alpha, C=joined.graph_better, reduce_C_function=np.mean,
                   gridsize=25, mincnt=1, cmap="RdYlBu_r", vmin=0, vmax=1, linewidths=.18, edgecolors="white")
    x = np.linspace(joined.regret.quantile(.01), joined.regret.quantile(.99), 180)
    ax.plot(x, 1.0 / (1.0 + np.exp(-x / .75)), "k--", lw=1.0, label="Cross fitted target")
    ax.set_xlabel("Cross fitted log rank regret")
    ax.set_ylabel("Quantum graph authority")
    ax.set_title("a  Quantum routing landscape", loc="left")
    ax.legend(frameon=True, edgecolor="0.3", loc="lower right")
    cb = fig.colorbar(hb, ax=ax, fraction=.046, pad=.03)
    cb.set_label("Observed graph advantage")

    ax = axes[0, 1]
    bins = np.linspace(0, 1, 11)
    for values, color, label, marker in [(joined.alpha.to_numpy(), COLORS["quantum_moe_selected"], "Quantum router", "o"),
                                         (joined.alpha_c.to_numpy(), COLORS["classical_moe"], "Classical router", "s")]:
        index = np.digitize(values, bins) - 1
        centers, observed, lower, upper = [], [], [], []
        for bucket in range(10):
            points = joined.graph_better.to_numpy()[index == bucket]
            if len(points) < 8:
                continue
            mean, lo, hi = bootstrap_mean(points, np.random.default_rng(101 + bucket))
            centers.append((bins[bucket] + bins[bucket + 1]) / 2)
            observed.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
        ax.errorbar(centers, observed, yerr=[lower, upper], fmt=marker + "-", color=color,
                    ms=4, lw=1.3, capsize=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=.9, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted graph authority")
    ax.set_ylabel("Observed graph advantage")
    ax.set_title("b  Reliability diagram", loc="left")
    ax.legend(frameon=True, edgecolor="0.3", loc="upper left")
    ax.grid(alpha=.22)

    ax = axes[1, 0]
    condition_order = ["complete", "random50", "hub50", "auth_only", "dns_only"]
    labels = ["Complete", "Query", "Hub", "Auth", "DNS"]
    if "history_random50" in set(joined.condition):
        condition_order.append("history_random50")
        labels.append("History")
    work = joined.groupby("condition", sort=False).agg(alpha=("alpha", "mean"), regret=("regret", "mean"), graph=("graph_better", "mean")).reindex(condition_order)
    x = np.arange(len(labels))
    ax.plot(x, work.alpha, color=COLORS["quantum_moe_selected"], marker="o", lw=2, label="Quantum authority")
    ax.plot(x, work.graph, color="#b15d45", marker="s", lw=1.7, label="Observed graph advantage")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Episode proportion")
    ax.set_title("c  Regime aligned authority", loc="left")
    ax.grid(alpha=.22)
    ax.legend(frameon=True, edgecolor="0.3", loc="lower right")

    ax = axes[1, 1]
    ablation = ["quantum_moe_selected", "quantum_no_graph_context", "quantum_no_temporal_context", "quantum_no_cross_expert_context", "quantum_moe_local_only"]
    metrics = []
    for model in ablation:
        data = results[results.model.eq(model)].merge(temporal, on=keys).merge(graph, on=keys)
        data = data[data.rank_t.ne(data.rank_g)]
        truth = (data.rank_g < data.rank_t).to_numpy(float)
        score = data.alpha.to_numpy(float)
        positive, negative = score[truth == 1], score[truth == 0]
        auc = (positive[:, None] > negative).mean() + .5 * (positive[:, None] == negative).mean()
        metrics.append(auc)
    labels = ["Full", "No graph", "No temporal", "No cross", "Local"]
    ax.barh(np.arange(len(labels)), metrics, color=[COLORS[m] for m in ablation], alpha=.9)
    ax.axvline(.5, color="k", lw=.9, ls="--")
    for i, value in enumerate(metrics):
        ax.text(value + .005, i, f"{value:.3f}", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlim(.5, .9)
    ax.set_xlabel("Routing AUC")
    ax.set_title("d  Context and observable ablation", loc="left")
    ax.grid(axis="x", alpha=.22)
    save(fig, output, "figure2_router_landscape")


def figure_quantum_stability(results: pd.DataFrame, resources: pd.DataFrame, output: Path) -> None:
    qmodel = "quantum_moe_selected"
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.1), constrained_layout=True)
    ax = axes[0, 0]
    shots = [32, 64, 128, 256, 512, 1024]
    display = [("complete", "Complete", "#1f3b5b"), ("random50", "Query loss", "#d97706")]
    if "history_random50" in set(results.condition):
        display.append(("history_random50", "History loss", "#0f766e"))
    else:
        display.append(("hub50", "Hub loss", "#0f766e"))
    shot_low, shot_high = [], []
    for condition, label, color in display:
        means, lows, highs = [], [], []
        for shot in shots:
            work = results[(results.condition.eq(condition)) & (results.model.eq(f"quantum_shot_{shot}"))].mrr.to_numpy()
            mean, low, high = bootstrap_mean(work, np.random.default_rng(shot))
            means.append(mean)
            lows.append(low)
            highs.append(high)
        shot_low.extend(lows)
        shot_high.extend(highs)
        ax.plot(shots, means, color=color, marker="o", lw=1.65, label=label)
        ax.fill_between(shots, lows, highs, color=color, alpha=.13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(shots, [str(v) for v in shots])
    span = max(shot_high) - min(shot_low)
    padding = max(0.02, 0.10 * span)
    ax.set_ylim(min(shot_low) - padding, max(shot_high) + padding)
    ax.set_xlabel("Measurement shots")
    ax.set_ylabel("Mean reciprocal rank")
    ax.set_title("a  Finite shot convergence", loc="left")
    ax.legend(frameon=True, edgecolor="0.3", loc="lower left")
    ax.grid(alpha=.22)

    ax = axes[0, 1]
    noise_models = sorted([m for m in results.model.unique() if m.startswith("quantum_noise_")])
    rows = []
    for model in noise_models:
        part = model.replace("quantum_noise_p", "").replace("_r", " ")
        p, r = [float(v) for v in part.split()]
        baseline = results[(results.model.eq(qmodel))].groupby(["fold", "seed", "episode", "condition"]).mrr.mean()
        noisy = results[results.model.eq(model)].groupby(["fold", "seed", "episode", "condition"]).mrr.mean()
        rows.append((p, r, float((noisy - baseline).mean())))
    table = pd.DataFrame(rows, columns=["p", "r", "delta"]).sort_values(["p", "r"])
    conditions = [("complete", "Complete", "#1f3b5b"), ("random50", "Query loss", "#d97706"),
                  ("hub50", "Hub loss", "#0f766e"), ("auth_only", "Auth only", "#7c4d79"),
                  ("dns_only", "DNS only", "#b85c38")]
    if "history_random50" in set(results.condition):
        conditions.append(("history_random50", "History loss", "#50585f"))
    labels = [f"p {p:.3f}\nr {r:.3f}" for p, r in table[["p", "r"]].itertuples(index=False)]
    for condition, label, color in conditions:
        baseline = results[(results.model.eq(qmodel)) & (results.condition.eq(condition))].groupby(["fold", "seed", "episode"]).mrr.mean()
        deltas = []
        for p, r in table[["p", "r"]].itertuples(index=False):
            name = f"quantum_noise_p{p:.3f}_r{r:.3f}"
            noisy = results[(results.model.eq(name)) & (results.condition.eq(condition))].groupby(["fold", "seed", "episode"]).mrr.mean()
            deltas.append(float((noisy - baseline).mean()))
        ax.plot(range(len(deltas)), deltas, marker="o", ms=3.5, lw=1.25, color=color, label=label)
    ax.axhline(0, color="black", lw=.9, ls="--")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("Depolarization and readout error schedule")
    ax.set_ylabel("Delta MRR from ideal")
    ax.set_title("b  Noisy circuit sensitivity", loc="left")
    ax.grid(alpha=.22)
    ax.legend(frameon=True, edgecolor="0.3", fontsize=6.5, loc="upper left", ncol=2)

    ax = axes[1, 0]
    grouped = resources.groupby("model", as_index=False).agg(params=("parameters", "mean"), seconds=("training_seconds", "mean"), loss=("validation_loss", "mean"))
    quantum = grouped[grouped.model.str.startswith("quantum")]
    preferred_labels = {
        "quantum_moe_selected": "Selected Q 4x2",
        "quantum_moe_q2d2": "Q 2x2",
        "quantum_moe_local_only": "Local only",
        "quantum_reservoir_r4": "Reservoir four",
    }
    for row in quantum.itertuples():
        selected = row.model == qmodel
        ax.scatter(row.seconds, row.loss, s=30 + row.params * 2.2,
                   color="#0f766e" if selected else "#a9ada8", edgecolor="black", linewidth=.35, zorder=3)
        if row.model in preferred_labels:
            ax.annotate(preferred_labels[row.model], (row.seconds, row.loss), textcoords="offset points", xytext=(4, 3), fontsize=7.1)
    classical = grouped[grouped.model.eq("classical_moe")].iloc[0]
    ax.scatter(classical.seconds, classical.loss, s=30 + classical.params * 2.2, marker="s", color="#557a95", edgecolor="black", linewidth=.35, zorder=3, label="Classical")
    ax.set_xlabel("Mean training seconds per fold and seed")
    ax.set_ylabel("Validation routing loss")
    ax.set_title("c  Architecture resource frontier", loc="left")
    ax.grid(alpha=.22)
    ax.legend(frameon=True, edgecolor="0.3", loc="lower right")

    ax = axes[1, 1]
    select = resources[resources.model.eq(qmodel)].selected_from.value_counts().sort_values(ascending=False)
    pretty = [x.replace("quantum_moe_", "Q ").replace("quantum_reservoir_", "R ") for x in select.index]
    ax.bar(np.arange(len(select)), select.to_numpy(), color=["#0f766e", "#356c9b", "#b69a5b", "#b85c38"][:len(select)], edgecolor="black", linewidth=.35)
    for i, count in enumerate(select):
        ax.text(i, count + .08, str(count), ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(select)), pretty, rotation=17, ha="right")
    ax.set_ylim(0, max(select) + 1.2)
    ax.set_ylabel("Fold seed selections")
    ax.set_title("d  Validation selected quantum family", loc="left")
    ax.grid(axis="y", alpha=.22)
    save(fig, output, "figure3_quantum_stability")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    args.output.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results / "interdependent_qrag_results.csv")
    resources = pd.read_csv(args.results / "interdependent_qrag_resources.csv")
    figure_rank_spectra(results, args.output)
    figure_router_landscape(results, args.output)
    figure_quantum_stability(results, resources, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
