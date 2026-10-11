"""Statistical audit and publication figures for Q-RAG-MoE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


CONDITIONS = ("complete", "auth_only", "dns_only", "random50", "hub50")
LABELS = {
    "complete": "Complete",
    "auth_only": "Auth only",
    "dns_only": "DNS only",
    "random50": "Random 50%",
    "hub50": "Hub loss 50%",
    "temporal": "Temporal",
    "fixed_curvature": "Fixed curvature",
    "classical_router": "Classical router",
    "quantum_router": "Quantum router",
    "quantum_256shot": "Quantum, 256 shots",
}
COLORS = {
    "temporal": "#20364f",
    "fixed_curvature": "#b68a3c",
    "classical_router": "#b65f42",
    "quantum_router": "#147d75",
    "quantum_256shot": "#6a8794",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8.3,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.3,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.75,
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", pad=6)
    ax.grid(True, color="#d8d8d3", linewidth=0.45, alpha=0.65)


def paired(frame, first, second, filters=None):
    work = frame.copy()
    for key, value in (filters or {}).items():
        work = work[work[key] == value]
    keys = [key for key in ("beta", "m", "fold", "seed", "episode", "condition") if key in work]
    pivot = work.pivot_table(index=keys, columns="model", values="mrr", aggfunc="mean").reset_index()
    pivot["delta"] = pivot[first] - pivot[second]
    return pivot


def two_axis_interval(frame, value="mrr", repetitions=4000, seed=2026):
    pivot = frame.pivot_table(index="episode", columns="seed", values=value, aggfunc="mean").to_numpy(dtype=float)
    mean = float(np.nanmean(pivot))
    rng = np.random.default_rng(seed)
    sample = np.empty(repetitions)
    for index in range(repetitions):
        rows = rng.integers(0, pivot.shape[0], pivot.shape[0])
        columns = rng.integers(0, pivot.shape[1], pivot.shape[1])
        sample[index] = np.nanmean(pivot[np.ix_(rows, columns)])
    return mean, float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def claim_table(controlled, lanl):
    rows = []
    for dataset, frame in (("controlled", controlled), ("LANL", lanl)):
        for condition in CONDITIONS:
            for first, second, comparison in (
                ("quantum_router", "temporal", "Q-T"),
                ("quantum_router", "classical_router", "Q-C"),
                ("quantum_256shot", "quantum_router", "S-Q"),
            ):
                delta = paired(frame, first, second, {"condition": condition})
                mean, low, high = two_axis_interval(delta, "delta", seed=1300 + len(rows))
                rows.append({
                    "dataset": dataset,
                    "condition": condition,
                    "comparison": comparison,
                    "episodes": int(delta.episode.nunique()),
                    "seeds": int(delta.seed.nunique()),
                    "delta_mrr": mean,
                    "ci_low": low,
                    "ci_high": high,
                })
    return pd.DataFrame(rows)


def summary_table(frame, dataset):
    rows = []
    group_keys = ["condition", "model"]
    if "beta" in frame:
        group_keys = ["beta", *group_keys]
    for key, cell in frame.groupby(group_keys):
        key = key if isinstance(key, tuple) else (key,)
        values = dict(zip(group_keys, key))
        mean, low, high = two_axis_interval(cell, "mrr", seed=2200 + len(rows))
        rows.append({
            "dataset": dataset,
            **values,
            "episodes": int(cell.episode.nunique()),
            "seeds": int(cell.seed.nunique()),
            "mrr": mean,
            "ci_low": low,
            "ci_high": high,
            "hit10": cell.hit10.mean(),
            "nll": cell.nll.mean(),
            "alpha": cell.alpha.mean(),
        })
    return pd.DataFrame(rows)


def figure_controlled(frame, claims, path):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    ax = axes[0, 0]
    for model in ("temporal", "classical_router", "quantum_router", "quantum_256shot"):
        group = frame[(frame.condition == "hub50") & (frame.model == model)]
        means = group.groupby("beta").mrr.mean().reindex((0.0, 1.0, 2.0))
        ax.plot(means.index, means, marker="o", linewidth=1.8, color=COLORS[model], label=LABELS[model])
    ax.set_xticks((0, 1, 2))
    ax.set_xlabel(r"Growth concentration $\beta$")
    ax.set_ylabel("Mean reciprocal rank")
    ax.legend(frameon=False)
    panel(ax, "a", "Hub-loss mechanism response")

    ax = axes[0, 1]
    q = frame.pivot_table(index=["beta", "seed", "episode", "condition"], columns="model", values="mrr").reset_index()
    q["delta"] = q.quantum_router - q.classical_router
    phase = q.groupby(["beta", "condition"]).delta.mean().unstack().reindex(index=(0.0, 1.0, 2.0), columns=CONDITIONS)
    bound = max(abs(float(phase.min().min())), abs(float(phase.max().max())), 0.01)
    image = ax.imshow(phase, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    for row in range(phase.shape[0]):
        for column in range(phase.shape[1]):
            ax.text(column, row, f"{phase.iloc[row, column]:+.3f}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(3), (r"$\beta=0$", r"$\beta=1$", r"$\beta=2$"))
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[item] for item in CONDITIONS], rotation=25, ha="right")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=r"$\Delta$MRR, quantum minus classical")
    panel(ax, "b", "Router-comparison phase map")

    ax = axes[1, 0]
    for model, marker in (("classical_router", "s"), ("quantum_router", "o")):
        for condition in ("complete", "random50", "hub50"):
            group = frame[(frame.model == model) & (frame.condition == condition)]
            mean = group.groupby("beta").alpha.mean().reindex((0.0, 1.0, 2.0))
            ax.plot(mean.index, mean, marker=marker, linewidth=1.4, color=COLORS[model],
                    linestyle={"complete": "-", "random50": "--", "hub50": ":"}[condition],
                    label=f"{LABELS[model]}, {LABELS[condition]}")
    ax.set_xticks((0, 1, 2))
    ax.set_ylim(-0.01, 0.42)
    ax.set_xlabel(r"Growth concentration $\beta$")
    ax.set_ylabel(r"Mean admission $\alpha$")
    ax.legend(frameon=False, ncol=2, fontsize=6.6)
    panel(ax, "c", "Conservative quantum admission")

    ax = axes[1, 1]
    paired_shot = paired(frame, "quantum_256shot", "quantum_router")
    difference = paired_shot.delta.to_numpy()
    bins = np.linspace(min(-0.03, difference.min()), max(0.03, difference.max()), 35)
    ax.hist(difference, bins=bins, color="#6a8794", alpha=0.72, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#222222", linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta$MRR, 256-shot minus ideal quantum")
    ax.set_ylabel("Episode-seed count")
    ax.text(0.03, 0.93, f"mean absolute shift = {np.mean(np.abs(difference)):.4f}", transform=ax.transAxes, va="top")
    panel(ax, "d", "Finite-shot stability")
    fig.suptitle("Q-RAG-MoE controlled quantum-routing audit", fontsize=12, fontweight="bold")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def figure_lanl(frame, claims, path):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(CONDITIONS))
    width = 0.19
    models = ("temporal", "classical_router", "quantum_router", "quantum_256shot")
    for offset, model in enumerate(models):
        means, lows, highs = [], [], []
        for condition in CONDITIONS:
            mean, low, high = two_axis_interval(frame[(frame.model == model) & (frame.condition == condition)], seed=3100 + offset)
            means.append(mean); lows.append(low); highs.append(high)
        means, lows, highs = map(np.asarray, (means, lows, highs))
        location = x + (offset - 1.5) * width
        ax.bar(location, means, width, color=COLORS[model], label=LABELS[model])
        ax.errorbar(location, means, yerr=(means - lows, highs - means), fmt="none", ecolor="#202020", capsize=1.6, linewidth=0.65)
    ax.set_xticks(x, [LABELS[item] for item in CONDITIONS], rotation=28, ha="right")
    ax.set_ylabel("Mean reciprocal rank")
    ax.legend(frameon=False, fontsize=6.4)
    panel(ax, "a", "Chronological LANL transfer")

    ax = axes[1]
    subset = claims[(claims.dataset == "LANL") & (claims.comparison == "Q-C")].set_index("condition").reindex(CONDITIONS)
    mean = subset.delta_mrr.to_numpy(); low = subset.ci_low.to_numpy(); high = subset.ci_high.to_numpy()
    ax.bar(x, mean, color=np.where(mean >= 0, COLORS["quantum_router"], COLORS["classical_router"]), width=0.65)
    ax.errorbar(x, mean, yerr=(mean - low, high - mean), fmt="none", ecolor="#202020", capsize=2, linewidth=0.8)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, [LABELS[item] for item in CONDITIONS], rotation=28, ha="right")
    ax.set_ylabel(r"Quantum minus classical $\Delta$MRR")
    panel(ax, "b", "Matched-budget router comparison")

    ax = axes[2]
    pivot = frame.pivot_table(index=["fold", "seed", "episode", "condition"], columns="model", values="mrr").reset_index()
    ax.scatter(pivot.quantum_router, pivot.quantum_256shot, s=20, color=COLORS["quantum_router"], alpha=0.72, edgecolor="white", linewidth=0.4)
    limit = max(float(pivot.quantum_router.max()), float(pivot.quantum_256shot.max()), 0.05)
    ax.plot((0, limit), (0, limit), linestyle="--", color="#555555", linewidth=0.9)
    ax.set_xlabel("Ideal quantum MRR")
    ax.set_ylabel("256-shot quantum MRR")
    ax.text(0.04, 0.94, "3 qubits\n2 layers\n16 parameters\n1 circuit per episode", transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#edf2f1", "edgecolor": "#7b8b89"})
    panel(ax, "c", "Shot-interface agreement")
    fig.suptitle("Q-RAG-MoE real-network transfer and resource audit", fontsize=11.5, fontweight="bold")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    controlled = pd.read_csv(args.results / "qrag_controlled_results.csv")
    lanl = pd.read_csv(args.results / "qrag_lanl_results.csv")
    claims = claim_table(controlled, lanl)
    controlled_summary = summary_table(controlled, "controlled")
    lanl_summary = summary_table(lanl, "LANL")
    claims.to_csv(args.output / "qrag_claims.csv", index=False)
    controlled_summary.to_csv(args.output / "qrag_controlled_table.csv", index=False)
    lanl_summary.to_csv(args.output / "qrag_lanl_table.csv", index=False)
    figure_controlled(controlled, claims, args.output / "figure_qrag_controlled.png")
    figure_lanl(lanl, claims, args.output / "figure_qrag_lanl.png")
    audit = {
        "model": "Q-RAG-MoE",
        "quantum_parameters": 16,
        "classical_router_parameters": 17,
        "qubits": 3,
        "circuit_depth": 2,
        "shots": 256,
        "quantum_hardware_used": False,
        "statevector_simulation": True,
        "controlled_test_episodes": int(controlled.episode.nunique()),
        "lanl_test_episodes": int(lanl.episode.nunique()),
        "seeds": sorted(int(seed) for seed in controlled.seed.unique()),
        "claims": claims.to_dict(orient="records"),
    }
    (args.output / "qrag_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"figures": 2, "tables": 3, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
