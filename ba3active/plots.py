from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .episodes import load_redteam, sessionize


PALETTE = {"random": "#a9a9a9", "activity": "#c98f34", "entropy": "#287d8e", "curvature": "#8b2f3c"}


def configure():
    mpl.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8, "axes.labelsize": 9,
        "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    })


def label_audit(redteam, output):
    rows = load_redteam(redteam)
    source = pd.Series([row[2] for row in rows]).value_counts()
    gaps = [900, 1800, 3600, 7200]
    episodes = [sessionize(redteam, gap) for gap in gaps]
    stats = pd.DataFrame({
        "gap_min": [gap // 60 for gap in gaps],
        "episodes": [len(items) for items in episodes],
        "origins": [len({item.origin for item in items}) for items in episodes],
        "events": [sum(item.count for item in items) for items in episodes],
    })
    stats.to_csv(output / "episode_audit.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.1), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    top = source.head(6)
    axes[0].barh(np.arange(len(top)), top.values, color="#34495e")
    axes[0].set_yticks(np.arange(len(top)), top.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Labelled red-team events")
    axes[0].set_title("(a) Raw source-machine imbalance", loc="left", fontweight="bold")
    axes[0].text(0.98, 0.05, f"Top machine: {top.iloc[0] / source.sum():.1%}", ha="right", transform=axes[0].transAxes, fontsize=7)
    axes[1].plot(stats.gap_min, stats.episodes, "o-", lw=1.8, color="#8b2f3c", label="Episodes")
    axes[1].plot(stats.gap_min, stats.origins, "s--", lw=1.5, color="#287d8e", label="Origins")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Session gap (minutes)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) Sessionization sensitivity", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right")
    axes[2].bar(stats.gap_min - 1.8, stats.events, width=3.6, color="#c98f34", alpha=0.9)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Session gap (minutes)")
    axes[2].set_ylabel("Retained labelled events")
    axes[2].set_title("(c) Labelled coverage retained", loc="left", fontweight="bold")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2, lw=0.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(output / "figure2_task_audit.png", bbox_inches="tight")
    plt.close(fig)


def _mean_ci(frame, column):
    grouped = frame.groupby(["policy", "budget"])[column]
    result = grouped.agg(["mean", "count", "std"]).reset_index()
    result["se"] = result["std"].fillna(0) / np.sqrt(result["count"].clip(lower=1))
    result["low"] = result["mean"] - 1.96 * result["se"]
    result["high"] = result["mean"] + 1.96 * result["se"]
    return result


def controlled_mechanism(synthetic_dir, output):
    frame = pd.read_csv(synthetic_dir / "synthetic_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25))
    for beta, color in ((0.0, "#777777"), (1.0, "#287d8e"), (2.0, "#8b2f3c")):
        subset = frame[(frame.beta == beta) & (frame.m == 2)]
        pivot = subset.pivot(index="budget", columns="policy", values="mrr").sort_index()
        axes[0].plot(pivot.index * 100, pivot["curvature"], marker="o", color=color, lw=1.8, label=rf"$\beta={beta:.0f}$")
        axes[0].plot(pivot.index * 100, pivot["random"], marker="o", color=color, alpha=0.32, lw=1.0, ls="--")
    grid = frame[frame.budget == 0.03].pivot_table(index="m", columns="beta", values="mrr", aggfunc="mean")
    image = axes[1].imshow(grid.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.3, float(grid.max().max())))
    axes[1].set_xticks(range(len(grid.columns)), [rf"$\beta={value:.0f}$" for value in grid.columns])
    axes[1].set_yticks(range(len(grid.index)), [f"m={value}" for value in grid.index])
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            axes[1].text(j, i, f"{grid.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes[1], pad=0.02, label="Curvature-policy MRR")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Observation budget (percent of nodes)")
    axes[0].set_ylabel("MRR")
    axes[0].set_title("(a) Controlled mechanism response at $m=2$", loc="left", fontweight="bold")
    axes[0].legend(title="Solid: audited; dashed: random", frameon=False, loc="best")
    axes[0].grid(alpha=0.25, lw=0.5)
    axes[1].set_title("(b) Three-percent budget stress grid", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.35)
    fig.savefig(output / "figure3_controlled_mechanism.png", bbox_inches="tight")
    plt.close(fig)


def main_results(result_dir, output):
    frame = pd.read_csv(result_dir / "active_results.csv")
    mrr = _mean_ci(frame, "mrr")
    hit = _mean_ci(frame, "hit5")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25))
    for policy, group in mrr.groupby("policy"):
        group = group.sort_values("budget")
        axes[0].plot(group.budget, group["mean"], marker="o", color=PALETTE[policy], lw=1.8, label=policy.title())
        axes[0].fill_between(group.budget, group.low, group.high, color=PALETTE[policy], alpha=0.12)
    for policy, group in hit.groupby("policy"):
        group = group.sort_values("budget")
        axes[1].plot(group.budget, group["mean"], marker="o", color=PALETTE[policy], lw=1.8, label=policy.title())
        axes[1].fill_between(group.budget, group.low, group.high, color=PALETTE[policy], alpha=0.12)
    for axis, label, title in zip(axes, ("Mean reciprocal rank", "Hit at 5"), ("(a) Ranking efficiency", "(b) Top-five recovery")):
        axis.set_xscale("log")
        axis.set_xlabel("Telemetry budget (percent of host-modality space)")
        axis.set_ylabel(label)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout(w_pad=1.3)
    fig.savefig(output / "figure4_lanl_budget_result.png", bbox_inches="tight")
    plt.close(fig)


def mechanism(result_dir, output):
    frame = pd.read_csv(result_dir / "active_results.csv")
    static = pd.read_csv(result_dir / "static_results.csv")
    subset = frame[frame.budget == 5.0]
    pivot = subset.pivot_table(index=["fold", "seed", "episode"], columns="policy", values="mrr").dropna().reset_index()
    rho = subset.groupby(["fold", "seed", "episode"]).reliability.first().reset_index()
    pivot = pivot.merge(rho, on=["fold", "seed", "episode"])
    pivot["delta"] = pivot.curvature - pivot.entropy
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25))
    for fold, group in pivot.groupby("fold"):
        axes[0].scatter(group.reliability, group.delta, s=15, alpha=0.55, label=f"Fold {fold + 1}")
    if len(pivot) > 2:
        coefficient = np.polyfit(pivot.reliability, pivot.delta, 1)
        x = np.linspace(pivot.reliability.min(), pivot.reliability.max(), 100)
        axes[0].plot(x, np.polyval(coefficient, x), color="#222222", lw=1.3)
    axes[0].axhline(0, color="#555555", lw=0.7, ls="--")
    axes[0].set_xlabel("Curvature reliability")
    axes[0].set_ylabel("MRR difference: curvature minus entropy")
    axes[0].set_title("(a) Curvature-conditioned policy effect", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2)
    summary = static.groupby("model").mrr.agg(["mean", "std", "count"]).reindex(["temporal", "blind", "audited"])
    axes[1].bar(np.arange(3), summary["mean"], yerr=1.96 * summary["std"].fillna(0) / np.sqrt(summary["count"]), capsize=2.5, color=["#287d8e", "#c98f34", "#8b2f3c"])
    axes[1].set_xticks(np.arange(3), ["Temporal", "Blind", "Audited"])
    axes[1].set_ylabel("MRR with full observed evidence")
    axes[1].set_title("(b) Structural-fusion ablation", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(w_pad=1.25)
    fig.savefig(output / "figure5_lanl_ablation.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redteam", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    label_audit(args.redteam, args.output)
    controlled_mechanism(args.synthetic_dir, args.output)
    main_results(args.result_dir, args.output)
    mechanism(args.result_dir, args.output)


if __name__ == "__main__":
    main()
