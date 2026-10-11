"""Create statistical tables and publication figures for the RAG-MoE study."""
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
    "temporal_mlp": "Temporal MLP",
    "structural": "Structural expert",
    "unconditional": "Unconditional fusion",
    "fixed_curvature": "Fixed-curvature fusion",
    "rag_moe": "RAG-MoE",
}
COLORS = {
    "temporal_mlp": "#20364f",
    "structural": "#9b9b96",
    "unconditional": "#b65f42",
    "fixed_curvature": "#b68a3c",
    "rag_moe": "#147d75",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.75,
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def two_way_interval(frame, value="mrr", repetitions=3000, seed=1701):
    """Bootstrap independent episode and optimizer-seed axes."""
    pivot = frame.pivot_table(index="episode", columns="seed", values=value, aggfunc="mean")
    array = pivot.to_numpy(dtype=float)
    if not array.size:
        return np.nan, np.nan, np.nan
    mean = float(np.nanmean(array))
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions)
    for i in range(repetitions):
        rows = rng.integers(0, array.shape[0], array.shape[0])
        cols = rng.integers(0, array.shape[1], array.shape[1])
        boot[i] = np.nanmean(array[np.ix_(rows, cols)])
    return mean, float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))


def paired_records(frame):
    keys = [column for column in ("beta", "m", "fold", "seed", "episode", "condition") if column in frame]
    metrics = frame.pivot_table(index=keys, columns="model", values=["mrr", "hit1", "hit5", "hit10", "rank", "nll"], aggfunc="mean")
    metrics.columns = [f"{metric}_{model}" for metric, model in metrics.columns]
    metrics = metrics.reset_index()
    meta = frame[frame.model == "rag_moe"][keys + ["gate_mean", "gate_truth", "rho", "se", "gini", "observed_fraction"]]
    return metrics.merge(meta, on=keys, how="left")


def paired_interval(paired, a, b, repetitions=4000, seed=2026):
    work = paired[["episode", "seed", a, b]].copy()
    work["delta"] = work[a] - work[b]
    return two_way_interval(work, value="delta", repetitions=repetitions, seed=seed)


def metric_interval(frame, model, condition, metric="mrr", **filters):
    group = frame[(frame.model == model) & (frame.condition == condition)]
    for key, value in filters.items():
        group = group[group[key] == value]
    token = "|".join((model, condition, metric, *(f"{k}={v}" for k, v in sorted(filters.items()))))
    seed = sum((index + 1) * ord(char) for index, char in enumerate(token)) % (2**32)
    return two_way_interval(group, metric, seed=seed)


def synthetic_table(frame):
    rows = []
    paired = paired_records(frame)
    for beta in sorted(frame.beta.unique()):
        for m in sorted(frame.m.unique()):
            for condition in CONDITIONS:
                cell = frame[(frame.beta == beta) & (frame.m == m) & (frame.condition == condition)]
                pair = paired[(paired.beta == beta) & (paired.m == m) & (paired.condition == condition)]
                values = {model: cell[cell.model == model].mrr.mean() for model in COLORS}
                delta, low, high = paired_interval(pair, "mrr_rag_moe", "mrr_temporal_mlp", seed=1000 + int(beta * 100) + m)
                rag = cell[cell.model == "rag_moe"]
                rows.append({
                    "beta": beta, "m": m, "condition": condition,
                    "episodes": int(rag.episode.nunique()), "seeds": int(rag.seed.nunique()),
                    "mrr_temporal": values["temporal_mlp"],
                    "mrr_structural": values["structural"],
                    "mrr_unconditional": values["unconditional"],
                    "mrr_fixed_curvature": values["fixed_curvature"],
                    "mrr_rag_moe": values["rag_moe"],
                    "delta_rag_temporal": delta, "ci_low": low, "ci_high": high,
                    "hit1_rag_moe": rag.hit1.mean(), "hit5_rag_moe": rag.hit5.mean(),
                    "gate_mean": rag.gate_mean.mean(), "rho": rag.rho.mean(),
                })
    return pd.DataFrame(rows)


def lanl_table(frame):
    rows = []
    for condition in CONDITIONS:
        for model in COLORS:
            cell = frame[(frame.condition == condition) & (frame.model == model)]
            mean, low, high = two_way_interval(cell, "mrr", seed=1200 + CONDITIONS.index(condition) * 10 + list(COLORS).index(model))
            rows.append({
                "condition": condition, "model": model,
                "episodes": int(cell.episode.nunique()), "seeds": int(cell.seed.nunique()),
                "mrr": mean, "ci_low": low, "ci_high": high,
                "hit1": cell.hit1.mean(), "hit5": cell.hit5.mean(), "hit10": cell.hit10.mean(),
                "median_rank": cell.groupby("episode")["rank"].mean().median(),
                "nll": cell.nll.mean(), "gate_mean": cell.gate_mean.mean(),
            })
    return pd.DataFrame(rows)


def overall_claims(synthetic, lanl):
    rows = []
    for dataset, frame in (("controlled", synthetic), ("LANL", lanl)):
        pair = paired_records(frame)
        for condition in CONDITIONS:
            cell = pair[pair.condition == condition]
            delta, low, high = paired_interval(cell, "mrr_rag_moe", "mrr_temporal_mlp", seed=2100 + CONDITIONS.index(condition))
            unconditional = float((cell.mrr_unconditional - cell.mrr_temporal_mlp).mean())
            fixed = float((cell.mrr_fixed_curvature - cell.mrr_temporal_mlp).mean())
            win = float((cell.mrr_rag_moe > cell.mrr_temporal_mlp).mean())
            loss = float((cell.mrr_rag_moe < cell.mrr_temporal_mlp).mean())
            rows.append({
                "dataset": dataset, "condition": condition, "pairs": len(cell),
                "delta_rag_temporal": delta, "ci_low": low, "ci_high": high,
                "delta_unconditional_temporal": unconditional,
                "delta_fixed_temporal": fixed,
                "rag_win_fraction": win, "rag_loss_fraction": loss,
                "gate_mean": cell.gate_mean.mean(),
            })
    return pd.DataFrame(rows)


def table_tex(table, path, kind):
    if kind == "synthetic":
        show = table[table.condition.isin(("auth_only", "dns_only", "random50", "hub50"))].copy()
        columns = ["beta", "m", "condition", "mrr_temporal", "mrr_structural", "mrr_unconditional", "mrr_fixed_curvature", "mrr_rag_moe", "delta_rag_temporal", "ci_low", "ci_high", "gate_mean"]
        headers = [r"$\beta$", "$m$", "Condition", "Temporal", "Structural", "Uncond.", "Fixed", "RAG-MoE", r"$\Delta$MRR", "CI low", "CI high", r"$\bar\alpha$"]
        caption = "Controlled mechanism results across growth and telemetry regimes. Confidence limits use two-way bootstrap resampling over held-out episodes and training seeds."
        label = "tab:rag_controlled"
    else:
        show = table.copy()
        columns = ["condition", "model", "mrr", "ci_low", "ci_high", "hit1", "hit5", "hit10", "median_rank", "nll", "gate_mean"]
        headers = ["Condition", "Model", "MRR", "CI low", "CI high", "H@1", "H@5", "H@10", "Med. rank", "NLL", r"$\bar\alpha$"]
        caption = "LANL endpoint-localization results under telemetry loss. Confidence limits use two-way bootstrap resampling over held-out incidents and training seeds."
        label = "tab:rag_lanl"
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for _, row in show.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column == "condition":
                cells.append(LABELS.get(value, str(value)))
            elif column == "model":
                cells.append(LABELS.get(value, str(value)))
            elif column in ("beta",):
                cells.append(f"{value:.1f}")
            elif column in ("m", "median_rank"):
                cells.append(f"{value:.0f}")
            else:
                cells.append(f"{value:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    path.write_text("\n".join(lines) + "\n")


def panel_label(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", pad=6)
    ax.grid(True, color="#d8d8d3", linewidth=0.45, alpha=0.65)


def figure_mechanism(syn, table, paired, path):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.1), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(len(CONDITIONS))
    for model in ("temporal_mlp", "unconditional", "fixed_curvature", "rag_moe"):
        means, lows, highs = [], [], []
        for condition in CONDITIONS:
            mean, low, high = metric_interval(syn, model, condition)
            means.append(mean); lows.append(low); highs.append(high)
        means, lows, highs = map(np.asarray, (means, lows, highs))
        ax.plot(x, means, marker="o", linewidth=1.8, markersize=4, color=COLORS[model], label=LABELS[model])
        ax.fill_between(x, lows, highs, color=COLORS[model], alpha=0.14, linewidth=0)
    ax.set_xticks(x, [LABELS[c] for c in CONDITIONS], rotation=22, ha="right")
    ax.set_ylabel("Mean reciprocal rank")
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, ncol=1)
    panel_label(ax, "a", "Controlled ranking under telemetry loss")

    ax = axes[0, 1]
    phase = table.groupby(["beta", "condition"]).delta_rag_temporal.mean().unstack().reindex(index=(0.0, 1.0, 2.0), columns=CONDITIONS)
    limit = max(abs(float(np.nanmin(phase))), abs(float(np.nanmax(phase))), 0.01)
    image = ax.imshow(phase, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    for i in range(phase.shape[0]):
        for j in range(phase.shape[1]):
            value = phase.iloc[i, j]
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_xticks(np.arange(len(CONDITIONS)), [LABELS[c] for c in CONDITIONS], rotation=22, ha="right")
    ax.set_yticks(np.arange(3), [r"$\beta=0$", r"$\beta=1$", r"$\beta=2$"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=r"$\Delta$MRR, RAG-MoE minus temporal")
    panel_label(ax, "b", "Conditional structural-benefit phase map")

    ax = axes[1, 0]
    condition_colors = dict(zip(CONDITIONS, ("#262d3a", "#527b88", "#8e6c54", "#b44f50", "#4d7b57")))
    for condition in CONDITIONS:
        group = paired[paired.condition == condition]
        means = group.groupby("beta").gate_mean.mean().reindex((0.0, 1.0, 2.0))
        ax.plot(means.index, means.values, marker="o", linewidth=1.5, color=condition_colors[condition], label=LABELS[condition])
    ax.set_xlabel(r"Growth concentration $\beta$")
    ax.set_ylabel(r"Mean structural admission $\bar\alpha$")
    ax.set_xticks((0, 1, 2))
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, ncol=2)
    panel_label(ax, "c", "Learned gate response")

    ax = axes[1, 1]
    work = paired.copy()
    work["helpful"] = (work.mrr_rag_moe > work.mrr_temporal_mlp).astype(float)
    work["alpha_bin"] = pd.qcut(work.gate_truth.rank(method="first"), 8, labels=False)
    calibration = work.groupby("alpha_bin").agg(alpha=("gate_truth", "mean"), helpful=("helpful", "mean"), delta=("mrr_rag_moe", "mean"), n=("helpful", "size"))
    temporal = work.groupby("alpha_bin").mrr_temporal_mlp.mean()
    calibration["delta"] -= temporal
    ax.bar(calibration.alpha, calibration.helpful, width=0.055, color="#d6d6d0", edgecolor="#777771", linewidth=0.6, label="Observed helpful fraction")
    ax.plot(calibration.alpha, calibration.helpful, color=COLORS["rag_moe"], marker="o", linewidth=1.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=0.9, label="Ideal reliability")
    ax.set_xlabel(r"Predicted gate admission $\alpha$")
    ax.set_ylabel("Fraction with positive rank gain")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    panel_label(ax, "d", "Gate utility calibration")
    fig.suptitle("RAG-MoE controlled-mechanism evaluation", fontsize=12, fontweight="bold")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def ecdf_rank(ax, frame, condition, title, letter):
    for model in ("temporal_mlp", "unconditional", "fixed_curvature", "rag_moe"):
        values = np.sort(
            frame[(frame.condition == condition) & (frame.model == model)]
            .groupby(["episode", "seed"])["rank"]
            .mean()
            .to_numpy()
        )
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", color=COLORS[model], linewidth=1.6, label=LABELS[model])
    ax.set_xscale("log")
    ax.set_xlabel("Source rank, lower is better")
    ax.set_ylabel(r"$P(R \leq r)$")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    panel_label(ax, letter, title)


def figure_spectra(syn, lanl, paired_syn, paired_lanl, path):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    ecdf_rank(axes[0, 0], syn, "random50", "Controlled random-loss rank spectrum", "a")
    ecdf_rank(axes[0, 1], syn, "hub50", "Controlled hub-loss rank spectrum", "b")

    ax = axes[1, 0]
    bins = np.linspace(-0.55, 0.55, 45)
    for beta, color in ((0.0, "#777777"), (1.0, "#b65f42"), (2.0, "#147d75")):
        group = paired_syn[(paired_syn.beta == beta) & paired_syn.condition.isin(("random50", "hub50", "auth_only", "dns_only"))]
        delta = group.mrr_rag_moe - group.mrr_temporal_mlp
        hist, edges = np.histogram(delta, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.step(centers, hist, where="mid", linewidth=1.5, color=color, label=rf"$\beta={beta:.0f}$")
        ax.fill_between(centers, 0, hist, step="mid", color=color, alpha=0.10)
    ax.axvline(0, color="#222222", linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_xlabel(r"Episode-level $\Delta$MRR")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    panel_label(ax, "c", "Residual-gain distribution")

    ax = axes[1, 1]
    x = np.arange(len(CONDITIONS))
    width = 0.24
    plotted = ("temporal_mlp", "fixed_curvature", "rag_moe")
    for offset, model in enumerate(plotted):
        means, lows, highs = [], [], []
        for condition in CONDITIONS:
            mean, low, high = metric_interval(lanl, model, condition)
            means.append(mean); lows.append(low); highs.append(high)
        means, lows, highs = map(np.asarray, (means, lows, highs))
        ax.bar(x + (offset - 1) * width, means, width=width, color=COLORS[model], alpha=0.90, label=LABELS[model])
        ax.errorbar(x + (offset - 1) * width, means, yerr=(means - lows, highs - means), fmt="none", ecolor="#202020", capsize=2, linewidth=0.7)
    ax.set_xticks(x, [LABELS[c] for c in CONDITIONS], rotation=22, ha="right")
    ax.set_ylabel("LANL mean reciprocal rank")
    ax.legend(frameon=False, ncol=1)
    panel_label(ax, "d", "Real-incident transfer and safety audit")
    fig.suptitle("Source-rank spectra and real-network transfer", fontsize=12, fontweight="bold")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def figure_safety(syn_table, claims, path):
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 3.4), constrained_layout=True)
    ax = axes[0]
    for beta, marker, color in ((0.0, "o", "#777777"), (1.0, "s", "#b65f42"), (2.0, "^", "#147d75")):
        group = syn_table[syn_table.beta == beta]
        x = group.mrr_unconditional - group.mrr_temporal
        y = group.mrr_rag_moe - group.mrr_temporal
        ax.scatter(x, y, s=28 + 90 * group.gate_mean, marker=marker, color=color, edgecolor="white", linewidth=0.5, alpha=0.88, label=rf"$\beta={beta:.0f}$")
    bounds = (-0.55, 0.55)
    ax.plot(bounds, bounds, color="#888888", linestyle=":", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8); ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlim(bounds); ax.set_ylim(bounds)
    ax.set_xlabel(r"Unconditional $\Delta$MRR")
    ax.set_ylabel(r"RAG-MoE $\Delta$MRR")
    ax.legend(frameon=False)
    panel_label(ax, "a", "Selective fusion versus always-on fusion")

    ax = axes[1]
    controlled = claims[claims.dataset == "controlled"].set_index("condition").reindex(CONDITIONS)
    x = np.arange(len(CONDITIONS))
    means = controlled.delta_rag_temporal.to_numpy()
    lows = controlled.ci_low.to_numpy(); highs = controlled.ci_high.to_numpy()
    colors = np.where(means >= 0, COLORS["rag_moe"], COLORS["unconditional"])
    ax.bar(x, means, color=colors, width=0.68, alpha=0.90)
    ax.errorbar(x, means, yerr=(means - lows, highs - means), fmt="none", color="#202020", capsize=2, linewidth=0.8)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, [LABELS[c] for c in CONDITIONS], rotation=28, ha="right")
    ax.set_ylabel(r"Paired $\Delta$MRR with 95% CI")
    panel_label(ax, "b", "Controlled two-way bootstrap")

    ax = axes[2]
    lanl = claims[claims.dataset == "LANL"].set_index("condition").reindex(CONDITIONS)
    means = lanl.delta_rag_temporal.to_numpy(); lows = lanl.ci_low.to_numpy(); highs = lanl.ci_high.to_numpy()
    colors = np.where(means >= 0, COLORS["rag_moe"], COLORS["unconditional"])
    ax.bar(x, means, color=colors, width=0.68, alpha=0.90)
    ax.errorbar(x, means, yerr=(means - lows, highs - means), fmt="none", color="#202020", capsize=2, linewidth=0.8)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, [LABELS[c] for c in CONDITIONS], rotation=28, ha="right")
    ax.set_ylabel(r"Paired $\Delta$MRR with 95% CI")
    panel_label(ax, "c", "LANL two-way bootstrap")
    fig.suptitle("Safety and conditional-utility diagnostics", fontsize=12, fontweight="bold")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    style()

    synthetic = pd.read_csv(args.results / "rag_synthetic_results.csv")
    lanl = pd.read_csv(args.results / "rag_lanl_results.csv")
    required = ("mrr", "nll", "entropy", "rank", "gate_mean", "gate_truth")
    for name, frame in (("synthetic", synthetic), ("LANL", lanl)):
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{name} results are missing required columns")
        if (frame.nll < -1e-9).any() or (frame.entropy < -1e-9).any():
            raise ValueError(f"{name} contains invalid unnormalized probability metrics")

    syn_table = synthetic_table(synthetic)
    real_table = lanl_table(lanl)
    claims = overall_claims(synthetic, lanl)
    syn_table.to_csv(args.tables / "table_rag_controlled_grid.csv", index=False)
    real_table.to_csv(args.tables / "table_rag_lanl_models.csv", index=False)
    claims.to_csv(args.tables / "table_rag_claim_audit.csv", index=False)
    table_tex(syn_table, args.tables / "table_rag_controlled_grid.tex", "synthetic")
    table_tex(real_table, args.tables / "table_rag_lanl_models.tex", "lanl")

    paired_syn = paired_records(synthetic)
    paired_lanl = paired_records(lanl)
    figure_mechanism(synthetic, syn_table, paired_syn, args.figures / "figure_rag_moe_mechanism.png")
    figure_spectra(synthetic, lanl, paired_syn, paired_lanl, args.figures / "figure_rag_moe_rank_spectra.png")
    figure_safety(syn_table, claims, args.figures / "figure_rag_moe_safety.png")

    audit = {
        "model": "Reliability-Aware Graph Mixture of Experts",
        "parameter_count": 682,
        "controlled_episodes": int(synthetic.episode.nunique()),
        "lanl_episodes": int(lanl.episode.nunique()),
        "training_seeds": sorted(int(x) for x in synthetic.seed.unique()),
        "conditions": list(CONDITIONS),
        "probability_metrics_valid": True,
        "claim_rows": claims.to_dict(orient="records"),
    }
    (args.tables / "rag_moe_statistical_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({
        "figures": sorted(path.name for path in args.figures.glob("figure_rag_*.png")),
        "tables": sorted(path.name for path in args.tables.glob("table_rag_*")),
        "audit": str(args.tables / "rag_moe_statistical_audit.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
