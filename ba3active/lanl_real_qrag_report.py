"""Publication audit for the real-event LANL Q-RAG-MoE benchmark.

The analysis keeps incidents and optimization seeds as separate sampling axes.
It therefore avoids reporting artificially narrow intervals obtained by treating
every seed-level prediction as an independent security event.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import wilcoxon


CONDITIONS = ("complete", "random50", "hub50", "auth_only", "dns_only")
CONDITION_LABELS = {
    "complete": "Complete",
    "random50": "Random loss",
    "hub50": "Hub loss",
    "auth_only": "Auth only",
    "dns_only": "DNS only",
}
MAIN_MODELS = (
    "degree_baseline",
    "temporal",
    "graph_only",
    "unconditional_fusion",
    "fixed_curvature",
    "classical_router",
    "q3d2_ring",
)
MODEL_LABELS = {
    "degree_baseline": "Degree",
    "temporal": "Temporal",
    "graph_only": "Graph",
    "unconditional_fusion": "Unconditional",
    "fixed_curvature": "Curvature",
    "logistic_router": "Logistic gate",
    "classical_router": "Classical MoE",
    "q2d1_none": "Q2 D1 none",
    "q3d1_linear": "Q3 D1 linear",
    "q3d2_ring": "Q3 D2 ring",
    "q4d2_ring": "Q4 D2 ring",
    "q4d3_ring": "Q4 D3 ring",
    "q3d2_no_graph_context": "No graph context",
    "q3d2_no_expert_context": "No expert context",
    "quantum_blind_reliability": "No reliability",
}
COLORS = {
    "degree_baseline": "#9b9b94",
    "temporal": "#23364d",
    "graph_only": "#a9a07d",
    "unconditional_fusion": "#b65f42",
    "fixed_curvature": "#c19a4b",
    "logistic_router": "#80736b",
    "classical_router": "#59758c",
    "q3d2_ring": "#147d75",
}
CIRCUITS = ("q2d1_none", "q3d1_linear", "q3d2_ring", "q4d2_ring", "q4d3_ring")
CLAIMS = (
    ("q3d2_ring", "temporal", "quantum_vs_temporal"),
    ("q3d2_ring", "classical_router", "quantum_vs_matched_classical"),
    ("q3d2_ring", "fixed_curvature", "quantum_vs_fixed_curvature"),
    ("q3d2_ring", "q3d2_no_graph_context", "graph_context_dependency"),
    ("q3d2_ring", "q3d2_no_expert_context", "expert_context_dependency"),
    ("q3d2_ring", "quantum_blind_reliability", "reliability_dependency"),
)


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.3,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.75,
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel(ax, letter: str, title: str, grid: bool = True) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", pad=5)
    if grid:
        ax.grid(True, color="#d8d8d3", linewidth=0.45, alpha=0.65, zorder=0)


def _paired(frame: pd.DataFrame, first: str, second: str, condition: str | None = None) -> pd.DataFrame:
    work = frame if condition is None else frame[frame.condition == condition]
    keys = ["fold", "seed", "episode", "condition"]
    wide = work[work.model.isin((first, second))].pivot_table(
        index=keys, columns="model", values=("mrr", "hit10", "nll", "rank"), aggfunc="mean"
    )
    if first not in wide["mrr"] or second not in wide["mrr"]:
        return pd.DataFrame()
    output = wide.index.to_frame(index=False)
    for metric in ("mrr", "hit10", "nll", "rank"):
        output[metric] = wide[metric][first].to_numpy() - wide[metric][second].to_numpy()
    return output


def _two_axis_interval(
    frame: pd.DataFrame,
    value: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap event-condition units and optimization seeds independently."""
    if frame.empty:
        return np.nan, np.nan, np.nan
    work = frame.copy()
    work["unit"] = work.episode.astype(str) + "|" + work.condition.astype(str)
    matrix = work.pivot_table(index="unit", columns="seed", values=value, aggfunc="mean").to_numpy(float)
    center = float(np.nanmean(matrix))
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        rows = rng.integers(0, matrix.shape[0], matrix.shape[0])
        columns = rng.integers(0, matrix.shape[1], matrix.shape[1])
        samples[index] = np.nanmean(matrix[np.ix_(rows, columns)])
    return center, float(np.nanquantile(samples, 0.025)), float(np.nanquantile(samples, 0.975))


def _holm(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[position]))
        adjusted[position] = running
    return adjusted.tolist()


def _incident_wilcoxon(delta: pd.DataFrame, metric: str) -> float:
    if delta.empty:
        return np.nan
    incident = delta.groupby(["episode", "condition"], as_index=False)[metric].mean()[metric].to_numpy()
    incident = incident[np.isfinite(incident)]
    if not incident.size or np.allclose(incident, 0.0):
        return 1.0
    return float(wilcoxon(incident, zero_method="zsplit", alternative="two-sided").pvalue)


def primary_summary(frame: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for model in MAIN_MODELS:
            cell = frame[(frame.condition == condition) & (frame.model == model)]
            if cell.empty:
                continue
            mrr, low, high = _two_axis_interval(cell, "mrr", repetitions, 1000 + len(rows))
            rows.append({
                "condition": condition,
                "model": model,
                "episodes": int(cell.episode.nunique()),
                "seeds": int(cell.seed.nunique()),
                "mrr": mrr,
                "mrr_ci_low": low,
                "mrr_ci_high": high,
                "hit1": float(cell.hit1.mean()),
                "hit5": float(cell.hit5.mean()),
                "hit10": float(cell.hit10.mean()),
                "mean_rank": float(cell["rank"].mean()),
                "nll": float(cell.nll.mean()),
                "entropy": float(cell.entropy.mean()),
                "alpha": float(cell.alpha.mean()),
            })
    return pd.DataFrame(rows)


def paired_claims(frame: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    for condition in (*CONDITIONS, "all"):
        raw_p = []
        local = []
        for first, second, claim in CLAIMS:
            delta = _paired(frame, first, second, None if condition == "all" else condition)
            if delta.empty:
                continue
            mean, low, high = _two_axis_interval(delta, "mrr", repetitions, 2000 + len(rows) + len(local))
            record = {
                "condition": condition,
                "claim": claim,
                "first": first,
                "second": second,
                "episodes": int(delta.episode.nunique()),
                "seeds": int(delta.seed.nunique()),
                "delta_mrr": mean,
                "mrr_ci_low": low,
                "mrr_ci_high": high,
                "delta_hit10": float(delta.hit10.mean()),
                "delta_nll": float(delta.nll.mean()),
                "delta_rank": float(delta["rank"].mean()),
                "win_rate": float((delta.mrr > 0).mean()),
                "tie_rate": float((delta.mrr == 0).mean()),
            }
            raw_p.append(_incident_wilcoxon(delta, "mrr"))
            local.append(record)
        adjusted = _holm(raw_p)
        for record, pvalue, corrected in zip(local, raw_p, adjusted):
            record["wilcoxon_p"] = pvalue
            record["holm_p"] = corrected
            rows.append(record)
    return pd.DataFrame(rows)


def model_group_summary(
    frame: pd.DataFrame,
    models: tuple[str, ...] | list[str],
    repetitions: int,
) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for model in models:
            cell = frame[(frame.condition == condition) & (frame.model == model)]
            if cell.empty:
                continue
            mean, low, high = _two_axis_interval(cell, "mrr", repetitions, 3000 + len(rows))
            rows.append({
                "condition": condition,
                "model": model,
                "episodes": int(cell.episode.nunique()),
                "seeds": int(cell.seed.nunique()),
                "mrr": mean,
                "ci_low": low,
                "ci_high": high,
                "hit10": float(cell.hit10.mean()),
                "nll": float(cell.nll.mean()),
                "alpha": float(cell.alpha.mean()),
            })
    return pd.DataFrame(rows)


def cold_start_summary(frame: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    labels = {
        (1, 1): "seen_source_seen_target",
        (0, 1): "cold_source",
        (1, 0): "cold_target",
        (0, 0): "cold_source_and_target",
    }
    for (source_seen, target_seen), label in labels.items():
        subset = frame[(frame.source_seen == source_seen) & (frame.target_seen == target_seen)]
        for model in ("temporal", "classical_router", "q3d2_ring"):
            cell = subset[subset.model == model]
            if cell.empty:
                continue
            mean, low, high = _two_axis_interval(cell, "mrr", repetitions, 4000 + len(rows))
            rows.append({
                "stratum": label,
                "model": model,
                "episodes": int(cell.episode.nunique()),
                "mrr": mean,
                "ci_low": low,
                "ci_high": high,
                "hit10": float(cell.hit10.mean()),
                "nll": float(cell.nll.mean()),
            })
    return pd.DataFrame(rows)


def resource_summary(resources: pd.DataFrame) -> pd.DataFrame:
    return resources.groupby("model", as_index=False).agg(
        runs=("seed", "size"),
        parameters=("parameters", "median"),
        gradient_norm_mean=("gradient_norm", "mean"),
        gradient_norm_std=("gradient_norm", "std"),
        temporal_seconds_mean=("temporal_seconds", "mean"),
        router_group_seconds_mean=("router_group_seconds", "mean"),
        train_episodes=("train_episodes", "median"),
        test_episodes=("test_episodes", "median"),
        information_scale=("information_scale", "median"),
    )


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def figure_main(frame: pd.DataFrame, summary: pd.DataFrame, claims: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    shown = ("degree_baseline", "temporal", "graph_only", "classical_router", "q3d2_ring")
    x = np.arange(len(CONDITIONS))
    width = 0.15
    for offset, model in enumerate(shown):
        cell = summary[summary.model == model].set_index("condition").reindex(CONDITIONS)
        mean = cell.mrr.to_numpy(float)
        low = cell.mrr_ci_low.to_numpy(float)
        high = cell.mrr_ci_high.to_numpy(float)
        location = x + (offset - (len(shown) - 1) / 2) * width
        ax.bar(location, mean, width, color=COLORS[model], label=MODEL_LABELS[model], zorder=2)
        ax.errorbar(location, mean, yerr=(mean - low, high - mean), fmt="none", ecolor="#222222",
                    capsize=1.5, linewidth=0.65, zorder=3)
    ax.set_xticks(x, [CONDITION_LABELS[item] for item in CONDITIONS], rotation=24, ha="right")
    ax.set_ylabel("Mean reciprocal rank")
    ax.legend(frameon=False, ncol=2)
    panel(ax, "a", "Real-event LANL ranking")

    ax = axes[0, 1]
    comparisons = (
        "quantum_vs_temporal",
        "quantum_vs_matched_classical",
        "quantum_vs_fixed_curvature",
    )
    labels = ("vs temporal", "vs classical", "vs curvature")
    y = np.arange(len(CONDITIONS))
    offsets = (-0.18, 0.0, 0.18)
    for offset, claim, label, marker in zip(offsets, comparisons, labels, ("o", "s", "D")):
        cell = claims[(claims.condition.isin(CONDITIONS)) & (claims.claim == claim)].set_index("condition").reindex(CONDITIONS)
        mean = cell.delta_mrr.to_numpy(float)
        low = cell.mrr_ci_low.to_numpy(float)
        high = cell.mrr_ci_high.to_numpy(float)
        ax.errorbar(mean, y + offset, xerr=(mean - low, high - mean), fmt=marker, markersize=4.5,
                    capsize=2, linewidth=1.0, label=label)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [CONDITION_LABELS[item] for item in CONDITIONS])
    ax.set_xlabel(r"Paired $\Delta$MRR for Q3 D2 ring")
    ax.legend(frameon=False)
    panel(ax, "b", "Hierarchical paired effects")

    ax = axes[1, 0]
    for model, linestyle in (("temporal", "--"), ("classical_router", "-."), ("q3d2_ring", "-")):
        values = frame[(frame.model == model) & (frame.condition == "hub50")]["rank"].to_numpy(float)
        values = np.sort(values)
        survival = 1.0 - np.arange(1, len(values) + 1) / len(values)
        ax.step(values, np.maximum(survival, 1.0 / len(values)), where="post", linewidth=1.8,
                linestyle=linestyle, color=COLORS.get(model, "#333333"), label=MODEL_LABELS[model])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True destination rank")
    ax.set_ylabel(r"Survival probability $P(R>r)$")
    ax.legend(frameon=False)
    panel(ax, "c", "Hub-loss rank spectrum")

    ax = axes[1, 1]
    paired = _paired(frame, "q3d2_ring", "temporal")
    metadata = frame[frame.model == "q3d2_ring"][
        ["fold", "seed", "episode", "condition", "rho", "alpha"]
    ].drop_duplicates()
    paired = paired.merge(metadata, on=["fold", "seed", "episode", "condition"], how="left")
    paired["rho_bin"] = pd.qcut(paired.rho, q=5, labels=False, duplicates="drop")
    grouped = paired.groupby("rho_bin").agg(rho=("rho", "mean"), delta=("mrr", "mean"), alpha=("alpha", "mean"))
    ax.plot(grouped.rho, grouped.delta, marker="o", color=COLORS["q3d2_ring"], linewidth=1.8,
            label=r"$\Delta$MRR")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Fisher reliability $\rho_F$")
    ax.set_ylabel(r"Quantum minus temporal $\Delta$MRR")
    twin = ax.twinx()
    twin.plot(grouped.rho, grouped.alpha, marker="s", color="#9a6b45", linewidth=1.4, label=r"admission $\alpha$")
    twin.set_ylabel(r"Mean admission $\alpha$")
    lines = ax.get_lines()[:1] + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    panel(ax, "d", "Reliability-conditioned routing")
    fig.suptitle("Q-RAG-MoE on observed LANL authentication and DNS events", fontsize=12, fontweight="bold")
    _save_figure(fig, path)


def figure_ablation(
    frame: pd.DataFrame,
    circuit: pd.DataFrame,
    resources: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    ax = axes[0, 0]
    for condition, marker in (("complete", "o"), ("random50", "s"), ("hub50", "D")):
        cell = circuit[circuit.condition == condition].set_index("model").reindex(CIRCUITS)
        ax.plot(range(len(CIRCUITS)), cell.mrr, marker=marker, linewidth=1.5, label=CONDITION_LABELS[condition])
        ax.fill_between(range(len(CIRCUITS)), cell.ci_low, cell.ci_high, alpha=0.12)
    ax.set_xticks(range(len(CIRCUITS)), [MODEL_LABELS[item] for item in CIRCUITS], rotation=24, ha="right")
    ax.set_ylabel("Mean reciprocal rank")
    ax.legend(frameon=False)
    panel(ax, "a", "Circuit family and depth")

    ax = axes[0, 1]
    dependency = ("q3d2_ring", "q3d2_no_graph_context", "q3d2_no_expert_context", "quantum_blind_reliability")
    labels = ("Full", "No graph context", "No expert context", "No reliability")
    for index, (model, label) in enumerate(zip(dependency, labels)):
        cell = frame[(frame.model == model) & (frame.condition == "hub50")]
        center, low, high = _two_axis_interval(cell, "mrr", 2000, 5100 + index)
        ax.bar(index, center, color=(COLORS["q3d2_ring"] if index == 0 else "#a6aaa5"), width=0.65)
        ax.errorbar(index, center, yerr=np.asarray([[center - low], [high - center]]), fmt="none",
                    color="#202020", capsize=2)
    ax.set_xticks(range(len(labels)), labels, rotation=22, ha="right")
    ax.set_ylabel("Hub-loss MRR")
    panel(ax, "b", "Coherent-context dependency")

    ax = axes[1, 0]
    resource = resources[resources.model.isin(CIRCUITS)].set_index("model").reindex(CIRCUITS)
    overall = circuit.groupby("model").mrr.mean().reindex(CIRCUITS)
    sizes = 35 + 170 * (resource.gradient_norm_mean / max(resource.gradient_norm_mean.max(), 1e-12))
    ax.scatter(resource.parameters, overall, s=sizes, color="#147d75", alpha=0.78,
               edgecolor="#20364f", linewidth=0.7)
    for model in CIRCUITS:
        ax.annotate(MODEL_LABELS[model], (resource.loc[model, "parameters"], overall.loc[model]),
                    xytext=(4, 4), textcoords="offset points", fontsize=6.7)
    ax.set_xlabel("Trainable router parameters")
    ax.set_ylabel("Condition-averaged MRR")
    panel(ax, "c", "Accuracy and gradient-scaled complexity")

    ax = axes[1, 1]
    grad = resource.gradient_norm_mean.to_numpy(float)
    grad_std = resource.gradient_norm_std.fillna(0).to_numpy(float)
    ax.bar(range(len(CIRCUITS)), grad, color="#59758c", width=0.68)
    ax.errorbar(range(len(CIRCUITS)), grad, yerr=grad_std, fmt="none", ecolor="#202020", capsize=2)
    ax.set_xticks(range(len(CIRCUITS)), [MODEL_LABELS[item] for item in CIRCUITS], rotation=24, ha="right")
    ax.set_ylabel(r"Gate-loss gradient norm $\Vert\nabla\mathcal{L}\Vert_2$")
    panel(ax, "d", "Trainability audit")
    fig.suptitle("Quantum router architecture and dependency ablations", fontsize=12, fontweight="bold")
    _save_figure(fig, path)


def figure_robustness(
    frame: pd.DataFrame,
    shots: pd.DataFrame,
    noise: pd.DataFrame,
    cold: pd.DataFrame,
    resources: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    ax = axes[0, 0]
    shot_values = sorted(int(model.split("_")[1]) for model in shots.model.unique())
    for condition, marker in (("complete", "o"), ("random50", "s"), ("hub50", "D")):
        cell = shots[shots.condition == condition].copy()
        cell["shots"] = cell.model.str.split("_").str[1].astype(int)
        cell = cell.set_index("shots").reindex(shot_values)
        ax.plot(shot_values, cell.mrr, marker=marker, linewidth=1.5, label=CONDITION_LABELS[condition])
        ax.fill_between(shot_values, cell.ci_low, cell.ci_high, alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Measurement shots")
    ax.set_ylabel("Mean reciprocal rank")
    ax.legend(frameon=False)
    panel(ax, "a", "Finite-shot convergence")

    ax = axes[0, 1]
    models = sorted(noise.model.unique(), key=lambda item: float(item.split("_p")[1].split("_")[0]))
    matrix = noise.pivot(index="model", columns="condition", values="mrr").reindex(index=models, columns=CONDITIONS)
    ideal = frame[frame.model == "q3d2_ring"].groupby("condition").mrr.mean().reindex(CONDITIONS)
    delta = matrix - ideal.to_numpy()
    bound = max(abs(float(np.nanmin(delta.to_numpy()))), abs(float(np.nanmax(delta.to_numpy()))), 1e-4)
    image = ax.imshow(delta, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    ax.set_xticks(range(len(CONDITIONS)), [CONDITION_LABELS[item] for item in CONDITIONS], rotation=24, ha="right")
    ax.set_yticks(range(len(models)), [item.replace("qnoise_", "").replace("_", "  ") for item in models])
    ax.set_ylabel("Depolarizing and readout noise")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=r"$\Delta$MRR from ideal")
    panel(ax, "b", "Noisy-circuit sensitivity", grid=False)

    ax = axes[1, 0]
    strata = list(cold.stratum.drop_duplicates())
    shown = ("temporal", "classical_router", "q3d2_ring")
    x = np.arange(len(strata))
    width = 0.24
    for offset, model in enumerate(shown):
        cell = cold[cold.model == model].set_index("stratum").reindex(strata)
        location = x + (offset - 1) * width
        ax.bar(location, cell.mrr, width, color=COLORS[model], label=MODEL_LABELS[model])
        ax.errorbar(location, cell.mrr, yerr=(cell.mrr - cell.ci_low, cell.ci_high - cell.mrr),
                    fmt="none", ecolor="#202020", capsize=1.5, linewidth=0.65)
    ax.set_xticks(x, [item.replace("_", "\n") for item in strata])
    ax.set_ylabel("Mean reciprocal rank")
    ax.legend(frameon=False)
    panel(ax, "c", "Chronological cold-start strata")

    ax = axes[1, 1]
    resource = resources[resources.model.isin(("logistic_router", "classical_router", *CIRCUITS))].copy()
    resource = resource.sort_values("parameters")
    ax.scatter(resource.parameters, resource.gradient_norm_mean, s=55, color="#59758c",
               edgecolor="#20364f", linewidth=0.6)
    for _, row in resource.iterrows():
        ax.annotate(MODEL_LABELS.get(row.model, row.model), (row.parameters, row.gradient_norm_mean),
                    xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    inset = ax.inset_axes([0.57, 0.08, 0.39, 0.38])
    times = resource[["temporal_seconds_mean", "router_group_seconds_mean"]].mean()
    inset.bar((0, 1), times, color=(COLORS["temporal"], COLORS["q3d2_ring"]), width=0.62)
    inset.set_xticks((0, 1), ("Temporal", "Routers"), rotation=20)
    inset.set_ylabel("s", fontsize=6.5)
    inset.tick_params(labelsize=6)
    ax.set_xlabel("Trainable router parameters")
    ax.set_ylabel("Mean gradient norm")
    panel(ax, "d", "Resource and trainability envelope")
    fig.suptitle("Finite-shot, noise, cold-start, and resource robustness", fontsize=12, fontweight="bold")
    _save_figure(fig, path)


def validate_inputs(frame: pd.DataFrame, resources: pd.DataFrame) -> list[str]:
    required_columns = {
        "fold", "seed", "episode", "condition", "model", "mrr", "hit1", "hit5", "hit10",
        "rank", "nll", "entropy", "alpha", "rho", "source_seen", "target_seen",
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    expected_models = set(MAIN_MODELS) | {first for first, _, _ in CLAIMS} | {second for _, second, _ in CLAIMS}
    expected_models |= set(CIRCUITS) | {f"qshot_{value}" for value in (32, 64, 128, 256, 512, 1024)}
    missing_models = sorted(expected_models.difference(frame.model.unique()))
    messages = []
    if missing_columns:
        messages.append(f"missing result columns: {', '.join(missing_columns)}")
    if missing_models:
        messages.append(f"missing evaluated models: {', '.join(missing_models)}")
    if resources.empty:
        messages.append("resource table is empty")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory containing LANL result CSV files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    apply_style()

    frame = pd.read_csv(args.results / "lanl_real_qrag_results.csv")
    resources_raw = pd.read_csv(args.results / "lanl_real_qrag_resources.csv")
    audit_path = args.results / "lanl_real_extraction_audit.json"
    extraction = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    issues = validate_inputs(frame, resources_raw)
    if issues:
        raise ValueError("; ".join(issues))

    summary = primary_summary(frame, args.bootstrap)
    claims = paired_claims(frame, args.bootstrap)
    circuit = model_group_summary(frame, list(CIRCUITS), args.bootstrap)
    shots = model_group_summary(frame, sorted(model for model in frame.model.unique() if model.startswith("qshot_")), args.bootstrap)
    noise = model_group_summary(frame, sorted(model for model in frame.model.unique() if model.startswith("qnoise_")), args.bootstrap)
    cold = cold_start_summary(frame, args.bootstrap)
    resources = resource_summary(resources_raw)

    outputs = {
        "table_primary_models.csv": summary,
        "table_paired_claims.csv": claims,
        "table_quantum_circuits.csv": circuit,
        "table_finite_shots.csv": shots,
        "table_quantum_noise.csv": noise,
        "table_cold_start.csv": cold,
        "table_resources.csv": resources,
    }
    for name, table in outputs.items():
        table.to_csv(args.output / name, index=False)

    figure_main(frame, summary, claims, args.output / "figure_real_lanl_main.png")
    figure_ablation(frame, circuit, resources, args.output / "figure_quantum_ablation.png")
    figure_robustness(frame, shots, noise, cold, resources, args.output / "figure_quantum_robustness.png")

    all_claims = claims[claims.condition == "all"]
    report = {
        "benchmark": "observed LANL authentication and DNS next-contact prediction",
        "synthetic_targets": False,
        "temporal_folds": int(frame.fold.nunique()),
        "optimization_seeds": sorted(int(value) for value in frame.seed.unique()),
        "test_episodes": int(frame.episode.nunique()),
        "conditions": list(CONDITIONS),
        "bootstrap_repetitions": args.bootstrap,
        "statistical_unit": "episode-condition with independent seed resampling",
        "extraction": extraction,
        "aggregate_claims": all_claims.to_dict(orient="records"),
        "figures": ["figure_real_lanl_main", "figure_quantum_ablation", "figure_quantum_robustness"],
        "tables": list(outputs),
    }
    (args.output / "lanl_real_qrag_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "figures": 3,
        "tables": len(outputs),
        "test_episodes": report["test_episodes"],
        "seeds": report["optimization_seeds"],
    }, indent=2))


if __name__ == "__main__":
    main()
