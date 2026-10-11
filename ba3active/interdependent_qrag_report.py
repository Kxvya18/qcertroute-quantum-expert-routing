"""Statistical tables and figures for the interdependent Q-RAG-MoE run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from .interdependent_qrag import CONDITIONS, MAIN_QUANTUM


LABELS = {
    "complete": "Complete",
    "random50": "Random loss",
    "hub50": "Hub loss",
    "auth_only": "Auth only",
    "dns_only": "DNS only",
    "history_random50": "History random loss",
    "degree_baseline": "Degree",
    "temporal": "Temporal expert",
    "graph_only": "Graph expert",
    "unconditional_mixture": "Equal mixture",
    "fixed_missingness_mixture": "Fixed missingness",
    "classical_moe": "Classical MoE",
    MAIN_QUANTUM: "Selected quantum router",
    "quantum_moe_q2d2": "Q2 D2",
    "quantum_moe_q4d1": "Q4 D1",
    "quantum_moe_q4d2": "Q4 D2",
    "quantum_moe_q4d3": "Q4 D3",
    "quantum_moe_local_only": "Local-only quantum",
    "quantum_reservoir_r2": "QELM R2",
    "quantum_reservoir_r4": "QELM R4",
    "quantum_no_graph_context": "No graph context",
    "quantum_no_temporal_context": "No temporal context",
    "quantum_no_cross_expert_context": "No cross-expert context",
}
COLORS = {
    "degree_baseline": "#9b9b94",
    "temporal": "#23364d",
    "graph_only": "#b49a62",
    "classical_moe": "#5c7890",
    MAIN_QUANTUM: "#147d75",
    "quantum_moe_q4d2": "#147d75",
    "fixed_missingness_mixture": "#a95642",
}
MAIN_MODELS = (
    MAIN_QUANTUM, "classical_moe", "temporal", "graph_only",
    "fixed_missingness_mixture", "degree_baseline",
)
CIRCUITS = (
    "quantum_moe_q2d2", "quantum_moe_q4d1", "quantum_moe_q4d2",
    "quantum_moe_q4d3", "quantum_moe_local_only",
    "quantum_reservoir_r2", "quantum_reservoir_r4",
)
CLAIMS = (
    (MAIN_QUANTUM, "temporal", "requires_graph_expert"),
    (MAIN_QUANTUM, "graph_only", "requires_temporal_expert"),
    (MAIN_QUANTUM, "classical_moe", "quantum_router_comparison"),
    (MAIN_QUANTUM, "fixed_missingness_mixture", "learned_routing_comparison"),
    (MAIN_QUANTUM, "quantum_no_graph_context", "graph_context_dependency"),
    (MAIN_QUANTUM, "quantum_no_temporal_context", "temporal_context_dependency"),
    (MAIN_QUANTUM, "quantum_no_cross_expert_context", "cross_expert_dependency"),
    (MAIN_QUANTUM, "quantum_moe_local_only", "parity_observable_dependency"),
)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8.2, "axes.titlesize": 9.1,
        "axes.labelsize": 8.3, "legend.fontsize": 6.9, "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1, "axes.linewidth": 0.75, "figure.dpi": 160,
        "savefig.dpi": 400, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def panel(ax, letter: str, title: str, grid: bool = True) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", pad=5)
    if grid:
        ax.grid(True, color="#d8d8d3", linewidth=0.45, alpha=0.65, zorder=0)


def paired(frame: pd.DataFrame, first: str, second: str, condition: str | None) -> pd.DataFrame:
    work = frame if condition is None else frame[frame.condition == condition]
    if work.empty:
        return pd.DataFrame()
    keys = ["fold", "seed", "episode", "timestamp", "condition"]
    wide = work[work.model.isin((first, second))].pivot_table(
        index=keys, columns="model", values=("mrr", "hit10", "nll", "rank"), aggfunc="mean"
    )
    if "mrr" not in wide or first not in wide["mrr"] or second not in wide["mrr"]:
        return pd.DataFrame()
    output = wide.index.to_frame(index=False)
    for metric in ("mrr", "hit10", "nll", "rank"):
        output[metric] = wide[metric][first].to_numpy() - wide[metric][second].to_numpy()
    return output


def block_interval(
    frame: pd.DataFrame,
    value: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if frame.empty:
        return np.nan, np.nan, np.nan, np.nan
    work = frame.copy()
    work["block"] = work.timestamp.astype(str) + "|" + work.condition.astype(str)
    matrix = work.pivot_table(index="block", columns="seed", values=value, aggfunc="mean").to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for index in range(repetitions):
        rows = rng.integers(0, matrix.shape[0], matrix.shape[0])
        columns = rng.integers(0, matrix.shape[1], matrix.shape[1])
        samples[index] = np.nanmean(matrix[np.ix_(rows, columns)])
    return (
        float(np.nanmean(matrix)),
        float(np.nanquantile(samples, 0.025)),
        float(np.nanquantile(samples, 0.975)),
        float(np.mean(samples > 0.0)),
    )


def summarize(frame: pd.DataFrame, models, repetitions: int) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for model in models:
            cell = frame[(frame.condition == condition) & (frame.model == model)]
            if cell.empty:
                continue
            center, low, high, _ = block_interval(cell, "mrr", repetitions, 1000 + len(rows))
            rows.append({
                "condition": condition, "model": model,
                "episodes": int(cell.episode.nunique()), "time_blocks": int(cell.timestamp.nunique()),
                "seeds": int(cell.seed.nunique()), "mrr": center, "ci_low": low, "ci_high": high,
                "hit1": float(cell.hit1.mean()), "hit5": float(cell.hit5.mean()),
                "hit10": float(cell.hit10.mean()), "mean_rank": float(cell["rank"].mean()),
                "nll": float(cell.nll.mean()), "entropy": float(cell.entropy.mean()),
                "alpha": float(cell.alpha.mean()),
            })
    return pd.DataFrame(rows)


def claim_table(frame: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    for condition in (*CONDITIONS, "all"):
        for first, second, claim in CLAIMS:
            delta = paired(frame, first, second, None if condition == "all" else condition)
            if delta.empty:
                continue
            center, low, high, probability = block_interval(delta, "mrr", repetitions, 2000 + len(rows))
            rows.append({
                "condition": condition, "claim": claim, "first": first, "second": second,
                "episodes": int(delta.episode.nunique()), "time_blocks": int(delta.timestamp.nunique()),
                "seeds": int(delta.seed.nunique()), "delta_mrr": center,
                "ci_low": low, "ci_high": high, "bootstrap_probability_positive": probability,
                "delta_hit10": float(delta.hit10.mean()), "delta_nll": float(delta.nll.mean()),
                "delta_rank": float(delta["rank"].mean()), "win_rate": float((delta.mrr > 0).mean()),
                "tie_rate": float((delta.mrr == 0).mean()),
            })
    return pd.DataFrame(rows)


def cold_table(frame: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    strata = {
        (1, 1): "seen_source_seen_target", (0, 1): "cold_source",
        (1, 0): "cold_target", (0, 0): "cold_source_and_target",
    }
    for (source_seen, target_seen), label in strata.items():
        subset = frame[(frame.source_seen == source_seen) & (frame.target_seen == target_seen)]
        for model in ("temporal", "graph_only", "classical_moe", MAIN_QUANTUM):
            cell = subset[subset.model == model]
            if cell.empty:
                continue
            center, low, high, _ = block_interval(cell, "mrr", repetitions, 3000 + len(rows))
            rows.append({
                "stratum": label, "model": model, "episodes": int(cell.episode.nunique()),
                "time_blocks": int(cell.timestamp.nunique()), "mrr": center,
                "ci_low": low, "ci_high": high, "hit10": float(cell.hit10.mean()),
                "nll": float(cell.nll.mean()),
            })
    return pd.DataFrame(rows)


def resources_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("model", as_index=False).agg(
        runs=("seed", "size"), parameters=("parameters", "median"),
        training_seconds_mean=("training_seconds", "mean"),
        training_seconds_std=("training_seconds", "std"), best_epoch=("best_epoch", "median"),
        validation_loss=("validation_loss", "mean"), train_records=("train_records", "median"),
        validation_records=("validation_records", "median"),
    )


def save(fig, path: Path) -> None:
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main_figure(frame: pd.DataFrame, summary: pd.DataFrame, claims: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.1), constrained_layout=True)
    ax = axes[0, 0]
    shown = ("temporal", "graph_only", "classical_moe", MAIN_QUANTUM)
    x = np.arange(len(CONDITIONS)); width = 0.19
    for offset, model in enumerate(shown):
        cell = summary[summary.model == model].set_index("condition").reindex(CONDITIONS)
        location = x + (offset - 1.5) * width
        ax.bar(location, cell.mrr, width, color=COLORS[model], label=LABELS[model], zorder=2)
        ax.errorbar(location, cell.mrr, yerr=(cell.mrr-cell.ci_low, cell.ci_high-cell.mrr),
                    fmt="none", ecolor="#222222", capsize=1.6, linewidth=0.65, zorder=3)
    ax.set_xticks(x, [LABELS[item] for item in CONDITIONS], rotation=24, ha="right")
    ax.set_ylabel("Mean reciprocal rank"); ax.legend(frameon=False, ncol=2)
    panel(ax, "a", "Real-event expert mixture")

    ax = axes[0, 1]
    selected = ("requires_graph_expert", "requires_temporal_expert", "quantum_router_comparison")
    names = ("vs temporal", "vs graph", "vs classical MoE")
    y = np.arange(len(CONDITIONS)); offsets = (-0.18, 0.0, 0.18)
    for offset, claim, label, marker in zip(offsets, selected, names, ("o", "s", "D")):
        cell = claims[(claims.condition.isin(CONDITIONS)) & (claims.claim == claim)].set_index("condition").reindex(CONDITIONS)
        ax.errorbar(cell.delta_mrr, y + offset, xerr=(cell.delta_mrr-cell.ci_low, cell.ci_high-cell.delta_mrr),
                    fmt=marker, capsize=2, linewidth=1.0, label=label)
    ax.axvline(0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [LABELS[item] for item in CONDITIONS]); ax.set_xlabel(r"Paired $\Delta$MRR")
    ax.legend(frameon=False); panel(ax, "b", "Component-necessity effects")

    ax = axes[1, 0]
    for model, linestyle in (("temporal", "--"), ("graph_only", ":"), ("classical_moe", "-."), (MAIN_QUANTUM, "-")):
        values = np.sort(frame[(frame.model == model) & (frame.condition == "random50")]["rank"].to_numpy())
        survival = np.maximum(1.0 - np.arange(1, len(values)+1) / len(values), 1.0 / len(values))
        ax.step(values, survival, where="post", linewidth=1.7, linestyle=linestyle,
                color=COLORS[model], label=LABELS[model])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("True destination rank")
    ax.set_ylabel(r"Survival probability $P(R>r)$"); ax.legend(frameon=False)
    panel(ax, "c", "Random-loss rank spectrum")

    ax = axes[1, 1]
    gate = frame[frame.model == MAIN_QUANTUM]
    grouped = gate.groupby("condition").agg(
        alpha=("alpha", "mean"), overlap=("top10_overlap", "mean"),
        history=("source_history", "mean")
    ).reindex(CONDITIONS)
    ax.plot(range(len(CONDITIONS)), grouped.alpha, marker="o", linewidth=1.8,
            color=COLORS[MAIN_QUANTUM], label=r"admission $\alpha$")
    ax.plot(range(len(CONDITIONS)), grouped.overlap, marker="s", linewidth=1.5,
            color="#a95642", label="top-10 expert overlap")
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[item] for item in CONDITIONS], rotation=24, ha="right")
    ax.set_ylabel("Router statistic"); ax.legend(frameon=False)
    panel(ax, "d", "Condition-adaptive arbitration")
    fig.suptitle("Interdependent quantum graph mixture on observed LANL events", fontsize=12, fontweight="bold")
    save(fig, path)


def ablation_figure(frame: pd.DataFrame, summary: pd.DataFrame, resources: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    ax = axes[0, 0]
    for condition, marker in (("complete", "o"), ("random50", "s"), ("hub50", "D")):
        cell = summary[(summary.condition == condition) & summary.model.isin(CIRCUITS)].set_index("model").reindex(CIRCUITS)
        ax.plot(range(len(CIRCUITS)), cell.mrr, marker=marker, linewidth=1.5, label=LABELS[condition])
        ax.fill_between(range(len(CIRCUITS)), cell.ci_low, cell.ci_high, alpha=0.12)
    ax.set_xticks(range(len(CIRCUITS)), [LABELS[item] for item in CIRCUITS], rotation=22, ha="right")
    ax.set_ylabel("Mean reciprocal rank"); ax.legend(frameon=False)
    panel(ax, "a", "Entanglement and parity ablation")

    ax = axes[0, 1]
    dependency = (
        MAIN_QUANTUM, "quantum_no_graph_context", "quantum_no_temporal_context",
        "quantum_no_cross_expert_context"
    )
    cell = summary[(summary.condition == "random50") & summary.model.isin(dependency)].set_index("model").reindex(dependency)
    ax.bar(range(len(dependency)), cell.mrr, color=[COLORS[MAIN_QUANTUM], "#a6aaa5", "#a6aaa5", "#a6aaa5"])
    ax.errorbar(range(len(dependency)), cell.mrr, yerr=(cell.mrr-cell.ci_low, cell.ci_high-cell.mrr),
                fmt="none", ecolor="#202020", capsize=2)
    ax.set_xticks(range(len(dependency)), [LABELS[item] for item in dependency], rotation=22, ha="right")
    ax.set_ylabel("Random-loss MRR"); panel(ax, "b", "Joint-context dependency")

    ax = axes[1, 0]
    shots = sorted(model for model in frame.model.unique() if model.startswith("quantum_shot_"))
    shot_values = sorted(int(model.rsplit("_", 1)[1]) for model in shots)
    for condition, marker in (("complete", "o"), ("random50", "s"), ("hub50", "D")):
        values=[]; low=[]; high=[]
        for shot in shot_values:
            cell=frame[(frame.condition==condition)&(frame.model==f"quantum_shot_{shot}")]
            center, lo, hi, _=block_interval(cell,"mrr",2000,5000+shot)
            values.append(center); low.append(lo); high.append(hi)
        values=np.asarray(values); low=np.asarray(low); high=np.asarray(high)
        ax.plot(shot_values,values,marker=marker,linewidth=1.5,label=LABELS[condition])
        ax.fill_between(shot_values,low,high,alpha=.12)
    ax.set_xscale("log",base=2); ax.set_xlabel("Measurement shots"); ax.set_ylabel("Mean reciprocal rank")
    ax.legend(frameon=False); panel(ax,"c","Finite-shot convergence")

    ax=axes[1,1]
    noise=sorted(model for model in frame.model.unique() if model.startswith("quantum_noise_"))
    matrix=[]
    ideal=frame[frame.model==MAIN_QUANTUM].groupby("condition").mrr.mean().reindex(CONDITIONS)
    for model in noise:
        matrix.append(frame[frame.model==model].groupby("condition").mrr.mean().reindex(CONDITIONS).to_numpy()-ideal.to_numpy())
    matrix=np.asarray(matrix); bound=max(float(np.abs(matrix).max()),1e-5)
    image=ax.imshow(matrix,aspect="auto",cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
    ax.set_xticks(range(len(CONDITIONS)),[LABELS[item] for item in CONDITIONS],rotation=22,ha="right")
    ax.set_yticks(range(len(noise)),[item.replace("quantum_noise_","").replace("_","  ") for item in noise])
    fig.colorbar(image,ax=ax,fraction=.046,pad=.03,label=r"$\Delta$MRR from ideal")
    panel(ax,"d","Noisy-circuit sensitivity",grid=False)
    fig.suptitle("Variational quantum routing, context dependence, and hardware audits",fontsize=12,fontweight="bold")
    save(fig,path)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=4000)
    parser.add_argument("--tables-only",action="store_true",help="Write statistical tables without extra figures")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True); style()
    frame=pd.read_csv(args.results/"interdependent_qrag_results.csv")
    raw_resources=pd.read_csv(args.results/"interdependent_qrag_resources.csv")
    expected=set(MAIN_MODELS)|set(CIRCUITS)|{
        "quantum_no_graph_context","quantum_no_temporal_context","quantum_no_cross_expert_context"
    }
    missing=sorted(expected.difference(frame.model.unique()))
    if missing:
        raise ValueError("missing evaluated models: "+", ".join(missing))
    all_models=sorted(expected)
    primary=summarize(frame,MAIN_MODELS,args.bootstrap)
    extended=summarize(frame,all_models,args.bootstrap)
    claims=claim_table(frame,args.bootstrap)
    cold=cold_table(frame,args.bootstrap)
    resources=resources_table(raw_resources)
    primary.to_csv(args.output/"table_primary_models.csv",index=False)
    extended.to_csv(args.output/"table_component_ablations.csv",index=False)
    claims.to_csv(args.output/"table_paired_claims.csv",index=False)
    cold.to_csv(args.output/"table_cold_start.csv",index=False)
    resources.to_csv(args.output/"table_resources.csv",index=False)
    if not args.tables_only:
        main_figure(frame,extended,claims,args.output/"figure_interdependent_main.png")
        ablation_figure(frame,extended,resources,args.output/"figure_interdependent_ablation.png")
    extraction_path=args.results/"lanl_real_extraction_audit.json"
    audit={
        "benchmark":"observed LANL next-authentication destination",
        "synthetic_targets":False,"test_episodes":int(frame.episode.nunique()),
        "time_blocks":int(frame.timestamp.nunique()),"seeds":sorted(int(v) for v in frame.seed.unique()),
        "statistical_unit":"timestamp-condition block with independent seed resampling",
        "main_quantum_model":MAIN_QUANTUM,
        "validation_selected_circuits":raw_resources[raw_resources.model==MAIN_QUANTUM]
        .get("selected_from",pd.Series(dtype=str)).dropna().value_counts().to_dict(),
        "claims":claims[claims.condition=="all"].to_dict(orient="records"),
        "extraction":json.loads(extraction_path.read_text()) if extraction_path.exists() else {},
    }
    (args.output/"interdependent_qrag_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"figures":0 if args.tables_only else 2,"tables":5,
                      "test_episodes":audit["test_episodes"],"time_blocks":audit["time_blocks"]},indent=2))


if __name__=="__main__":
    main()
