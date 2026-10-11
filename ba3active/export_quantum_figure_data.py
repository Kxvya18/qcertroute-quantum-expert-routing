"""Export non-tabular figure data from the observed LANL quantum-routing results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


QMODEL = "quantum_moe_selected"
ORDER = ("complete", "random50", "hub50", "auth_only", "dns_only")


def write(frame: pd.DataFrame, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = pd.read_csv(args.results)
    keys = ["fold", "seed", "episode", "timestamp", "condition"]
    quantum = results[results.model.eq(QMODEL)].copy()

    context_columns = keys + [
        "alpha", "mrr", "rank", "nll", "candidates", "missing_fraction",
        "source_history", "source_novelty", "temporal_entropy", "graph_entropy",
        "temporal_margin", "graph_margin", "top10_overlap",
    ]
    write(quantum[context_columns].sort_values(keys), args.output, "figure1_authority_context_raw.csv")

    x = quantum.temporal_entropy.to_numpy()
    y = quantum.graph_entropy.to_numpy()
    z = quantum.alpha.to_numpy()
    xb = np.linspace(np.quantile(x, .02), np.quantile(x, .98), 12)
    yb = np.linspace(np.quantile(y, .02), np.quantile(y, .98), 12)
    surface = []
    for i in range(len(xb) - 1):
        for j in range(len(yb) - 1):
            take = (x >= xb[i]) & (x < xb[i + 1]) & (y >= yb[j]) & (y < yb[j + 1])
            if take.sum() >= 6:
                surface.append({
                    "temporal_entropy_center": (xb[i] + xb[i + 1]) / 2,
                    "graph_entropy_center": (yb[j] + yb[j + 1]) / 2,
                    "mean_quantum_authority": z[take].mean(),
                    "records": int(take.sum()),
                })
    write(pd.DataFrame(surface), args.output, "figure2_authority_surface_bins.csv")

    temporal = results[results.model.eq("temporal")][keys + ["rank"]].rename(columns={"rank": "temporal_rank"})
    graph = results[results.model.eq("graph_only")][keys + ["rank", "mrr"]].rename(columns={"rank": "graph_rank", "mrr": "graph_mrr"})
    route = quantum[keys + ["alpha", "mrr"]].rename(columns={"mrr": "quantum_mrr"}).merge(temporal, on=keys).merge(graph, on=keys)
    route["regret"] = np.log1p(route.temporal_rank) - np.log1p(route.graph_rank)
    route["graph_beats_temporal"] = (route.graph_rank < route.temporal_rank).astype(int)
    route["regret_bin"] = pd.qcut(route.regret, q=14, duplicates="drop")
    regret = route.groupby("regret_bin", observed=True).agg(
        mean_regret=("regret", "mean"), mean_authority=("alpha", "mean"),
        authority_std=("alpha", "std"), graph_win_rate=("graph_beats_temporal", "mean"), records=("alpha", "size"),
    ).reset_index(drop=True)
    regret["authority_se"] = regret.authority_std.fillna(0) / np.sqrt(regret.records)
    write(regret, args.output, "figure2_cross_fitted_regret_response.csv")

    temporal_mrr = results[results.model.eq("temporal")].groupby("condition").mrr.mean()
    graph_mrr = results[results.model.eq("graph_only")].groupby("condition").mrr.mean()
    alignment = quantum.groupby("condition").agg(
        mean_quantum_authority=("alpha", "mean"), authority_std=("alpha", "std"), records=("alpha", "size"),
    ).reset_index()
    alignment["graph_minus_temporal_mrr"] = alignment.condition.map(graph_mrr - temporal_mrr)
    alignment["authority_se"] = alignment.authority_std / np.sqrt(alignment.records)
    write(alignment, args.output, "figure2_condition_utility_alignment.csv")

    exact = quantum.set_index(keys)
    shot_rows = []
    for shots in (32, 64, 128, 256, 512, 1024):
        work = results[results.model.eq(f"quantum_shot_{shots}")].set_index(keys)
        for condition in ORDER:
            item = work[work.index.get_level_values("condition") == condition]
            base = exact.reindex(item.index)
            shot_rows.append({
                "condition": condition,
                "shots": shots,
                "mean_absolute_mrr_drift": np.abs(item.mrr - base.mrr).mean(),
                "mean_absolute_authority_drift": np.abs(item.alpha - base.alpha).mean(),
                "records": len(item),
            })
    write(pd.DataFrame(shot_rows), args.output, "figure3_finite_shot_stability.csv")

    noise_rows = []
    for p, r in ((.000, .000), (.005, .010), (.010, .020), (.020, .030), (.050, .050)):
        work = results[results.model.eq(f"quantum_noise_p{p:.3f}_r{r:.3f}")].set_index(keys)
        for condition in ORDER:
            item = work[work.index.get_level_values("condition") == condition]
            base = exact.reindex(item.index)
            noise_rows.append({
                "condition": condition,
                "depolarizing_probability": p,
                "readout_error": r,
                "mean_mrr_shift": (item.mrr - base.mrr).mean(),
                "mean_absolute_authority_drift": np.abs(item.alpha - base.alpha).mean(),
                "records": len(item),
            })
    write(pd.DataFrame(noise_rows), args.output, "figure3_noise_sensitivity.csv")


if __name__ == "__main__":
    main()
