"""Fast go or stop audit for the interdependent Q-RAG-MoE pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from .interdependent_qrag import CONDITIONS, MAIN_QUANTUM


def audit_checkpoint(results: pd.DataFrame) -> dict:
    summary = results.groupby(["condition", "model"], as_index=False).agg(
        mrr=("mrr", "mean"), alpha=("alpha", "mean"), episodes=("episode", "nunique")
    )
    pivot = summary.pivot(index="condition", columns="model", values="mrr")
    required = {
        MAIN_QUANTUM, "temporal", "graph_only", "classical_moe",
        "quantum_no_cross_expert_context", "quantum_moe_local_only", "quantum_shot_256",
    }
    missing = sorted(required.difference(pivot.columns))
    missing_conditions = sorted(set(CONDITIONS).difference(pivot.index))
    if missing or missing_conditions:
        return {"decision": "STOP", "missing_models": missing, "missing_conditions": missing_conditions}

    equal_weight_mrr = pivot.loc[list(CONDITIONS)].mean(axis=0)
    gate = summary[summary.model == MAIN_QUANTUM].set_index("condition").alpha
    graph_advantage = pivot["graph_only"] - pivot["temporal"]
    graph_favored = graph_advantage[graph_advantage > 0].index
    temporal_favored = graph_advantage[graph_advantage <= 0].index
    specialization = float(gate.loc[graph_favored].mean() - gate.loc[temporal_favored].mean())
    regime_alignment = float(graph_advantage.corr(gate, method="spearman"))
    gate_range = float(gate.max() - gate.min())
    quantum_mean = float(equal_weight_mrr[MAIN_QUANTUM])
    strongest_expert = float(max(equal_weight_mrr["temporal"], equal_weight_mrr["graph_only"]))
    classical_mean = float(equal_weight_mrr["classical_moe"])
    context_mrr_gain = quantum_mean - float(equal_weight_mrr["quantum_no_cross_expert_context"])
    parity_mrr_gain = quantum_mean - float(equal_weight_mrr["quantum_moe_local_only"])
    shot_drift = abs(float(equal_weight_mrr["quantum_shot_256"]) - quantum_mean)

    keys = ["fold", "seed", "episode", "timestamp", "condition"]
    expert_rank = results[results.model.isin(("temporal", "graph_only"))].pivot_table(
        index=keys, columns="model", values="rank", aggfunc="mean"
    ).reset_index()

    def routing_metrics(model: str) -> dict[str, float]:
        routed = results[results.model == model][keys + ["alpha"]].merge(expert_rank, on=keys)
        routed = routed[routed.temporal != routed.graph_only]
        truth = (routed.graph_only < routed.temporal).to_numpy(dtype=float)
        probability = np.clip(routed.alpha.to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        log_loss = float(np.mean(-(truth * np.log(probability) + (1.0 - truth) * np.log(1.0 - probability))))
        brier = float(np.mean((probability - truth) ** 2))
        positive, negative = probability[truth == 1.0], probability[truth == 0.0]
        auc = float(
            (positive[:, None] > negative).mean()
            + 0.5 * (positive[:, None] == negative).mean()
        ) if len(positive) and len(negative) else float("nan")
        return {"log_loss": log_loss, "brier": brier, "auc": auc, "non_tied": int(len(truth))}

    routing = {
        model: routing_metrics(model)
        for model in (
            MAIN_QUANTUM, "classical_moe", "quantum_no_graph_context",
            "quantum_no_temporal_context", "quantum_no_cross_expert_context",
            "quantum_moe_local_only",
        )
    }
    context_gain = (
        routing["quantum_no_cross_expert_context"]["log_loss"]
        - routing[MAIN_QUANTUM]["log_loss"]
    )
    context_auc_gain = (
        routing[MAIN_QUANTUM]["auc"]
        - routing["quantum_no_cross_expert_context"]["auc"]
    )
    parity_gain = (
        routing["quantum_moe_local_only"]["log_loss"]
        - routing[MAIN_QUANTUM]["log_loss"]
    )

    checks = {
        "gate_not_collapsed": gate_range >= 0.20,
        "correct_regime_ordering": routing[MAIN_QUANTUM]["auc"] >= 0.65 and specialization >= 0.10,
        "competitive_with_best_expert": quantum_mean >= strongest_expert - 0.005,
        "noninferior_to_classical": quantum_mean >= classical_mean - 0.005,
        "quantum_context_dependency": context_gain >= 0.020 or context_auc_gain >= 0.050,
        "parity_observable_dependency": parity_gain >= 0.020,
        "finite_shot_stability": shot_drift <= 0.010,
    }
    advantage_checks = all(checks.values())
    limits_checks = (
        checks["gate_not_collapsed"]
        and routing[MAIN_QUANTUM]["auc"] >= 0.65
        and specialization >= 0.10
        and checks["quantum_context_dependency"]
        and checks["finite_shot_stability"]
    )
    if advantage_checks:
        decision = "GO_QUANTUM_ADVANTAGE_STUDY"
    elif limits_checks:
        decision = "GO_QUANTUM_LIMITS_STUDY"
    else:
        decision = "STOP"
    return {
        "decision": decision,
        "primary_claim": (
            "A validation-selected quantum router learns the telemetry-to-topology regime transition "
            "and remains finite-shot stable; matched classical, parity, and reservoir controls determine "
            "whether this constitutes an advantage or a documented quantum limitation."
        ),
        "checks": checks,
        "conditions": list(CONDITIONS),
        "quantum_model": MAIN_QUANTUM,
        "equal_weight_mrr": {
            "quantum": quantum_mean,
            "temporal": float(equal_weight_mrr["temporal"]),
            "graph": float(equal_weight_mrr["graph_only"]),
            "classical": classical_mean,
        },
        "gate_by_condition": {key: float(gate[key]) for key in CONDITIONS},
        "gate_range": gate_range,
        "regime_specialization": specialization,
        "regime_alignment_spearman": regime_alignment,
        "graph_advantage_by_condition": {
            key: float(graph_advantage[key]) for key in CONDITIONS
        },
        "quantum_context_gain": context_gain,
        "quantum_context_auc_gain": context_auc_gain,
        "parity_observable_gain": parity_gain,
        "quantum_context_mrr_gain": context_mrr_gain,
        "parity_observable_mrr_gain": parity_mrr_gain,
        "routing_metrics": routing,
        "shot_256_absolute_drift": shot_drift,
        "episodes_per_condition": int(summary[summary.model == MAIN_QUANTUM].episodes.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.results / "interdependent_qrag_results.csv"
    if not result_path.exists():
        parser.error(f"missing checkpoint results at {result_path}")
    audit = audit_checkpoint(pd.read_csv(result_path))
    output = args.results / "checkpoint_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["decision"].startswith("GO_") else 2)


if __name__ == "__main__":
    main()
