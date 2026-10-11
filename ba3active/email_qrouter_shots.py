"""Joint-readout adaptive shots for the trained four-qubit email router."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .email_qrouter import CONDITIONS, load_episodes
from .qrag_moe import EntangledQuantumRouter


LOOKS = (16, 32, 64, 128, 256, 512)


def _topk(values: np.ndarray, k: int = 10) -> frozenset[int]:
    # A fixed node-id tie break makes endpoint-set invariance well defined.
    order = np.lexsort((np.arange(len(values)), -values))
    return frozenset(order[:k].tolist())


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


def _set(temporal: np.ndarray, graph: np.ndarray, gate: float) -> frozenset[int]:
    return _topk((1.0 - gate) * temporal + gate * graph)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("outputs/email_qrouter_v1/episodes.npz"))
    parser.add_argument("--training", type=Path, default=Path("outputs/email_qrouter_v1/train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/email_qrouter_v1/shots"))
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--shot-seed", type=int, default=4909)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error("confidence must be strictly between zero and one")
    data = load_episodes(args.episodes)
    with np.load(args.training / "predictions.npz", allow_pickle=False) as fit:
        split = fit["split"]
        mean, scale = fit["feature_mean"], fit["feature_scale"]
    test_episodes = np.flatnonzero(split == "test")
    model = EntangledQuantumRouter(feature_count=data.features.shape[-1], n_qubits=4, depth=2)
    model.load_state_dict(torch.load(args.training / f"quantum_seed{args.model_seed}.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    weight = model.readout.weight.detach().numpy().ravel()
    bias = float(model.readout.bias.detach().numpy()[0])
    signs = model.observable_signs.detach().numpy().T
    delta = 1.0 - args.confidence
    outcome_readout = signs @ weight
    readout_range = float(outcome_readout.max() - outcome_readout.min())
    rng = np.random.default_rng(args.shot_seed)
    rows = []
    with torch.no_grad():
        for episode in test_episodes:
            for condition_id, condition in enumerate(CONDITIONS):
                features = np.clip((data.features[episode, condition_id] - mean) / scale,
                                   -5.0, 5.0).astype(np.float32)
                temporal = data.temporal[episode, condition_id]
                graph = data.graph[episode, condition_id]
                agreement_set = _set(temporal, graph, 0.0)
                zero_shot = agreement_set == _set(temporal, graph, 1.0)
                probabilities = model.state_probabilities(torch.from_numpy(features)).numpy().ravel()
                exact_gate = _sigmoid(float(probabilities @ outcome_readout + bias))
                exact_set = _set(temporal, graph, exact_gate)
                draws = rng.choice(len(probabilities), size=LOOKS[-1], p=probabilities)
                observed = np.cumsum(outcome_readout[draws])
                observed_signs = np.cumsum(signs[draws], axis=0)
                selected_budget = 0 if zero_shot else LOOKS[-1]
                selected_gate = "" if zero_shot else 0.0
                selected_set = agreement_set if zero_shot else frozenset()
                certificate = "expert_agreement" if zero_shot else "unresolved"
                if not zero_shot:
                    for shots in LOOKS:
                        logit = float(observed[shots - 1] / shots + bias)
                        gate = _sigmoid(logit)
                        # Hoeffding for the bounded joint-readout scalar, unioned over looks.
                        radius = readout_range * np.sqrt(
                            np.log(2.0 * len(LOOKS) / delta) / (2.0 * shots))
                        left = _sigmoid(logit - radius)
                        right = _sigmoid(logit + radius)
                        selected_budget, selected_gate = shots, gate
                        selected_set = _set(temporal, graph, gate)
                        if _set(temporal, graph, left) == _set(temporal, graph, right):
                            certificate = "confidence"
                            break
                observable_budget = 0 if zero_shot else LOOKS[-1]
                observable_certified = bool(zero_shot)
                if not zero_shot:
                    for shots in LOOKS:
                        center = float(observed_signs[shots - 1] @ weight / shots + bias)
                        radius = min(2.0, np.sqrt(
                            2.0 * np.log(2.0 * signs.shape[1] * len(LOOKS) / delta) / shots))
                        error = radius * float(np.abs(weight).sum())
                        observable_budget = shots
                        if _set(temporal, graph, _sigmoid(center - error)) == _set(
                                temporal, graph, _sigmoid(center + error)):
                            observable_certified = True
                            break
                fixed_matches = {}
                for shots in LOOKS:
                    gate = _sigmoid(float(observed[shots - 1] / shots + bias))
                    fixed_matches[f"fixed{shots}_matches_exact"] = int(
                        _set(temporal, graph, gate) == exact_set)
                rows.append({
                    "episode": int(episode), "day": int(data.day[episode]),
                    "condition": condition, "shots": selected_budget,
                    "certificate": certificate,
                    "certified": int(certificate != "unresolved"),
                    "zero_shot": int(zero_shot),
                    "observable_bound_shots": observable_budget,
                    "observable_bound_certified": int(observable_certified),
                    "exact_gate": exact_gate, "measured_gate": selected_gate,
                    "absolute_gate_error": abs(selected_gate - exact_gate) if not zero_shot else "",
                    "top10_matches_exact": int(selected_set == exact_set),
                    **fixed_matches,
                    "top10_contains_target": int(data.truth[episode] in selected_set),
                    "exact_top10_contains_target": int(data.truth[episode] in exact_set),
                })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    certified = [row for row in rows if row["certified"]]
    summary = {"test_episodes": int(len(test_episodes)), "decisions": len(rows),
               "model_seed": args.model_seed, "shot_seed": args.shot_seed,
               "confidence": args.confidence, "looks": LOOKS,
               "average_shots": float(np.mean([row["shots"] for row in rows])),
               "median_shots": float(np.median([row["shots"] for row in rows])),
               "zero_shot_fraction": float(np.mean([row["zero_shot"] for row in rows])),
               "certified_fraction": float(len(certified) / len(rows)),
               "certified_top10_match_exact": (float(np.mean([
                   row["top10_matches_exact"] for row in certified])) if certified else None),
               "certified_hit10": (float(np.mean([
                   row["top10_contains_target"] for row in certified])) if certified else None),
               "exact_hit10_all": float(np.mean([
                   row["exact_top10_contains_target"] for row in rows])),
               "forced_top10_match_exact": float(np.mean([row["top10_matches_exact"] for row in rows])),
               "fixed512_match_exact": float(np.mean([row["fixed512_matches_exact"] for row in rows])),
               "observable_bound_average_shots": float(np.mean([row["observable_bound_shots"] for row in rows])),
               "observable_bound_certified_fraction": float(np.mean([row["observable_bound_certified"] for row in rows])),
               "fixed_budget_match_exact": {str(shots): float(np.mean([
                   row[f"fixed{shots}_matches_exact"] for row in rows])) for shots in LOOKS},
               "by_condition": {condition: {
                   "average_shots": float(np.mean([row["shots"] for row in rows if row["condition"] == condition])),
                   "certified_fraction": float(np.mean([row["certified"] for row in rows if row["condition"] == condition])),
                   "zero_shot_fraction": float(np.mean([row["zero_shot"] for row in rows if row["condition"] == condition])),
               } for condition in CONDITIONS}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
