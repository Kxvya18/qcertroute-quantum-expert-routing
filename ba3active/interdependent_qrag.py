"""Quantum-first contextual routing benchmark on observed LANL events.

The task predicts a real next authentication destination from strictly earlier
LANL authentication and DNS events.  A temporal expert and a source-conditioned
graph-diffusion expert emit separate probability distributions. The primary
decision policy is a validation-selected variational quantum circuit that
observes telemetry availability, source history, expert confidence, and
top-ranked overlap, then controls their probability-level mixture.
Cross-fitted router training prevents the temporal expert from supplying
in-sample confidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .lanl_real_qrag import (
    REAL_SEEDS,
    RealEpisode,
    chronological_folds,
    load_episode_cache,
    save_episode_cache,
    _counter_sum,
    _read_stream,
    _values,
)
from .metrics import ranking, softmax
from .qrag_moe import EntangledQuantumRouter, InteractionClassicalRouter, QuantumReservoirRouter
from .rag_moe import CONDITIONS as TELEMETRY_CONDITIONS, EpisodeData, _gini, mask_for, scaler_fit, tensors, train_temporal


FEATURE_NAMES = (
    "missing_fraction",
    "source_history",
    "source_novelty",
    "temporal_entropy",
    "graph_entropy",
    "temporal_margin",
    "graph_margin",
    "top10_overlap",
    "graph_history_retention",
)
HISTORY_CONDITIONS = ("history_random50",)
CONDITIONS = (*TELEMETRY_CONDITIONS, *HISTORY_CONDITIONS)
TEMPORAL_CONDITION = {condition: "complete" for condition in HISTORY_CONDITIONS}
MAIN_QUANTUM = "quantum_moe_selected"
FULL_CONTEXT_QUANTUM = "quantum_moe_q4d2"
QUANTUM_SPECS = {
    "quantum_moe_q2d2": dict(n_qubits=2, depth=2, entanglement="ring", parity=True),
    "quantum_moe_q4d1": dict(n_qubits=4, depth=1, entanglement="linear", parity=True),
    FULL_CONTEXT_QUANTUM: dict(n_qubits=4, depth=2, entanglement="ring", parity=True),
    "quantum_moe_q4d3": dict(n_qubits=4, depth=3, entanglement="ring", parity=True),
    "quantum_moe_local_only": dict(n_qubits=4, depth=2, entanglement="ring", parity=False),
}
RESERVOIR_SPECS = {
    "quantum_reservoir_r2": dict(reservoirs=2, n_qubits=4, depth=2, parity=True),
    "quantum_reservoir_r4": dict(reservoirs=4, n_qubits=4, depth=2, parity=True),
}
FEATURE_MASKS = {
    FULL_CONTEXT_QUANTUM: np.ones(len(FEATURE_NAMES), dtype=np.float32),
    "quantum_no_graph_context": np.asarray((1, 0, 0, 1, 0, 1, 0, 0, 0), dtype=np.float32),
    "quantum_no_temporal_context": np.asarray((1, 1, 1, 0, 1, 0, 1, 0, 1), dtype=np.float32),
    "quantum_no_cross_expert_context": np.asarray((1, 1, 1, 1, 1, 1, 1, 0, 1), dtype=np.float32),
}
SHOT_GRID = (32, 64, 128, 256, 512, 1024)
NOISE_GRID = ((0.000, 0.000), (0.005, 0.010), (0.010, 0.020), (0.020, 0.030), (0.050, 0.050))
EVIDENCE_SEEDS = (131, 271, 419)
MASK_SEED = 20260922


@dataclass
class RouterRecords:
    features: np.ndarray
    temporal_truth: np.ndarray
    graph_truth: np.ndarray
    temporal_square: np.ndarray
    graph_square: np.ndarray
    cross_product: np.ndarray
    identifiers: np.ndarray
    condition_ids: np.ndarray
    temporal_rank: np.ndarray
    graph_rank: np.ndarray


class TemporalEvidenceEnsemble(nn.Module):
    """Fixed evidence ensemble shared by every router initialization."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def temporal(self, features: torch.Tensor) -> torch.Tensor:
        probabilities = torch.stack(
            [torch.softmax(member.temporal(features).squeeze(-1), dim=0) for member in self.members]
        ).mean(dim=0)
        return torch.log(torch.clamp(probabilities, min=1e-9)).unsqueeze(-1)


def train_temporal_ensemble(
    train: list[EpisodeData],
    mean: np.ndarray,
    std: np.ndarray,
    seed_offset: int,
    epochs: int,
    device: str,
) -> TemporalEvidenceEnsemble:
    members = [
        train_temporal(
            train, mean, std, evidence_seed + seed_offset,
            epochs=epochs, device=device, batch_size=48,
        ).to("cpu").eval()
        for evidence_seed in EVIDENCE_SEEDS
    ]
    return TemporalEvidenceEnsemble(members).eval()


def _stable_unit(*parts: object) -> float:
    """Deterministic edge masking that is stable across Python processes."""
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2**64


def _retain_history_edge(condition: str, bucket: int, source: str, target: str) -> bool:
    if condition == "history_random50":
        return _stable_unit(condition, bucket, source, target) < 0.5
    return True


def _weighted_transition(auth, dns, buckets, candidates, observation, condition="complete"):
    index = {node: position for position, node in enumerate(candidates)}
    adjacency = np.zeros((len(candidates), len(candidates)), dtype=np.float32)
    total_weight = retained_weight = 0.0
    for stream, modality_weight in ((auth, 1.0), (dns, 0.35)):
        for bucket in buckets:
            decay = float(np.exp(-0.35 * max(observation - bucket, 0)))
            for source, destinations in stream.contacts[bucket].items():
                source_index = index.get(source)
                if source_index is None:
                    continue
                for target, count in destinations.items():
                    target_index = index.get(target)
                    if target_index is not None and target_index != source_index:
                        weight = modality_weight * decay * count
                        total_weight += weight
                        if _retain_history_edge(condition, bucket, source, target):
                            adjacency[source_index, target_index] += weight
                            retained_weight += weight
    row_sum = adjacency.sum(axis=1, keepdims=True)
    transition = np.divide(adjacency, row_sum, out=np.zeros_like(adjacency), where=row_sum > 0)
    return transition, adjacency.sum(axis=0), index, retained_weight / max(total_weight, 1e-12)


def _source_restart(auth, dns, buckets, source, candidates, index, observation, condition="complete"):
    restart = np.zeros(len(candidates), dtype=np.float64)
    for stream, modality_weight in ((auth, 1.0), (dns, 0.35)):
        for bucket in buckets:
            decay = float(np.exp(-0.35 * max(observation - bucket, 0)))
            for target, count in stream.contacts[bucket].get(source, Counter()).items():
                target_index = index.get(target)
                if target_index is not None and target != source and _retain_history_edge(
                    condition, bucket, source, target
                ):
                    restart[target_index] += modality_weight * decay * count
    history = float(restart.sum())
    source_index = index.get(source)
    if source_index is not None:
        restart[source_index] += max(1.0, 0.15 * history)
    if restart.sum() <= 0:
        restart[:] = 1.0
    restart /= restart.sum()
    return restart, history


def _personalized_diffusion(transition, restart, teleport=0.18, iterations=30):
    probability = restart.copy()
    for _ in range(iterations):
        walk = probability @ transition
        missing_mass = max(0.0, 1.0 - float(walk.sum()))
        updated = teleport * restart + (1.0 - teleport) * (walk + missing_mass * restart)
        if np.abs(updated - probability).sum() < 1e-9:
            probability = updated
            break
        probability = updated
    probability = np.clip(probability, 1e-12, None)
    return probability / probability.sum()


def extract_diffusion_episodes(
    auth_path: Path,
    dns_path: Path,
    bucket_seconds: int = 600,
    max_seconds: int = 43_200,
    candidate_cap: int = 400,
    episodes_per_bucket: int = 8,
) -> tuple[list[RealEpisode], dict]:
    """Create source-conditioned real next-contact episodes using only past edges."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="lanl-stream") as pool:
        auth_future = pool.submit(_read_stream, auth_path, "auth", bucket_seconds, max_seconds)
        dns_future = pool.submit(_read_stream, dns_path, "dns", bucket_seconds, max_seconds)
        auth, dns = auth_future.result(), dns_future.result()
    episodes, identifier = [], 0
    potential_labels = retained_labels = 0
    final_bucket = min(auth.max_time, dns.max_time, max_seconds - 1) // bucket_seconds
    for target_bucket in range(9, final_bucket + 1):
        calibration = range(target_bucket - 9, target_bucket - 3)
        fitting = range(target_bucket - 3, target_bucket - 1)
        history_buckets = list(range(target_bucket - 9, target_bucket - 1))
        observation = target_bucket - 1
        calibration_activity = _counter_sum(auth.activity, calibration)
        calibration_activity.update(_counter_sum(dns.activity, calibration))
        fitting_activity = _counter_sum(auth.activity, fitting)
        fitting_activity.update(_counter_sum(dns.activity, fitting))
        observed_activity = auth.activity[observation] + dns.activity[observation]
        candidate_score = calibration_activity + fitting_activity + observed_activity
        candidates = [node for node, _ in candidate_score.most_common(candidate_cap)]
        if len(candidates) < 50:
            continue
        transition, indegree, candidate_index, _ = _weighted_transition(
            auth, dns, history_buckets, candidates, observation
        )
        history_transitions = {}
        for history_condition in HISTORY_CONDITIONS:
            history_transitions[history_condition] = _weighted_transition(
                auth, dns, history_buckets, candidates, observation, history_condition
            )
        source_score = auth.source_activity[observation] + dns.source_activity[observation]
        eligible = []
        for source, target in auth.first_destination[target_bucket].items():
            if source_score.get(source, 0) < 2:
                continue
            potential_labels += 1
            if target not in candidate_index:
                continue
            retained_labels += 1
            eligible.append((source_score[source], source, target))
        eligible.sort(key=lambda item: (-item[0], item[1], item[2]))
        if len(eligible) > episodes_per_bucket:
            eligible = [eligible[i] for i in np.linspace(0, len(eligible) - 1, episodes_per_bucket, dtype=int)]
        auth_global, dns_global = auth.activity[observation], dns.activity[observation]

        for _, source, target in eligible:
            restart, source_history = _source_restart(
                auth, dns, history_buckets, source, candidates, candidate_index, observation
            )
            graph_probability = _personalized_diffusion(transition, restart)
            keep = np.asarray([node != source for node in candidates], dtype=bool)
            episode_candidates = [node for node in candidates if node != source]
            graph_probability = graph_probability[keep]
            graph_probability /= graph_probability.sum()
            degree = indegree[keep]
            structural_by_condition = {}
            history_retention = {}
            for history_condition, (history_transition, _, history_index, retained_fraction) in history_transitions.items():
                history_restart, _ = _source_restart(
                    auth, dns, history_buckets, source, candidates, history_index, observation,
                    history_condition,
                )
                history_probability = _personalized_diffusion(history_transition, history_restart)[keep]
                history_probability /= history_probability.sum()
                structural_by_condition[history_condition] = np.log(
                    np.clip(history_probability, 1e-12, 1.0)
                ).astype(np.float32)
                history_retention[history_condition] = float(retained_fraction)
            query = np.column_stack((
                np.log1p(_values(auth_global, episode_candidates)),
                np.log1p(_values(auth.contacts[observation].get(source, Counter()), episode_candidates)),
                np.log1p(_values(dns_global, episode_candidates)),
                np.log1p(_values(dns.contacts[observation].get(source, Counter()), episode_candidates)),
            )).astype(np.float32)
            pre = np.log1p(_values(calibration_activity, episode_candidates)).astype(np.float32)
            data = EpisodeData(
                identifier=identifier,
                x=np.column_stack((pre, query)).astype(np.float32),
                structural=np.log(np.clip(graph_probability, 1e-12, 1.0)).astype(np.float32),
                degree=degree.astype(np.float32),
                truth=episode_candidates.index(target),
                rho=0.0,
                se=0.0,
                gini=_gini(degree),
                source_history=float(np.log1p(source_history)),
                source_novelty=float(source_history <= 0.0),
                structural_by_condition=structural_by_condition,
                graph_history_retention=history_retention,
            )
            episodes.append(RealEpisode(
                data=data,
                timestamp=target_bucket * bucket_seconds,
                source=source,
                target=target,
                beta=float("nan"),
                information=source_history,
                candidates=len(episode_candidates),
            ))
            identifier += 1
    target_counts = Counter(item.target for item in episodes)
    audit = {
        "graph_expert": "source-conditioned recency-weighted personalized diffusion",
        "history_stress_conditions": {
            "history_random50": "deterministic independent retention of historical candidate edges",
        },
        "teleport_probability": 0.18,
        "diffusion_iterations": 30,
        "auth_lines": auth.lines,
        "dns_lines": dns.lines,
        "bucket_seconds": bucket_seconds,
        "max_seconds": max_seconds,
        "candidate_cap": candidate_cap,
        "episodes_per_bucket": episodes_per_bucket,
        "episodes": len(episodes),
        "potential_labels": potential_labels,
        "candidate_retained_labels": retained_labels,
        "candidate_label_recall": retained_labels / max(potential_labels, 1),
        "unique_sources": len({item.source for item in episodes}),
        "unique_targets": len(target_counts),
        "top_target_fraction": max(target_counts.values(), default=0) / max(len(episodes), 1),
        "first_timestamp": min((item.timestamp for item in episodes), default=None),
        "last_timestamp": max((item.timestamp for item in episodes), default=None),
        "extraction_seconds": time.perf_counter() - started,
        "threads": 2,
    }
    return episodes, audit


def _normalized_entropy(probability: np.ndarray) -> float:
    denominator = max(float(np.log(max(len(probability), 2))), 1e-12)
    return float(-np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0))) / denominator)


def _confidence_margin(probability: np.ndarray) -> float:
    largest = np.partition(probability, -2)[-2:]
    return float(largest.max() - largest.min())


def _top_overlap(first: np.ndarray, second: np.ndarray, count: int = 10) -> float:
    count = min(count, len(first), len(second))
    first_top = set(np.argpartition(first, -count)[-count:].tolist())
    second_top = set(np.argpartition(second, -count)[-count:].tolist())
    return float(len(first_top.intersection(second_top)) / max(count, 1))


def expert_evidence(
    model: nn.Module,
    data: EpisodeData,
    condition: str,
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    telemetry_condition = TEMPORAL_CONDITION.get(condition, condition)
    mask = mask_for(data, telemetry_condition, np.random.default_rng(seed + data.identifier))
    x, _, _, _ = tensors(data, mask, mean, std)
    model.eval()
    with torch.no_grad():
        temporal = softmax(model.temporal(x).squeeze(-1).numpy())
    structural_variants = getattr(data, "structural_by_condition", {})
    retention_variants = getattr(data, "graph_history_retention", {})
    graph = softmax(structural_variants.get(condition, data.structural))
    history_retention = float(retention_variants.get(condition, 1.0))
    feature = np.asarray((
        1.0 - float(mask.mean()),
        data.source_history,
        data.source_novelty,
        _normalized_entropy(temporal),
        _normalized_entropy(graph),
        _confidence_margin(temporal),
        _confidence_margin(graph),
        _top_overlap(temporal, graph),
        history_retention,
    ), dtype=np.float32)
    return temporal, graph, feature


def _append_router_record(
    rows: dict[str, list],
    temporal: np.ndarray,
    graph: np.ndarray,
    feature: np.ndarray,
    data: EpisodeData,
    condition_id: int,
) -> None:
    rows["features"].append(feature)
    rows["temporal_truth"].append(temporal[data.truth])
    rows["graph_truth"].append(graph[data.truth])
    rows["temporal_square"].append(float(np.dot(temporal, temporal)))
    rows["graph_square"].append(float(np.dot(graph, graph)))
    rows["cross_product"].append(float(np.dot(temporal, graph)))
    rows["identifiers"].append(data.identifier)
    rows["condition_ids"].append(condition_id)
    rows["temporal_rank"].append(ranking(temporal, data.truth)["rank"])
    rows["graph_rank"].append(ranking(graph, data.truth)["rank"])


def cross_fitted_router_records(
    train: list[EpisodeData],
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
    temporal_device: str,
) -> tuple[nn.Module, RouterRecords]:
    rows = {name: [] for name in RouterRecords.__dataclass_fields__}
    ordered = list(train)
    halves = (ordered[::2], ordered[1::2])
    for split, (target, source) in enumerate(((halves[0], halves[1]), (halves[1], halves[0]))):
        source = source or target
        cross_model = train_temporal_ensemble(
            source, mean, std, 1000 * split, epochs=8, device=temporal_device
        )
        for data in target:
            for offset, condition in enumerate(CONDITIONS):
                temporal, graph, feature = expert_evidence(
                    cross_model, data, condition, mean, std, MASK_SEED + 1009 * offset
                )
                _append_router_record(rows, temporal, graph, feature, data, offset)
    final_model = train_temporal_ensemble(
        train, mean, std, 5000, epochs=12, device=temporal_device
    )
    records = RouterRecords(**{
        name: np.asarray(
            values,
            dtype=(np.int64 if name in {"identifiers", "condition_ids"} else np.float32),
        )
        for name, values in rows.items()
    })
    return final_model, records


def regret_supervision(records: RouterRecords, temperature: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Return soft graph-admission targets from cross-fitted rank regret.

    Positive log-rank regret means that the graph expert ranked the true node
    above the temporal expert. Ties receive little weight, preventing the many
    equivalent sparse-regime rankings from forcing an arbitrary gate value.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    regret = np.log1p(records.temporal_rank) - np.log1p(records.graph_rank)
    target = 1.0 / (1.0 + np.exp(-regret / temperature))
    weight = 0.05 + 0.95 * np.tanh(np.abs(regret) / temperature)
    return target.astype(np.float32), weight.astype(np.float32)


def _proper_mixture_loss(alpha: torch.Tensor, tensors_: dict[str, torch.Tensor]) -> torch.Tensor:
    temporal_truth = tensors_["temporal_truth"]
    graph_truth = tensors_["graph_truth"]
    truth_probability = (1.0 - alpha) * temporal_truth + alpha * graph_truth
    nll = -torch.log(torch.clamp(truth_probability, min=1e-9))
    square = (
        (1.0 - alpha).square() * tensors_["temporal_square"]
        + alpha.square() * tensors_["graph_square"]
        + 2.0 * alpha * (1.0 - alpha) * tensors_["cross_product"]
    )
    brier = square - 2.0 * truth_probability + 1.0
    return nll.mean() + 0.10 * brier.mean()


def _regret_router_loss(alpha: torch.Tensor, tensors_: dict[str, torch.Tensor]) -> torch.Tensor:
    target = tensors_["gate_target"]
    weight = tensors_["gate_weight"]
    binary = nn.functional.binary_cross_entropy(alpha, target, reduction="none")
    supervised = (weight * binary).sum() / torch.clamp(weight.sum(), min=1e-8)
    balance_terms = []
    for condition_id in torch.unique(tensors_["condition_ids"]):
        selected = tensors_["condition_ids"] == condition_id
        balance_terms.append((alpha[selected].mean() - target[selected].mean()).square())
    balance = torch.stack(balance_terms).mean()
    # Proper scoring remains a weak secondary objective. Rank regret, rather
    # than likelihood concentration, determines which expert controls a case.
    return supervised + 0.15 * balance + 0.04 * _proper_mixture_loss(alpha, tensors_)


def train_router(
    model: nn.Module,
    records: RouterRecords,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_mask: np.ndarray,
    seed: int,
    device: str,
    epochs: int = 120,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    model = model.to(device)
    scaled = ((records.features - feature_mean) / feature_std) * feature_mask
    gate_target, gate_weight = regret_supervision(records)
    identifiers = np.unique(records.identifiers)
    validation_ids = set(identifiers[max(1, int(0.8 * len(identifiers))):].tolist())
    validation_mask = np.asarray([identifier in validation_ids for identifier in records.identifiers])
    if not validation_mask.any() or validation_mask.all():
        validation_mask[-max(1, len(validation_mask) // 5):] = True
    train_mask = ~validation_mask
    device_object = torch.device(device)

    def pack(mask):
        return {
            "features": torch.tensor(scaled[mask], dtype=torch.float32, device=device_object),
            "temporal_truth": torch.tensor(records.temporal_truth[mask], device=device_object),
            "graph_truth": torch.tensor(records.graph_truth[mask], device=device_object),
            "temporal_square": torch.tensor(records.temporal_square[mask], device=device_object),
            "graph_square": torch.tensor(records.graph_square[mask], device=device_object),
            "cross_product": torch.tensor(records.cross_product[mask], device=device_object),
            "missing_fraction": torch.tensor(records.features[mask, 0], device=device_object),
            "gate_target": torch.tensor(gate_target[mask], device=device_object),
            "gate_weight": torch.tensor(gate_weight[mask], device=device_object),
            "condition_ids": torch.tensor(records.condition_ids[mask], device=device_object),
        }

    training, validation = pack(train_mask), pack(validation_mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.018, weight_decay=1e-4)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        alpha = model(training["features"])
        loss = _regret_router_loss(alpha, training)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(_regret_router_loss(model(validation["features"]), validation).item())
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 25:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    model = model.to("cpu").eval()
    return model, {
        "training_seconds": elapsed,
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "train_records": int(train_mask.sum()),
        "validation_records": int(validation_mask.sum()),
        "gate_target_mean": float(gate_target[train_mask].mean()),
        "gate_target_validation_mean": float(gate_target[validation_mask].mean()),
    }


def fit_experts_and_routers(
    train: list[EpisodeData],
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
    temporal_device: str,
    router_device: str,
) -> tuple[nn.Module, dict[str, nn.Module], np.ndarray, np.ndarray, list[dict]]:
    temporal, records = cross_fitted_router_records(train, mean, std, seed, temporal_device)
    feature_mean = records.features.mean(axis=0)
    feature_std = records.features.std(axis=0).clip(1e-5)
    models: dict[str, nn.Module] = {}
    resources = []

    def quantum_model(name: str) -> nn.Module:
        if name in QUANTUM_SPECS:
            return EntangledQuantumRouter(feature_count=len(FEATURE_NAMES), **QUANTUM_SPECS[name])
        if name in RESERVOIR_SPECS:
            return QuantumReservoirRouter(feature_count=len(FEATURE_NAMES), **RESERVOIR_SPECS[name])
        raise KeyError(f"unknown quantum architecture {name}")

    candidates: list[tuple[str, nn.Module, np.ndarray]] = [
        ("classical_moe", InteractionClassicalRouter(feature_count=len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES), dtype=np.float32)),
    ]
    candidates.extend(
        (name, quantum_model(name), np.ones(len(FEATURE_NAMES), dtype=np.float32))
        for name in QUANTUM_SPECS
    )
    candidates.extend(
        (name, quantum_model(name), np.ones(len(FEATURE_NAMES), dtype=np.float32))
        for name in RESERVOIR_SPECS
    )
    for index, (name, model, feature_mask) in enumerate(candidates):
        fitted, resource = train_router(
            model, records, feature_mean, feature_std, feature_mask,
            seed + 401 + index * 53, router_device,
        )
        models[name] = fitted
        resources.append({
            "model": name,
            "parameters": sum(parameter.numel() for parameter in fitted.parameters() if parameter.requires_grad),
            "feature_mask": "".join(str(int(value)) for value in feature_mask),
            "router_device": router_device,
            **resource,
        })
    eligible_names = set(QUANTUM_SPECS) | set(RESERVOIR_SPECS)
    eligible = [row for row in resources if row["model"] in eligible_names]
    selected = min(eligible, key=lambda row: (row["validation_loss"], row["parameters"]))
    models[MAIN_QUANTUM] = deepcopy(models[selected["model"]])
    resources.append({
        **selected,
        "model": MAIN_QUANTUM,
        "selected_from": selected["model"],
    })
    # Context ablations must share the selected circuit family. Changing both
    # the input mask and the quantum architecture would not identify context
    # dependence.
    for offset, (name, feature_mask) in enumerate(
        (item for item in FEATURE_MASKS.items() if item[0] != FULL_CONTEXT_QUANTUM)
    ):
        fitted, resource = train_router(
            quantum_model(selected["model"]), records, feature_mean, feature_std,
            feature_mask, seed + 1301 + offset * 53, router_device,
        )
        models[name] = fitted
        resources.append({
            "model": name,
            "parameters": sum(
                parameter.numel() for parameter in fitted.parameters() if parameter.requires_grad
            ),
            "feature_mask": "".join(str(int(value)) for value in feature_mask),
            "router_device": router_device,
            "selected_from": selected["model"],
            **resource,
        })
    return temporal, models, feature_mean, feature_std, resources


def _mixed(temporal: np.ndarray, graph: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * temporal + alpha * graph


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_interim_report(results: pd.DataFrame, output: Path) -> None:
    if results.empty:
        return
    summary = results.groupby(["condition", "model"], as_index=False).agg(
        episodes=("episode", "nunique"),
        seeds=("seed", "nunique"),
        mrr=("mrr", "mean"),
        hit10=("hit10", "mean"),
        mean_rank=("rank", "mean"),
        nll=("nll", "mean"),
        alpha=("alpha", "mean"),
    )
    _atomic_csv(summary, output / "interim_summary.csv")
    comparisons = []
    for condition in CONDITIONS:
        work = results[results.condition == condition]
        keys = ["fold", "seed", "episode", "timestamp"]
        baselines = (
            "temporal", "graph_only", "classical_moe", "quantum_no_graph_context",
            "quantum_no_temporal_context", "quantum_no_cross_expert_context",
            "quantum_moe_local_only",
        )
        wide = work[work.model.isin((MAIN_QUANTUM, *baselines))].pivot_table(
            index=keys, columns="model", values="mrr", aggfunc="mean"
        )
        if MAIN_QUANTUM not in wide:
            continue
        for baseline in baselines:
            if baseline in wide:
                delta = wide[MAIN_QUANTUM] - wide[baseline]
                comparisons.append({
                    "condition": condition,
                    "comparison": f"{MAIN_QUANTUM}_minus_{baseline}",
                    "episodes": int(len(delta)),
                    "delta_mrr": float(delta.mean()),
                    "win_rate": float((delta > 0).mean()),
                    "tie_rate": float((delta == 0).mean()),
                })
    claim_frame = pd.DataFrame(comparisons)
    _atomic_csv(claim_frame, output / "interim_claims.csv")
    focus = claim_frame[claim_frame.comparison == f"{MAIN_QUANTUM}_minus_classical_moe"]
    message = " | ".join(
        f"{row.condition}:Q-C={row.delta_mrr:+.4f}" for row in focus.itertuples()
    )
    print(f"interim checkpoint | {message}", flush=True)


def evaluate_interdependent(
    episodes: list[RealEpisode],
    output: Path,
    temporal_device: str,
    router_device: str,
    folds: int,
    seeds: tuple[int, ...],
    resume: bool = False,
    max_runs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {item.data.identifier: item for item in episodes}
    result_path = output / "interdependent_qrag_results.csv"
    resource_path = output / "interdependent_qrag_resources.csv"
    if resume and result_path.exists() and resource_path.exists():
        result_rows = pd.read_csv(result_path).to_dict(orient="records")
        resource_rows = pd.read_csv(resource_path).to_dict(orient="records")
    else:
        result_rows, resource_rows = [], []
    completed = {(int(row["fold"]), int(row["seed"])) for row in resource_rows}
    runs_this_invocation = 0
    for fold, raw_train, raw_test in chronological_folds(episodes, folds):
        train = [item.data for item in raw_train]
        test = [item.data for item in raw_test]
        train_sources = {item.source for item in raw_train}
        train_targets = {item.target for item in raw_train}
        mean, std = scaler_fit(train)
        for seed in seeds:
            if (fold, seed) in completed:
                print(f"skip completed fold={fold} seed={seed}", flush=True)
                continue
            if max_runs is not None and runs_this_invocation >= max_runs:
                results = pd.DataFrame(result_rows)
                resources = pd.DataFrame(resource_rows)
                _write_interim_report(results, output)
                return results, resources
            print(
                f"fold={fold} seed={seed} train={len(train)} test={len(test)} "
                f"temporal={temporal_device} router={router_device}", flush=True
            )
            temporal, routers, feature_mean, feature_std, resources = fit_experts_and_routers(
                train, mean, std, seed, temporal_device, router_device
            )
            pair_resources = []
            for resource in resources:
                pair_resources.append({
                    "fold": fold,
                    "seed": seed,
                    "train_episodes": len(train),
                    "test_episodes": len(test),
                    **resource,
                })

            pair_rows = []
            for data in test:
                meta = metadata[data.identifier]
                for offset, condition in enumerate(CONDITIONS):
                    temporal_probability, graph_probability, raw_feature = expert_evidence(
                        temporal, data, condition, mean, std, MASK_SEED + 1009 * offset
                    )
                    scaled_feature = (raw_feature - feature_mean) / feature_std
                    probabilities = {
                        "degree_baseline": softmax(np.log1p(data.degree)),
                        "temporal": temporal_probability,
                        "graph_only": graph_probability,
                        "unconditional_mixture": _mixed(temporal_probability, graph_probability, 0.5),
                        "fixed_missingness_mixture": _mixed(
                            temporal_probability, graph_probability, float(raw_feature[0])
                        ),
                    }
                    admissions = {
                        "degree_baseline": 0.0,
                        "temporal": 0.0,
                        "graph_only": 1.0,
                        "unconditional_mixture": 0.5,
                        "fixed_missingness_mixture": float(raw_feature[0]),
                    }
                    for name, router in routers.items():
                        mask = FEATURE_MASKS.get(name, np.ones(len(FEATURE_NAMES), dtype=np.float32))
                        feature_tensor = torch.tensor(scaled_feature * mask, dtype=torch.float32)
                        with torch.no_grad():
                            alpha = float(router(feature_tensor).item())
                        probabilities[name] = _mixed(temporal_probability, graph_probability, alpha)
                        admissions[name] = alpha
                        if name == MAIN_QUANTUM:
                            for shots in SHOT_GRID:
                                shot_alpha = router.finite_shot(
                                    feature_tensor, shots, np.random.default_rng(seed + data.identifier + shots * 17)
                                )
                                key = f"quantum_shot_{shots}"
                                probabilities[key] = _mixed(temporal_probability, graph_probability, shot_alpha)
                                admissions[key] = shot_alpha
                            for depolarizing, readout in NOISE_GRID:
                                noise_alpha = router.finite_shot(
                                    feature_tensor, 256,
                                    np.random.default_rng(
                                        seed + data.identifier + int(1e5 * depolarizing) + int(1e4 * readout)
                                    ),
                                    depolarizing=depolarizing,
                                    readout_error=readout,
                                )
                                key = f"quantum_noise_p{depolarizing:.3f}_r{readout:.3f}"
                                probabilities[key] = _mixed(temporal_probability, graph_probability, noise_alpha)
                                admissions[key] = noise_alpha

                    for model_name, probability in probabilities.items():
                        row = {
                            "fold": fold,
                            "seed": seed,
                            "episode": data.identifier,
                            "timestamp": meta.timestamp,
                            "condition": condition,
                            "model": model_name,
                            "alpha": admissions[model_name],
                            "candidates": meta.candidates,
                            "source_seen": int(meta.source in train_sources),
                            "target_seen": int(meta.target in train_targets),
                            **dict(zip(FEATURE_NAMES, raw_feature)),
                        }
                        row.update(ranking(probability, data.truth))
                        pair_rows.append(row)

            result_rows.extend(pair_rows)
            resource_rows.extend(pair_resources)
            runs_this_invocation += 1
            results = pd.DataFrame(result_rows)
            resource_frame = pd.DataFrame(resource_rows)
            _atomic_csv(results, result_path)
            _atomic_csv(resource_frame, resource_path)
            _write_interim_report(results, output)
            print(
                f"checkpoint saved fold={fold} seed={seed} "
                f"completed_pairs={len({(int(row['fold']), int(row['seed'])) for row in resource_rows})}",
                flush=True,
            )

    results = pd.DataFrame(result_rows)
    resources = pd.DataFrame(resource_rows)
    _atomic_csv(results, result_path)
    _atomic_csv(resources, resource_path)
    _write_interim_report(results, output)
    return results, resources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=Path)
    parser.add_argument("--dns", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--temporal-device", default="mps")
    parser.add_argument("--router-device", default="mps")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--bucket-seconds", type=int, default=600)
    parser.add_argument("--max-seconds", type=int, default=43200)
    parser.add_argument("--candidate-cap", type=int, default=400)
    parser.add_argument("--episodes-per-bucket", type=int, default=8)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in REAL_SEEDS))
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, help="Stop after this many new fold-seed pairs")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))

    if args.cache and args.cache.exists():
        episodes, audit = load_episode_cache(args.cache)
        if not episodes or any(
            condition not in getattr(item.data, "structural_by_condition", {})
            for item in episodes for condition in HISTORY_CONDITIONS
        ):
            parser.error("cache lacks historical-edge stress evidence; use a fresh cache and the original LANL streams")
    else:
        if args.auth is None or args.dns is None:
            parser.error("--auth and --dns are required when --cache is absent")
        for source_path in (args.auth, args.dns):
            if not source_path.is_file():
                parser.error(f"LANL stream is unavailable: {source_path}")
        episodes, audit = extract_diffusion_episodes(
            args.auth, args.dns, args.bucket_seconds, args.max_seconds,
            args.candidate_cap, args.episodes_per_bucket,
        )
        if args.cache:
            save_episode_cache(args.cache, episodes, audit)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "lanl_real_extraction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.extract_only:
        print(json.dumps({**audit, "output": str(args.output)}, indent=2))
        return
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    results, resources = evaluate_interdependent(
        episodes, args.output, args.temporal_device, args.router_device, args.folds, seeds,
        resume=args.resume, max_runs=args.max_runs,
    )
    completed_pairs = resources[["fold", "seed"]].drop_duplicates().shape[0] if not resources.empty else 0
    print(json.dumps({
        **audit,
        "result_rows": len(results),
        "resource_rows": len(resources),
        "temporal_device": args.temporal_device,
        "router_device": args.router_device,
        "batched_quantum_statevectors": True,
        "completed_fold_seed_pairs": completed_pairs,
        "expected_fold_seed_pairs": len(chronological_folds(episodes, args.folds)) * len(seeds),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
