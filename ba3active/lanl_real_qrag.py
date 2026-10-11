"""Real-event LANL benchmark for reliability-aware quantum graph routing.

The benchmark uses only observed LANL authentication and DNS events.  Each
episode predicts the first external authentication destination in the next
time bucket from strictly earlier graph and telemetry windows.  Controlled
missingness changes observability but never changes the graph or target label.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .metrics import ranking, softmax
from .prior import fit_endpoint_choice_prior, reliability
from .qrag_moe import (
    ClassicalReliabilityRouter,
    QuantumReliabilityRouter,
    _episode_feature,
    _train_small_router,
    prepare_router_training,
)
from .rag_moe import CONDITIONS, EpisodeData, _gini, mask_for, scaler_fit


REAL_SEEDS = (7, 19, 42, 73, 101)
QUANTUM_SPECS = {
    "q2d1_none": {"n_qubits": 2, "depth": 1, "entanglement": "none"},
    "q3d1_linear": {"n_qubits": 3, "depth": 1, "entanglement": "linear"},
    "q3d2_ring": {"n_qubits": 3, "depth": 2, "entanglement": "ring"},
    "q4d2_ring": {"n_qubits": 4, "depth": 2, "entanglement": "ring"},
    "q4d3_ring": {"n_qubits": 4, "depth": 3, "entanglement": "ring"},
}
QUANTUM_FEATURE_MASKS = {
    "q3d2_ring": np.ones(6, dtype=np.float32),
    "q3d2_no_graph_context": np.asarray((0, 1, 1, 1, 0, 0), dtype=np.float32),
    "q3d2_no_expert_context": np.asarray((1, 0, 0, 0, 1, 1), dtype=np.float32),
}
SHOT_GRID = (32, 64, 128, 256, 512, 1024)
NOISE_GRID = (
    (0.000, 0.000),
    (0.005, 0.010),
    (0.010, 0.020),
    (0.020, 0.030),
    (0.050, 0.050),
)


@dataclass
class StreamBuckets:
    activity: list[Counter]
    destination: list[Counter]
    contacts: list[dict[str, Counter]]
    first_destination: list[dict[str, str]]
    source_activity: list[Counter]
    lines: int
    max_time: int


@dataclass
class RealEpisode:
    data: EpisodeData
    timestamp: int
    source: str
    target: str
    beta: float
    information: float
    candidates: int


def _episode_payload(item: RealEpisode) -> dict:
    """Return a module-independent cache representation."""
    data = item.data
    return {
        "identifier": data.identifier,
        "x": data.x,
        "structural": data.structural,
        "degree": data.degree,
        "truth": data.truth,
        "rho": data.rho,
        "se": data.se,
        "gini": data.gini,
        "source_history": getattr(data, "source_history", 0.0),
        "source_novelty": getattr(data, "source_novelty", 0.0),
        "structural_by_condition": getattr(data, "structural_by_condition", {}),
        "graph_history_retention": getattr(data, "graph_history_retention", {}),
        "timestamp": item.timestamp,
        "source": item.source,
        "target": item.target,
        "beta": item.beta,
        "information": item.information,
        "candidates": item.candidates,
    }


def _episode_from_payload(payload: dict) -> RealEpisode:
    data = EpisodeData(
        identifier=int(payload["identifier"]),
        x=np.asarray(payload["x"], dtype=np.float32),
        structural=np.asarray(payload["structural"], dtype=np.float32),
        degree=np.asarray(payload["degree"], dtype=np.float32),
        truth=int(payload["truth"]),
        rho=float(payload["rho"]),
        se=float(payload["se"]),
        gini=float(payload["gini"]),
        source_history=float(payload.get("source_history", 0.0)),
        source_novelty=float(payload.get("source_novelty", 0.0)),
        structural_by_condition={
            str(key): np.asarray(value, dtype=np.float32)
            for key, value in payload.get("structural_by_condition", {}).items()
        },
        graph_history_retention={
            str(key): float(value)
            for key, value in payload.get("graph_history_retention", {}).items()
        },
    )
    return RealEpisode(
        data=data,
        timestamp=int(payload["timestamp"]),
        source=str(payload["source"]),
        target=str(payload["target"]),
        beta=float(payload["beta"]),
        information=float(payload["information"]),
        candidates=int(payload["candidates"]),
    )


def save_episode_cache(path: Path, episodes: list[RealEpisode], audit: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "lanl-real-qrag-v2",
        "episodes": [_episode_payload(item) for item in episodes],
        "audit": audit,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_episode_cache(path: Path) -> tuple[list[RealEpisode], dict]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and str(payload.get("format", "")).startswith("lanl-real-qrag-v"):
        return [_episode_from_payload(item) for item in payload["episodes"]], payload["audit"]
    # Backward compatibility for caches created before the portable format.
    episodes, audit = payload
    return episodes, audit


class LogisticRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, 1), nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _empty_buckets(count: int):
    return (
        [Counter() for _ in range(count)],
        [Counter() for _ in range(count)],
        [defaultdict(Counter) for _ in range(count)],
        [dict() for _ in range(count)],
        [Counter() for _ in range(count)],
    )


def _read_stream(path: Path, kind: str, bucket_seconds: int, max_seconds: int) -> StreamBuckets:
    count = max_seconds // bucket_seconds + 2
    activity, destination, contacts, first_destination, source_activity = _empty_buckets(count)
    lines = 0
    observed_max = 0
    try:
        with gzip.open(path, "rt", errors="ignore") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                parts = line.rstrip().split(",")
                try:
                    timestamp = int(parts[0])
                except (ValueError, IndexError):
                    continue
                if timestamp >= max_seconds:
                    break
                if kind == "auth":
                    if len(parts) < 9 or parts[-1] != "Success":
                        continue
                    source, target = parts[3], parts[4]
                else:
                    if len(parts) < 3:
                        continue
                    source, target = parts[1], parts[2]
                if not source.startswith("C") or not target.startswith("C") or source == target:
                    continue
                bucket = timestamp // bucket_seconds
                activity[bucket][source] += 1
                activity[bucket][target] += 1
                destination[bucket][target] += 1
                contacts[bucket][source][target] += 1
                source_activity[bucket][source] += 1
                first_destination[bucket].setdefault(source, target)
                lines += 1
                observed_max = max(observed_max, timestamp)
    except EOFError:
        # The LANL mirrors are intentionally bounded gzip streams.
        pass
    return StreamBuckets(activity, destination, contacts, first_destination, source_activity, lines, observed_max)


def _counter_sum(values, indices) -> Counter:
    result = Counter()
    for index in indices:
        if 0 <= index < len(values):
            result.update(values[index])
    return result


def _values(counter: Counter, candidates: list[str]) -> np.ndarray:
    return np.asarray([counter.get(node, 0.0) for node in candidates], dtype=float)


def extract_real_episodes(
    auth_path: Path,
    dns_path: Path,
    bucket_seconds: int = 600,
    max_seconds: int = 43_200,
    candidate_cap: int = 400,
    episodes_per_bucket: int = 8,
) -> tuple[list[RealEpisode], dict]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="lanl-stream") as pool:
        auth_future = pool.submit(_read_stream, auth_path, "auth", bucket_seconds, max_seconds)
        dns_future = pool.submit(_read_stream, dns_path, "dns", bucket_seconds, max_seconds)
        auth, dns = auth_future.result(), dns_future.result()

    episodes: list[RealEpisode] = []
    identifier = 0
    potential_labels = 0
    retained_labels = 0
    final_bucket = min(auth.max_time, dns.max_time, max_seconds - 1) // bucket_seconds
    for target_bucket in range(9, final_bucket + 1):
        calibration = range(target_bucket - 9, target_bucket - 3)
        fitting = range(target_bucket - 3, target_bucket - 1)
        observation = target_bucket - 1
        calibration_activity = _counter_sum(auth.activity, calibration)
        calibration_activity.update(_counter_sum(dns.activity, calibration))
        fitting_activity = _counter_sum(auth.activity, fitting)
        fitting_activity.update(_counter_sum(dns.activity, fitting))
        observed_activity = auth.activity[observation] + dns.activity[observation]
        candidate_score = calibration_activity + fitting_activity + observed_activity
        candidates = [node for node, _ in candidate_score.most_common(candidate_cap)]
        candidate_index = {node: index for index, node in enumerate(candidates)}
        if len(candidates) < 50:
            continue

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

        fit_choices = _counter_sum(auth.destination, fitting)
        fit_choices.update(_counter_sum(dns.destination, fitting))
        degree = _values(calibration_activity, candidates)
        choices = _values(fit_choices, candidates)
        beta, information, probability = fit_endpoint_choice_prior(degree, choices)
        if information <= 0.0 or not np.isfinite(information):
            continue
        structural = np.log(np.clip(probability, 1e-12, 1.0))
        structural -= structural.mean()
        auth_global = auth.activity[observation]
        dns_global = dns.activity[observation]

        if len(eligible) <= episodes_per_bucket:
            selected = eligible
        else:
            # Span the activity spectrum instead of repeatedly choosing only hubs.
            selected = [eligible[index] for index in np.linspace(0, len(eligible) - 1, episodes_per_bucket, dtype=int)]

        for _, source, target in selected:
            query = np.column_stack((
                np.log1p(_values(auth_global, candidates)),
                np.log1p(_values(auth.contacts[observation].get(source, Counter()), candidates)),
                np.log1p(_values(dns_global, candidates)),
                np.log1p(_values(dns.contacts[observation].get(source, Counter()), candidates)),
            )).astype(np.float32)
            pre = np.log1p(_values(calibration_activity, candidates)).astype(np.float32)
            data = EpisodeData(
                identifier=identifier,
                x=np.column_stack((pre, query)).astype(np.float32),
                structural=structural.astype(np.float32),
                degree=degree.astype(np.float32),
                truth=candidate_index[target],
                rho=0.0,
                se=1.0 / np.sqrt(max(information, 1e-12)),
                gini=_gini(degree),
            )
            episodes.append(RealEpisode(
                data=data,
                timestamp=target_bucket * bucket_seconds,
                source=source,
                target=target,
                beta=beta,
                information=information,
                candidates=len(candidates),
            ))
            identifier += 1

    audit = {
        "auth_lines": auth.lines,
        "dns_lines": dns.lines,
        "auth_max_time": auth.max_time,
        "dns_max_time": dns.max_time,
        "bucket_seconds": bucket_seconds,
        "max_seconds": max_seconds,
        "candidate_cap": candidate_cap,
        "episodes_per_bucket": episodes_per_bucket,
        "episodes": len(episodes),
        "potential_labels": potential_labels,
        "candidate_retained_labels": retained_labels,
        "candidate_label_recall": retained_labels / max(potential_labels, 1),
        "unique_sources": len({item.source for item in episodes}),
        "unique_targets": len({item.target for item in episodes}),
        "top_target_fraction": max(Counter(item.target for item in episodes).values(), default=0) / max(len(episodes), 1),
        "top_five_target_fraction": sum(value for _, value in Counter(item.target for item in episodes).most_common(5)) / max(len(episodes), 1),
        "first_timestamp": min((item.timestamp for item in episodes), default=None),
        "last_timestamp": max((item.timestamp for item in episodes), default=None),
        "extraction_seconds": time.perf_counter() - started,
        "threads": 2,
    }
    return episodes, audit


def chronological_folds(episodes: list[RealEpisode], folds: int = 3):
    ordered = sorted(episodes, key=lambda item: (item.timestamp, item.data.identifier))
    test_size = max(20, len(ordered) // 10)
    first_end = max(60, len(ordered) - folds * test_size)
    result = []
    for fold in range(folds):
        train_end = first_end + fold * test_size
        test_end = min(train_end + test_size, len(ordered))
        train = ordered[:train_end]
        test = ordered[train_end:test_end]
        if len(train) >= 30 and len(test) >= 10:
            result.append((fold, train, test))
    return result


def _rescale(train: list[RealEpisode], test: list[RealEpisode]):
    scale = float(np.median([item.information for item in train]))
    convert = lambda item: replace(item.data, rho=reliability(item.information, scale))
    return [convert(item) for item in train], [convert(item) for item in test], scale


def _gradient_norm(model: nn.Module, features: torch.Tensor, labels: list[float]) -> float:
    model.zero_grad(set_to_none=True)
    prediction = model(features)
    target = torch.tensor(labels, dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy(prediction, target)
    loss.backward()
    squared = sum(float(parameter.grad.detach().square().sum()) for parameter in model.parameters() if parameter.grad is not None)
    model.zero_grad(set_to_none=True)
    return float(np.sqrt(squared))


def evaluate_real(
    episodes: list[RealEpisode],
    output: Path,
    device: str = "mps",
    folds: int = 3,
    seeds=REAL_SEEDS,
):
    output.mkdir(parents=True, exist_ok=True)
    metadata = {item.data.identifier: item for item in episodes}
    rows, resources = [], []
    for fold, raw_train, raw_test in chronological_folds(episodes, folds=folds):
        train, test, information_scale = _rescale(raw_train, raw_test)
        train_sources = {item.source for item in raw_train}
        train_targets = {item.target for item in raw_train}
        mean, std = scaler_fit(train)
        for seed in seeds:
            print(f"fold={fold} seed={seed} train={len(train)} test={len(test)} device={device}", flush=True)
            training_started = time.perf_counter()
            temporal, features, labels, feature_mean, feature_std = prepare_router_training(
                train, mean, std, seed, temporal_device=device
            )
            temporal_seconds = time.perf_counter() - training_started
            router_started = time.perf_counter()
            classical = _train_small_router(ClassicalReliabilityRouter(), features, labels, seed + 601)
            logistic = _train_small_router(LogisticRouter(), features, labels, seed + 701)
            quantum = {
                name: _train_small_router(QuantumReliabilityRouter(**spec), features, labels, seed + 301 + index * 37)
                for index, (name, spec) in enumerate(QUANTUM_SPECS.items())
            }
            base_spec = QUANTUM_SPECS["q3d2_ring"]
            for index, (name, mask) in enumerate(QUANTUM_FEATURE_MASKS.items()):
                if name == "q3d2_ring":
                    continue
                quantum[name] = _train_small_router(
                    QuantumReliabilityRouter(**base_spec), features * torch.tensor(mask), labels,
                    seed + 901 + index * 41,
                )
            router_seconds = time.perf_counter() - router_started
            for name, model in {"logistic_router": logistic, "classical_router": classical, **quantum}.items():
                feature_mask = QUANTUM_FEATURE_MASKS.get(name, np.ones(6, dtype=np.float32))
                resources.append({
                    "fold": fold,
                    "seed": seed,
                    "model": name,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "gradient_norm": _gradient_norm(model, features * torch.tensor(feature_mask), labels),
                    "temporal_seconds": temporal_seconds,
                    "router_group_seconds": router_seconds,
                    "device": device,
                    "train_episodes": len(train),
                    "test_episodes": len(test),
                    "information_scale": information_scale,
                    "feature_mask": "".join(str(int(value)) for value in feature_mask),
                })

            for data in test:
                meta = metadata[data.identifier]
                for condition_index, condition in enumerate(CONDITIONS):
                    temporal_score, raw_feature = _episode_feature(
                        temporal, data, condition, mean, std, seed + 1009 * condition_index
                    )
                    feature = (torch.tensor(raw_feature) - feature_mean) / feature_std
                    model_scores = {
                        "degree_baseline": (np.log1p(data.degree), 0.0),
                        "temporal": (temporal_score, 0.0),
                        "graph_only": (data.rho * data.structural, 1.0),
                        "unconditional_fusion": (temporal_score + data.structural, 1.0),
                        "fixed_curvature": (temporal_score + data.rho * data.structural, 1.0),
                    }
                    with torch.no_grad():
                        alpha_logistic = float(logistic(feature).item())
                        alpha_classical = float(classical(feature).item())
                    model_scores["logistic_router"] = (
                        temporal_score + alpha_logistic * data.rho * data.structural, alpha_logistic
                    )
                    model_scores["classical_router"] = (
                        temporal_score + alpha_classical * data.rho * data.structural, alpha_classical
                    )
                    for name, model in quantum.items():
                        model.eval()
                        feature_mask = torch.tensor(QUANTUM_FEATURE_MASKS.get(name, np.ones(6, dtype=np.float32)))
                        model_feature = feature * feature_mask
                        with torch.no_grad():
                            alpha = float(model(model_feature).item())
                        model_scores[name] = (temporal_score + alpha * data.rho * data.structural, alpha)
                        if name == "q3d2_ring":
                            model_scores["quantum_blind_reliability"] = (
                                temporal_score + alpha * data.structural, alpha
                            )
                            for shots in SHOT_GRID:
                                shot_alpha = model.finite_shot(
                                    model_feature, shots, np.random.default_rng(seed + data.identifier + shots * 17)
                                )
                                model_scores[f"qshot_{shots}"] = (
                                    temporal_score + shot_alpha * data.rho * data.structural, shot_alpha
                                )
                            for depolarizing, readout in NOISE_GRID:
                                noise_alpha = model.noisy_finite_shot(
                                    model_feature, 256, depolarizing, readout,
                                    np.random.default_rng(seed + data.identifier + int(1e5 * depolarizing) + int(1e4 * readout)),
                                )
                                model_scores[f"qnoise_p{depolarizing:.3f}_r{readout:.3f}"] = (
                                    temporal_score + noise_alpha * data.rho * data.structural, noise_alpha
                                )

                    for model_name, (score, alpha) in model_scores.items():
                        result = {
                            "fold": fold,
                            "seed": seed,
                            "episode": data.identifier,
                            "timestamp": meta.timestamp,
                            "condition": condition,
                            "model": model_name,
                            "alpha": alpha,
                            "rho": data.rho,
                            "beta": meta.beta,
                            "candidates": meta.candidates,
                            "source_seen": int(meta.source in train_sources),
                            "target_seen": int(meta.target in train_targets),
                        }
                        result.update(ranking(softmax(score), data.truth))
                        rows.append(result)

    frame = pd.DataFrame(rows)
    resource_frame = pd.DataFrame(resources)
    frame.to_csv(output / "lanl_real_qrag_results.csv", index=False)
    resource_frame.to_csv(output / "lanl_real_qrag_resources.csv", index=False)
    return frame, resource_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=Path, required=True)
    parser.add_argument("--dns", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--bucket-seconds", type=int, default=600)
    parser.add_argument("--max-seconds", type=int, default=43200)
    parser.add_argument("--candidate-cap", type=int, default=400)
    parser.add_argument("--episodes-per-bucket", type=int, default=8)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in REAL_SEEDS))
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    if args.cache and args.cache.exists():
        episodes, audit = load_episode_cache(args.cache)
    else:
        episodes, audit = extract_real_episodes(
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
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    frame, resources = evaluate_real(
        episodes, args.output, device=args.device, folds=args.folds, seeds=selected_seeds
    )
    print(json.dumps({
        **audit,
        "result_rows": len(frame),
        "resource_rows": len(resources),
        "output": str(args.output),
        "device": args.device,
    }, indent=2))


if __name__ == "__main__":
    main()
