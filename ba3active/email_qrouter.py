"""Small real-network benchmark for quantum allocation of expert authority.

Each label is the first observed recipient of a sender on a day. All evidence
comes from earlier days. The temporal expert uses the preceding seven days;
the graph expert uses days eight through ninety before the label day.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags


CONDITIONS = ("complete", "query_half", "history_half", "joint_half")
FEATURE_NAMES = (
    "query_retention", "history_retention", "source_query_activity",
    "source_history_activity", "temporal_entropy", "graph_entropy",
    "temporal_margin", "graph_margin", "top_ten_overlap",
)


@dataclass
class EmailEpisodes:
    temporal: np.ndarray
    graph: np.ndarray
    features: np.ndarray
    truth: np.ndarray
    day: np.ndarray
    source: np.ndarray
    target: np.ndarray
    candidate_count: np.ndarray
    audit: dict


def _daily_edges(frame: pd.DataFrame, count: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    complete = [np.empty((0, 2), dtype=np.int32) for _ in range(count)]
    half = [np.empty((0, 2), dtype=np.int32) for _ in range(count)]
    rng = np.random.default_rng(20260927)
    keep = rng.random(len(frame)) < 0.5
    for day, indices in frame.groupby("day", sort=True).indices.items():
        pairs = frame.iloc[indices][["src", "dst"]].to_numpy(dtype=np.int32)
        complete[int(day)] = pairs
        half[int(day)] = pairs[keep[indices]]
    return complete, half


def _update(matrix: np.ndarray, pairs: np.ndarray, sign: int) -> None:
    if pairs.size:
        np.add.at(matrix, (pairs[:, 0], pairs[:, 1]), sign)


def _normalize(score: np.ndarray, candidates: np.ndarray, source: int) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).copy()
    score[~candidates] = 0.0
    score[source] = 0.0
    score[candidates] += 1e-7
    score[source] = 0.0
    total = score.sum()
    if total <= 0:
        score[candidates] = 1.0
        score[source] = 0.0
        total = score.sum()
    return (score / total).astype(np.float32)


def _daily_graph(matrix: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    graph = csr_matrix(matrix, dtype=np.float64)
    rows = np.asarray(graph.sum(axis=1)).ravel()
    transition = diags(1.0 / np.maximum(rows, 1.0)) @ graph
    popularity = np.asarray(graph.sum(axis=0)).ravel()
    popularity /= max(popularity.sum(), 1.0)
    return transition.tocsr(), popularity


def _temporal_distribution(matrix: np.ndarray, source: int, popularity: np.ndarray,
                           candidates: np.ndarray) -> np.ndarray:
    direct = matrix[source].astype(np.float64)
    direct /= max(direct.sum(), 1.0)
    return _normalize(0.82 * direct + 0.18 * popularity, candidates, source)


def _graph_distribution(transition: csr_matrix, popularity: np.ndarray, source: int,
                        candidates: np.ndarray) -> np.ndarray:
    direct = transition.getrow(source).toarray().ravel()
    second = np.asarray(direct @ transition).ravel()
    return _normalize(0.68 * direct + 0.27 * second + 0.05 * popularity, candidates, source)


def _features(temporal: np.ndarray, graph: np.ndarray, query_retention: float,
              history_retention: float, query_activity: float, history_activity: float) -> np.ndarray:
    def entropy(values: np.ndarray) -> float:
        positive = values[values > 0]
        return float(-(positive * np.log(positive)).sum() / np.log(max(len(values), 2)))

    def margin(values: np.ndarray) -> float:
        largest = np.partition(values, -2)[-2:]
        return float(largest.max() - largest.min())

    top_temporal = set(np.argpartition(temporal, -10)[-10:].tolist())
    top_graph = set(np.argpartition(graph, -10)[-10:].tolist())
    return np.asarray((
        query_retention, history_retention, np.log1p(query_activity),
        np.log1p(history_activity), entropy(temporal), entropy(graph),
        margin(temporal), margin(graph), len(top_temporal & top_graph) / 10.0,
    ), dtype=np.float32)


def extract(path: Path, max_episodes_per_day: int = 8, first_day: int = 90,
            last_day: int | None = None) -> EmailEpisodes:
    frame = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                        names=["src", "dst", "timestamp"])
    frame["original_row"] = np.arange(len(frame))
    frame.sort_values(["timestamp", "original_row"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    frame["day"] = frame.timestamp // 86400
    nodes = int(max(frame.src.max(), frame.dst.max()) + 1)
    observed_nodes = int(np.unique(frame[["src", "dst"]].to_numpy()).size)
    days = int(frame.day.max()) + 1
    if first_day < 90:
        raise ValueError("at least ninety prior days are required")
    if last_day is None:
        last_day = days - 1
    if last_day < first_day or last_day >= days:
        raise ValueError("last_day is outside the available date range")
    full, half = _daily_edges(frame, days)
    recent = [np.zeros((nodes, nodes), dtype=np.int32) for _ in range(2)]
    history = [np.zeros((nodes, nodes), dtype=np.int32) for _ in range(2)]
    for day in range(first_day - 7, first_day):
        _update(recent[0], full[day], 1)
        _update(recent[1], half[day], 1)
    for day in range(first_day - 90, first_day - 7):
        _update(history[0], full[day], 1)
        _update(history[1], half[day], 1)
    seen = np.zeros(nodes, dtype=bool)
    for day in range(first_day):
        pairs = full[day]
        if pairs.size:
            seen[pairs.ravel()] = True

    distributions_t, distributions_g, features = [], [], []
    truths, episode_days, sources, targets, candidate_counts = [], [], [], [], []
    possible = covered = eligible = sampled = 0
    active_days = 0
    for day in range(first_day, last_day + 1):
        if day > first_day:
            for matrix, payload in zip(recent, (full, half)):
                _update(matrix, payload[day - 1], 1)
                _update(matrix, payload[day - 8], -1)
            for matrix, payload in zip(history, (full, half)):
                _update(matrix, payload[day - 8], 1)
                _update(matrix, payload[day - 91], -1)
            if full[day - 1].size:
                seen[full[day - 1].ravel()] = True
        day_edges = full[day]
        if not day_edges.size:
            continue
        active_days += 1
        first_recipient = {}
        for source, target in day_edges:
            first_recipient.setdefault(int(source), int(target))
        possible += len(first_recipient)
        sender_pool = [source for source in first_recipient if seen[source]
                       and recent[0][source].sum() + history[0][source].sum() >= 2]
        eligible += len(sender_pool)
        rng = np.random.default_rng(3021 + day)
        if len(sender_pool) > max_episodes_per_day:
            sender_pool = sorted(rng.choice(sender_pool, max_episodes_per_day, replace=False).tolist())
        sampled += len(sender_pool)
        if not sender_pool:
            continue
        graph_full, history_pop_full = _daily_graph(history[0])
        graph_half, history_pop_half = _daily_graph(history[1])
        recent_pop = [matrix.sum(axis=0).astype(np.float64) for matrix in recent]
        for popularity in recent_pop:
            popularity /= max(popularity.sum(), 1.0)
        query_retention = float(recent[1].sum() / max(recent[0].sum(), 1))
        history_retention = float(history[1].sum() / max(history[0].sum(), 1))
        for source in sender_pool:
            target = first_recipient[source]
            if not seen[target] or target == source:
                continue
            covered += 1
            temporal = [
                _temporal_distribution(recent[index], source, recent_pop[index], seen)
                for index in range(2)
            ]
            structural = [
                _graph_distribution(graph, popularity, source, seen)
                for graph, popularity in ((graph_full, history_pop_full), (graph_half, history_pop_half))
            ]
            select = ((0, 0), (1, 0), (0, 1), (1, 1))
            distributions_t.append(np.stack([temporal[q] for q, _ in select]))
            distributions_g.append(np.stack([structural[h] for _, h in select]))
            features.append(np.stack([
                _features(temporal[q], structural[h], 1.0 if q == 0 else query_retention,
                          1.0 if h == 0 else history_retention,
                          recent[q][source].sum(), history[h][source].sum())
                for q, h in select
            ]))
            truths.append(target)
            episode_days.append(day)
            sources.append(source)
            targets.append(target)
            candidate_counts.append(int(seen.sum() - 1))

    audit = {
        "dataset": "SNAP email-Eu-core-temporal",
        "source_url": "https://snap.stanford.edu/data/email-Eu-core-temporal.html",
        "raw_events": len(frame), "nodes": observed_nodes,
        "node_id_slots": nodes, "active_days": active_days,
        "earliest_episode_day": first_day, "last_episode_day": last_day,
        "candidate_policy": "all previously observed members except sender",
        "label_policy": "first email recipient per sender per day",
        "possible_sender_days": possible, "eligible_sender_days": eligible,
        "sampled_sender_days": sampled,
        "episodes": covered, "conditions": CONDITIONS,
        "query_window_days": 7, "graph_window_days": [8, 90],
        "query_half_mask": "fixed seeded independent event retention",
        "history_half_mask": "fixed seeded independent event retention",
        "sampled_sources_per_day": max_episodes_per_day,
    }
    return EmailEpisodes(
        np.asarray(distributions_t, dtype=np.float32),
        np.asarray(distributions_g, dtype=np.float32),
        np.asarray(features, dtype=np.float32),
        np.asarray(truths, dtype=np.int32), np.asarray(episode_days, dtype=np.int32),
        np.asarray(sources, dtype=np.int32), np.asarray(targets, dtype=np.int32),
        np.asarray(candidate_counts, dtype=np.int32), audit,
    )


def save_episodes(path: Path, episodes: EmailEpisodes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, temporal=episodes.temporal, graph=episodes.graph,
                        features=episodes.features, truth=episodes.truth, day=episodes.day,
                        source=episodes.source, target=episodes.target,
                        candidate_count=episodes.candidate_count,
                        audit=np.asarray(json.dumps(episodes.audit)))


def load_episodes(path: Path) -> EmailEpisodes:
    with np.load(path, allow_pickle=False) as data:
        return EmailEpisodes(data["temporal"], data["graph"], data["features"],
                             data["truth"], data["day"], data["source"],
                             data["target"], data["candidate_count"],
                             json.loads(str(data["audit"])))


def expert_summary(episodes: EmailEpisodes) -> pd.DataFrame:
    records = []
    row = np.arange(len(episodes.truth))
    for condition_id, condition in enumerate(CONDITIONS):
        for name, values in (("temporal", episodes.temporal), ("graph", episodes.graph)):
            probabilities = values[row, condition_id]
            true_probability = probabilities[row, episodes.truth]
            ranks = 1 + (probabilities > true_probability[:, None]).sum(axis=1)
            records.append({
                "condition": condition, "expert": name, "episodes": len(ranks),
                "mrr": float(np.mean(1.0 / ranks)),
                "hit10": float(np.mean(ranks <= 10)),
                "median_rank": float(np.median(ranks)),
                "nll": float(np.mean(-np.log(np.maximum(true_probability, 1e-12)))),
            })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/email-Eu-core-temporal.txt.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/email_qrouter_v1"))
    parser.add_argument("--max-episodes-per-day", type=int, default=8)
    parser.add_argument("--first-day", type=int, default=90)
    parser.add_argument("--last-day", type=int)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"dataset not found at {args.input}")
    episodes = extract(args.input, args.max_episodes_per_day, args.first_day, args.last_day)
    args.output.mkdir(parents=True, exist_ok=True)
    save_episodes(args.output / "episodes.npz", episodes)
    (args.output / "extraction_audit.json").write_text(json.dumps(episodes.audit, indent=2) + "\n")
    summary = expert_summary(episodes)
    summary.to_csv(args.output / "expert_summary.csv", index=False)
    print(json.dumps(episodes.audit, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
