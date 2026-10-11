from __future__ import annotations

import argparse
import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .episodes import rolling_folds
from .metrics import paired_bootstrap, paired_permutation, ranking, softmax
from .prior import fit_endpoint_choice_prior, reliability


MODALITIES = ("auth", "dns")
COST = np.asarray([1.0, 1.4])
BUDGETS = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
SEEDS = (7, 19, 42, 73, 101)


class _CompatibleUnpickler(pickle.Unpickler):
    """Load the initial extraction artifact created through python -m."""
    def find_class(self, module, name):
        if module == "__main__" and name == "ExtractedEpisode":
            from .extract import ExtractedEpisode
            return ExtractedEpisode
        if module == "__main__" and name == "Episode":
            from .episodes import Episode
            return Episode
        return super().find_class(module, name)


def load_records(path):
    with gzip.open(path, "rb") as handle:
        return _CompatibleUnpickler(handle).load()


@dataclass
class Example:
    identifier: int
    candidates: list[str]
    truth: int
    base: np.ndarray
    query: np.ndarray
    reliability: float
    beta: float
    information: float


def _counter_value(counter, nodes):
    return np.log1p(np.asarray([counter.get(node, 0.0) for node in nodes], dtype=float))


def make_example(record, candidate_cap, curvature_scale):
    pool = record.pre_activity + record.calibration_degree + record.fit_choices
    candidates = [node for node, _ in pool.most_common(candidate_cap) if node.startswith("C")]
    if record.episode.origin not in candidates:
        return None
    degree = np.asarray([record.calibration_degree.get(node, 0.0) for node in candidates], dtype=float)
    choices = np.asarray([record.fit_choices.get(node, 0.0) for node in candidates], dtype=float)
    beta, information, probability = fit_endpoint_choice_prior(degree, choices)
    rho = reliability(information, curvature_scale)
    structural_residual = np.log(np.clip(probability, 1e-12, 1.0))
    structural_residual -= structural_residual.mean()
    base = np.column_stack((
        _counter_value(record.pre_activity, candidates),
        rho * structural_residual,
        np.log1p(degree),
    ))
    query = np.column_stack((
        _counter_value(record.post_activity["auth"], candidates),
        _counter_value(record.post_contact["auth"], candidates),
        _counter_value(record.post_activity["dns"], candidates),
        _counter_value(record.post_contact["dns"], candidates),
    ))
    return Example(record.episode.identifier, candidates, candidates.index(record.episode.origin), base, query, rho, beta, information)


class Ranker:
    def __init__(self, structural_mode, seed):
        self.structural_mode = structural_mode
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=seed
        )

    def design(self, example, mask):
        base = example.base.copy()
        if self.structural_mode == "temporal":
            base[:, 1] = 0.0
            base[:, 2] = 0.0
        elif self.structural_mode == "blind":
            base[:, 1] /= max(example.reliability, 1e-12)
        revealed = example.query * mask
        return np.hstack((base, revealed, mask))

    def fit(self, examples):
        rng = np.random.default_rng(self.seed)
        designs, labels = [], []
        for example in examples:
            nontruth = np.delete(np.arange(len(example.candidates)), example.truth)
            negatives = rng.choice(nontruth, size=min(2000, len(nontruth)), replace=False)
            selected = np.concatenate(([example.truth], negatives))
            label = np.zeros(len(selected), dtype=int)
            label[0] = 1
            for fraction in (0.0, 0.01, 0.03, 0.1, 0.25, 1.0):
                mask = (rng.random((len(example.candidates), 4)) < fraction).astype(float)
                designs.append(self.design(example, mask)[selected])
                labels.append(label)
        x = np.vstack(designs)
        y = np.concatenate(labels)
        self.scaler.fit(x)
        self.model.fit(self.scaler.transform(x), y)
        return self

    def probability(self, example, mask):
        logits = self.model.decision_function(self.scaler.transform(self.design(example, mask)))
        return softmax(logits)

    def utility_scale(self):
        weights = np.abs(self.model.coef_[0, 3:7] / self.scaler.scale_[3:7])
        return np.asarray([weights[0] + weights[1], weights[2] + weights[3]])


def acquire(example, ranker, policy, budget_percent, rng):
    mask = np.zeros((len(example.candidates), 4), dtype=float)
    probability = ranker.probability(example, mask)
    uncertainty = probability * (1.0 - probability)
    prior = example.base[:, 1]
    prior_rank = np.argsort(np.argsort(prior)) / max(len(prior) - 1, 1)
    posterior_rank = np.argsort(np.argsort(probability)) / max(len(probability) - 1, 1)
    importance = ranker.utility_scale()
    query_pairs = [(row, modality) for row in range(len(example.candidates)) for modality in range(2)]
    if policy == "random":
        score = rng.random(len(query_pairs))
    elif policy == "activity":
        score = np.asarray([example.base[row, 0] / COST[modality] for row, modality in query_pairs])
    else:
        score = np.asarray([
            uncertainty[row] * importance[modality] / COST[modality] for row, modality in query_pairs
        ])
        if policy == "curvature":
            score *= np.asarray([
                1.0 + example.reliability * abs(prior_rank[row] - posterior_rank[row])
                for row, _ in query_pairs
            ])
    order = np.argsort(-score, kind="stable")
    limit = len(example.candidates) * float(COST.sum()) * budget_percent / 100.0
    spent = 0.0
    selected = 0
    for index in order:
        row, modality = query_pairs[int(index)]
        if spent + COST[modality] > limit:
            continue
        mask[row, 2 * modality : 2 * modality + 2] = 1.0
        spent += COST[modality]
        selected += 1
    return mask, spent, selected


def static_scores(example, ranker):
    all_mask = np.ones((len(example.candidates), 4), dtype=float)
    return ranker.probability(example, all_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-cap", type=int, default=20000)
    args = parser.parse_args()
    records = load_records(args.input)

    episodes = [record.episode for record in records]
    record_by_id = {record.episode.identifier: record for record in records}
    rows = []
    static_rows = []
    recalled = set()
    fold_scales = []
    for fold_id, train_episodes, test_episodes in rolling_folds(episodes):
        raw_train = [make_example(record_by_id[item.identifier], args.candidate_cap, 1.0) for item in train_episodes]
        information = np.asarray([item.information for item in raw_train if item is not None and item.information > 0])
        scale = float(np.median(information)) if len(information) else 1.0
        fold_scales.append(scale)
        train = [make_example(record_by_id[item.identifier], args.candidate_cap, scale) for item in train_episodes]
        test = [make_example(record_by_id[item.identifier], args.candidate_cap, scale) for item in test_episodes]
        train = [item for item in train if item is not None]
        test = [item for item in test if item is not None]
        recalled.update(item.identifier for item in train + test)
        if len(train) < 3 or not test:
            continue
        for seed in SEEDS:
            audited = Ranker("audited", seed).fit(train)
            temporal = Ranker("temporal", seed).fit(train)
            blind = Ranker("blind", seed).fit(train)
            for example in test:
                for policy in ("random", "activity", "entropy", "curvature"):
                    for budget in BUDGETS:
                        mask, spent, selected = acquire(
                            example, audited, policy, budget, np.random.default_rng(seed * 10000 + example.identifier)
                        )
                        result = ranking(audited.probability(example, mask), example.truth)
                        rows.append({
                            "fold": fold_id, "seed": seed, "episode": example.identifier, "policy": policy,
                            "budget": budget, "cost": spent, "queries": selected, "reliability": example.reliability,
                            "beta": example.beta, "information": example.information, **result,
                        })
                for name, model in (("temporal", temporal), ("blind", blind), ("audited", audited)):
                    result = ranking(static_scores(example, model), example.truth)
                    static_rows.append({
                        "fold": fold_id, "seed": seed, "episode": example.identifier, "model": name,
                        "reliability": example.reliability, "beta": example.beta, "information": example.information, **result,
                    })
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    static = pd.DataFrame(static_rows)
    frame.to_csv(args.output / "active_results.csv", index=False)
    static.to_csv(args.output / "static_results.csv", index=False)
    summary = frame.groupby(["policy", "budget"]).agg(
        instances=("episode", "size"), episodes=("episode", "nunique"), mrr=("mrr", "mean"),
        hit1=("hit1", "mean"), hit5=("hit5", "mean"), hit10=("hit10", "mean"),
        nll=("nll", "mean"), entropy=("entropy", "mean"), cost=("cost", "mean"), queries=("queries", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "active_summary.csv", index=False)
    static_summary = static.groupby("model").agg(
        instances=("episode", "size"), episodes=("episode", "nunique"), mrr=("mrr", "mean"),
        hit1=("hit1", "mean"), hit5=("hit5", "mean"), hit10=("hit10", "mean"), nll=("nll", "mean"),
    ).reset_index()
    static_summary.to_csv(args.output / "static_summary.csv", index=False)

    pivot = frame.pivot_table(index=["fold", "seed", "episode", "budget"], columns="policy", values="mrr").dropna()
    comparisons = []
    for budget, group in pivot.groupby(level="budget"):
        mean, low, high = paired_bootstrap(group["curvature"], group["entropy"], seed=int(budget * 1000))
        comparisons.append({
            "budget": budget, "pairs": len(group), "delta_mrr": mean, "ci_low": low, "ci_high": high,
            "permutation_p": paired_permutation(group["curvature"], group["entropy"], seed=int(budget * 1000)),
        })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output / "paired_comparison.csv", index=False)

    aubc = []
    for (fold, seed, episode, policy), group in frame.groupby(["fold", "seed", "episode", "policy"]):
        ordered = group.sort_values("budget")
        x = ordered.budget.to_numpy(dtype=float) / 100.0
        y = ordered.mrr.to_numpy(dtype=float)
        aubc.append({"fold": fold, "seed": seed, "episode": episode, "policy": policy, "aubc": np.trapezoid(y, x) / (x[-1] - x[0])})
    aubc = pd.DataFrame(aubc)
    aubc.to_csv(args.output / "aubc.csv", index=False)
    pair = aubc.pivot_table(index=["fold", "seed", "episode"], columns="policy", values="aubc").dropna()
    mean, low, high = paired_bootstrap(pair["curvature"], pair["entropy"], seed=991)
    audit = {
        "episodes_total": len(records), "candidate_recalled": len(recalled),
        "candidate_recall": len(recalled) / max(len(records), 1), "rolling_folds": 4,
        "episode_instances": int(len(pair)), "candidate_cap": args.candidate_cap,
        "curvature_scale_median": float(np.median(fold_scales)), "aubc_delta_curvature_minus_entropy": mean,
        "aubc_ci_low": low, "aubc_ci_high": high,
        "aubc_permutation_p": paired_permutation(pair["curvature"], pair["entropy"], seed=991),
        "central_claim_supported": bool(low > 0),
    }
    (args.output / "claim_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
