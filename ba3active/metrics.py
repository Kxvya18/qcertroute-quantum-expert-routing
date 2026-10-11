from __future__ import annotations

import numpy as np


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def ranking(probability, truth_index):
    probability = np.asarray(probability, dtype=float)
    rank = int(np.argsort(-probability, kind="stable").tolist().index(truth_index) + 1)
    return {
        "mrr": 1.0 / rank,
        "hit1": float(rank == 1),
        "hit5": float(rank <= 5),
        "hit10": float(rank <= 10),
        "rank": rank,
        "nll": float(-np.log(max(probability[truth_index], 1e-12))),
        "entropy": float(-np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)))),
    }


def paired_bootstrap(a, b, repetitions=2000, seed=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions)
    for i in range(repetitions):
        index = rng.integers(0, len(a), len(a))
        deltas[i] = np.mean(a[index] - b[index])
    return float(np.mean(a - b)), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def paired_permutation(a, b, repetitions=10000, seed=0):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(float(delta.mean()))
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1.0, 1.0], size=(repetitions, len(delta))) * delta).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (repetitions + 1))
