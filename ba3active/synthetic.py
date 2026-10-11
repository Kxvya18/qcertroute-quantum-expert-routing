"""Controlled, reproducible mechanism study for curvature-audited acquisition.

This module is intentionally separate from the LANL experiment.  It creates
known-origin cascades on generalized preferential-attachment graphs and labels
every result as semi-synthetic rather than as an operational attack benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .metrics import ranking, softmax
from .prior import fisher_identity, reliability


def grow(n, m, beta, rng):
    degree = np.ones(m + 1, dtype=float) * m
    history = []
    for new in range(m + 1, n):
        probability = softmax(beta * np.log1p(degree))
        chosen = rng.choice(len(degree), size=m, replace=False, p=probability)
        counts = np.zeros(len(degree), dtype=float)
        counts[chosen] = 1.0
        history.append((degree.copy(), counts))
        degree[chosen] += 1.0
        degree = np.append(degree, float(m))
    return degree, history


def run_trial(n, m, beta, budget, seed):
    rng = np.random.default_rng(seed)
    degree, history = grow(n, m, beta, rng)
    def objective(value):
        total = 0.0
        for before, chosen in history:
            log_degree = np.log1p(before)
            logits = value * log_degree
            maximum = logits.max()
            total -= float(chosen @ logits - m * (maximum + np.log(np.exp(logits - maximum).sum())))
        return total
    beta_hat = float(minimize_scalar(objective, bounds=(-2.0, 4.0), method="bounded").x)
    probability = softmax(beta_hat * np.log1p(degree))
    information = sum(m * fisher_identity(before, beta_hat) for before, _ in history)
    scale = sum(m * fisher_identity(before, beta_hat) for before, _ in history[-50:])
    rho = reliability(information, max(scale, 1.0))
    source = int(rng.choice(n, p=softmax(beta * np.log1p(degree))))
    # A queried node emits a noisy local incident marker.  Before acquisition,
    # all temporal scores are neutral, making the budget constraint explicit.
    marker = rng.binomial(1, np.where(np.arange(n) == source, 0.82, 0.018))
    queried = max(1, int(round(budget * n)))
    entropy_order = rng.permutation(n)
    curvature_order = np.argsort(-probability)
    rows = []
    for policy, order in (("random", entropy_order), ("curvature", curvature_order)):
        mask = np.zeros(n, dtype=bool)
        mask[order[:queried]] = True
        temporal = np.where(mask, marker * 5.0, 0.0)
        temporal_score = temporal
        audited_score = temporal + rho * np.log(np.clip(probability, 1e-12, 1.0))
        score = audited_score if policy == "curvature" else temporal_score
        result = ranking(score, source)
        rows.append({"n": n, "m": m, "beta": beta, "budget": budget, "seed": seed,
                     "policy": policy, "beta_hat": beta_hat, "information": information,
                     "reliability": rho, **result})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    rows = []
    for beta in (0.0, 1.0, 2.0):
        for m in (1, 2, 3):
            for budget in (0.01, 0.03, 0.10):
                for seed in range(args.repetitions):
                    rows.extend(run_trial(500, m, beta, budget, 100000 * int(beta + 1) + 1000 * m + 100 * int(budget * 100) + seed))
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "synthetic_results.csv", index=False)
    summary = frame.groupby(["beta", "m", "budget", "policy"]).agg(
        trials=("seed", "size"), mrr=("mrr", "mean"), hit1=("hit1", "mean"), hit10=("hit10", "mean"),
        beta_hat=("beta_hat", "mean"), information=("information", "mean"), reliability=("reliability", "mean")
    ).reset_index()
    summary.to_csv(args.output / "synthetic_summary.csv", index=False)
    print(json.dumps({"trials": len(frame), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
