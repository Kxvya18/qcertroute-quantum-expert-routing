from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _logits(degree, beta, alpha):
    return beta * np.log(np.asarray(degree, dtype=float) + alpha)


def fit_endpoint_choice_prior(degree, future_counts, alpha=1.0):
    """Fit P_beta(v) proportional to (k_v + alpha)^beta on held-out choices."""
    degree = np.asarray(degree, dtype=float)
    counts = np.asarray(future_counts, dtype=float)
    active = (degree > 0) | (counts > 0)
    degree = degree[active]
    counts = counts[active]
    total = float(counts.sum())
    if len(degree) < 2 or total <= 0:
        return 0.0, 0.0, np.full(len(active), 1.0 / max(len(active), 1))

    log_degree = np.log(degree + alpha)

    def negative(beta):
        logits = beta * log_degree
        maximum = logits.max()
        log_normalizer = maximum + np.log(np.exp(logits - maximum).sum())
        return float(-(counts @ logits - total * log_normalizer))

    result = minimize_scalar(negative, bounds=(-2.0, 4.0), method="bounded")
    beta = float(result.x)
    logits = beta * log_degree
    weights = np.exp(logits - logits.max())
    probability = weights / weights.sum()
    information = float(total * (probability @ (log_degree**2) - (probability @ log_degree) ** 2))
    full_probability = np.zeros(len(active), dtype=float)
    full_probability[active] = probability
    return beta, max(information, 0.0), full_probability


def reliability(information, scale):
    return float(information / (information + max(scale, 1e-12)))


def fisher_identity(degree, beta, alpha=1.0):
    """One-step Fisher identity used in the manuscript proposition."""
    values = np.log(np.asarray(degree, dtype=float) + alpha)
    logits = beta * values
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    return float(probability @ (values**2) - (probability @ values) ** 2)
