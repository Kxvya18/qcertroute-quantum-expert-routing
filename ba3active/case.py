"""Selective evidence-fusion analysis for the existing BA3 experiments.

CASE learns, from controlled graph regimes only, whether the curvature policy
is expected to outperform uncertainty-only acquisition.  LANL is never used
to fit this selector; it is a transfer safety evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["beta_hat", "log_information", "reliability", "budget", "m"]


def feature_frame(frame):
    result = frame.copy()
    result["log_information"] = np.log1p(result.information)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--lanl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    syn = pd.read_csv(args.synthetic)
    pivot = syn.pivot_table(index=["n", "m", "beta", "budget", "seed", "beta_hat", "information", "reliability"], columns="policy", values="mrr").reset_index()
    pivot["helpful"] = (pivot.curvature > pivot.random).astype(int)
    pivot = feature_frame(pivot)
    # Independent graph realizations are separated by seed parity.
    train, test = pivot[pivot.seed % 2 == 0], pivot[pivot.seed % 2 == 1]
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=17))
    model.fit(train[FEATURES], train.helpful)
    test = test.copy()
    test["p_helpful"] = model.predict_proba(test[FEATURES])[:, 1]
    test["case_action"] = np.where(test.p_helpful >= 0.5, "curvature", "nonstructural")
    test["case_mrr"] = np.where(test.case_action.eq("curvature"), test.curvature, test.random)
    test["oracle_mrr"] = np.maximum(test.curvature, test.random)

    lanl = pd.read_csv(args.lanl)
    pair = lanl.pivot_table(index=["fold", "seed", "episode", "budget", "reliability", "beta", "information"], columns="policy", values="mrr").dropna().reset_index()
    pair = pair.rename(columns={"beta": "beta_hat"})
    pair["m"] = 2.0
    pair = feature_frame(pair)
    pair["p_helpful"] = model.predict_proba(pair[FEATURES])[:, 1]
    pair["case_action"] = np.where(pair.p_helpful >= 0.5, "curvature", "nonstructural")
    pair["case_mrr"] = np.where(pair.case_action.eq("curvature"), pair.curvature, pair.entropy)
    args.output.mkdir(parents=True, exist_ok=True)
    test.to_csv(args.output / "case_controlled.csv", index=False)
    pair.to_csv(args.output / "case_lanl.csv", index=False)
    test.groupby(["beta", "m", "budget"]).agg(case_mrr=("case_mrr", "mean"), curvature_mrr=("curvature", "mean"), temporal_mrr=("random", "mean"), selected=("case_action", lambda x: float((x == "curvature").mean()))).reset_index().to_csv(args.output / "case_controlled_summary.csv", index=False)
    pair.groupby("budget").agg(case_mrr=("case_mrr", "mean"), curvature_mrr=("curvature", "mean"), temporal_mrr=("entropy", "mean"), selected=("case_action", lambda x: float((x == "curvature").mean()))).reset_index().to_csv(args.output / "case_lanl_summary.csv", index=False)


if __name__ == "__main__":
    main()
