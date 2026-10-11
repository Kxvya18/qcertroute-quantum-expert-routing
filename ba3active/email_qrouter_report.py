"""Day-block uncertainty for paired quantum readout protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _estimates(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_shots": float(frame.shots.mean()),
        "savings_vs_fixed512_percent": float(100.0 * (1.0 - frame.shots.mean() / 512.0)),
        "shots_saved_vs_observable_bound": float(
            (frame.observable_bound_shots - frame.shots).mean()),
        "certified_fraction": float(frame.certified.mean()),
        "certification_gain_percentage_points": float(
            100.0 * (frame.certified - frame.observable_bound_certified).mean()),
        "certified_exact_match": float(
            frame.loc[frame.certified == 1, "top10_matches_exact"].mean()),
        "zero_shot_fraction": float(frame.zero_shot.mean()),
    }


def _day_bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> dict[str, tuple[float, float]]:
    daily = frame.assign(
        shot_gain=frame.observable_bound_shots - frame.shots,
        coverage_gain=frame.certified - frame.observable_bound_certified,
        certified_match=frame.certified * frame.top10_matches_exact,
    ).groupby("day", sort=True).agg(
        count=("shots", "size"), shot_sum=("shots", "sum"),
        shot_gain_sum=("shot_gain", "sum"),
        certified_sum=("certified", "sum"),
        coverage_gain_sum=("coverage_gain", "sum"),
        certified_match_sum=("certified_match", "sum"),
    )
    values = daily.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(draws, len(values)))
    totals = values[sample].sum(axis=1)
    count = totals[:, 0]
    mean_shots = totals[:, 1] / count
    certified_count = totals[:, 3]
    metrics = {
        "mean_shots": mean_shots,
        "savings_vs_fixed512_percent": 100.0 * (1.0 - mean_shots / 512.0),
        "shots_saved_vs_observable_bound": totals[:, 2] / count,
        "certified_fraction": certified_count / count,
        "certification_gain_percentage_points": 100.0 * totals[:, 4] / count,
        "certified_exact_match": np.divide(
            totals[:, 5], certified_count,
            out=np.full(draws, np.nan), where=certified_count > 0),
    }
    return {name: tuple(np.nanquantile(result, (0.025, 0.975)).tolist())
            for name, result in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="One decisions.csv per distinct trained model seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=719)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("at least 100 bootstrap draws are required")
    rows, condition_rows = [], []
    reference_keys = None
    for model_id, path in enumerate(args.inputs):
        frame = pd.read_csv(path)
        keys = set(zip(frame.episode, frame.condition))
        if len(keys) != len(frame):
            raise ValueError(f"duplicate episode-condition keys in {path}")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("model seeds do not cover the same episode-condition pairs")
        point = _estimates(frame)
        intervals = _day_bootstrap(frame, args.bootstrap, args.seed + model_id)
        row = {"model": path.parent.name, "decisions": len(frame),
               "days": int(frame.day.nunique()), **point}
        for name, (low, high) in intervals.items():
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        rows.append(row)
        for condition, subset in frame.groupby("condition", sort=False):
            condition_rows.append({"model": path.parent.name,
                                   "condition": condition, "decisions": len(subset),
                                   **_estimates(subset)})
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "seed_day_block_summary.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(args.output / "condition_summary.csv", index=False)
    core = ("mean_shots", "savings_vs_fixed512_percent",
            "shots_saved_vs_observable_bound", "certified_fraction",
            "certification_gain_percentage_points")
    aggregate = {name: {
        "mean_across_model_seeds": float(np.mean([row[name] for row in rows])),
        "min_model_seed": float(min(row[name] for row in rows)),
        "max_model_seed": float(max(row[name] for row in rows)),
    } for name in core}
    aggregate["independent_test_days"] = rows[0]["days"]
    aggregate["paired_test_decisions_per_model"] = rows[0]["decisions"]
    aggregate["bootstrap_note"] = (
        "Percentile intervals resample days within each model seed. "
        "Three model seeds are summarized as a range, not independent test datasets.")
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
