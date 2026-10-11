from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def write_tables(result_dir: Path, figure_dir: Path):
    active = pd.read_csv(result_dir / "active_summary.csv")
    static = pd.read_csv(result_dir / "static_summary.csv")
    paired = pd.read_csv(result_dir / "paired_comparison.csv")
    audit = json.loads((result_dir / "claim_audit.json").read_text())
    episode = pd.read_csv(figure_dir / "episode_audit.csv")
    primary = active[active.budget.isin([1.0, 5.0, 25.0])].copy()
    primary = primary[["policy", "budget", "mrr", "hit1", "hit5", "nll", "cost", "queries"]]
    primary.to_csv(result_dir / "table_main_results.csv", index=False)
    static.to_csv(result_dir / "table_ablation.csv", index=False)
    metadata = pd.DataFrame([{
        "primary_gap_minutes": 15,
        "episodes": int(episode.iloc[0].episodes),
        "origins": int(episode.iloc[0].origins),
        "candidate_recall": audit["candidate_recall"],
        "rolling_folds": audit["rolling_folds"],
        "episode_instances": audit["episode_instances"],
        "candidate_cap": audit["candidate_cap"],
    }])
    metadata.to_csv(result_dir / "table_protocol.csv", index=False)
    paired.to_csv(result_dir / "table_paired_test.csv", index=False)
    return metadata, primary, static, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    for table in write_tables(args.result_dir, args.figure_dir):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
