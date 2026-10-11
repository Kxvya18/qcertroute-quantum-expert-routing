"""Render compact IEEE tables from the verified SNAP email CSV summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    "complete": "Complete", "query_half": "Query half",
    "history_half": "History half", "joint_half": "Joint half",
    "all": "All",
}
MODELS = {
    "temporal": "Recent", "graph": "Graph", "fixed_mix": "Fixed mix",
    "quantum": "Quantum", "classical": "Classical",
}


def _escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _pm(mean: float, sd: float, runs: int) -> str:
    if runs == 1:
        return f"${mean:.3f}$"
    return rf"${mean:.3f}\!\pm\!{sd:.3f}$"


def _table1(frame: pd.DataFrame) -> str:
    rows = []
    previous = None
    for row in frame.itertuples(index=False):
        if previous is not None and row.block != previous:
            rows.append(r"\midrule")
        rows.append(f"{_escape(row.block)} & {_escape(row.measure)} & "
                    f"{_number(row.value)} & {_escape(row.definition)} " + r"\\")
        previous = row.block
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[!t]
\caption{{Observed network and evaluation protocol. Calendar gaps are retained in the data but omitted from the active-day plot axis.}}
\label{{tab:protocol}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{3.2pt}}\renewcommand{{\arraystretch}}{{0.89}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}llrl}}
\toprule
Block & Measure & Value & Definition \\
\midrule
{body}
\bottomrule
\end{{tabular*}}
\end{{table*}}"""


def _table2(frame: pd.DataFrame) -> str:
    rows = []
    previous = None
    for row in frame.itertuples(index=False):
        if previous is not None and row.condition != previous:
            rows.append(r"\midrule")
        label = LABELS[row.condition]
        model = MODELS[row.model]
        n = int(row.model_seeds)
        rows.append(f"{label} & {model} & {_pm(row.mrr_mean, row.mrr_sd, n)} & "
                    f"{_pm(row.hit10_mean, row.hit10_sd, n)} & "
                    f"{_pm(row.nll_mean, row.nll_sd, n)} & "
                    f"${row.mean_gate_mean:.3f}$ & {n} " + r"\\")
        previous = row.condition
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[!t]
\caption{{Held-out endpoint performance on 697 sender-day episodes per condition. Means and model-seed standard deviations are shown where applicable. Rank metrics use optimistic score ties. Lower NLL is better.}}
\label{{tab:predictive}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{2.6pt}}\renewcommand{{\arraystretch}}{{0.86}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}llrrrrr}}
\toprule
Condition & Policy & MRR & Hit at ten & NLL & Mean gate & Runs \\
\midrule
{body}
\bottomrule
\end{{tabular*}}
\end{{table*}}"""


def _table3(frame: pd.DataFrame) -> str:
    rows = []
    previous = None
    for row in frame.itertuples(index=False):
        if previous is not None and row.model_seed != previous:
            rows.append(r"\midrule")
        rows.append(f"{int(row.model_seed)} & {LABELS[row.condition]} & "
                    f"{row.mean_shots:.1f} & {row.savings_vs_fixed512_pct:.1f} & "
                    f"{row.observable_bound_mean_shots:.1f} & "
                    f"{row.certified_pct:.1f} & {row.observable_certified_pct:.1f} & "
                    f"{row.fixed512_exact_match_pct:.1f} & "
                    f"{int(row.certified_mismatches)} " + r"\\")
        previous = row.model_seed
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[!t]
\caption{{Adaptive quantum measurement on 2788 paired test decisions per model seed. Savings compare with a fixed 512-shot policy. The alternative bound treats eight observables separately. Match denotes agreement with the exact circuit top-ten set.}}
\label{{tab:shots}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{2.6pt}}\renewcommand{{\arraystretch}}{{0.87}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}llrrrrrrr}}
\toprule
Seed & Condition & Shots & Saved \% & Separate shots & Cert. \% & Separate cert. \% & Fixed match \% & Cert. errors \\
\midrule
{body}
\bottomrule
\end{{tabular*}}
\end{{table*}}"""


def _table4(frame: pd.DataFrame) -> str:
    rows = []
    previous = None
    for row in frame.itertuples(index=False):
        if previous is not None and row.condition != previous:
            rows.append(r"\midrule")
        rows.append(f"{LABELS[row.condition]} & {row.expert_top10_overlap} & "
                    f"{int(row.paired_decisions)} & {row.mean_shots:.1f} & "
                    f"{row.shots_saved_vs_separate_observables:.1f} & "
                    f"{row.certified_pct:.1f} & {row.observable_certified_pct:.1f} & "
                    f"{row.admitted_hit10_pct:.1f} & {row.exact_hit10_all_pct:.1f} & "
                    f"{int(row.certified_mismatches)} " + r"\\")
        previous = row.condition
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[!t]
\caption{{Measurement cost and certification by telemetry condition and fixed-expert top-ten overlap. Counts are per model seed. Hit rates use deterministic set membership. Certified hit rate is selected-case performance, not unconditional improvement.}}
\label{{tab:strata}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{2.3pt}}\renewcommand{{\arraystretch}}{{0.85}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}llrrrrrrrr}}
\toprule
Condition & Overlap & $n$ & Shots & Saved vs sep. & Cert. \% & Sep. cert. \% & Cert. H10 \% & Exact H10 \% & Errors \\
\midrule
{body}
\bottomrule
\end{{tabular*}}
\end{{table*}}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("email_quantum_paper/assets"))
    parser.add_argument("--output", type=Path,
                        default=Path("email_quantum_paper/tables_generated.tex"))
    args = parser.parse_args()
    blocks = [
        _table1(pd.read_csv(args.assets / "table1_data_protocol.csv")),
        _table2(pd.read_csv(args.assets / "table2_predictive_audit.csv")),
        _table3(pd.read_csv(args.assets / "table3_shots_certification.csv")),
        _table4(pd.read_csv(args.assets / "table4_disagreement_strata.csv")),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, block in enumerate(blocks, 1):
        (args.output.parent / f"table{index}.tex").write_text(block + "\n")
    args.output.write_text("\n\n".join(blocks) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
