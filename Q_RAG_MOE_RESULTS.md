# Q-RAG-MoE Results Manifest

## Quantum contribution

Q-RAG-MoE replaces the classical reliability gate with a shallow hybrid
quantum-classical router. Six episode-level reliability features are encoded in
a three-qubit circuit through two data-reuploading layers. Each layer contains
trainable `RY` and `RZ` rotations followed by ring CNOT entanglement. Three
local Pauli-Z expectations are passed to a trainable affine sigmoid readout.

The quantum router has 16 trainable parameters. The matched classical router
has 17 parameters. The circuit is executed once per episode and telemetry
condition, not once per candidate endpoint. Quantum-call complexity is
therefore constant in the candidate-set size.

All reported quantum results use an exact local statevector simulator written
in PyTorch. A second evaluation estimates each local observable with 256 shots.
No physical quantum hardware was used, and the results do not establish a
quantum computational advantage.

## Evaluation protocol

- Controlled networks use beta in 0, 1, and 2 with attachment count m equal to
  2. Each regime has 20 training and 10 test episodes.
- LANL uses four expanding chronological folds and 20 distinct held-out
  red-team-derived endpoint episodes.
- Three optimization seeds are used for the compact quantum experiment.
- Conditions are complete, authentication-only, DNS-only, random 50 percent
  loss, and hub-targeted 50 percent loss.
- Comparators are temporal-only, fixed-curvature fusion, a 17-parameter
  classical router, ideal statevector quantum routing, and 256-shot routing.
- Confidence intervals resample episodes and optimizer seeds independently.

## Defensible findings

Across controlled regimes, the quantum router exceeds the matched classical
router under DNS-only telemetry by 0.0629 MRR with a 95 percent two-axis
bootstrap interval of 0.0055 to 0.1475. The authentication-only difference is
0.0312 but its interval crosses zero. The benefit is concentrated in the
super-preferential regime, where conservative quantum admission avoids some of
the over-fusion produced by the classical gate. The uniform regime is exactly
unchanged because its centered structural residual is zero.

On LANL, ideal quantum and classical routing are effectively tied in
authentication-only, DNS-only, random-loss, and hub-loss conditions. The
quantum router is higher by 0.0083 MRR under complete telemetry, but the
interval crosses zero. Relative to temporal-only ranking, the quantum router
improves DNS-only MRR by 0.0222 with an interval of 0.0011 to 0.0525. This is a
structural-fusion effect, not evidence of quantum superiority, because the
classical router obtains essentially the same result.

Finite-shot behavior is stable. The mean absolute MRR difference between the
256-shot and ideal circuits is 0.00027 in the controlled study and 0.000013 on
LANL. The rankings are identical in 97.6 percent of controlled episode-seed
pairs and 67.3 percent of LANL pairs.

## Submission-safe claim

A shallow variational circuit can implement a parameter-matched, constant-call
reliability router for mechanistic graph evidence. It matches the classical
router on chronological enterprise episodes, is more conservative in selected
controlled growth regimes, and remains stable under finite-shot measurement.
The study supports feasibility and resource-aware hybrid routing, not quantum
advantage.

## Artifacts

- Implementation: `ba3active/qrag_moe.py`
- Statistical report: `ba3active/qrag_moe_report.py`
- Tests: `tests/test_qrag_moe.py`
- Raw outputs: `outputs/qrag_moe_v2/`
- Claim intervals: `outputs/qrag_moe_v2/report/qrag_claims.csv`
- Controlled plot: `outputs/qrag_moe_v2/report/figure_qrag_controlled.png`
- LANL plot: `outputs/qrag_moe_v2/report/figure_qrag_lanl.png`
- Machine-readable audit: `outputs/qrag_moe_v2/report/qrag_audit.json`
