# RAG-MoE Results Manifest

## Implemented system

RAG-MoE contains a 553-parameter temporal MLP and a 129-parameter reliability
gate. The gate receives normalized temporal entropy, temporal-structural rank
disagreement, Fisher reliability, missingness, fitted-parameter standard error,
and degree Gini. Its output is bounded by a sigmoid. The final score is

`temporal logit + gate * Fisher reliability * centered structural residual`.

The bounded form guarantees that the structural displacement of any node logit
cannot exceed the magnitude of its Fisher-weighted structural residual. The
gate is trained against cross-fitted admission targets produced only from the
training split. The LANL test incidents are not used to train the temporal
expert, scaler, or selector.

## Evaluation

- Controlled mechanisms use beta in 0, 1, and 2 and attachment count in 1, 2,
  and 3. Each cell has 24 training and 12 held-out episodes.
- LANL uses four rolling chronological folds and 20 distinct held-out incident
  episodes.
- Every principal comparison uses five fixed training seeds.
- Conditions are complete, auth-only, DNS-only, random 50 percent loss, and
  hub-targeted 50 percent loss.
- Baselines are temporal MLP, structural expert, unconditional fusion, and
  fixed-curvature fusion.
- Confidence intervals use two-way bootstrap resampling over episodes and
  training seeds. Seeds are not treated as additional incident samples.

## Defensible findings

Across controlled mechanisms, RAG-MoE improves MRR over the temporal expert by
0.0505 under random loss and 0.1547 under hub loss. The corresponding 95 percent
intervals are 0.0237 to 0.0843 and 0.0800 to 0.2393. At beta zero, mean gate
admission is below 0.0033 in every cell and RAG-MoE reproduces the temporal
ranking, demonstrating suppression when no preferential signal exists.

On LANL, RAG-MoE improves DNS-only MRR by 0.0341 with a 95 percent interval of
0.0013 to 0.1026 and hub-loss MRR by 0.0209 with an interval of 0.0009 to
0.0541. Complete, auth-only, and random-loss intervals cross zero. The LANL
evidence therefore supports conditional transfer under specific missingness
mechanisms, not universal superiority.

The controlled beta-one grid contains harmful cells, most clearly DNS-only at
attachment count three. This is retained as a limitation. Learned admission
reduces but does not eliminate every structural failure mode.

## Artifacts

- Raw controlled results are in `outputs/rag_moe_v5/rag_synthetic_results.csv`.
- Raw LANL results are in `outputs/rag_moe_v5/rag_lanl_results.csv`.
- Claim intervals are in `outputs/rag_moe_v5/report/table_rag_claim_audit.csv`.
- Full controlled and LANL tables are available as CSV and LaTeX fragments in
  `outputs/rag_moe_v5/report`.
- Publication figures are in `figures_rag_moe_v5` as 400 dpi PNG and vector PDF.
- The machine-readable audit is
  `outputs/rag_moe_v5/report/rag_moe_statistical_audit.json`.

## Claim boundary

The evidence supports a lightweight learned selector that conditionally admits
a mechanistic structural expert under missing telemetry. It does not establish
that LANL follows a Barabasi-Albert process, that structural fusion is always
beneficial, or that the current small incident set is sufficient for broad
operational deployment claims.
