# Decision-certified quantum routing

This is the proposed replacement for the LANL-dependent paper. It is a design and run protocol, not a claim that the new experiment has succeeded. The old manuscript at `quantum_forward_paper/main.tex` still reports LANL results and must not be submitted as if it described this benchmark.

## Research question

Can a small variational quantum circuit arbitrate temporal and historical graph evidence for real next-recipient retrieval while using fewer state preparations than a fixed-shot quantum router? The primary output is a top-ten candidate **set** for triage. Ordering within that set and full probability calibration are secondary tasks.

The quantum claim is **measurement efficiency conditional on using a quantum router**, not quantum computational speedup or predictive superiority over a classical model. A similar-capacity classical router remains an essential comparator. The quantum circuit has 25 trainable parameters and the two-hidden-unit classical router has 23. If the classical router is better on the primary predictive task, say so.

## Literature boundary

- Quantum MoE and quantum gating are already proposed in [QMoE](https://arxiv.org/abs/2507.05190) and [hybrid quantum-classical MoE](https://arxiv.org/abs/2512.22296). We must not claim to invent quantum MoE.
- Adaptive allocation of quantum measurements is established in [AEQuO](https://quantum-journal.org/papers/q-2023-01-26-906/) and [gCANS](https://arxiv.org/abs/2108.10434). Our specific question is when the *downstream candidate set*, rather than the expectation value, is determined.
- The need for honest classical comparison follows [Power of data in quantum machine learning](https://www.nature.com/articles/s41467-021-22539-9) and [The inductive bias of quantum kernels](https://proceedings.neurips.cc/paper/2021/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html).
- The formal sequential confidence argument is a finite-look union bound. More elaborate anytime-valid confidence sequences are available in [Howard et al.](https://arxiv.org/abs/1810.08240). The present implementation uses only six prespecified looks and does not claim a new concentration inequality.

## Real data and leakage control

The [SNAP email-Eu-core-temporal](https://snap.stanford.edu/data/email-Eu-core-temporal.html) dataset is a real timestamped directed email network. It contains 332,334 observed events and 986 distinct members. Node IDs have gaps, so arrays have 1,005 slots. A label is the first recipient of a sampled sender on an active day. The recent temporal expert uses only days 1 to 7 before the label; the graph expert uses only days 8 to 90 before the label. Recipient candidates were observed on earlier days. The two evidence windows are disjoint. Complete, query-half, history-half, and joint-half conditions are paired on the same sender-day labels. Event masks are fixed without using outcomes.

The first full extraction yielded 3,503 episodes. The chronological split is 2,106 train, 700 validation, and 697 test episodes. These are empirical benchmark facts, not quantum findings.

## Quantum-first method

The four-qubit, depth-two data-reuploading circuit maps nine episode descriptors to one mixing coefficient. Its local Z and adjacent ZZ observables commute. Therefore a single computational-basis bitstring yields all eight observable values jointly. The trained readout for bitstring `b` is the bounded scalar `Y(b) = sum_j w_j O_j(b)`. Estimating the mean of this scalar is tighter than separately bounding every observable and unioning eight intervals.

For candidate `v`, the fused probability is `p_v(a) = (1-a) pT_v + a pG_v`. Its dependence on `a` is affine. Under fixed node-ID tie breaking, if the top-ten set is identical at both ends of an interval `[aL,aU]`, it is identical throughout the interval. For every selected node and every excluded node, their score difference is affine and nonnegative at both endpoints, so it cannot be negative between them.

The decision protocol first checks `a=0` and `a=1`. If their top-ten sets agree, it emits that set with **zero circuit shots**. Otherwise it measures jointly at prespecified budgets 16, 32, 64, 128, 256, and 512. Let `R` be the exact range of `Y(b)` over all sixteen computational-basis outcomes. At budget `n`, a simultaneous finite-look radius is `R sqrt(log(2L/delta)/(2n))`, where `L=6`. The sigmoid maps the logit interval to a gate interval. If the endpoint top-ten sets agree, the decision is certified and measurement stops. If 512 shots do not certify it, the system **abstains** from a certified top-ten output; a forced 512-shot set is recorded only for diagnostic comparison.

The nominal probability statement is conditional on fixed experts, a fixed trained circuit, independent ideal bitstring measurements, and the six prespecified looks. It does not cover readout noise or model error. It certifies top-ten set invariance, not the internal ranking of that set or superiority to a classical router.

## Required comparisons

1. Temporal expert, historical graph expert, validation-selected fixed mix, three quantum seeds, and three similar-capacity classical seeds on identical chronological splits.
2. Fixed 16, 32, 64, 128, 256, and 512 shot top-ten agreement against the exact circuit.
3. The old per-observable simultaneous bound versus the new direct joint-readout scalar bound, with the same zero-shot bypass.
4. Certified coverage, zero-shot fraction, unresolved fraction, mean shots over *all* decisions, and conditional certified agreement with the exact circuit. Do not report agreement only among easy accepted cases without the coverage denominator.
5. Stratification by the four telemetry conditions and day-block uncertainty intervals. Do not count four masks of one sender-day as independent examples.

Acceptance of the new primary claim requires meaningful shot savings versus fixed 512, high certified coverage, and candidate-set agreement near the nominal confidence level across independent measurement seeds. Predictive utility should be reported separately. If the classical router is stronger, the paper is a quantum resource-efficiency and limits study, not a quantum-advantage paper.

## First decision checkpoint

The first completed decision run used quantum model seed 0 and measurement seed 4909 on all 697 held-out sender-day episodes under four paired conditions, giving 2,788 decisions. It passed the two invariance tests. Mean cost was 226.48 shots per decision, including unresolved decisions that spent the full 512-shot budget. This is 55.76 percent fewer state preparations than fixed 512. Median cost was 128 shots. The direct joint-readout bound certified 71.92 percent of decisions, compared with 67.72 percent for the separate-observable bound, and used 226.48 rather than 254.81 mean shots. All certified sets matched the ideal simulator on this run, but 28.08 percent remained uncertified and the nominal 95 percent guarantee is per decision under ideal independent measurements.

The zero-shot bypass fired on only 4 of 2,788 decisions. It is mathematically valid but empirically negligible, so it should **not** be presented as the main efficiency mechanism. The meaningful observed mechanism is the scalar joint-readout confidence bound. The 75.86 percent target hit rate among certified decisions is selected-case performance; it must not be compared with the 73.31 percent exact-circuit hit rate over all decisions as though it were a predictive gain. Forced adaptive and fixed-512 decisions both matched the exact top-ten set in 97.56 percent of cases using the same simulated bitstrings. No hardware result or classical-computational speedup has been established.

This first checkpoint used one trained model and one measurement stream. The subsequent section reports additional seeds; day-block uncertainty intervals remain to be checked. The older predictive test showed the classical router outperforming the quantum gate, so the new paper must not claim predictive quantum superiority unless a new controlled test changes that finding.

## Three-model checkpoint

The later runs cover model seeds 0, 1, and 2 on the same 697 held-out sender-day episodes, plus an independent measurement repeat for seed 0. Across the three distinct trained models, the unweighted mean shot count is 202.83, a 60.39 percent reduction relative to fixed 512. Individual model means are 226.48, 172.62, and 209.38. Their certified fractions are 71.92, 79.56, and 74.43 percent. The corresponding separate-observable rule uses 254.81, 233.36, and 239.17 mean shots and certifies 67.72, 69.84, and 69.94 percent. Thus the direct scalar readout uses 39.62 fewer shots on average and increases coverage by 6.13 percentage points across these three models. The independent seed-0 measurement repeat gives 225.02 shots and 72.02 percent certification, close to the original 226.48 and 71.92 percent.

All certified sets matched the ideal simulator in these particular runs. This does not prove error-free operation or validate physical hardware noise. The same test episodes are reused across trained models and four masks, so the 8,364 model-mask decisions are **not** independent samples. The day-block report below provides paired uncertainty within each model seed. The primary paper claim can now be stated as a reproducible *ideal-simulator shot-efficiency finding at partial coverage*, subject to those intervals and the existing classical-prediction limitation.

## Run order

Use the existing `cdg` environment. All outputs below are new; they do not overwrite the LANL paper.

```bash
cd /Users/boolu/Desktop/simulations/ba3_upgrade
DATA=/Users/boolu/Desktop/simulations/ba3_upgrade/data/email-Eu-core-temporal.txt.gz
test -f "$DATA"
/opt/anaconda3/envs/cdg/bin/python -m unittest discover -s tests -p test_email_qrouter_decision.py
/opt/anaconda3/envs/cdg/bin/python -m ba3active.email_qrouter --input "$DATA" --output outputs/email_qrouter_v1
/opt/anaconda3/envs/cdg/bin/python -m ba3active.email_qrouter_train --episodes outputs/email_qrouter_v1/episodes.npz --output outputs/email_qrouter_v1/train --epochs 80 --seeds 3
/opt/anaconda3/envs/cdg/bin/python -m ba3active.email_qrouter_shots --episodes outputs/email_qrouter_v1/episodes.npz --training outputs/email_qrouter_v1/train --output outputs/email_qrouter_v2/shots_model0_repeat0 --model-seed 0 --shot-seed 4909
/opt/anaconda3/envs/cdg/bin/python -m ba3active.email_qrouter_shots --episodes outputs/email_qrouter_v1/episodes.npz --training outputs/email_qrouter_v1/train --output outputs/email_qrouter_v2/shots_model1_repeat0 --model-seed 1 --shot-seed 4910
/opt/anaconda3/envs/cdg/bin/python -m ba3active.email_qrouter_shots --episodes outputs/email_qrouter_v1/episodes.npz --training outputs/email_qrouter_v1/train --output outputs/email_qrouter_v2/shots_model2_repeat0 --model-seed 2 --shot-seed 4911
/opt/anaconda3/envs/cdg/bin/python -m ba3active.email_qrouter_report --inputs outputs/email_qrouter_v2/shots_model0_repeat0/decisions.csv outputs/email_qrouter_v2/shots_model1_repeat0/decisions.csv outputs/email_qrouter_v2/shots_model2_repeat0/decisions.csv --output outputs/email_qrouter_v2/report --bootstrap 2000
```

Run extraction and training only if their existing outputs are missing or need regeneration. The shot runs above have now completed; only the lightweight `email_qrouter_report` command remains. It writes paired day-block intervals and condition summaries. The old paper and its three figures have not been replaced because they describe LANL rather than this benchmark.
