# QCertRoute: Shot-Adaptive Quantum Expert Routing with Certified Top-Ten Decisions on Temporal Networks

**Kavya Kalavala**, Manipal Institute of Technology Bengaluru (IEEE conference manuscript).

Finite quantum measurements need not estimate every observable precisely to keep an ordered shortlist correct.
QCertRoute routes between experts on temporal communication networks with a shallow variational circuit that fuses a
recent-contact expert and a historical-diffusion expert through local and pairwise readouts of shared-basis
bitstrings, and a sequential confidence rule that spends measurement shots only until the top-ten decision is certified.

Main modules (in `ba3active/`): `email_qrouter.py` (router), `email_qrouter_train.py`, `email_qrouter_shots.py`
(shot-adaptive certification), `email_qrouter_report.py`, `quantum_forward_figures.py`; tests in `tests/`.
The same package also contains the cost-aware active-telemetry and RAG-MoE studies it grew out of (described below).
Datasets are not redistributed.

---

# BA3 Active Telemetry Evidence Suite

This package implements the compact result-producing study for the active
telemetry manuscript. The primary task is leakage-controlled recovery of the
first red-team destination endpoint within a sessionized incident. It does not
claim that LANL is a Barabasi-Albert network. The structural component is a
Boltzmann endpoint-choice prior fitted on pre-incident authentication history.

## Primary protocol

- 15-minute red-team sessionization
- One-hour calibration history and one-hour held-out pre-event fit history
- Two-hour post-event evidence window
- Auth and DNS as separately priced modalities
- Four rolling chronological folds
- Five fixed seeds and paired inference across held-out episode instances

The controller result is considered supportive only when the paired lower
confidence limit for AUBC versus entropy-only acquisition exceeds zero.

## Reliability-Aware Graph Mixture of Experts

The `ba3active.rag_moe` module adds a compact mixture-of-experts model for
source ranking under incomplete telemetry. A temporal MLP remains the primary
expert. A centered Boltzmann endpoint-choice residual is the structural expert,
and a learned gate admits that residual according to Fisher reliability,
temporal uncertainty, rank disagreement, missingness, standard error, and
degree inequality. The temporal expert is pretrained and frozen before gate
training. Gate targets are generated out of fold on training incidents only.

The final evaluation uses five model seeds, five telemetry conditions, 108
controlled held-out episodes, and 20 chronologically held-out LANL incidents.
Two-way bootstrap intervals resample both episodes and training seeds. The
result is conditional rather than universal. RAG-MoE suppresses structural
evidence for uniform attachment, improves controlled random and hub loss, and
improves LANL DNS-only and hub-loss settings. Other LANL intervals cross zero.

```bash
PYTHONPATH=. python -m ba3active.rag_moe \
  --input outputs/extracted_episodes.pkl.gz \
  --output outputs/rag_moe_v5

PYTHONPATH=. python -m ba3active.rag_moe_report \
  --results outputs/rag_moe_v5 \
  --figures figures_rag_moe_v5 \
  --tables outputs/rag_moe_v5/report
```

See `RAG_MOE_RESULTS.md` for the claim boundary and artifact inventory.

## Quantum-forward LANL routing study

The `ba3active.interdependent_qrag` experiment treats quantum routing as the
primary method and temporal and graph-diffusion models as evidence-producing
experts. It compares validation-selected data-reuploading variational circuits,
quantum extreme-learning reservoirs, a parameter-matched classical router,
context ablations, local-only measurements, finite-shot execution, and noisy
measurement models. Router targets are cross-fitted rank regrets, so test labels
never determine expert admission.

The experiment distinguishes two publishable outcomes. An advantage study
requires every predeclared superiority and dependency check to pass. A limits
study requires a noncollapsed, context-dependent, finite-shot-stable quantum
regime transition, while reporting classical superiority or nonbeneficial
entanglement as negative findings rather than hiding them.

```bash
PYTHONPATH=. python -m ba3active.interdependent_qrag \
  --cache outputs/interdependent_qrag_v6/diffusion_episodes.pkl \
  --output outputs/quantum_forward_confirmatory \
  --temporal-device mps --router-device mps --folds 3 --seeds 19,42,73

PYTHONPATH=. python -m ba3active.interdependent_checkpoint \
  --results outputs/quantum_forward_confirmatory
```
