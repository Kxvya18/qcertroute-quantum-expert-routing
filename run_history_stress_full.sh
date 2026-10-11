#!/usr/bin/env bash
set -euo pipefail

cd /Users/boolu/Desktop/simulations/ba3_upgrade

export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR=/tmp/mpl-iqrag-history
export OMP_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export PYTHONPATH=/Users/boolu/Desktop/simulations/ba3_upgrade

OUT=outputs/quantum_history_stress_v1
test -f "$OUT/diffusion_episodes_history_stress.pkl"

caffeinate -dimsu /opt/anaconda3/envs/cdg/bin/python -u -m ba3active.interdependent_qrag \
  --cache "$OUT/diffusion_episodes_history_stress.pkl" \
  --output "$OUT" \
  --temporal-device mps \
  --router-device mps \
  --cpu-threads 8 \
  --folds 3 \
  --seeds 19,73,101 \
  --resume 2>&1 | tee "$OUT/full.log"

/opt/anaconda3/envs/cdg/bin/python -m ba3active.interdependent_checkpoint --results "$OUT"

mkdir -p "$OUT/report"
/opt/anaconda3/envs/cdg/bin/python -m ba3active.interdependent_qrag_report \
  --results "$OUT" --output "$OUT/report" --bootstrap 4000 --tables-only

/opt/anaconda3/envs/cdg/bin/python -m ba3active.quantum_forward_figures \
  --results "$OUT" \
  --output quantum_forward_paper/figures
