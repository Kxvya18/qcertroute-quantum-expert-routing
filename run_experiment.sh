#!/usr/bin/env bash
set -euo pipefail

cd /Users/boolu/Desktop/simulations/ba3_upgrade

export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR=/tmp/mpl-iqrag
export OMP_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export PYTHONPATH=/Users/boolu/Desktop/simulations/ba3_upgrade

mkdir -p outputs/interdependent_qrag_v7

echo "Running 3-seed confirmatory experiment (resuming from checkpoint)..."
caffeinate -dimsu \
/opt/anaconda3/envs/cdg/bin/python -u -m ba3active.interdependent_qrag \
  --cache outputs/interdependent_qrag_v6/diffusion_episodes.pkl \
  --output outputs/interdependent_qrag_v7 \
  --temporal-device mps \
  --router-device mps \
  --cpu-threads 8 \
  --folds 3 \
  --seeds 7,19,42 \
  --resume 2>&1 | tee outputs/interdependent_qrag_v7/experiment.log

echo "Experiment complete. Run ./run_report.sh to generate figures and tables."
