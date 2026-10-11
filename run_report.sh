#!/usr/bin/env bash
set -euo pipefail

cd /Users/boolu/Desktop/simulations/ba3_upgrade

export MPLCONFIGDIR=/tmp/mpl-iqrag
export PYTHONPATH=/Users/boolu/Desktop/simulations/ba3_upgrade

mkdir -p outputs/interdependent_qrag_v7/report

echo "Generating publication figures and statistical tables..."
/opt/anaconda3/envs/cdg/bin/python -u -m ba3active.interdependent_qrag_report \
  --results outputs/interdependent_qrag_v7 \
  --output outputs/interdependent_qrag_v7/report \
  --bootstrap 4000 2>&1 | tee outputs/interdependent_qrag_v7/report.log

echo "Report generated in outputs/interdependent_qrag_v7/report/"
