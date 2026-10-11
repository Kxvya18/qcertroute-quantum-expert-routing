#!/usr/bin/env bash
set -euo pipefail

cd /Users/boolu/Desktop/simulations/ba3_upgrade

export PYTHONPATH=/Users/boolu/Desktop/simulations/ba3_upgrade

if [ ! -f outputs/interdependent_qrag_v7/interdependent_qrag_results.csv ]; then
  echo "No results found in outputs/interdependent_qrag_v7 yet."
  echo "Please run ./run_pilot.sh first!"
  exit 1
fi

echo "=== CHECKPOINT AUDIT ==="
/opt/anaconda3/envs/cdg/bin/python -m ba3active.interdependent_checkpoint \
  --results outputs/interdependent_qrag_v7

echo ""
echo "=== INTERIM CLAIMS ==="
if [ -f outputs/interdependent_qrag_v7/interim_claims.csv ]; then
  column -s, -t outputs/interdependent_qrag_v7/interim_claims.csv
fi

echo ""
echo "=== INTERIM SUMMARY ==="
if [ -f outputs/interdependent_qrag_v7/interim_summary.csv ]; then
  column -s, -t outputs/interdependent_qrag_v7/interim_summary.csv
fi
