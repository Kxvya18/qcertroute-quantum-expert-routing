#!/usr/bin/env bash
set -euo pipefail

cd /Users/boolu/Desktop/simulations/ba3_upgrade

LANL_AUTH=${LANL_AUTH:-/Volumes/Seagate/simulations/ALL_PROJECTS/ba3_active_telemetry/data/raw/lanl/auth.txt.mirror.gz}
LANL_DNS=${LANL_DNS:-/Volumes/Seagate/simulations/ALL_PROJECTS/ba3_active_telemetry/data/raw/lanl/dns.txt.mirror.gz}
for source_file in "$LANL_AUTH" "$LANL_DNS"; do
  if [[ ! -r "$source_file" ]]; then
    printf 'LANL source file is unavailable: %s\n' "$source_file" >&2
    printf 'Mount Seagate or set LANL_AUTH and LANL_DNS to the readable .gz files.\n' >&2
    exit 1
  fi
done

export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR=/tmp/mpl-iqrag-history
export OMP_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export PYTHONPATH=/Users/boolu/Desktop/simulations/ba3_upgrade

OUT=outputs/quantum_history_stress_v1
mkdir -p "$OUT"

# Fresh extraction is required because historical edge masking cannot be
# recovered from a cache that stores only the already-computed graph scores.
caffeinate -dimsu /opt/anaconda3/envs/cdg/bin/python -u -m ba3active.interdependent_qrag \
  --auth "$LANL_AUTH" \
  --dns "$LANL_DNS" \
  --cache "$OUT/diffusion_episodes_history_stress.pkl" \
  --output "$OUT" \
  --temporal-device mps \
  --router-device mps \
  --cpu-threads 8 \
  --folds 3 \
  --seeds 19 \
  --max-runs 1 2>&1 | tee "$OUT/pilot.log"

/opt/anaconda3/envs/cdg/bin/python -m ba3active.interdependent_checkpoint --results "$OUT" || true
