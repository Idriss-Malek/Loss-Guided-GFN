#!/usr/bin/env bash
set -euo pipefail

# ---- settings you can tweak ----
EDGE_PROB=0.4
ALGORITHMS=("lggfn" "AT")
SEEDS=(1 2 3)
SCRIPT="examples/bayesian_structure.py"   # <--- change if your filename is different
DEVICE_STR="cuda"        # set to "cuda" if you want GPU
# --------------------------------

mkdir -p logs

for n in {5..13}; do
  # num_edges = round(p * n*(n-1)/2)  -- matches how your script computes edge_prob
  m=$(python - <<PY
n=$n
p=$EDGE_PROB
print(round(p * n*(n-1)/2))
PY
)
  for algo in "${ALGORITHMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ts=$(date +%Y%m%d-%H%M%S)
      log="logs/${algo}_n${n}_m${m}_seed${seed}_${ts}.log"
      echo "[$(date)] n=$n m=$m algo=$algo seed=$seed -> $log"
      python -u "$SCRIPT" \
        --algo "$algo" \
        --num_nodes "$n" \
        --num_edges "$m" \
        --seed "$seed" \
        --graph_name erdos_renyi_lingauss \
        --prior_name uniform \
        --num_samples_posterior 1000 \
        --batch_size 256 \
        --iterations 1000 \
        --device_str "$DEVICE_STR" \
        --lamda 0.1\
        >> "$log" 2>&1
    done
  done
done

echo "All runs finished."
