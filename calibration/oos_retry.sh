#!/bin/bash
# Retry A100 allocation until it succeeds (up to ~3 h), run the frozen OOS benchmark, fetch results, stop.
set -u
cd "$(dirname "$0")"
S=accel-oos-a100; TAG=a100_2026-10-05_seq_oos
for i in $(seq 1 18); do
  if ./colab_job.sh new "$S" --gpu A100 | grep -qv "NEW FAILED" && colab sessions 2>&1 | grep -q "$S"; then
    echo "allocated on attempt $i at $(date)"
    ./colab_job.sh launch "$S" gpu_bench.py seq_oos "$TAG"
    for j in $(seq 1 40); do
      sleep 60
      out=$(./colab_job.sh poll "$S" "$TAG" 3)
      echo "$out" | grep -q "bench.py" || break
    done
    ./colab_job.sh get "$S" "$TAG" "measurements/$TAG"
    colab stop -s "$S" 2>&1 | tail -1
    echo "DONE $(date)"; exit 0
  fi
  echo "attempt $i failed at $(date)"; sleep 600
done
echo "GAVE UP $(date)"; exit 1
