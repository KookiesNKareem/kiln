#!/bin/bash
set -u
S=$1; GPU=$2
cd "$(dirname "$0")"
export NO_COLOR=1
out=$(colab new -s "$S" --gpu "$GPU" 2>&1); echo "$out" | tail -5
colab sessions 2>&1 | grep -q "\[$S\]" || { echo "NEW FAILED"; exit 1; }
colab upload -s "$S" gpu_bench.py /content/gpu_bench.py 2>&1 | tail -1
colab upload -s "$S" oplist.json /content/oplist.json 2>&1 | tail -1
cat > /tmp/launch_$S.py <<'PY'
import subprocess
subprocess.Popen("nohup python /content/gpu_bench.py /content/oplist.json /content/results.json > /content/bench.log 2>&1 &", shell=True)
print("launched")
PY
colab exec -s "$S" -f /tmp/launch_$S.py 2>&1 | tail -3
