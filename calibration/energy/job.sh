#!/bin/bash
# Energy bench on a Colab A100 (all compute remote).
#   job.sh new <session>                     create A100 session
#   job.sh launch <session> <tag> [args...]  upload code, run `energy_bench.py <args>` detached -> /content/<tag>.{json,log}
#   job.sh sh <session> "<shell cmd>" [timeout]
#   job.sh get <session> <tag>               -> ../measurements/energy/<tag>.{json,log}
set -u
export NO_COLOR=1
cd "$(dirname "$0")"
cmd=$1; S=$2; shift 2
case $cmd in
new)
  colab new -s "$S" --gpu A100 2>&1 | tail -3
  colab sessions 2>&1 | grep -q "\[$S\]" || { echo "NEW FAILED"; exit 1; } ;;
launch)
  TAG=$1; shift
  for f in kernels.cu energy_bench.py ncu_check.sh; do colab upload -s "$S" "$f" "/content/$f" 2>&1 | tail -1; done
  cat > "/tmp/elaunch_$S.py" <<PY
import subprocess
subprocess.Popen("cd /content && nohup $* >> /content/$TAG.log 2>&1 &", shell=True)
print("launched $TAG")
PY
  colab exec -s "$S" -f "/tmp/elaunch_$S.py" 2>&1 | tail -2 ;;
sh)
  C=$1; TO=${2:-60}
  python3 -c 'import sys,json; print("import subprocess\nr=subprocess.run(%s, shell=True, capture_output=True, text=True)\nprint(r.stdout[-6000:], r.stderr[-3000:])" % json.dumps(sys.argv[1]))' "$C" > "/tmp/esh_$S.py"
  colab exec -s "$S" -f "/tmp/esh_$S.py" --timeout "$TO" 2>&1 ;;
get)
  TAG=$1
  for e in json log; do colab download -s "$S" "/content/$TAG.$e" "../measurements/energy/$TAG.$e" 2>&1 | tail -1; done ;;
*) echo "unknown $cmd"; exit 2 ;;
esac
