#!/bin/bash
# Colab job helper (all compute stays on Colab).
#   colab_job.sh new    <session> --gpu A100 | --tpu v5e1|v6e1
#   colab_job.sh launch <session> <runner.py> <suite> <tag>     -> /content/<tag>.json, /content/<tag>.log (detached)
#                 (LAUNCH_ENV="K=V ..." is prefixed to the remote command)
#   colab_job.sh poll   <session> <tag> [lines]
#   colab_job.sh get    <session> <tag> <local_base>           -> <local_base>.json/.log
set -u
export NO_COLOR=1
cd "$(dirname "$0")"
cmd=$1; S=$2; shift 2
case $cmd in
new)
  colab new -s "$S" "$@" 2>&1 | tail -3
  colab sessions 2>&1 | grep -q "$S" || { echo "NEW FAILED"; exit 1; } ;;
launch)
  R=$1; SUITE=$2; TAG=$3
  colab upload -s "$S" "$R" "/content/$R" 2>&1 | tail -1
  colab upload -s "$S" oplist.json /content/oplist.json 2>&1 | tail -1
  cat > "/tmp/launch_$S.py" <<PY
import subprocess
subprocess.Popen("cd /content && ${LAUNCH_ENV:-} nohup python /content/$R $SUITE /content/$TAG.json /content/oplist.json >> /content/$TAG.log 2>&1 &", shell=True)
print("launched $TAG")
PY
  colab exec -s "$S" -f "/tmp/launch_$S.py" 2>&1 | tail -2 ;;
poll)
  TAG=$1; N=${2:-8}
  cat > "/tmp/poll_$S.py" <<PY
import os, subprocess
print(subprocess.run("tail -n $N /content/$TAG.log; ls -la /content/$TAG.json; pgrep -af 'bench.py' | grep -v pgrep | head -3", shell=True, capture_output=True, text=True).stdout)
PY
  colab exec -s "$S" -f "/tmp/poll_$S.py" --timeout 60 2>&1 | tail -n $((N + 6)) ;;
get)
  TAG=$1; L=$2
  colab download -s "$S" "/content/$TAG.json" "$L.json" 2>&1 | tail -1
  colab download -s "$S" "/content/$TAG.log" "$L.log" 2>&1 | tail -1 ;;
*) echo "unknown $cmd"; exit 2 ;;
esac
