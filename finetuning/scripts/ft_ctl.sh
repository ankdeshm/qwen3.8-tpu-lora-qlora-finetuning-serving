#!/bin/bash
# Host-side (Mac) orchestration for the lora-ft dev Job: push code to the 4 pods, start / stop runs, tail logs.
# usage:
#   ft_ctl.sh push
#   ft_ctl.sh start <config> <run_name> [extra torchtitan overrides...]
#   ft_ctl.sh tail <run_name> [node]      # tail log of node (default 0)
#   ft_ctl.sh stop                        # kill torchrun / python trainers in all 4 pods
#   ft_ctl.sh grep <run_name> <regex>     # grep node0 log
set -uo pipefail
CTX=gke_YOUR_GCP_PROJECT_ID_us-central1_tpu-v5p-cluster
K="kubectl --context ${CTX}"
HERE="$(cd "$(dirname "$0")" && pwd)"
JOB=${LORA_FT_JOB:-lora-ft-dev}
pods() { for i in 0 1 2 3; do ${K} get pods -l job-name=${JOB},batch.kubernetes.io/job-completion-index=${i} --field-selector=status.phase=Running -o name | head -1 | sed 's#pod/##'; done; }
cmd=$1; shift
case "$cmd" in
  push)
    for p in $(pods); do
      for f in lora_tpu_recipes.py launch_ft.sh; do
        for attempt in 1 2 3 4 5; do
          if ${K} exec -i "$p" -c trainer -- bash -c "mkdir -p /dev/shm/code /dev/shm/logs && cat > /dev/shm/code/${f}.tmp && mv /dev/shm/code/${f}.tmp /dev/shm/code/${f} && chmod +x /dev/shm/code/${f}" < "${HERE}/${f}"; then break; fi
          echo "retry push $f -> $p ($attempt)"; sleep 2
        done
      done
      echo "pushed -> $p ($(${K} exec "$p" -c trainer -- md5sum /dev/shm/code/lora_tpu_recipes.py | cut -c1-8))"
    done ;;
  start)
    config=$1; run=$2; shift 2
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    extra="$*"
    for p in $(pods); do
      ${K} exec "$p" -c trainer -- bash -c "LORA_FT_TS=${ts} nohup /dev/shm/code/launch_ft.sh ${config} ${run} ${extra} >/dev/null 2>&1 &" &
    done
    wait
    echo "started ${config} as ${run} ts=${ts}" ;;
  tail)
    run=$1; node=${2:-0}; p=$(pods | sed -n "$((node + 1))p")
    ${K} exec "$p" -c trainer -- bash -c "grep -v -E '^(I|W)0000|^WARNING: All log' /dev/shm/logs/${run}.node${node}.log | tail -${TAIL_N:-40} | cut -c1-400" ;;
  grep)
    run=$1; rx=$2; p=$(pods | sed -n 1p)
    ${K} exec "$p" -c trainer -- bash -c "grep -E '${rx}' /dev/shm/logs/${run}.node0.log | cut -c1-400 | tail -${TAIL_N:-80}" ;;
  stop)
    for p in $(pods); do ${K} exec "$p" -c trainer -- bash -c "pkill -f torchrun; pkill -f torchtitan.experiments.tpu.train; sleep 2; pkill -9 -f torchtitan.experiments.tpu.train; true" & done; wait; echo stopped ;;
  fetch)
    run=$1; dst=${2:-${HERE}/logs}; mkdir -p "$dst"; i=0
    for p in $(pods); do ${K} cp "default/${p}:/dev/shm/logs/${run}.node${i}.log" "${dst}/${run}.node${i}.log" -c trainer >/dev/null 2>&1; i=$((i + 1)); done
    ls -la "$dst" | grep "$run" ;;
  *) echo "unknown cmd $cmd"; exit 2 ;;
esac
