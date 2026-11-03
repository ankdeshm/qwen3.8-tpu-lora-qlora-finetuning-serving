#!/bin/bash
# Sequentially run fine-tuning configs on the lora-ft dev Job, each starting when the previous finished.
# usage: ft_chain.sh <wait_for_run_or_-> <config1>:<run1> [<config2>:<run2> ...]
# A run whose name starts with "dbg_" must succeed (rc=0 + >=5 logged steps) for the chain to continue.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CTX=gke_YOUR_GCP_PROJECT_ID_us-central1_tpu-v5p-cluster
K="kubectl --context ${CTX}"
P0=$(${K} get pods -l job-name=lora-ft-dev,batch.kubernetes.io/job-completion-index=0 --field-selector=status.phase=Running -o name | head -1 | sed 's#pod/##')
wait_done() {  # $1 = run name; waits for the node0 "[launch_ft] rc=" line, echoes rc
  local run=$1
  while true; do
    local line
    line=$(${K} exec "${P0}" -c trainer -- bash -c "grep -a '\[launch_ft\] rc=' /dev/shm/logs/${run}.node0.log 2>/dev/null | tail -1" 2>/dev/null)
    if [ -n "${line}" ]; then echo "${line}" | sed -E 's/.*rc=([0-9]+).*/\1/'; return; fi
    sleep 30
  done
}
first=$1; shift
if [ "${first}" != "-" ]; then
  echo "[chain] $(date -u +%FT%TZ) waiting for ${first}"; rc=$(wait_done "${first}"); echo "[chain] ${first} rc=${rc}"
fi
for item in "$@"; do
  cfg=${item%%:*}; run=${item##*:}
  "${HERE}/ft_ctl.sh" stop >/dev/null 2>&1; sleep 5
  echo "[chain] $(date -u +%FT%TZ) start ${cfg} as ${run}"
  "${HERE}/ft_ctl.sh" start "${cfg}" "${run}"
  sleep 60
  rc=$(wait_done "${run}")
  nsteps=$(${K} exec "${P0}" -c trainer -- bash -c "grep -a -c '\[titan\].*step:' /dev/shm/logs/${run}.node0.log" 2>/dev/null)
  echo "[chain] $(date -u +%FT%TZ) ${run} rc=${rc} logged_steps=${nsteps}"
  if [[ "${run}" == dbg_* ]] && { [ "${rc}" != "0" ] || [ "${nsteps:-0}" -lt 5 ]; }; then
    echo "[chain] smoke test ${run} failed -> stopping chain"; exit 1
  fi
done
echo "[chain] $(date -u +%FT%TZ) all done"
