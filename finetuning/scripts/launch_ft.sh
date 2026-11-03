#!/bin/bash
# Per-host launcher for the LoRA / QLoRA fine-tuning runs (run in each of the 4 pods of the
# lora-ft Indexed Job; node rank = JOB_COMPLETION_INDEX).
# usage: launch_ft.sh <config> <run_name> [extra torchtitan CLI overrides...]
set -uo pipefail
CONFIG=$1
RUN=$2
shift 2
NODE_RANK=${JOB_COMPLETION_INDEX:-0}
NNODES=${LORA_FT_NNODES:-4}
NPROC=${LORA_FT_NPROC:-4}
MASTER=${LORA_FT_MASTER:-lora-ft-dev-0.lora-ft-dev-svc.default.svc.cluster.local}
PORT=${LORA_FT_PORT:-29511}
CODE=${LORA_FT_CODE:-/dev/shm/code}
LOGDIR=${LORA_FT_LOGDIR:-/dev/shm/logs}
export PYTHONPATH=${CODE}:/torchtitan-native
export LORA_FT_RUN_NAME=${RUN}
export LORA_FT_TS=${LORA_FT_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
export PJRT_DEVICE=TPU
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/${RUN}.node${NODE_RANK}.log
cd /torchtitan-native
run() {
  echo "[launch_ft] $(date -u +%FT%TZ) host=$(hostname) node_rank=${NODE_RANK} config=${CONFIG} run=${RUN} master=${MASTER}:${PORT} extra=$*"
  torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC}" --rdzv_backend=static \
    --rdzv_endpoint="${MASTER}:${PORT}" --node_rank="${NODE_RANK}" \
    -m torchtitan.experiments.tpu.train --module=lora_tpu_recipes --config="${CONFIG}" "$@"
  rc=$?
  echo "[launch_ft] rc=${rc} $(date -u +%FT%TZ)"
  return ${rc}
}
if [ "${LORA_FT_TEE:-0}" = "1" ]; then
  run "$@" 2>&1 | tee "${LOG}"
  exit "${PIPESTATUS[0]}"
else
  run "$@" > "${LOG}" 2>&1
fi
