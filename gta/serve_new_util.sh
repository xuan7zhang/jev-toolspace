#!/usr/bin/env bash
# OpenAI-compatible server from the new vLLM, for models lmdeploy and the old
# venv cannot load. Run ON the node (inside srun). Env: MODEL (dir), NAME
# (served name), PORT, GPUS ("0" or "0,1"), TP (1|2), MAXLEN (default 32768),
# NOTHINK=1 to strip reasoning via the chat template default.
set -uo pipefail
source /datasets/omni_pretraining/gta2/scripts_extra/new_model_env.sh
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
# never ask for more context than the model declares (phi-4: 16384)
MAXLEN=${MAXLEN:-32768}
CAP=$(python3 -c "import json,sys; c=json.load(open('$MODEL/config.json')); c=c.get('text_config',c); print(int(c.get('max_position_embeddings',1e9)))" 2>/dev/null || echo 1000000000)
[ "$CAP" -lt "$MAXLEN" ] && MAXLEN=$CAP
EXTRA=""
[ "${TP:-1}" -gt 1 ] && EXTRA="--tensor-parallel-size ${TP}"
[ -n "${NOTHINK:-}" ] && EXTRA="$EXTRA --default-chat-template-kwargs {\"enable_thinking\":false}"
exec $NEW_PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$NAME" "$MODEL" --port "${PORT:-8800}" \
  --gpu-memory-utilization ${UTIL:-0.55} --max-model-len "$MAXLEN" --enforce-eager \
  --dtype bfloat16 --trust-remote-code $EXTRA --no-enable-log-requests
