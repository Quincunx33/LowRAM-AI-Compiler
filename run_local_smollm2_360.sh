#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${LOWRAM_MODEL:-${ROOT_DIR}/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
export LOWRAM_MODEL="$MODEL_PATH"
export LOWRAM_MAX_CONTEXT="${LOWRAM_MAX_CONTEXT:-512}"
export LOWRAM_MAX_RAM_MB="${LOWRAM_MAX_RAM_MB:-1024}"
export LOWRAM_HOST="${LOWRAM_HOST:-127.0.0.1}"
export LOWRAM_PORT="${LOWRAM_PORT:-8000}"

if [[ ! -f "$LOWRAM_MODEL" ]]; then
  echo "Model not found: $LOWRAM_MODEL" >&2
  echo "Download Llama-3.2-1B-Instruct-Q4_K_M.gguf and place it in models/, or set LOWRAM_MODEL." >&2
  exit 1
fi

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m lowram_ai.api
