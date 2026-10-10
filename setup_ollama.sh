#!/usr/bin/env bash
set -e
VLM="${1:-qwen3-vl:4b}"; TXT="${2:-}"
command -v ollama >/dev/null || { curl -fsSL https://ollama.com/install.sh | sh; }
if ! curl -sf localhost:11434/api/tags >/dev/null; then
  OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve > ollama.log 2>&1 &
  for i in $(seq 1 30); do curl -sf localhost:11434/api/tags >/dev/null && break; sleep 1; done
fi
ollama pull "$VLM"
[ -n "$TXT" ] && ollama pull "$TXT"
echo "Pronto. Modelos instalados:"; ollama list