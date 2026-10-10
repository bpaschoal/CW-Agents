#!/usr/bin/env bash
# Roda as três condições e calcula as métricas.
#   MOCK=1 bash run_all.sh                          # sem modelo (valida o fluxo; números sem significado)
#   PROVIDER=ollama bash run_all.sh                 # Ollama local (padrão: qwen3-vl:4b); gratuito, sem chave
#   PROVIDER=ollama MODEL=qwen3-vl:8b CW_MODEL=qwen3:8b bash run_all.sh
#   REPEATS=1 bash run_all.sh                       # teste pequeno (REPEATS padrão = 3)
#   bash run_all.sh                                 # API da Anthropic (precisa de ANTHROPIC_API_KEY)
set -e
REPEATS="${REPEATS:-3}"; TASKS="${TASKS:-data/tasks.json}"; PROVIDER="${PROVIDER:-anthropic}"; EXTRA=""
if [ "${MOCK:-0}" = "1" ]; then
  EXTRA="--mock"
elif [ "$PROVIDER" = "ollama" ]; then
  HOST="${OLLAMA_HOST:-localhost:11434}"; case "$HOST" in http*) ;; *) HOST="http://$HOST";; esac
  TAGS=$(curl -sf "$HOST/api/tags") || { echo "ERRO: Ollama não responde em $HOST. Inicie com: nohup ollama serve > ollama.log 2>&1 &"; exit 1; }
  MODEL="${MODEL:-qwen3-vl:4b}"
  echo "$TAGS" | grep -q "\"$MODEL\"" || { echo "ERRO: modelo $MODEL não baixado. Rode: ollama pull $MODEL"; exit 1; }
  EXTRA="--provider ollama --model $MODEL"
  if [ -n "$CW_MODEL" ]; then
    echo "$TAGS" | grep -q "\"$CW_MODEL\"" || { echo "ERRO: modelo $CW_MODEL não baixado. Rode: ollama pull $CW_MODEL"; exit 1; }
    EXTRA="$EXTRA --cw-model $CW_MODEL"
  fi
elif [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "ERRO: defina ANTHROPIC_API_KEY, ou use PROVIDER=ollama, ou MOCK=1."; exit 1
fi
if [ -d runs ]; then echo "ERRO: a pasta runs/ já existe (os logs seriam duplicados). Apague com: rm -rf runs results"; exit 1; fi
python3 cw_agents.py --tasks "$TASKS" --repeats "$REPEATS" --out runs $EXTRA
ARGS=""
[ -f data/gold_cw.csv ] && ARGS="$ARGS --gold-cw data/gold_cw.csv"
[ -f data/gold_components.json ] && ARGS="$ARGS --gold-components data/gold_components.json"
python3 metrics.py --runs runs $ARGS
