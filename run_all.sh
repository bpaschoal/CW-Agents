#!/usr/bin/env bash
# Roda as três condições e calcula as métricas.
#   bash run_all.sh                 # experimento com a API (REPEATS=3)
#   REPEATS=1 bash run_all.sh       # teste pequeno de custo/funcionamento
#   MOCK=1 bash run_all.sh          # sem API (valida o fluxo; números sem significado)
set -e
REPEATS="${REPEATS:-3}"
TASKS="${TASKS:-data/tasks.json}"
EXTRA=""
if [ "${MOCK:-0}" = "1" ]; then EXTRA="--mock"
elif [ -z "$ANTHROPIC_API_KEY" ]; then echo "ERRO: defina ANTHROPIC_API_KEY (ou use MOCK=1)."; exit 1; fi
if [ -d runs ]; then echo "ERRO: a pasta runs/ já existe (os logs seriam duplicados). Apague com: rm -rf runs results"; exit 1; fi
python3 cw_agents.py --tasks "$TASKS" --repeats "$REPEATS" --out runs $EXTRA
ARGS=""
[ -f data/gold_cw.csv ] && ARGS="$ARGS --gold-cw data/gold_cw.csv"
[ -f data/gold_components.json ] && ARGS="$ARGS --gold-components data/gold_components.json"
python3 metrics.py --runs runs $ARGS
