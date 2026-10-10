# CW-Agents — piloto rápido

Pipeline da fase de Análise do Percurso Cognitivo: **Orchestrator → Perception (3 condições) → CW1–CW4 independentes → Report**, com logs JSONL por execução e script de métricas.

## 1. Instalar
```bash
pip install anthropic pillow numpy pandas opencv-python pytesseract tabulate
sudo apt install tesseract-ocr          # (ou o equivalente no seu sistema)
export ANTHROPIC_API_KEY=...            # modelo padrão: claude-sonnet-5-5 (troque com --model ou CW_MODEL)
# opcional: pip install ultralytics  e passe --yolo-weights <pesos de detector de UI>
```

## 2. Testar sem API (resultados sem significado, só valida o fluxo)
```bash
python make_demo_data.py
python cw_agents.py --tasks data/tasks.json --mock --repeats 2 --out runs
python metrics.py --runs runs --gold-cw data/gold_cw.csv --gold-components data/gold_components.json
```

## 3. Rodar com seus dados
1. Monte `data/tasks.json` (formato em `make_demo_data.py`): por tarefa, persona e passos; cada passo tem **uma ação atômica** e as telas `pre`/`post` (caminhos relativos à pasta do JSON).
2. Gabaritos: `python metrics.py --make-gold-template --tasks data/tasks.json` gera `gold_cw.csv` para o especialista preencher (yes/no). Para a Percepção, crie `gold_components.json` com os componentes visíveis (texto + bbox) em ~10 telas.
3. Rodar as três condições, 3 repetições:
```bash
python cw_agents.py --tasks data/tasks.json --repeats 3 --out runs
python metrics.py --runs runs --gold-cw data/gold_cw.csv --gold-components data/gold_components.json
```
Saídas: `runs/<COND>.jsonl` (logs completos, Seção 3.3), `results/per_run.csv`, `results/summary.csv|md` (média ± desvio padrão por condição, kappa contra o especialista).

## Decisões que afetam a interpretação (cite no artigo)
- **CW1–CW4 recebem só a descrição da Percepção por padrão**, para isolar a contribuição de cada condição. Com `--cw-sees-image` eles também veem as telas (como na limitação da Seção 5.5 da monografia); o ganho marginal da Percepção tende a diminuir.
- **Cache da Percepção** por (condição, imagem): as repetições só variam nos agentes CW. Use `--no-cache` para medir também a variação da Percepção (mais caro).
- **Detector:** sem `--yolo-weights`, a detecção é um baseline por contornos (OpenCV), não um detector treinado. Declare isso como limitação.
- **Casamento de componentes (recall/precisão):** IoU ≥ 0,3 *ou* rótulo do gabarito contido na descrição / similaridade textual ≥ 0,7. Bbox de VLM é aproximada; por isso o critério textual.
- **action_hyp_alignment** (monografia): CW1 devolve também `hypothesized_action`; o alinhamento é aproximado por sobreposição de tokens (Jaccard ≥ 0,4). No artigo, o CW1 foca na formação de objetivo; se seguir o artigo, ignore essa coluna.
- Temperatura 0; o modelo e os prompts ficam registrados nos logs para reprodutibilidade.
