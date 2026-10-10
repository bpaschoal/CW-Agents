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


## Rodar com Ollama (gratuito, local)
```bash
bash setup_ollama.sh qwen3-vl:4b          # instala o Ollama, sobe o servidor e baixa o modelo (≈3,3 GB)
PROVIDER=ollama REPEATS=1 bash run_all.sh # teste pequeno; depois apague com: rm -rf runs results
PROVIDER=ollama bash run_all.sh           # experimento completo
```
- `MODEL=` troca o modelo de Percepção (VLM); `CW_MODEL=` usa outro modelo (pode ser só de texto) para CW1–CW4.
- Sem GPU a velocidade cai bastante: faça o teste com 1 repetição e meça o tempo antes de planejar o experimento.
- O cache da Percepção inclui o nome do modelo; trocar de modelo não reaproveita resultados antigos.
- Modelos locais pequenos erram o formato JSON às vezes; o código tenta de novo até 3 vezes e descarta componentes inválidos.

## Capturar telas de sites automaticamente (fase de Preparação)
O CW-Agents **não navega**: ele analisa telas prontas (princípio de projeto do artigo). Para ganhar tempo, `capture.py` automatiza a *Preparação*:
abre as URLs que você listou, executa a lista de ações pré-definida e salva as telas antes/depois de cada passo, além do `tasks.json`.
```bash
bash setup_capture.sh                                  # uma vez
cp sites.example.json data/sites.json                  # edite: allowed_domains, url, goal, passos
python3 capture.py --spec data/sites.json              # gera data/screens/*.png e data/tasks.json
cat data/capture_report.json                           # validações rápidas (HTTP, passo ok/falhou, % de pixels que mudou)
```
- Cada passo tem `action` (texto da ação, que os agentes veem), `gesture` (click, type, press, hover, select, scroll) e `selector` (Playwright: `text=Sign in`, `#id`, css, `role=button[name="Save"]`). **O seletor só é usado para executar a ação; os agentes nunca o veem.**
- `allowed_domains` é obrigatório: o script recusa URLs fora da lista e avisa se uma ação levar a outro domínio.
- Passo que falha interrompe só aquela tarefa (tarefas sem nenhum passo ok não entram no `tasks.json`).
- Aviso "a tela não mudou": a ação não teve efeito visível ou o `wait_ms` foi curto; confira antes de rodar o experimento.
- Respeite os termos de uso e o robots.txt dos sites; evite login, dados pessoais e fluxos de pagamento reais.
- Teste local: abra o site de demonstração com `python3 -m http.server 8765 --bind 127.0.0.1` dentro de `demo_site/` e rode `python3 capture.py --spec sites.example.json --out /tmp/cap`.

### Lojas Americanas (`sites.americanas.json`)
```bash
python3 capture.py --spec sites.americanas.json --out data
```
4 tarefas / 10 passos: busca de produto, informar CEP, menu de departamentos, abrir produto e adicionar ao carrinho. **Os seletores não foram testados no site real**
(foram escritos a partir do texto da página: "busque aqui seu produto", "informe seu CEP", "todos os departamentos"). Rode uma vez e leia `data/capture_report.json`:
passos com `FALHOU` precisam de um seletor melhor (botão direito → Inspecionar no navegador) e passos com "a tela não mudou" precisam de `wait_ms` maior.
Se o site bloquear o navegador automatizado (HTTP 403, página de verificação), não contorne: capture as telas manualmente e monte o `tasks.json` à mão.
Confira os termos de uso e o robots.txt do site antes de capturar. Para na tarefa do carrinho: o script não faz login, não preenche pagamento e não finaliza compras.

## Decisões que afetam a interpretação (cite no artigo)
- **CW1–CW4 recebem só a descrição da Percepção por padrão**, para isolar a contribuição de cada condição. Com `--cw-sees-image` eles também veem as telas (como na limitação da Seção 5.5 da monografia); o ganho marginal da Percepção tende a diminuir.
- **Cache da Percepção** por (condição, imagem): as repetições só variam nos agentes CW. Use `--no-cache` para medir também a variação da Percepção (mais caro).
- **Detector:** sem `--yolo-weights`, a detecção é um baseline por contornos (OpenCV), não um detector treinado. Declare isso como limitação.
- **Casamento de componentes (recall/precisão):** IoU ≥ 0,3 *ou* rótulo do gabarito contido na descrição / similaridade textual ≥ 0,7. Bbox de VLM é aproximada; por isso o critério textual.
- **action_hyp_alignment** (monografia): CW1 devolve também `hypothesized_action`; o alinhamento é aproximado por sobreposição de tokens (Jaccard ≥ 0,4). No artigo, o CW1 foca na formação de objetivo; se seguir o artigo, ignore essa coluna.
- Temperatura 0; o modelo e os prompts ficam registrados nos logs para reprodutibilidade.
