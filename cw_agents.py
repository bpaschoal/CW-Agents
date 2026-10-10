#!/usr/bin/env python3
"""CW-Agents: pipeline mínimo da fase de Análise do Percurso Cognitivo.

Orchestrator -> Perception (DET_OCR | VLM | COMBINED) -> CW1..CW4 (independentes) -> Report.
Cada execução gera um JSONL de logs (Seção 3.3 do artigo).

Uso:
  python cw_agents.py --tasks data/tasks.json --conditions DET_OCR VLM COMBINED --repeats 3 --out runs
  python cw_agents.py --mock ...      # sem API: valida o pipeline (resultados sem significado)
"""
import argparse, base64, hashlib, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONDITIONS = ["DET_OCR", "VLM", "COMBINED"]
DEFAULT_MODEL = os.environ.get("CW_MODEL", "claude-sonnet-5-5")

QUESTIONS = {
    "CW1": "Will the user try to achieve the right effect? (goal formation: given the task and the current screen, would the persona form the intention that this Action serves?)",
    "CW2": "Will the user notice that the correct action is available? (visibility and salience of the cues for the predefined Action; the action may be implicit, e.g. a keyboard shortcut)",
    "CW3": "Will the user associate the correct action with the effect they are trying to achieve? (meaning conveyed by labels, icons, layout and other cues)",
    "CW4": "If the correct action is performed, will the user perceive that progress is being made toward the task goal? (compare the PRE-action and POST-action screens and their descriptions)",
}

# ----------------------------------------------------------------------------- LLM
class LLM:
    def __init__(self, model, mock=False, temperature=0.0):
        self.model, self.mock, self.temperature = model, mock, temperature
        self.client = None
        if not mock:
            import anthropic
            self.client = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY

    def ask(self, system, text, images=(), max_tokens=1500):
        if self.mock:
            return self._mock(text)
        content = []
        for p in images:
            data = base64.standard_b64encode(Path(p).read_bytes()).decode()
            mt = "image/png" if str(p).lower().endswith("png") else "image/jpeg"
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        content.append({"type": "text", "text": text})
        for attempt in range(4):
            try:
                r = self.client.messages.create(model=self.model, max_tokens=max_tokens,
                        temperature=self.temperature, system=system,
                        messages=[{"role": "user", "content": content}])
                return "".join(b.text for b in r.content if b.type == "text")
            except Exception as e:
                if attempt == 3: raise
                time.sleep(2 ** attempt)

    @staticmethod
    def _mock(text):
        rnd = random.Random(hashlib.md5(text.encode()).hexdigest())
        if "COMPONENTS_JSON" in text:
            comps = [{"description": f"mock component {i}", "bbox": [10 + 80 * i, 10, 70, 30]} for i in range(rnd.randint(2, 5))]
            return json.dumps({"components": comps})
        return json.dumps({"answer": "yes" if rnd.random() > 0.3 else "no",
                           "justification": "mock", "hypothesized_action": "mock action"})

def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: raise ValueError("sem JSON na resposta: " + txt[:200])
    return json.loads(m.group(0))

# ------------------------------------------------------------------------ Perception
def ocr_components(img_path):
    """OCR por linha (pytesseract), na imagem original e invertida (texto claro sobre fundo escuro, ex.: botões)."""
    import pytesseract
    from PIL import Image, ImageOps
    base = Image.open(img_path).convert("RGB"); out, seen = [], set()
    for im in (base, ImageOps.invert(base)):
        d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
        lines = {}
        for i, t in enumerate(d["text"]):
            if t.strip() and float(d["conf"][i]) > 30:
                k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
                lines.setdefault(k, []).append((t, d["left"][i], d["top"][i], d["width"][i], d["height"][i]))
        for ws in lines.values():
            txt = " ".join(w[0] for w in ws)
            if txt.lower() in seen: continue
            seen.add(txt.lower())
            x0 = min(w[1] for w in ws); y0 = min(w[2] for w in ws)
            x1 = max(w[1] + w[3] for w in ws); y1 = max(w[2] + w[4] for w in ws)
            out.append({"description": "text: " + txt, "bbox": [x0, y0, x1 - x0, y1 - y0], "source": "ocr"})
    return out

def detect_components(img_path, yolo_weights=None):
    """Detecção de elementos. Com --yolo-weights usa Ultralytics; senão, contornos OpenCV (baseline simples)."""
    if yolo_weights:
        from ultralytics import YOLO
        res = YOLO(yolo_weights)(img_path, verbose=False)[0]
        return [{"description": f"{res.names[int(b.cls)]} (conf {float(b.conf):.2f})",
                 "bbox": [int(v) for v in (b.xyxy[0][0], b.xyxy[0][1], b.xyxy[0][2] - b.xyxy[0][0], b.xyxy[0][3] - b.xyxy[0][1])],
                 "source": "det"} for b in res.boxes]
    import cv2
    img = cv2.imread(img_path); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(g, 50, 150), None, iterations=2)
    cs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = g.shape; out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if 12 <= w < W * 0.9 and 12 <= h < H * 0.5:
            out.append({"description": "ui element (contour)", "bbox": [x, y, w, h], "source": "det"})
    return out

VLM_SYS = ("You describe what is visually perceivable on a rendered user-interface screenshot, as seen by a human user. "
           "Never infer hidden markup. Reply with JSON only.")

def vlm_components(llm, img_path, persona, hint=None):
    extra = ""
    if hint:
        extra = ("\nCandidate elements from an object detector and OCR (bbox=[x,y,w,h] in pixels):\n"
                 + json.dumps(hint)[:6000]
                 + "\nMerge them: keep real elements, drop false detections, add anything visible that is missing, "
                   "and describe the FUNCTION of each component (especially icons without text).")
    prompt = ("COMPONENTS_JSON\nList every visible interface component (buttons, links, icons, fields, menus, labels, feedback messages). "
              "For each give a natural-language description (include visible text and apparent function) and bbox=[x,y,w,h] in pixels (best effort).\n"
              f"User persona (affects what is understandable): {persona}\n"
              'Format: {"components":[{"description":"...","bbox":[x,y,w,h]}]}' + extra)
    comps = parse_json(llm.ask(VLM_SYS, prompt, [img_path]))["components"]
    for c in comps: c.setdefault("source", "vlm")
    return comps

def perceive(llm, cond, img_path, persona, yolo_weights, cache_dir, use_cache=True):
    key = hashlib.md5(cond.encode() + Path(img_path).read_bytes()).hexdigest()
    cf = Path(cache_dir) / f"{cond}_{key}.json"
    if use_cache and cf.exists(): 
        r = json.loads(cf.read_text()); r["cached"] = True; return r
    t0 = time.time()
    if cond == "DET_OCR":
        comps = ocr_components(img_path) + detect_components(img_path, yolo_weights)
    elif cond == "VLM":
        comps = vlm_components(llm, img_path, persona)
    elif cond == "COMBINED":
        hint = ocr_components(img_path) + detect_components(img_path, yolo_weights)
        comps = vlm_components(llm, img_path, persona, hint)
    else: raise ValueError(cond)
    r = {"components": comps, "seconds": round(time.time() - t0, 3), "cached": False}
    cf.parent.mkdir(parents=True, exist_ok=True); cf.write_text(json.dumps(r))
    return r

def render(comps):
    return "\n".join(f"- [{i}] {c['description']} (bbox {c.get('bbox')})" for i, c in enumerate(comps))

# ------------------------------------------------------------------------- CW agents
CW_SYS = ("You are a usability inspector applying the Cognitive Walkthrough. You reason ONLY as the given persona would, "
          "from what is perceivable on screen; you answer exactly ONE canonical question. The Action under evaluation "
          "is fixed in advance: you never choose or execute actions. Reply with JSON only.")

def cw_agent(llm, qid, task, step, persona, pre_desc, post_desc, images):
    p = (f"Persona: {persona}\nTask: {task['goal']}\nStep {step['id']} - predefined Action: {step['action']} "
         f"(target: {step.get('target','')}, gesture: {step.get('gesture','')})\n\n"
         f"PRE-action screen, perceivable components:\n{pre_desc}\n")
    if qid == "CW4":
        p += f"\nPOST-action screen, perceivable components:\n{post_desc}\n"
    p += f"\nQuestion ({qid}): {QUESTIONS[qid]}\n"
    fmt = '{"answer":"yes"|"no","justification":"2-3 sentences citing the on-screen evidence"'
    if qid == "CW1":
        fmt += ',"hypothesized_action":"the single action this persona would try at this point, in tutorial-level wording"'
    p += "Reply as JSON: " + fmt + "}"
    t0 = time.time()
    out = parse_json(llm.ask(CW_SYS, p, images))
    out["answer"] = "yes" if str(out.get("answer", "")).lower().startswith("y") else "no"
    out["seconds"] = round(time.time() - t0, 3)
    return out

# ----------------------------------------------------------------------- Orchestrator
def run_task(llm, task, cond, run_id, args):
    persona = task.get("persona") or "A first-time user with no training, who does not read documentation and explores the interface using visible cues."
    t_start = time.time(); steps_out = []
    for st in task["steps"]:
        base = Path(args.tasks).parent
        pre, post = str(base / st["pre"]), str(base / st["post"])
        t0 = time.time()
        per_pre = perceive(llm, cond, pre, persona, args.yolo_weights, args.cache, not args.no_cache)
        per_post = perceive(llm, cond, post, persona, args.yolo_weights, args.cache, not args.no_cache)
        t_perc = time.time() - t0
        pre_d, post_d = render(per_pre["components"]), render(per_post["components"])
        imgs = {"CW1": [pre], "CW2": [pre], "CW3": [pre], "CW4": [pre, post]} if args.cw_sees_image else {k: [] for k in QUESTIONS}
        with ThreadPoolExecutor(4) as ex:   # CW1..CW4 independentes: não veem as saídas uns dos outros
            futs = {q: ex.submit(cw_agent, llm, q, task, st, persona, pre_d, post_d, imgs[q]) for q in QUESTIONS}
            cw = {q: f.result() for q, f in futs.items()}
        steps_out.append({"step_id": st["id"], "action": st["action"], "pre": st["pre"], "post": st["post"],
                          "perception_pre": per_pre, "perception_post": per_post,
                          "perception_seconds_wall": round(t_perc, 3), "cw": cw})
    return {"run_id": run_id, "task_id": task["id"], "app": task.get("app"), "condition": cond,
            "model": llm.model if not llm.mock else "MOCK", "persona": persona, "cw_sees_image": args.cw_sees_image,
            "steps": steps_out, "total_seconds": round(time.time() - t_start, 3), "report": report(steps_out)}

def report(steps):
    neg = {q: 0 for q in QUESTIONS}; flagged = 0
    for s in steps:
        n = [q for q in QUESTIONS if s["cw"][q]["answer"] == "no"]
        flagged += bool(n)
        for q in n: neg[q] += 1
    return {"steps": len(steps), "steps_flagged": flagged, "negatives_by_question": neg,
            "descriptive": [{"step": s["step_id"], "action": s["action"],
                             **{q: {"answer": s["cw"][q]["answer"], "why": s["cw"][q]["justification"]} for q in QUESTIONS}} for s in steps]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--out", default="runs")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--yolo-weights")
    ap.add_argument("--cw-sees-image", action="store_true", help="CW1-4 também recebem a(s) tela(s) (ganho marginal da Percepção tende a diminuir; ver limitação da Seção 5.5)")
    ap.add_argument("--no-cache", action="store_true", help="refaz a Percepção a cada repetição (mede variação estocástica dela)")
    ap.add_argument("--cache", default=".perception_cache"); ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks).read_text())["tasks"]
    llm = LLM(args.model, args.mock); Path(args.out).mkdir(exist_ok=True, parents=True)
    jobs = [(t, c, r) for c in args.conditions for t in tasks for r in range(args.repeats)]
    lock = threading.Lock()
    def job(j):
        t, c, r = j; rid = f"{c}_{t['id']}_r{r}"
        res = run_task(llm, t, c, rid, args)
        with lock, open(Path(args.out) / f"{c}.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
        print(f"ok {rid} ({res['total_seconds']}s)", flush=True)
    with ThreadPoolExecutor(args.workers) as ex: list(ex.map(job, jobs))
    print("logs em", args.out)

if __name__ == "__main__": main()
