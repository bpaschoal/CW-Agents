#!/usr/bin/env python3
"""Métricas da Tabela 5.2 a partir dos logs JSONL.

  python metrics.py --runs runs --gold-cw data/gold_cw.csv --gold-components data/gold_components.json
  python metrics.py --make-gold-template --tasks data/tasks.json     # gera data/gold_cw.csv para o especialista preencher

Formatos do gabarito:
  gold_cw.csv:          task_id,step_id,q,answer          (q = CW1..CW4; answer = yes|no)
  gold_components.json: {"<caminho da tela, como em tasks.json>": [{"text":"Edit address","bbox":[x,y,w,h]}, ...]}
"""
import argparse, csv, difflib, glob, json, re
from pathlib import Path
import numpy as np, pandas as pd

Q = ["CW1", "CW2", "CW3", "CW4"]

def toks(s): return set(re.findall(r"\w+", s.lower()))
def jaccard(a, b):
    a, b = toks(a), toks(b); return len(a & b) / len(a | b) if a | b else 0.0
def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy; u = aw * ah + bw * bh - inter
    return inter / u if u else 0.0

def matches(pred, gold, iou_thr=0.3, txt_thr=0.7):
    """Casa por bbox (IoU) OU por texto (rótulo contido na descrição, ou similaridade)."""
    pb, gb = pred.get("bbox"), gold.get("bbox")
    if pb and gb and len(pb) == 4 and len(gb) == 4 and iou(pb, gb) >= iou_thr: return True
    g = gold.get("text", "").lower().strip(); p = pred["description"].lower()
    if g and (g in p or difflib.SequenceMatcher(None, g, p).ratio() >= txt_thr): return True
    return False

def perception_pr(steps, gold_comp):
    rec, prec = [], []
    for s in steps:
        g = gold_comp.get(s["pre"])
        if not g: continue
        preds = s["perception_pre"]["components"]
        if not preds: rec.append(0.0); prec.append(0.0); continue
        rec.append(sum(any(matches(p, gi) for p in preds) for gi in g) / len(g))
        prec.append(sum(any(matches(p, gi) for gi in g) for p in preds) / len(preds))
    return (np.mean(rec) if rec else np.nan, np.mean(prec) if prec else np.nan)

def kappa(a, b):
    a, b = np.array(a), np.array(b); po = (a == b).mean()
    pe = sum(((a == v).mean()) * ((b == v).mean()) for v in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan

def load_runs(path):
    runs = []
    for f in sorted(glob.glob(str(Path(path) / "*.jsonl"))):
        runs += [json.loads(l) for l in open(f) if l.strip()]
    return runs

def make_template(tasks_path):
    tasks = json.loads(Path(tasks_path).read_text())["tasks"]
    out = Path(tasks_path).parent / "gold_cw.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task_id", "step_id", "q", "answer"])
        for t in tasks:
            for s in t["steps"]:
                for q in Q: w.writerow([t["id"], s["id"], q, ""])
    print("template em", out, "(preencher com yes/no)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--gold-cw"); ap.add_argument("--gold-components")
    ap.add_argument("--make-gold-template", action="store_true"); ap.add_argument("--tasks")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.make_gold_template: return make_template(a.tasks)
    gold_cw = None
    if a.gold_cw:
        g = pd.read_csv(a.gold_cw, dtype=str).dropna(subset=["answer"])
        gold_cw = {(r.task_id, r.step_id, r.q): r.answer.strip().lower()[0] for r in g.itertuples()}
    gold_comp = json.loads(Path(a.gold_components).read_text()) if a.gold_components else {}
    rows, pairs = [], []
    for r in load_runs(a.runs):
        steps = r["steps"]; n = len(steps)
        row = {"run_id": r["run_id"], "condition": r["condition"], "task_id": r["task_id"], "total_s": r["total_seconds"],
               "perception_s": sum(s["perception_pre"]["seconds"] + s["perception_post"]["seconds"] for s in steps
                                   if not s["perception_pre"]["cached"]) ,
               "neg_rate_any": sum(any(s["cw"][q]["answer"] == "no" for q in Q) for s in steps) / n}
        for q in Q: row[f"neg_{q}"] = sum(s["cw"][q]["answer"] == "no" for s in steps) / n
        hyp = [jaccard(s["cw"]["CW1"].get("hypothesized_action", ""), s["action"]) >= 0.4 for s in steps if "hypothesized_action" in s["cw"]["CW1"]]
        row["action_hyp_alignment"] = np.mean(hyp) if hyp else np.nan
        if gold_cw:
            m = [(s["cw"][q]["answer"][0] == gold_cw[(r["task_id"], s["step_id"], q)], q) for s in steps for q in Q
                 if (r["task_id"], s["step_id"], q) in gold_cw]
            row["expert_agreement"] = np.mean([x for x, _ in m]) if m else np.nan
            for q in Q:
                mq = [x for x, qq in m if qq == q]; row[f"agree_{q}"] = np.mean(mq) if mq else np.nan
            for s in steps:
                for q in Q:
                    k = (r["task_id"], s["step_id"], q)
                    if k in gold_cw: pairs.append((r["condition"], s["cw"][q]["answer"][0], gold_cw[k]))
        rec, prec = perception_pr(steps, gold_comp)
        row["perc_recall"], row["perc_precision"] = rec, prec
        rows.append(row)
    df = pd.DataFrame(rows); Path(a.out).mkdir(exist_ok=True)
    df.to_csv(Path(a.out) / "per_run.csv", index=False)
    cols = [c for c in df.columns if c not in ("run_id", "condition", "task_id") and df[c].notna().any()]
    g = df.groupby("condition")[cols]
    summ = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
    summ["n_runs"] = df.groupby("condition").size()
    if pairs:
        p = pd.DataFrame(pairs, columns=["condition", "agent", "expert"])
        summ["kappa_vs_expert"] = p.groupby("condition").apply(lambda d: round(kappa(d.agent, d.expert), 3))
    summ.to_csv(Path(a.out) / "summary.csv")
    (Path(a.out) / "summary.md").write_text(summ.T.to_markdown() if hasattr(summ, "to_markdown") else summ.T.to_string())
    print(summ.T.to_string())
    print("\n(média ± desvio padrão entre execuções; kappa agregado sobre todos os pares; arquivos em", a.out + ")")

if __name__ == "__main__": main()
