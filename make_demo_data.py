#!/usr/bin/env python3
"""Gera dados de demonstração (telas sintéticas + tasks.json + gabaritos) para testar o pipeline sem API."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
D = Path("data"); (D / "screens").mkdir(parents=True, exist_ok=True)
try: F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
except Exception: F = ImageFont.load_default()

def screen(name, items, banner=None):
    im = Image.new("RGB", (900, 500), "#f4f4f4"); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 900, 50], fill="#223355"); d.text((20, 12), "Checkout", fill="white", font=F)
    gold = []
    for kind, label, (x, y, w, h) in items:
        if kind == "button":
            d.rectangle([x, y, x + w, y + h], fill="#2a7fff"); d.text((x + 12, y + 10), label, fill="white", font=F)
        elif kind == "icon":
            d.ellipse([x, y, x + w, y + h], outline="#444", width=3); d.line([x + 8, y + h - 8, x + w - 8, y + 8], fill="#444", width=3)
        elif kind == "text":
            d.text((x, y), label, fill="black", font=F)
        elif kind == "field":
            d.rectangle([x, y, x + w, y + h], fill="white", outline="#888"); d.text((x + 8, y + 8), label, fill="#333", font=F)
        gold.append({"text": label, "bbox": [x, y, w, h]})
    if banner: d.rectangle([0, 450, 900, 500], fill="#2e9e5b"); d.text((20, 462), banner, fill="white", font=F)
    im.save(D / "screens" / f"{name}.png"); return gold

G = {}
def reg(n, items, banner=None): G[f"screens/{n}.png"] = screen(n, items, banner)
addr = ("text", "Shipping address: Rua A, 100", (40, 90, 400, 30))
reg("s1_pre", [addr, ("button", "Edit address", (500, 85, 170, 40)), ("button", "Pay now", (40, 400, 140, 40))])
reg("s1_post", [("field", "Rua A, 100", (40, 90, 420, 40)), ("button", "Save", (480, 90, 100, 40))])
reg("s2_pre", [("field", "Rua A, 100", (40, 90, 420, 40)), ("button", "Save", (480, 90, 100, 40))])
reg("s2_post", [addr, ("button", "Pay now", (40, 400, 140, 40))], "Address updated")
reg("t2s1_pre", [addr, ("icon", "edit icon", (500, 85, 36, 36)), ("button", "Pay now", (40, 400, 140, 40))])
reg("t2s1_post", [("field", "Rua A, 100", (40, 90, 420, 40)), ("button", "Save", (480, 90, 100, 40))])
persona = "A first-time visitor of the shop, with general familiarity with online shopping but no prior exposure to this site."
tasks = {"tasks": [
  {"id": "T1_labelled", "app": "demo_shop", "goal": "Change the delivery address before completing checkout", "persona": persona, "steps": [
    {"id": "1", "action": "Click the 'Edit address' button next to the shipping address", "target": "Edit address button", "gesture": "click", "pre": "screens/s1_pre.png", "post": "screens/s1_post.png"},
    {"id": "2", "action": "Click the 'Save' button next to the address field", "target": "Save button", "gesture": "click", "pre": "screens/s2_pre.png", "post": "screens/s2_post.png"}]},
  {"id": "T2_icon", "app": "demo_shop", "goal": "Change the delivery address before completing checkout", "persona": persona, "steps": [
    {"id": "1", "action": "Click the pencil icon next to the shipping address", "target": "pencil icon", "gesture": "click", "pre": "screens/t2s1_pre.png", "post": "screens/t2s1_post.png"}]}]}
(D / "tasks.json").write_text(json.dumps(tasks, indent=1)); (D / "gold_components.json").write_text(json.dumps(G, indent=1))
rows = ["task_id,step_id,q,answer"]
gold_ans = {("T1_labelled", "1"): "yyyy", ("T1_labelled", "2"): "yyyy", ("T2_icon", "1"): "ynny"}
for (t, s), a in gold_ans.items():
    for q, v in zip(["CW1", "CW2", "CW3", "CW4"], a): rows.append(f"{t},{s},{q},{'yes' if v == 'y' else 'no'}")
(D / "gold_cw.csv").write_text("\n".join(rows) + "\n")
print("dados de demonstração em ./data (gold_cw.csv aqui é ilustrativo)")
