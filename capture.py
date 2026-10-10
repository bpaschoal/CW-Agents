import argparse, json, sys, time
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

def host_ok(url, allowed):
    h = urlparse(url).hostname or ""
    return any(h == d or h.endswith("." + d) for d in allowed)

def changed_ratio(a, b):
    A, B = np.asarray(Image.open(a).convert("L"), dtype=int), np.asarray(Image.open(b).convert("L"), dtype=int)
    if A.shape != B.shape: return 1.0
    return float((np.abs(A - B) > 24).mean())

def do_action(page, st, timeout):
    g = st.get("gesture", "click")
    if g == "scroll":
        page.mouse.wheel(0, int(st.get("value", 600))); return
    loc = page.locator(st["selector"]).filter(visible=True).first   # ignora duplicatas ocultas (ex.: cabeçalho mobile)
    loc.wait_for(state="visible", timeout=timeout)
    if g == "click": loc.click(timeout=timeout)
    elif g == "hover": loc.hover(timeout=timeout)
    elif g == "type": loc.fill(st["value"], timeout=timeout)
    elif g == "press": loc.press(st["value"], timeout=timeout)
    elif g == "select": loc.select_option(st["value"], timeout=timeout)
    else: raise ValueError(f"gesto desconhecido: {g}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--out", default="data")
    ap.add_argument("--tasks-out"); ap.add_argument("--force", action="store_true", help="sobrescrever tasks.json existente")
    ap.add_argument("--timeout", type=int, default=8000, help="ms para o elemento aparecer")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text()); out = Path(a.out); (out / "screens").mkdir(parents=True, exist_ok=True)
    tasks_out = Path(a.tasks_out) if a.tasks_out else out / "tasks.json"
    if tasks_out.exists() and not a.force: sys.exit(f"{tasks_out} já existe (use --force ou --tasks-out outro.json)")
    allowed = spec.get("allowed_domains") or sys.exit("defina 'allowed_domains' no spec: só esses domínios serão visitados")
    vw, vh = spec.get("viewport", [1280, 800]); tasks, report = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for t in spec["tasks"]:
            if not host_ok(t["url"], allowed):
                report.append({"task": t["id"], "error": f"URL fora de allowed_domains: {t['url']}"}); continue
            ctx = browser.new_context(viewport={"width": vw, "height": vh}, locale=spec.get("locale", "en-US"))
            page = ctx.new_page(); rep = {"task": t["id"], "url": t["url"], "steps": []}
            try:
                resp = page.goto(t["url"], wait_until="domcontentloaded", timeout=30000)
                rep["http_status"] = resp.status if resp else None
                page.wait_for_timeout(1200)
                for sel in t.get("dismiss", []):   # banners de cookies etc. (melhor esforço)
                    try: page.locator(sel).first.click(timeout=2000); page.wait_for_timeout(400)
                    except Exception: pass
                steps = []
                for st in t["steps"]:
                    sid = str(st["id"]); pre = f"screens/{t['id']}_s{sid}_pre.png"; post = f"screens/{t['id']}_s{sid}_post.png"
                    srep = {"step": sid, "ok": False}
                    try:
                        page.screenshot(path=str(out / pre)); t0 = time.time()
                        do_action(page, st, a.timeout)
                        try: page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except PWTimeout: pass
                        page.wait_for_timeout(int(st.get("wait_ms", 800)))
                        page.screenshot(path=str(out / post))
                        if not host_ok(page.url, allowed): srep["warning"] = f"navegou para fora do domínio permitido: {page.url}"
                        srep.update(ok=True, seconds=round(time.time() - t0, 2), changed_ratio=round(changed_ratio(out / pre, out / post), 4),
                                    final_url=page.url)
                        if srep["changed_ratio"] == 0: srep["warning"] = "a tela não mudou após a ação (ação sem efeito visível ou espera curta)"
                        steps.append({"id": sid, "action": st["action"], "target": st.get("target", ""), "gesture": st.get("gesture", "click"),
                                      "pre": pre, "post": post})
                    except Exception as e:
                        srep["error"] = f"{type(e).__name__}: {str(e)[:200]}"; rep["steps"].append(srep); break   # interrompe a tarefa
                    rep["steps"].append(srep)
                if steps:
                    tasks.append({"id": t["id"], "app": t.get("app", urlparse(t["url"]).hostname), "goal": t["goal"],
                                  **({"persona": t["persona"]} if t.get("persona") else {}), "steps": steps})
            except Exception as e:
                rep["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            finally:
                ctx.close(); report.append(rep)
        browser.close()
    tasks_out.write_text(json.dumps({"tasks": tasks}, indent=1, ensure_ascii=False))
    (out / "capture_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for r in report:
        print(f"[{r['task']}]", r.get("error") or f"HTTP {r.get('http_status')}")
        for s in r.get("steps", []):
            print(f"   passo {s['step']}: {'ok' if s['ok'] else 'FALHOU'}", f"(mudou {s['changed_ratio']*100:.1f}% dos pixels)" if s["ok"] else s.get("error", ""), "| " + s["warning"] if s.get("warning") else "")
    print("tarefas capturadas:", len(tasks), "→", tasks_out)

if __name__ == "__main__": main()
