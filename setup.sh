#!/usr/bin/env bash
set -e
sudo apt-get update -y && sudo apt-get install -y tesseract-ocr
pip install --quiet anthropic pillow numpy pandas opencv-python-headless pytesseract tabulate
echo "Instalação concluída."
python3 - <<'PY'
import shutil, importlib
print("tesseract:", shutil.which("tesseract") or "FALTANDO")
for m in ["anthropic","PIL","numpy","pandas","cv2","pytesseract"]:
    importlib.import_module(m); print(m, "ok")
PY
