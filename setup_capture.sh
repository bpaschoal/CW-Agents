#!/usr/bin/env bash
set -e
pip install --quiet playwright
python3 -m playwright install --with-deps chromium
echo "Captura pronta. Veja a seção 'Capturar telas' no README."
