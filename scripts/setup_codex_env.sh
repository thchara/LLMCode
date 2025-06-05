#!/usr/bin/env bash
set -euo pipefail

echo "[*] Installing system packages needed for C-extensions…"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    zlib1g-dev \
    pkg-config \
    curl \
    git

echo "[*] Creating and activating virtualenv…"
python -m venv venv          # uses system Python 3.12
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install --upgrade "pip>=24.0"

echo "[*] Installing project dependencies…"
# Make sure matplotlib resolves to a wheel that supports cp312
python -m pip install --only-binary :all: "matplotlib==3.8.4"
python -m pip install -r requirements.txt

echo "[*] Installing dev / test tooling…"
python -m pip install \
    pytest pytest-cov coverage[toml] \
    black flake8 isort mypy \
    pre-commit \
    python-dotenv \
    presidio-analyzer presidio-anonymizer

pre-commit install

echo "[*] Environment ready:"
python -V
pip -V
python - <<'PY'
import matplotlib, sys, platform
print("matplotlib", matplotlib.__version__)
print("sys.executable", sys.executable)
print("platform", platform.platform())
PY
