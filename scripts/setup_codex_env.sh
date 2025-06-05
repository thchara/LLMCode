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
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "[*] Installing project dependencies (excluding matplotlib)…"
# Skip matplotlib; it fails to build in Codex container
python -m pip install --no-deps -r requirements.txt || true
python -m pip install numpy pandas scikit-learn hdbscan titlecase

echo "[*] Installing dev / test tooling…"
python -m pip install \
    pytest pytest-cov coverage[toml] \
    black flake8 isort mypy \
    pre-commit \
    python-dotenv \
    presidio-analyzer presidio-anonymizer

pre-commit install || true

echo "[*] Environment ready:"
python -V
pip -V
