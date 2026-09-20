#!/usr/bin/env bash
# End-to-end pipeline. Run from the repository root.
set -e

echo "=== 0/5  Verifying setup ==="
python -m src.verify_setup

echo "=== 1/5  Preprocessing ==="
python -m src.preprocess

echo "=== 2/5  Signal branch ==="
python -m src.train --branch signal

echo "=== 3/5  Image branch ==="
python -m src.train --branch image

echo "=== 4/5  Ensemble ==="
python -m src.ensemble

echo "Done. See outputs/reports/"
