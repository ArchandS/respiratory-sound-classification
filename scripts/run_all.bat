@echo off
REM End-to-end pipeline. Run from the repository root.
echo === 0/5  Verifying setup ===
python -m src.verify_setup || exit /b 1
echo === 1/5  Preprocessing ===
python -m src.preprocess || exit /b 1
echo === 2/5  Signal branch ===
python -m src.train --branch signal || exit /b 1
echo === 3/5  Image branch ===
python -m src.train --branch image || exit /b 1
echo === 4/5  Ensemble ===
python -m src.ensemble || exit /b 1
echo Done. See outputs\reports\
