# Reproducibility

1. Create a Python 3.10–3.13 environment.
2. Install the project without reinstalling already satisfied packages when appropriate:
   `pip install -e . --no-deps`
3. Run `pytest tests/v3`.
4. Run `python scripts/validate_v3.py`.
5. Generate development synthetic data if standalone files are needed:
   `python scripts/generate_synthetic_v3.py --seeds 3000 3001 3002 3003 3004`.
6. Run or resume full development:
   `python scripts/run_development_v3.py --workers 6`.
7. Rebuild aggregate tables only:
   `python scripts/summarize_development_v3.py`.

Per-series checkpoints are stored under `results/development/full_v3/series/`. Existing completed series are skipped by default.

Confirmatory V3 is not authorized and has not been run.
