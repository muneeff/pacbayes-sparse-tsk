# Datasets

## Synthetic V3

The exact synthetic definitions and parameters are fixed in `configs/v3/synthetic_v3.yaml`. Development seeds and confirmatory seeds are disjoint.

## M4, NN5, and Tourism

The final real-benchmark protocol will preserve official validation/test horizons where available. Confirmatory V3 has not been run.

## Tetouan power-consumption case

- Official source: UCI Machine Learning Repository, dataset 849.
- DOI: `10.24432/C5B034`.
- License: CC BY 4.0.
- Raw observations: 52,416 records at ten-minute frequency.
- Case-study transformation: nonoverlapping two-hour means, yielding 4,368 rows for each of the three zones.
- Tracked processed file: `data/processed/tetouan_two_hour.csv`.
- Raw file: excluded from Git by default.
- Exact hashes and the provenance limitation of the execution environment are recorded in `docs/TETOUAN_DATASET_MANIFEST.json`.

The UCI metadata does not specify a physical unit for the three target columns. Results therefore use “source units” and do not relabel them as kW.
