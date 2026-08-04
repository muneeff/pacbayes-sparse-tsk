# V6 Pre-outcome Amendment 001 — NESO ISO Date Representation

## Status

- Amendment ID: `V6-001`
- Scope: input date parser only
- Classification: pre-outcome implementation correction
- Raw data refresh: forbidden
- Model, split, structural prior, certificate, gate, and test policy: unchanged

## Trigger

The single authorized NESO download completed on
`2026-08-04T18:52:14.714702+00:00`.

The downloaded raw file is fixed as:

- Resource ID: `b2bde559-3455-4021-b179-dfe60c0337b0`
- Size: `1,869,771` bytes
- SHA-256:
  `8b8be946bbb084bc28fbf95a71fad0fceb5543b98e02755235130a42beaee4e0`

The first deterministic preparation attempt aborted before creating any
processed data or preparation audit. Pandas rejected the frozen
`dayfirst=True` parser because every observed `SETTLEMENT_DATE` value uses
the strict ISO representation `YYYY-MM-DD`.

This correction is a pre-outcome parser amendment.

## Evidence inspected before this amendment

Only transport/schema evidence was displayed:

- raw column names;
- raw row count: `17,520`;
- first and last distinct date strings;
- distinct date count: `365`;
- date-representation pattern counts;
- absence of processed and temporary output artifacts.

No `ND` values, aggregates, candidate certificates, validation metrics,
test outcomes, or deployment decisions were displayed or reviewed.

Observed date evidence:

- first date: `2025-01-01`;
- last date: `2025-12-31`;
- all `17,520` rows matched `^\d{4}-\d{2}-\d{2}$`;
- no slash-formatted or timestamp-bearing dates were observed.

## Frozen correction

Replace inference with an exact parser:

```python
pd.to_datetime(
    frame["SETTLEMENT_DATE"],
    format="%Y-%m-%d",
    exact=True,
    errors="raise",
)
```

The configuration is amended from `parse_dates_day_first: true` to:

```yaml
date_format: "%Y-%m-%d"
parse_dates_day_first: false
```

Any non-ISO date representation must abort.

## Audit chain

The original protocol snapshot and final pre-download execution snapshot
remain preserved and must not be replaced.

Parent final pre-download execution snapshot SHA-256:

`c0bb169c88073854fd3d936190238d110a2d5bce5931e2bebd6e7db42d338d34`

A new chained amendment execution snapshot must be created after this
amendment, its implementation, and its tests are committed:

`artifacts/prospective_execution_snapshot_v6_amendment_001.json`

Preparation and the prospective run must use that amendment snapshot while
retaining `artifacts/prospective_execution_snapshot_v6.json` as the parent
snapshot recorded by the original download audit. The downloaded raw bytes
must not be refreshed.

## Disclosure requirement

Any manuscript or release describing the NESO experiment must disclose
that the source date representation required this pre-outcome parser
amendment after download and before processed data, model fitting,
certificate evaluation, validation review, or test opening.
