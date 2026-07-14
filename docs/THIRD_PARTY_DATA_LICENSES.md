# Third-party data

## Power Consumption of Tetouan City

- Creators: Abdulwahed Salam and Abdelaaziz El Hibaoui.
- Repository: UCI Machine Learning Repository, dataset 849.
- DOI: `10.24432/C5B034`.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Use in this project: deterministic hourly means of the three power-consumption zones.

The processed file retains the source attribution and may be shared under the
terms of CC BY 4.0. No claim is made that the units are kW because the UCI
metadata identifies the variables as power consumption but does not state a
unit.


## Hourly Energy Consumption / PJM regional load

- Primary collection: Kaggle `robikscube/hourly-energy-consumption`.
- Underlying source: PJM regional hourly electricity-load series.
- License shown by the primary collection: CC0 1.0 Universal.
- Transport mirror used by the frozen experiment: `ping543f/ren-energy`.
- Included series: AEP, COMED, DAYTON, and PJME.
- Use in this project: deterministic duplicate collapse followed by daily-mean aggregation over the frozen common interval.

The mirror is used only to transport the exact bytes recorded by SHA-256; it is not treated as the source of the underlying data license.
