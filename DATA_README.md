# Orchard IoT microclimate dataset (weather regimes)

Two data files support the paper *"Feature treatment and bootstrap stability in
clustering orchard microclimate time series"*. Full variable definitions, derivation
rules and analysis thresholds are in [`CODEBOOK.md`](CODEBOOK.md).

| File | Role | Description |
| --- | --- | --- |
| `data/IoT_Sensor_Hourly.csv` | Raw input | Hourly orchard microclimate record (4,409 rows), 13 Dec 2025 – 15 Jun 2026: air temperature (°C), relative humidity (% RH), rainfall (mm), wind speed (m s⁻¹), vapour-pressure deficit (kPa) and light intensity (klx). |
| `data/daily_features.csv` | Derived | Analysis-ready daily table (183 rows), 14 Dec 2025 – 14 Jun 2026. Twelve features per day. This is the table the manuscript reports on, and the one `weather_regimes.py` treats as canonical. |

## Design

One IoT weather station in a single durian orchard (cv. Monthong) in Chanthaburi,
eastern Thailand, over one growing season (2025/26). **One station, one orchard, one
season** — nothing here supports generalisation to other sites or years, and the
manuscript frames the study accordingly.

The record is dry-season dominated. Of the 183 days, 54 carry some measurable rain
and only 8 clear the 5 mm wet-day threshold, so any rain-related cluster rests on
very few observations. This is stated as a limitation rather than worked around.

## Field observation

Flowering was observed at the orchard on **11 February 2026**, 56 days after the
onset of the December–February dry period (17 December 2025). This is a single
phenological event at a single site, reported as a field observation and not as a
statistically supported result. Note also that those 56 days span two dry spells
separated by rain on 18 January 2026, not one continuous run.

## Missing values and one full-day outage

Gaps of ≤ 6 h are filled by linear interpolation; missing rainfall is set to zero.

**18 February 2026 is a complete sensor outage** — all 24 hours are empty on every
channel except rain. The archived daily table still contains a row for that date, so
those values are imputed across a full-day gap rather than measured.
`weather_regimes.py` prints an `AUDIT:` line naming the date on every run so the fact
cannot be lost. See [`CODEBOOK.md`](CODEBOOK.md) §1.

## Rebuilding the daily table

`weather_regimes.py` uses the archived `daily_features.csv` when it is present,
because that is the file carrying the DOI. When only the hourly record is available
it rebuilds the daily table from scratch and reports any day it has to drop.

## Archive

> Thongnim, P. (2026). *Orchard IoT Microclimate Dataset (Weather Regimes)*. OSF.
> <https://osf.io/7thq4/> (DOI: 10.17605/OSF.IO/7THQ4), CC-BY 4.0
