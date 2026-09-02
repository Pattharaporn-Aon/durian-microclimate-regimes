# Codebook — `IoT_Sensor_Hourly.csv` and `daily_features.csv`

Companion to the manuscript *"Feature treatment and bootstrap stability in clustering
orchard microclimate time series"* (Thongnim & Srinil).

This codebook lists every variable in both data files, states its unit and type, and
records the derivation rules and thresholds that the published figures depend on.
Every threshold named here is a named constant at the top of
[`src/weather_regimes.py`](src/weather_regimes.py); none of them is hard-coded inside
a function.

**Encoding:** UTF-8. **Delimiter:** comma. **Decimal separator:** period.
**Time zone:** Asia/Bangkok (UTC+7) throughout. **Missing-value marker:** empty cell.

---

## 1. `IoT_Sensor_Hourly.csv` — raw hourly record

**Unit of observation:** one hour at one sensor station.
**Extent:** 4,409 rows, 13 Dec 2025 – 15 Jun 2026 (185 calendar days).
**Station:** one IoT weather station in a durian orchard (cv. Monthong), Chanthaburi,
eastern Thailand. There is only one station; nothing in this file supports
between-site comparison.

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `datetime` | timestamp | — | Start of the hour, local time (Asia/Bangkok) |
| `Temp(C)` | continuous | °C | Air temperature |
| `Humid(%RH)` | continuous | % | Relative humidity |
| `Rain(mm)` | continuous | mm | Rainfall accumulated **within** that hour (not cumulative) |
| `WindSpeed(m/s)` | continuous | m s⁻¹ | Wind speed |
| `VPD(kpa)` | continuous | kPa | Vapour-pressure deficit — see the caveat below |
| `Lux(klux)` | continuous | klx | Light intensity |

### `VPD(kpa)` is a derived channel, not an independent measurement

The station computes VPD on board from its own temperature and humidity readings.
The FAO-56 consistency check reported in the manuscript (`fig7_sensorqc`) recomputes

```
es  = 0.6108 · exp(17.27 · T / (T + 237.3))
VPD = es − es · RH / 100
```

and compares it against the stored channel. Agreement (mean residual −0.006 kPa,
no early/late drift) **certifies the derivation only**. It cannot detect drift in the
underlying temperature or humidity sensors, because those are the same readings the
station used. This limitation is stated in the manuscript and must not be dropped in
revision.

### Missing data and the 18 February 2026 outage

Gaps of **≤ 6 h** (`MAX_GAP_HOURS`) are filled by linear interpolation on the time
index; missing rainfall is set to zero.

**`2026-02-18` is a complete sensor outage.** All 24 hours of that day carry no
reading on any of temperature, humidity, wind, VPD or light — only the rain channel
records (as zero). The archived `daily_features.csv` nevertheless contains a row for
that date, so **that row is imputed across a 24-hour gap, not measured**. This
exceeds the ≤ 6 h interpolation rule and is flagged at runtime by
`weather_regimes.py`, which prints an `AUDIT:` line naming the date on every run.

---

## 2. `daily_features.csv` — derived daily features

**Unit of observation:** one calendar day.
**Extent:** 183 rows, 14 Dec 2025 – 14 Jun 2026.

Days at the two boundaries of the hourly record (13 Dec 2025, 7 h; 15 Jun 2026, 10 h)
carry fewer than `MIN_HOURS_PER_DAY = 20` hours of record and are dropped. Every
other calendar day in range has a full 24 hours.

| Column | Type | Unit | Derivation |
| --- | --- | --- | --- |
| `date` | date | — | Calendar date |
| `temp_mean` | continuous | °C | Mean of the 24 hourly temperatures |
| `temp_max` | continuous | °C | Daily maximum temperature |
| `temp_range` | continuous | °C | Diurnal range, `temp_max − temp_min` |
| `humid_mean` | continuous | % | Mean relative humidity |
| `rain_sum` | continuous | mm | Daily rainfall total |
| `rain_hours` | count | h | Hours with rain > 0 mm |
| `wind_mean` | continuous | m s⁻¹ | Mean wind speed |
| `vpd_mean` | continuous | kPa | Mean vapour-pressure deficit |
| `vpd_stress_h` | count | h | Hours with VPD > `VPD_STRESS_THRESHOLD` = 2.0 kPa |
| `hot_hours` | count | h | Hours with temperature > `HOT_HOUR_THRESHOLD` = 35 °C |
| `light_sum` | continuous | klx·h | Daily light integral, sum of the hourly klx values |
| `night_tmin` | continuous | °C | Minimum temperature over `NIGHT_HOURS` = 19:00–06:00 |

**Column order.** `night_tmin` is stored **last** in the file, after `light_sum`.
An earlier version of the OSF data README listed it after `temp_range`; the file, not
that listing, is authoritative. `weather_regimes.py` selects columns by name, so the
stored order does not affect any result.

---

## 3. Analysis thresholds

These are not properties of the data; they are decisions. Both of the headline
rainfall figures in the manuscript depend on them, and **neither threshold appears in
the current manuscript text or in the OSF data README** — stating them is a required
revision.

| Constant | Value | Meaning | What it produces |
| --- | --- | --- | --- |
| `DRY_DAY_MM` | 1.0 mm | A day with **less** rain than this counts as dry | Longest dry spell = **40 days**, 22 Mar – 30 Apr 2026 |
| `WET_DAY_MM` | 5.0 mm | A day with **at least** this much rain counts as wet | **8 wet days** in the record |

Two consequences follow, and both change how the results may be worded:

1. **The 40-day spell is not rain-free.** Five days inside it carry measurable rain
   (0.2–0.8 mm on 22 Mar, 6 Apr, 7 Apr, 20 Apr, 23 Apr and 30 Apr). Under a strict
   rain-free test (0 mm) the longest spell in the record is **32 days**
   (17 Dec 2025 – 17 Jan 2026). The manuscript's phrase "the longest *rain-free*
   spell ran to 40 days" is therefore incorrect as written; the pipeline prints both
   figures side by side so the distinction cannot be lost again.
2. **54 days carry some measurable rain**, against 8 that clear the 5 mm wet-day
   threshold. "Eight wet days" is a statement about the threshold, not about how
   often it rained.

### Other analysis constants

| Constant | Value | Role |
| --- | --- | --- |
| `RANDOM_STATE` | 42 | Seed for k-means, the bootstrap, the GMM and the isolation forest |
| `N_INIT` | 50 | k-means restarts, set high for run-to-run determinism |
| `K_RANGE` | 2–8 | Candidate cluster counts scanned |
| `K_FINAL` | 4 | Reported partition — the **inertia elbow**, not the silhouette optimum (silhouette monotonically favours k = 2 on this dataset; `fig2_kselect` shows both) |
| `DAMPED_COLS` | `rain_sum`, `rain_hours`, `vpd_stress_h`, `hot_hours` | Zero-inflated counts and totals that the physics-informed treatment passes through `log1p` before z-scoring |
| `N_BOOTSTRAP` | 500 | Hennig bootstrap replicates |
| `MIN_REGIME_DAYS` | 5 | Minimum size for a cluster to be described as a regime |
| `N_ANOMALIES` | 10 | Days retained from the isolation forest |

### Reading the stability numbers

Cluster-wise bootstrap Jaccard is reported **per cluster**, against the Hennig (2007)
bands: ≥ 0.75 stable, 0.60–0.75 doubtful, < 0.60 dissolved. A mean over clusters is
also printed, but it is the per-cluster values that license the word "stable" — a
mean can sit comfortably above 0.75 while one small cluster dissolves beneath it.
`weather_regimes.py` prints an explicit warning whenever fewer than all `K_FINAL`
regimes clear the threshold.

---

## 4. Provenance

Both files are archived on OSF with a permanent DOI:

> Thongnim, P. (2026). *Orchard IoT Microclimate Dataset (Weather Regimes)*. OSF.
> <https://doi.org/10.17605/OSF.IO/7THQ4> — CC-BY 4.0

The copies in `data/` are identical to the archived version. Cite the DOI, not this
repository, when referring to the data itself.
