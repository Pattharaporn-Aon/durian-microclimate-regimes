# Feature Treatment and Bootstrap Stability in Clustering Orchard Microclimate Time Series

Analysis code and data for the manuscript:

> Thongnim, P., & Srinil, P. *Feature treatment and bootstrap stability in clustering
> orchard microclimate time series.* Manuscript in preparation.

---

## Description

IoT stations record orchard microclimate continuously, but those long hourly streams
are rarely condensed into interpretable day-types a grower can act on. This
repository takes a six-month hourly record from a single durian orchard, aggregates
it into 183 daily feature vectors, and asks what has to be true before the recovered
day-types can be called regimes.

The central claim is about **feature treatment**, not algorithm choice. On a record
dominated by the dry season, a naive distance-based clustering is pulled apart by the
handful of rainy days: rainfall dominates the distances and the partition collapses.
A physics-informed treatment that log-damps the zero-inflated rain and stress-hour
counts recovers four seasonal regimes instead.

The pipeline then tests, rather than assumes, that those regimes are real:

- **cluster-wise** bootstrap Jaccard stability against the Hennig (2007) bands, not a
  single mean that a fragile cluster can hide beneath;
- cross-algorithm agreement (k-means vs Ward vs a Gaussian mixture) by adjusted Rand
  index, reported whether or not it flatters the argument;
- Markov transition probabilities and dwell times for regime persistence;
- dry-spell detection under an explicitly stated rainfall threshold;
- an FAO-56 recomputation of the station's vapour-pressure deficit, which certifies
  the derivation and — as the code and codebook both say — nothing more.

All nine manuscript figures are reproduced by the single script described below.

---

## Dataset Information

Full column definitions and every derivation rule are in
[`DATA_README.md`](DATA_README.md) and [`CODEBOOK.md`](CODEBOOK.md).

| File | Role | Description |
| --- | --- | --- |
| `data/IoT_Sensor_Hourly.csv` | Raw input | Hourly orchard microclimate record (4,409 rows), 13 Dec 2025 – 15 Jun 2026: temperature, humidity, rainfall, wind speed, VPD and light. |
| `data/daily_features.csv` | Derived | Analysis-ready daily table (183 rows × 12 features), 14 Dec 2025 – 14 Jun 2026. Canonical input: this is the table carrying the DOI and the one the manuscript reports on. |

**Design.** One station, one orchard (cv. Monthong, Chanthaburi, eastern Thailand),
one growing season. Nothing here supports generalisation to other sites or years.

**Class balance.** The record is dry-season dominated: 54 of 183 days carry some
measurable rain and only 8 clear the 5 mm wet-day threshold. Any rain-related cluster
therefore rests on very few observations, and the paper says so.

**One full-day outage.** 18 February 2026 has no reading on any channel except rain.
The archived daily table still contains that row, so its values are imputed across a
24-hour gap rather than measured — beyond the ≤ 6 h interpolation rule. The pipeline
prints an `AUDIT:` line naming the date on every run.

**Archive.** The dataset is deposited on OSF: <https://osf.io/7thq4/>
(DOI: 10.17605/OSF.IO/7THQ4), CC-BY 4.0.

---

## Reproducing the analysis

```bash
git clone https://github.com/Pattharaporn-Aon/durian-microclimate-regimes.git
cd durian-microclimate-regimes
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/weather_regimes.py
```

Figures `fig1`–`fig9` are written to `outputs_regimes/`; every reported number is
printed to stdout. `RANDOM_STATE = 42` and `N_INIT = 50` are set at the top of the
script, so runs are reproducible.

The GitHub Actions workflow in `.github/workflows/run-pipeline.yml` runs the same
pipeline on every push to `main`, uploads the outputs as a downloadable artifact and
commits the regenerated figures back to the repository.

### Analysis decisions are constants, not magic numbers

Every choice the results depend on is a named constant in the configuration block at
the top of [`src/weather_regimes.py`](src/weather_regimes.py) and is documented in
[`CODEBOOK.md`](CODEBOOK.md) §3 — the damped columns, the stress thresholds, the
bootstrap replicate count, the minimum regime size, and in particular:

| Constant | Value | Why it matters |
| --- | --- | --- |
| `DRY_DAY_MM` | 1.0 mm | Produces the 40-day dry spell of 22 Mar – 30 Apr 2026. Five days inside that run carry 0.2–0.8 mm of rain, so it is a **dry** spell, not a rain-free one; with a strict 0 mm test the longest run is **32 days**. The script prints both. |
| `WET_DAY_MM` | 5.0 mm | Produces the eight wet days. 54 days carry some measurable rain. |
| `K_FINAL` | 4 | The **inertia elbow**. Silhouette monotonically favours k = 2 on this dataset; `fig2_kselect` plots both curves and marks the elbow rather than hiding the disagreement. |

---

## Repository layout

```
data/                     the two CSVs (also archived on OSF, CC-BY 4.0)
src/weather_regimes.py    the complete pipeline
outputs_regimes/          fig1..fig9, regenerated by the script and by CI
CODEBOOK.md               every variable, derivation rule and threshold
DATA_README.md            dataset description and provenance
CITATION.cff              how to cite the software
.zenodo.json              Zenodo deposition metadata
```

---

## Licence

- **Code** (`src/`): MIT — see [LICENSE](LICENSE)
- **Data** (`data/`): CC-BY 4.0, matching the OSF archive — see [data/LICENSE](data/LICENSE)

## Citation

Please cite both the paper and the dataset. See [CITATION.cff](CITATION.cff).

## Contact

Pattharaporn Thongnim — Department of Mathematics, Faculty of Science,
Burapha University, Chon Buri, Thailand — pattharaporn@buu.ac.th
