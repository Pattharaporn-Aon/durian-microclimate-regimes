# src/

`weather_regimes.py` — the complete analysis pipeline. One script reproduces every
number and all nine figures in the manuscript.

```bash
pip install -r ../requirements.txt
python weather_regimes.py
```

It looks for `daily_features.csv` and `IoT_Sensor_Hourly.csv` in `.`, `./data`,
`/content` and its own directory.

- The **archived daily table is canonical** — it carries the DOI and is what the
  manuscript reports on. The hourly record is used to audit that table and to run the
  FAO-56 VPD provenance check.
- With only the hourly record, the daily table is rebuilt from scratch and any day
  that has to be dropped is named.
- With only the daily table, everything runs except the VPD check.

Figures are written to `../outputs_regimes/` as `fig1`–`fig9`.

## Reading the output

Three lines deserve attention on every run:

- `AUDIT:` — names any day present in the archived table that cannot be rebuilt from
  the hourly record (currently 2026-02-18, a full-day sensor outage).
- `top-3 dry spells, rain < 1.0 mm` printed directly above `top-3 with NO measurable
  rain` — the two are different numbers (40 days vs 32 days) and the manuscript must
  not conflate them.
- `=== CLUSTER-WISE STABILITY (Hennig bands) ===` — per-cluster Jaccard with an
  explicit verdict, plus a warning if fewer than all four regimes clear 0.75.

## Configuration

Every analysis decision is a named constant in the block at the top of the file, not
a value buried in a function. `CODEBOOK.md` §3 documents what each one does and which
published figure depends on it. `RANDOM_STATE = 42` and `N_INIT = 50` keep runs
reproducible.
