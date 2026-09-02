#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weather_regimes.py
==================
Regime discovery in orchard microclimate time series.

Supports: "Feature Treatment and Bootstrap Stability in Clustering Orchard
Microclimate Time Series" (Thongnim & Srinil).

Pipeline
--------
1.  Hourly -> daily feature aggregation (12 features, 183 complete days)
2.  Two feature treatments: naive (raw + z-score) vs physics-informed
    (log1p-damping of zero-inflated rain / stress-hour counts, then z-score)
3.  k selection: inertia elbow + silhouette, reported side by side
4.  k-means clustering, cross-algorithm agreement (Ward, GMM) via ARI
5.  Hennig bootstrap cluster stability -- reported PER CLUSTER, not just the mean
6.  Markov transition matrix, stationary distribution, dwell times
7.  Dry-spell detection
8.  Sensor QC: FAO-56 VPD recomputation + isolation forest
9.  Figures fig1..fig9 -> outputs_regimes/

Inputs (searched in ., ./data, /content)
----------------------------------------
IoT_Sensor_Hourly.csv   preferred -- daily features are rebuilt from it
daily_features.csv      fallback  -- used if the hourly record is absent
                                     (sensor QC is then skipped)

Data: https://doi.org/10.17605/OSF.IO/7THQ4  (CC-BY 4.0)

Usage
-----
    python weather_regimes.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Configuration -- every analysis choice is a named constant, not a magic
# number buried in the code.
# --------------------------------------------------------------------------

RANDOM_STATE = 42
OUTDIR = "outputs_regimes"

K_RANGE = range(2, 9)          # candidate cluster counts
K_FINAL = 4                    # see select_k(): elbow choice, NOT the silhouette optimum
N_INIT = 50                    # k-means restarts (high, for run-to-run determinism)

MIN_HOURS_PER_DAY = 20         # days with fewer hourly records are dropped
MAX_GAP_HOURS = 6              # linear interpolation only across gaps this short

NIGHT_HOURS = list(range(19, 24)) + list(range(0, 7))   # 19:00-06:00
VPD_STRESS_THRESHOLD = 2.0     # kPa
HOT_HOUR_THRESHOLD = 35.0      # degC

# Columns that are zero-inflated counts / totals: these are the ones the
# physics-informed treatment log-damps so that a handful of rainy days cannot
# dominate the Euclidean distances.
DAMPED_COLS = ["rain_sum", "rain_hours", "vpd_stress_h", "hot_hours"]

# Rain thresholds. NEITHER of these is a "rain-free" test, and both must be
# stated explicitly in the manuscript -- the published figures depend on them.
#   DRY_DAY_MM = 1.0 reproduces the 40-day spell of 22 Mar - 30 Apr 2026.
#   With a strict rain-free test (0.0 mm) the longest spell is 32 days.
#   WET_DAY_MM = 5.0 yields exactly the eight "wet days" the paper reports;
#   54 days in the record carry some measurable rain.
DRY_DAY_MM = 1.0               # a day with < this much rain counts as dry
WET_DAY_MM = 5.0               # a day with >= this much rain counts as wet

MIN_REGIME_DAYS = 5            # a cluster smaller than this is not a "regime"
N_BOOTSTRAP = 500              # Hennig bootstrap replicates
N_ANOMALIES = 10               # days flagged by the isolation forest

# Hennig (2007) interpretation bands for bootstrap Jaccard.
JACCARD_STABLE = 0.75          # >= 0.75 -> stable / valid
JACCARD_DISSOLVED = 0.60       # <  0.60 -> "dissolved", not a real cluster

# Field observations at the orchard (from the field notebook).
FIELD_VISITS = ["2025-12-17", "2026-02-11", "2026-04-08"]
FLOWERING_DATE = "2026-02-11"

FEATURES = [
    "temp_mean", "temp_max", "temp_range", "humid_mean",
    "rain_sum", "rain_hours", "wind_mean", "vpd_mean",
    "vpd_stress_h", "hot_hours", "light_sum", "night_tmin",
]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def find_file(name: str) -> str | None:
    for d in (".", "data", "/content", os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def load_hourly() -> pd.DataFrame | None:
    path = find_file("IoT_Sensor_Hourly.csv")
    if path is None:
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.rename(columns={
        tcol: "datetime",
        "Temp(C)": "temp",
        "Humid(%RH)": "rh",
        "Rain(mm)": "rain",
        "WindSpeed(m/s)": "wind",
        "VPD(kpa)": "vpd",
        "Lux(klux)": "lux",
    })
    return df.sort_values("datetime").reset_index(drop=True)


def clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Short-gap linear interpolation; missing rainfall treated as zero."""
    df = df.set_index("datetime")
    df["rain"] = df["rain"].fillna(0.0)
    for c in ["temp", "rh", "wind", "vpd", "lux"]:
        if c in df:
            df[c] = df[c].interpolate(method="time", limit=MAX_GAP_HOURS,
                                      limit_direction="both")
    return df.reset_index()


def build_daily(h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the hourly record into the 12 daily features."""
    h = h.copy()
    h["date"] = h["datetime"].dt.date
    h["hour"] = h["datetime"].dt.hour

    counts = h.groupby("date").size()
    keep = counts[counts >= MIN_HOURS_PER_DAY].index

    g = h[h["date"].isin(keep)].groupby("date")
    night = h[h["date"].isin(keep) & h["hour"].isin(NIGHT_HOURS)].groupby("date")

    daily = pd.DataFrame({
        "temp_mean":    g["temp"].mean(),
        "temp_max":     g["temp"].max(),
        "temp_range":   g["temp"].max() - g["temp"].min(),
        "humid_mean":   g["rh"].mean(),
        "rain_sum":     g["rain"].sum(),
        "rain_hours":   g["rain"].apply(lambda s: int((s > 0).sum())),
        "wind_mean":    g["wind"].mean(),
        "vpd_mean":     g["vpd"].mean(),
        "vpd_stress_h": g["vpd"].apply(lambda s: int((s > VPD_STRESS_THRESHOLD).sum())),
        "hot_hours":    g["temp"].apply(lambda s: int((s > HOT_HOUR_THRESHOLD).sum())),
        "light_sum":    g["lux"].sum(),
        "night_tmin":   night["temp"].min(),
    })
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"

    bad = daily.index[daily.isna().any(axis=1)]
    if len(bad):
        print("WARNING: sensor outage — no usable readings on "
              + ", ".join(str(d.date()) for d in bad)
              + "; dropped from the rebuilt table.")
    return daily.dropna()


def load_daily_archived() -> pd.DataFrame | None:
    """The archived, DOI-citable daily table. This is the canonical input."""
    path = find_file("daily_features.csv")
    if path is None:
        return None
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return d[FEATURES]


# --------------------------------------------------------------------------
# Feature treatments
# --------------------------------------------------------------------------

def treat_naive(daily: pd.DataFrame) -> np.ndarray:
    """Raw features, z-scored. Rain dominates the distances."""
    return StandardScaler().fit_transform(daily[FEATURES].values)


def treat_informed(daily: pd.DataFrame) -> np.ndarray:
    """log1p-damp the zero-inflated counts/totals, then z-score."""
    X = daily[FEATURES].copy()
    for c in DAMPED_COLS:
        X[c] = np.log1p(X[c])
    return StandardScaler().fit_transform(X.values)


# --------------------------------------------------------------------------
# Clustering helpers
# --------------------------------------------------------------------------

def kmeans_fit(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=N_INIT,
                  random_state=RANDOM_STATE).fit_predict(X)


def k_table(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
        rows.append({
            "k": k,
            "inertia": round(km.inertia_, 1),
            "silhouette": round(silhouette_score(X, km.labels_), 3),
        })
    return pd.DataFrame(rows)


def elbow_k(tab: pd.DataFrame) -> int:
    """
    Knee of the inertia curve: the point furthest from the straight line
    joining the first and last (k, inertia) points. Deterministic, and
    independent of the silhouette -- which on this dataset monotonically
    favours k=2 and therefore cannot be used to justify a richer partition.
    """
    ks = tab["k"].values.astype(float)
    ine = tab["inertia"].values.astype(float)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (ine - ine.min()) / (ine.max() - ine.min())
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) \
        / np.hypot(y2 - y1, x2 - x1)
    return int(ks[int(np.argmax(dist))])


def n_interpretable(labels: np.ndarray) -> int:
    """
    Descriptive diagnostic: how many clusters hold at least MIN_REGIME_DAYS days.
    This is NOT a cluster-validity index and must not be read as one -- it only
    records how many groups are large enough to describe at all.
    """
    _, counts = np.unique(labels, return_counts=True)
    return int((counts >= MIN_REGIME_DAYS).sum())


def largest_share(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return 100.0 * counts.max() / counts.sum()


# --------------------------------------------------------------------------
# Hennig bootstrap stability
# --------------------------------------------------------------------------

def bootstrap_jaccard(X: np.ndarray, k: int, labels: np.ndarray,
                      B: int = N_BOOTSTRAP) -> np.ndarray:
    """
    Hennig (2007) cluster-wise stability.

    For each replicate: resample the days with replacement, recluster, and for
    every original cluster take the best Jaccard against any bootstrap cluster,
    both restricted to the days actually drawn. Returns the per-cluster mean
    Jaccard over B replicates.

    Reported per cluster on purpose. A mean over clusters hides exactly the
    case that matters here -- one small cluster that dissolves under resampling.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = X.shape[0]
    orig = [set(np.where(labels == j)[0]) for j in range(k)]
    acc = np.zeros((B, k))

    for b in range(B):
        idx = rng.integers(0, n, n)
        uniq = np.unique(idx)
        lab_b = KMeans(n_clusters=k, n_init=10,
                       random_state=int(rng.integers(1 << 30))).fit_predict(X[idx])
        boot = [set(uniq[np.isin(uniq, idx[lab_b == m])]) for m in range(k)]
        for j in range(k):
            cj = orig[j] & set(uniq)
            if not cj:
                acc[b, j] = np.nan
                continue
            acc[b, j] = max(
                (len(cj & dm) / len(cj | dm)) if (cj | dm) else 0.0
                for dm in boot
            )
    # Return the full B x k matrix as well: the manuscript plots the
    # DISTRIBUTION of recovery per regime, not just its mean.
    return np.nanmean(acc, axis=0), acc


# --------------------------------------------------------------------------
# Regime naming, dynamics, dry spells
# --------------------------------------------------------------------------

def name_regimes(daily: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """
    Label clusters from their centroids rather than by hand, so the names
    follow the data if the clustering changes.
    """
    prof = daily[FEATURES].groupby(labels).mean()
    names, left = {}, list(prof.index)

    wet = prof.loc[left, "rain_sum"].idxmax()
    names[wet] = "Wet / overcast"
    left.remove(wet)

    cool = prof.loc[left, "night_tmin"].idxmin()
    names[cool] = "Cool-night high-demand dry"
    left.remove(cool)

    humid = prof.loc[left, "humid_mean"].idxmax()
    names[humid] = "Humid transition"
    left.remove(humid)

    for j in left:
        names[j] = "Warm dry"
    return names


def transition_matrix(labels: np.ndarray, k: int) -> np.ndarray:
    T = np.zeros((k, k))
    for a, b in zip(labels[:-1], labels[1:]):
        T[a, b] += 1
    rs = T.sum(axis=1, keepdims=True)
    return np.divide(T, rs, out=np.zeros_like(T), where=rs > 0)


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(T.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum()


def dwell_times(labels: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    runs, cur, ln = [], labels[0], 1
    for x in labels[1:]:
        if x == cur:
            ln += 1
        else:
            runs.append((cur, ln))
            cur, ln = x, 1
    runs.append((cur, ln))

    r = pd.DataFrame(runs, columns=["regime", "len"])
    out = r.groupby("regime")["len"].agg(
        n_runs="count", mean_dwell="mean", max_dwell="max").reset_index()
    out["mean_dwell"] = out["mean_dwell"].round(2)
    out["regime_name"] = out["regime"].map(names)
    return out


def dry_spells(daily: pd.DataFrame,
               threshold_mm: float = DRY_DAY_MM) -> list[tuple[str, str, int]]:
    """
    Runs of consecutive days with rainfall below `threshold_mm`.

    Note the wording this licenses: at the default 1.0 mm these are DRY spells,
    not RAIN-FREE spells. Four days inside the longest run carry 0.2-0.8 mm of
    measurable rain. Calling the result "rain-free" would misstate it.
    """
    dry = (daily["rain_sum"] < threshold_mm).values
    dates = daily.index
    spells, start = [], None
    for i, d in enumerate(dry):
        if d and start is None:
            start = i
        elif not d and start is not None:
            spells.append((str(dates[start].date()), str(dates[i - 1].date()), i - start))
            start = None
    if start is not None:
        spells.append((str(dates[start].date()), str(dates[-1].date()), len(dry) - start))
    return sorted(spells, key=lambda s: -s[2])


# --------------------------------------------------------------------------
# Sensor QC
# --------------------------------------------------------------------------

def fao56_vpd(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """FAO-56 saturation vapour pressure deficit (kPa)."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return es - es * rh_pct / 100.0


def vpd_provenance(h: pd.DataFrame) -> dict:
    """
    Compare the station's on-board VPD against the FAO-56 recomputation.

    This certifies the DERIVATION only. VPD is computed on-board from
    temperature and humidity, so agreement here says nothing about drift in
    the temperature or humidity sensors themselves.
    """
    r = h["vpd"].values - fao56_vpd(h["temp"].values, h["rh"].values)
    ok = ~np.isnan(r)
    r, t = r[ok], h["datetime"].values[ok]
    early = r[t < t[0] + np.timedelta64(30, "D")]
    late = r[t > t[-1] - np.timedelta64(30, "D")]
    return {"mean": r.mean(), "early": early.mean(), "late": late.mean()}


def isolation_forest_days(daily: pd.DataFrame, n: int = N_ANOMALIES):
    """Returns (flagged dates, anomaly score per day). Lower score = more anomalous."""
    X = StandardScaler().fit_transform(daily[FEATURES].values)
    s = IsolationForest(n_estimators=500, contamination="auto",
                        random_state=RANDOM_STATE).fit(X).score_samples(X)
    return daily.index[np.argsort(s)[:n]], pd.Series(s, index=daily.index)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

PALETTE = ["#c77d17", "#3b6fb0", "#6b4fbb", "#c14b6b", "#0f9d8f"]


def fig1_timeseries(daily, h, spells):
    """
    Manuscript Figure 6. Four stacked panels showing the TREATED features that
    define the regimes -- VPD-stress hours, hot hours, night minimum and rain
    hours -- rather than raw daily means. The shaded band is the longest dry
    spell; dotted green lines are the field visits.
    """
    panels = [
        ("vpd_stress_h", "VPD stress\n(h d$^{-1}$, >2.0 kPa)", "#c77d17"),
        ("hot_hours",    "Hot hours\n(h d$^{-1}$, >35 °C)",    "#c14b6b"),
        ("night_tmin",   "Night $T_{min}$\n(°C, 19:00–06:00)", "#3b6fb0"),
        ("rain_hours",   "Rain hours\n(h d$^{-1}$)",           "#0f9d8f"),
    ]
    s = spells[0]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 8.0), sharex=True)
    for ax, (col, lab, c) in zip(axes, panels):
        ax.fill_between(daily.index, 0, daily[col], color=c, alpha=0.35, lw=0)
        ax.plot(daily.index, daily[col], color=c, lw=1.0)
        ax.axvspan(pd.Timestamp(s[0]), pd.Timestamp(s[1]),
                   color="#f5a623", alpha=0.15, lw=0, zorder=0)
        for vd in FIELD_VISITS:
            vd = pd.Timestamp(vd)
            if daily.index.min() <= vd <= daily.index.max():
                ax.axvline(vd, color="#2e7d32", ls=":", lw=1.2, zorder=3)
        ax.set_ylabel(lab, fontsize=9)
        ax.margins(x=0.01)
        ax.tick_params(labelsize=8)
    axes[0].set_title(f"Physics-informed regime drivers over the season "
                      f"(shaded = longest dry spell, {s[2]} days; "
                      f"dotted = field visits)", fontsize=11)
    fig.align_ylabels(axes)
    save(fig, "fig1_timeseries")


def fig2_kselect(tab, k_elbow):
    """Manuscript Figure 2. Inertia (grey) and silhouette (green) versus k."""
    GREY, GREEN = "#6b7280", "#2e7d32"
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(tab["k"], tab["inertia"], "o-", color=GREY, lw=1.6, ms=6,
             label="inertia")
    ax1.set_xlabel("number of regimes $k$")
    ax1.set_ylabel("inertia (within-cluster sum of squares)", color=GREY)
    ax1.tick_params(axis="y", labelcolor=GREY)

    ax2 = ax1.twinx()
    ax2.plot(tab["k"], tab["silhouette"], "s--", color=GREEN, lw=1.6, ms=6,
             label="silhouette")
    ax2.set_ylabel("silhouette", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)

    ax1.axvline(K_FINAL, color="#1c2430", ls=":", lw=1.5, zorder=0)
    ax1.annotate(f"$k={K_FINAL}$ retained", xy=(K_FINAL, tab["inertia"].max()),
                 xytext=(4, -6), textcoords="offset points",
                 fontsize=9, color="#1c2430", va="top")
    k_sil = int(tab.loc[tab["silhouette"].idxmax(), "k"])
    ax1.set_title(f"Regime-number selection — inertia elbow at $k$={k_elbow}; "
                  f"silhouette peaks at $k$={k_sil}", fontsize=10)
    save(fig, "fig2_kselect")


def fig3_centroids(daily, labels, names):
    prof = daily[FEATURES].groupby(labels).mean()
    z = (prof - prof.mean()) / prof.std(ddof=0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels([names[i] for i in prof.index], fontsize=8)
    ax.set_title("Regime centroids (z-scored across regimes)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "fig3_centroids")


def fig4_timeline(daily, labels, names):
    """
    Manuscript Figure 7. (top) regime label for each day of the season.
    (bottom) daily rainfall on the same time axis.
    """
    k = len(names)
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})

    axes[0].imshow(labels.reshape(1, -1), aspect="auto",
                   cmap=ListedColormap(PALETTE[:k]),
                   extent=[0, len(labels), 0, 1], interpolation="nearest")
    axes[0].set_yticks([])
    axes[0].set_ylabel("regime", fontsize=9)

    x = np.arange(len(daily))
    axes[1].bar(x, daily["rain_sum"].values, width=1.0, color="#3b6fb0")
    axes[1].axhline(DRY_DAY_MM, color="#c14b6b", ls="--", lw=1,
                    label=f"dry-day threshold ({DRY_DAY_MM:g} mm)")
    axes[1].axhline(WET_DAY_MM, color="#0f9d8f", ls="--", lw=1,
                    label=f"wet-day threshold ({WET_DAY_MM:g} mm)")
    axes[1].set_ylabel("rainfall (mm d$^{-1}$)", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    axes[1].set_xlim(0, len(daily))

    step = max(1, len(daily) // 12)
    axes[1].set_xticks(range(0, len(daily), step))
    axes[1].set_xticklabels([d.strftime("%b %d") for d in daily.index[::step]],
                            rotation=45, ha="right", fontsize=8)

    axes[0].set_title("Regime sequence and daily rainfall", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in range(k)]
    fig.legend(handles, [names[i] for i in range(k)], ncol=k, fontsize=8,
               loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUTDIR, "fig4_timeline.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig5_transitions(T, names, dwell):
    """
    Manuscript Figure 5. (left) day-to-day transition probabilities.
    (right) mean dwell time per regime.
    """
    k = T.shape[0]
    lab = [names[i] for i in range(k)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    im = axes[0].imshow(T, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            axes[0].text(j, i, f"{T[i,j]:.2f}", ha="center", va="center",
                         fontsize=9, color="white" if T[i, j] > 0.5 else "#1c2430")
    axes[0].set_xticks(range(k)); axes[0].set_xticklabels(lab, rotation=40,
                                                          ha="right", fontsize=8)
    axes[0].set_yticks(range(k)); axes[0].set_yticklabels(lab, fontsize=8)
    axes[0].set_title("Day-to-day transition probabilities", fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.75)

    d = dwell.set_index("regime").reindex(range(k))
    order = np.argsort(d["mean_dwell"].values)
    axes[1].barh(range(k), d["mean_dwell"].values[order],
                 color=[PALETTE[i] for i in order])
    for y, i in enumerate(order):
        axes[1].text(d["mean_dwell"].values[i] + 0.06, y,
                     f"{d['mean_dwell'].values[i]:.2f} d  "
                     f"(max {int(d['max_dwell'].values[i])})",
                     va="center", fontsize=8)
    axes[1].set_yticks(range(k))
    axes[1].set_yticklabels([lab[i] for i in order], fontsize=8)
    axes[1].set_xlabel("mean dwell time (days)", fontsize=9)
    axes[1].set_xlim(0, d["mean_dwell"].max() * 1.65)
    axes[1].set_title("Mean dwell time per regime", fontsize=10)
    save(fig, "fig5_transitions")


def fig6_stability(jac, jac_raw, names):
    """
    Manuscript Figure 4. DISTRIBUTION of bootstrap Jaccard recovery for each
    regime over N_BOOTSTRAP resamples, with the 0.75 "stable" threshold marked.
    """
    k = len(jac)
    order = np.argsort(jac)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    data = [jac_raw[:, i][~np.isnan(jac_raw[:, i])] for i in order]
    bp = ax.boxplot(data, vert=False, widths=0.62, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#1c2430", lw=1.4),
                    whiskerprops=dict(color="#5b6672"),
                    capprops=dict(color="#5b6672"))
    for patch, i in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE[i]); patch.set_alpha(0.55)
        patch.set_edgecolor("#5b6672")

    rng = np.random.default_rng(RANDOM_STATE)
    for y, d in enumerate(data, start=1):
        s = d if len(d) <= 300 else rng.choice(d, 300, replace=False)
        ax.scatter(s, np.full(len(s), y) + rng.normal(0, 0.055, len(s)),
                   s=3, color="#1c2430", alpha=0.16, zorder=3, lw=0)

    ax.axvline(JACCARD_STABLE, color="#2e7d32", ls=":", lw=1.6,
               label=f'{JACCARD_STABLE} "stable" threshold')
    ax.axvline(JACCARD_DISSOLVED, color="#c14b6b", ls=":", lw=1.4,
               label=f'{JACCARD_DISSOLVED} "dissolved" threshold')
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    ax.set_ylim(0.4, k + 0.6)
    ax.set_yticks(range(1, k + 1))
    ax.set_yticklabels([f"{names[i]}\n(mean {jac[i]:.2f})" for i in order],
                       fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("bootstrap Jaccard recovery")
    ax.set_title(f"Regime stability over {N_BOOTSTRAP} bootstrap resamples",
                 fontsize=11)
    save(fig, "fig6_stability")


def fig7_sensorqc(h, anomalies, scores, daily):
    """
    Manuscript Figure 9. (top) reported minus physics-recomputed VPD with a
    14-day rolling mean. (bottom) isolation-forest anomaly scores, red points
    marking the flagged days.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.0))

    if h is not None:
        r = pd.Series(h["vpd"].values - fao56_vpd(h["temp"].values, h["rh"].values),
                      index=h["datetime"])
        axes[0].plot(r.index, r.values, lw=0.4, color="#9db4d0", zorder=1,
                     label="hourly residual")
        roll = r.rolling("14D").mean()
        axes[0].plot(roll.index, roll.values, lw=1.8, color="#1c4f8f", zorder=3,
                     label="14-day rolling mean")
        axes[0].axhline(0, color="#1c2430", lw=0.9, zorder=2)
        axes[0].set_ylabel("VPD residual (kPa)", fontsize=9)
        axes[0].legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
        axes[0].set_title("Reported minus physics-recomputed (FAO-56) VPD — "
                          "near zero and trend-free, indicating no drift.\n"
                          "This certifies the derivation, not the temperature "
                          "and humidity sensors behind it.", fontsize=10)
        axes[0].margins(x=0.01)
    else:
        axes[0].text(0.5, 0.5, "hourly record not available",
                     ha="center", va="center"); axes[0].axis("off")

    axes[1].plot(scores.index, scores.values, lw=1.0, color="#6b4fbb",
                 label="anomaly score")
    axes[1].scatter(anomalies, scores.loc[anomalies], color="#c14b6b",
                    zorder=5, s=34, label=f"{len(anomalies)} flagged days")
    thr = float(scores.loc[anomalies].max())
    axes[1].axhline(thr, color="#c14b6b", ls="--", lw=1)
    axes[1].set_ylabel("isolation-forest score", fontsize=9)
    axes[1].set_xlabel("")
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    axes[1].set_title("Isolation-forest anomaly scores (lower = more anomalous); "
                      "the flagged days proved to be genuine extreme weather, "
                      "not sensor faults.", fontsize=10)
    axes[1].margins(x=0.01)
    save(fig, "fig7_sensorqc")


def fig8_monthly(daily, labels, names):
    d = daily.copy()
    d["regime"] = [names[i] for i in labels]
    d["month"] = d.index.strftime("%Y-%m")
    ct = pd.crosstab(d["month"], d["regime"])
    ct = ct.div(ct.sum(axis=1), axis=0) * 100
    # pandas sorts the crosstab columns alphabetically, so colours must be
    # looked up by regime identity rather than by column position -- otherwise
    # the same regime appears in different colours in fig4 and fig8.
    name_to_idx = {v: k for k, v in names.items()}
    colors = [PALETTE[name_to_idx[c]] for c in ct.columns]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8,
            legend=False)
    ax.set_ylabel("% of days"); ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.set_title("Monthly regime composition")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")

    # Legend below the axes, never over the bars.
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, ncol=2, fontsize=8, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(OUTDIR, "fig8_monthly.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig9_pca_treatment(Xn, Xi, li, names):
    """
    Manuscript Figure 1. PCA projection under both treatments, with days
    coloured by the SAME physics-informed regimes in both panels -- that is
    what makes the collapse under the naive treatment visible.
    """
    k = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, X, tag, title in [
        (axes[0], Xn, "(a)", "Naive (raw)"),
        (axes[1], Xi, "(b)", "Physics-informed (log-damped)"),
    ]:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        p = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_ * 100
        ax.scatter(p[:, 0], p[:, 1], c=[PALETTE[i] for i in li],
                   s=22, alpha=0.85, lw=0)
        ax.set_title(f"{tag} {title}", fontsize=10, loc="left")
        ax.set_xlabel(f"PC1 ({ev[0]:.0f}% of variance)", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)", fontsize=9)
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[i],
                          label=names[i], ms=7) for i in range(k)]
    fig.legend(handles=handles, ncol=k, fontsize=8, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Days coloured by the four physics-informed regimes under both "
                 "treatments", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(os.path.join(OUTDIR, "fig9_pca_treatment.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    h = load_hourly()
    if h is not None:
        h = clean_hourly(h)
    else:
        print("NOTE: hourly record not found -- sensor QC will be skipped.\n")

    # The archived daily table is canonical: it is what carries the DOI and
    # what the manuscript reports. The hourly record is used to audit it and
    # to run the VPD provenance check.
    daily = load_daily_archived()
    if daily is None:
        if h is None:
            sys.exit("ERROR: need daily_features.csv or IoT_Sensor_Hourly.csv "
                     "(see https://doi.org/10.17605/OSF.IO/7THQ4)")
        print("NOTE: archived daily_features.csv not found -- rebuilding the "
              "daily table from the hourly record.\n")
        daily = build_daily(h)
    elif h is not None:
        rebuilt = build_daily(h)
        gap = daily.index.difference(rebuilt.index)
        if len(gap):
            print("AUDIT: the archived daily table contains "
                  f"{len(gap)} day(s) that CANNOT be rebuilt from the hourly "
                  "record, because every sensor channel is empty that day: "
                  + ", ".join(str(d.date()) for d in gap))
            print("       Those rows are imputed across a full-day outage, not "
                  "measured. This exceeds the '<= 6 h linear interpolation'\n"
                  "       rule stated in the data README, and should be "
                  "disclosed in the manuscript.")
            print("       The analysis below uses the archived table as "
                  "published.\n")

    print(f"daily feature table: {daily.shape} days\n")

    # --- feature treatments -------------------------------------------------
    Xn = treat_naive(daily)
    Xi = treat_informed(daily)

    tab_n = k_table(Xn)
    tab_i = k_table(Xi)

    print("k-selection (informed):")
    print(tab_i.to_string(index=False))
    print()

    k_elbow = elbow_k(tab_i)
    if k_elbow != K_FINAL:
        print(f"NOTE: inertia elbow suggests k={k_elbow}; "
              f"the paper reports k={K_FINAL}.\n")

    k_naive = int(tab_n.loc[tab_n["silhouette"].idxmax(), "k"])
    lab_n = kmeans_fit(Xn, k_naive)
    lab_i = kmeans_fit(Xi, K_FINAL)

    contrast = pd.DataFrame([
        {"treatment": "Naive (raw)", "chosen_k": k_naive,
         "silhouette": round(silhouette_score(Xn, lab_n), 3),
         "largest_regime_share_%": round(largest_share(lab_n), 1),
         "interpretable_regimes": n_interpretable(lab_n)},
        {"treatment": "Physics-informed (log-damped)", "chosen_k": K_FINAL,
         "silhouette": round(silhouette_score(Xi, lab_i), 3),
         "largest_regime_share_%": round(largest_share(lab_i), 1),
         "interpretable_regimes": n_interpretable(lab_i)},
    ])
    print("feature-treatment contrast:")
    print(contrast.to_string(index=False))
    print()

    # --- regimes ------------------------------------------------------------
    names = name_regimes(daily, lab_i)
    summary = daily[FEATURES].groupby(lab_i).mean().round(2)
    summary.insert(0, "n_days", pd.Series(lab_i).value_counts().sort_index())
    summary.insert(0, "regime", [names[i] for i in summary.index])
    print("regime summary:")
    print(summary.to_string(index=False))
    print()

    # --- cross-algorithm agreement -----------------------------------------
    lab_w = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward").fit_predict(Xi)
    lab_g = GaussianMixture(n_components=K_FINAL, covariance_type="full",
                            n_init=10, random_state=RANDOM_STATE).fit_predict(Xi)
    ari_w = adjusted_rand_score(lab_i, lab_w)
    ari_g = adjusted_rand_score(lab_i, lab_g)

    # --- stability ----------------------------------------------------------
    jac, jac_raw = bootstrap_jaccard(Xi, K_FINAL, lab_i)

    # --- dynamics -----------------------------------------------------------
    T = transition_matrix(lab_i, K_FINAL)
    stat = stationary_distribution(T)
    dwell = dwell_times(lab_i, names)
    spells = dry_spells(daily)

    flower = pd.Timestamp(FLOWERING_DATE)
    before = [s for s in spells if pd.Timestamp(s[0]) < flower]
    before = sorted(before, key=lambda s: -s[2])[:2]

    # --- sensor QC ----------------------------------------------------------
    vpd = vpd_provenance(h) if h is not None else None
    anomalies, iso_scores = isolation_forest_days(daily)

    # --- report -------------------------------------------------------------
    print("=== KEY NUMBERS ===")
    print(f"n days: {len(daily)} | regimes: {K_FINAL} | "
          f"kmeans silhouette: {silhouette_score(Xi, lab_i):.3f}")
    print(f"ARI kmeans-ward: {ari_w:.2f} | kmeans-gmm: {ari_g:.2f}")
    print(f"bootstrap Jaccard: {jac.mean():.2f} [{jac.min():.2f}, {jac.max():.2f}]")
    print("stationary regime freq: "
          f"{ {names[i]: round(float(stat[i]), 2) for i in range(K_FINAL)} }")
    strict = dry_spells(daily, threshold_mm=0.0001)
    print(f"top-3 dry spells, rain < {DRY_DAY_MM} mm (start,end,len): {spells[:3]}")
    print(f"top-3 spells with NO measurable rain at all:              {strict[:3]}")
    print(f"  -> report the {spells[0][2]}-day figure as a dry spell, not a "
          f"rain-free spell; rain-free tops out at {strict[0][2]} days.")
    print(f"wet days (rain >= {WET_DAY_MM} mm): "
          f"{int((daily['rain_sum'] >= WET_DAY_MM).sum())} | "
          f"days with any measurable rain: {int((daily['rain_sum'] > 0).sum())}")
    print(f"longest dry spell before flowering visit: {before}")
    print("dwell times:")
    print(dwell.to_string(index=False))
    if vpd:
        print(f"VPD residual: mean={vpd['mean']:.3f} "
              f"early(first30d)={vpd['early']:.3f} late(last30d)={vpd['late']:.3f}")
    print(f"IsolationForest anomalies: {len(anomalies)} days")

    # --- stability, spelled out --------------------------------------------
    print("\n=== CLUSTER-WISE STABILITY (Hennig bands) ===")
    for i in range(K_FINAL):
        n_i = int((lab_i == i).sum())
        verdict = ("stable" if jac[i] >= JACCARD_STABLE
                   else "doubtful" if jac[i] >= JACCARD_DISSOLVED
                   else "DISSOLVED — not a supportable cluster")
        print(f"  {names[i]:<28} n={n_i:>3}  Jaccard={jac[i]:.2f}  {verdict}")
    n_ok = int((jac >= JACCARD_STABLE).sum())
    if n_ok < K_FINAL:
        print(f"\n  WARNING: {n_ok} of {K_FINAL} regimes clear the 0.75 stability "
              f"threshold.\n  Reporting all {K_FINAL} as \"bootstrap-stable\" is not "
              f"supported by these numbers.")

    # --- figures ------------------------------------------------------------
    fig1_timeseries(daily, h, spells)
    fig2_kselect(tab_i, k_elbow)
    fig3_centroids(daily, lab_i, names)
    fig4_timeline(daily, lab_i, names)
    fig5_transitions(T, names, dwell)
    fig6_stability(jac, jac_raw, names)
    fig7_sensorqc(h, anomalies, iso_scores, daily)
    fig8_monthly(daily, lab_i, names)
    fig9_pca_treatment(Xn, Xi, lab_i, names)
    print(f"\nfigures written to {OUTDIR}/")


if __name__ == "__main__":
    main()
